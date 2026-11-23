"""v10 destroy-and-repair moves for the frozen policy.

Why this module exists
----------------------
The frozen policy has a fixed point.  Measured (``models/probe_flip_rate.py``):
it flips 1.7-2.2 % of the nodes per step and its density is already at the
plateau after 50 steps, so the remaining ~2800 steps of a 1 s run only jitter.
The one-flip neighbourhood is exhausted there, and the stronger *exact* move
(Koenig recombination over the archived solutions, ``models/mis_v7_recombine``)
returns exactly zero gain on an 80-solution archive -- the archive is
recombine-closed.  Every remedy that keeps the move set (temperature, learned
annealing, exchange heads, PAVE, wide populations, kernel reduction) measured
non-positive.

So the move itself has to change.  This module injects one larger move:
empty a radius-``r`` ball of the incumbent, *forbid* its nodes for a few blocks,
and let the policy refill the hole.  The tabu mask is the only difference from
the v8 refill probe, which walked back into the same solution in 96 of 100
graphs; forbidding the evacuated nodes makes that outcome impossible by
construction, which is exactly the failure mode the move has to break.

Acceptance is free: ``PaddedSearch`` keeps a monotone per-particle archive
(``best_states``/``best_energy``), so a round that repairs worse simply never
replaces the incumbent.
"""


def ball(neighbours, seed, radius):
    """Node bitset of the BFS ball of ``radius`` around ``seed``."""
    frontier = 1 << int(seed)
    seen = frontier
    for _ in range(int(radius)):
        grown = 0
        rest = frontier
        while rest:
            low = rest & -rest
            grown |= neighbours[low.bit_length() - 1]
            rest ^= low
        frontier = grown & ~seen
        seen |= grown
        if not frontier:
            break
    return seen


def nodes_of(mask):
    """Row indices of a node bitset."""
    rows = []
    while mask:
        low = mask & -mask
        rows.append(low.bit_length() - 1)
        mask ^= low
    return rows


class LNSController:
    """Drives the destroy/repair cycle from the block boundary hook.

    One cycle is two ``period`` block boundaries: the first evacuates the ball
    and sets the tabu, the second releases the tabu so the policy can move on.
    Both boundaries are cheap (a bitset BFS plus a ``(rows, K)`` mask write), so
    the cycle can run dozens of times inside a 1 s budget.
    """

    def __init__(self, engine, neighbours, *, radius=2, period=2, particle=0,
                 seed=0, restore=True, verbose=False):
        self.engine = engine
        self.neighbours = neighbours
        self.radius = int(radius)
        self.period = max(1, int(period))
        self.particle = int(particle)
        self.restore = bool(restore)
        self.random = __import__("random").Random(seed)
        self.verbose = verbose
        self.open = False
        self.rounds = 0
        self.evicted = 0
        self.restored = 0

    def _selected(self):
        """Selected rows of the influencing particle, as a bitset."""
        column = self.engine.state[:, self.particle].detach().to("cpu").numpy()
        mask = 0
        for row in range(int(self.engine.nodes)):
            if column[row] > 0.5:
                mask |= 1 << int(row)
        return mask

    def _destroy(self):
        selected = self._selected()
        if not selected:
            return False
        pool = nodes_of(selected)
        seed = pool[self.random.randrange(len(pool))]
        region = ball(self.neighbours, seed, self.radius)
        rows = nodes_of(region)
        self.engine.evict(rows)
        self.rounds += 1
        self.evicted += sum(1 for row in rows if selected >> row & 1)
        return True

    def on_block(self, blocks):
        if blocks % self.period:
            return
        if self.open:
            self.engine.release()
            # Rejection: the archive is monotone per particle, so restoring it
            # turns the cycle into a propose/accept loop that can never report a
            # worse answer than the incumbent it started from.  Without the
            # restore, a too-short repair window accumulates damage (measured:
            # -0.275 at period 2, exactly 0.000 at period 10).
            if self.restore:
                self.engine.state.copy_(self.engine.best_states)
                self.restored += 1
            self.open = False
            return
        if self._destroy():
            self.open = True

    def __repr__(self):
        return (f"LNSController(radius={self.radius}, period={self.period}, "
                f"rounds={self.rounds}, evicted={self.evicted})")
