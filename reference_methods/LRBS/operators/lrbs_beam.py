"""LRBS 核心算子重写 —— 推理期 beam search + 限步 rollout 打分（MIS 版）。

论文：Scaling Combinatorial Optimization Neural Improvement Heuristics with Online
Search and Adaptation, AAAI 2025, doi 10.1609/aaai.v39i25.34921（本地 PDF:
docs/参考文献/学习型局部搜索/LRBS-AAAI25.pdf）。

该论文未见公开代码（arXiv 摘要只给 anonymous URL），本文件是按论文 §3 与 Algorithm 1
（p3）重写的算子，不声称复现作者实现。对应关系：

* Algorithm 1, p3：以当前解为根做 beam search；beam 宽度 beta 表示每层保留的活跃节点数，
  每个节点扩展 alpha 个子节点 —— 见 beam_improve() 的 beta / alpha 参数。
* §3 “limited rollout”：子节点不用完整 rollout 打分（那等于 MCTS 的 simulated playout，
  代价高），而是用 ns 步的**限步** rollout 估值 —— 见 _limited_rollout()。
* §3 “offline/online adaptation”：用同一搜索过程微调策略参数（论文 §3.2）。本文件只保留
  搜索脚手架，适配部分未实现。

与 PAVE 的关系：PAVE 的 extend 阶段是“在平台解上重复 size+1 交换直到不动点”，搜索宽度
固定为 1（horizon = 1）。本算子把同一个 repair 动作做成**宽度受限的 beam**：horizon 从
1 提到 beta × alpha，但子节点用 ns 步 rollout 排序，避免把预算全花在深 rollout 上。
size+1 交换在实现上 = 强制 v 进入、踢掉 N(v)、并禁止这些被踢点在本次补全中被重新加回
（PAVE 里靠 tabu），见 _moves()。

依赖：只用标准库（random / math），纯 CPU、无 torch。
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set

Graph = Dict[int, Set[int]]


def _is_independent(adj: Graph, sol: Iterable[int]) -> bool:
    s = set(sol)
    return all(not (adj[v] & s) for v in s)


def _greedy_complete(
    adj: Graph,
    forced: Set[int],
    rng: random.Random,
    forbid: FrozenSet[int] = frozenset(),
    noise: float = 0.0,
) -> Set[int]:
    """把强制选择集合补成极大独立集；forbid 里的点不许被加回（对应 PAVE 的 tabu）。"""
    sol = set(forced)
    blocked: Set[int] = set()
    for v in sol:
        blocked |= adj[v]
    cand = [v for v in adj if v not in sol and v not in blocked and v not in forbid]
    rng.shuffle(cand)
    cand.sort(key=lambda v: len(adj[v]) + noise * rng.random())
    for v in cand:
        if v not in blocked:
            sol.add(v)
            blocked |= adj[v]
    return sol


def _limited_rollout(
    adj: Graph, forced: Set[int], ns: int, rng: random.Random, forbid: FrozenSet[int] = frozenset()
) -> float:
    """限步 rollout 估值（论文 §3 的 limited rollout，步数 ns）。

    ns = 0 时只看立即规模；ns >= 补全规模时等于完整 rollout；中间按前 ns 次添加 + 剩余
    可加空间的粗估。估值只用于排序子节点，不要求精确。
    """
    full = _greedy_complete(adj, forced, rng, forbid)
    if ns <= 0:
        return float(len(forced))
    if ns >= len(full):
        return float(len(full))
    sol = set(forced)
    blocked: Set[int] = set()
    for v in sol:
        blocked |= adj[v]
    added = 0
    for v in sorted(full - forced, key=lambda v: len(adj[v])):
        if added >= ns:
            break
        if v not in blocked and v not in forbid:
            sol.add(v)
            blocked |= adj[v]
            added += 1
    remaining = [v for v in adj if v not in sol and v not in blocked and v not in forbid]
    if not remaining:
        return float(len(sol))
    avg_deg = sum(len(adj[v]) for v in remaining) / len(remaining)
    return float(len(sol)) + len(remaining) / (1.0 + avg_deg)


def _moves(adj: Graph, sol: Set[int], rng: random.Random) -> List[tuple[Set[int], FrozenSet[int]]]:
    """生成 (forced, forbid) 动作：对每个可加入点 v 做一次 size+1 交换尝试。

    forced = sol ∪ {v} − N(v)；forbid = N(v) \\ sol（本次补全不许把它们加回，避免像
    v10 里“受禁破坏-修复”那样逐位回到原解）。
    """
    out: List[tuple[Set[int], FrozenSet[int]]] = []
    seen: Set[FrozenSet[int]] = set()
    for v in adj:
        if v in sol:  # 已选点不作为“加入点”
            continue
        kicked = adj[v] & sol
        forced = (sol - kicked) | {v}
        key = frozenset(forced)
        if key in seen:
            continue
        seen.add(key)
        out.append((forced, frozenset(adj[v] - sol)))
    rng.shuffle(out)
    return out


def beam_improve(
    adj: Graph,
    x0: Sequence[int] | Set[int],
    beta: int = 4,
    alpha: int = 4,
    ns: int = 3,
    rounds: int = 1,
    rng: random.Random | None = None,
    accept_equal: bool = True,
) -> Set[int]:
    """LRBS 风格改进：beam(beta) × children(alpha) × 限步 rollout(ns)。

    x0           : 初始独立集（对应 PAVE 的 ref0）
    beta         : 每层保留的活跃节点数（论文 Algorithm 1 的 beam 宽度）
    alpha        : 每个节点扩展的子节点数
    ns           : 限步 rollout 步数（论文 §3；0 表示只看立即收益）
    rounds       : 外层轮数（每轮都保留 beam，等价于在平台上做多步扩展）
    accept_equal : True 时等规模的新平台解也进入 beam（平台搜索；PAVE 的核心假设）
    """
    rng = rng or random.Random(0)
    best = set(x0)
    if not _is_independent(adj, best):
        raise ValueError("x0 不是独立集")
    beam: List[Set[int]] = [best]
    for _ in range(max(1, rounds)):
        scored: List[tuple[float, Set[int]]] = []
        for node in beam:
            for forced, forbid in _moves(adj, node, rng)[:alpha]:
                child = _greedy_complete(adj, forced, rng, forbid)
                scored.append((_limited_rollout(adj, forced, ns, rng, forbid), child))
        if not scored:
            break
        if not accept_equal:
            scored = [(s, c) for s, c in scored if len(c) > len(best)]
        if not scored:
            break
        scored.sort(key=lambda sc: (sc[0], len(sc[1])), reverse=True)
        beam = [c for _, c in scored[:beta]]
        if len(beam[0]) > len(best):
            best = set(beam[0])
    return best


def _random_graph(n: int, deg: int, seed: int) -> Graph:
    rng = random.Random(seed)
    adj: Graph = {i: set() for i in range(n)}
    for i in range(n):
        while len(adj[i]) < deg:
            j = rng.randrange(n)
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _selftest() -> None:
    """三组对照：稀疏/稠密图上的收敛解，以及“人为劣化的输入”。

    第三组模拟我们的实际场景：输入本身次优（这里随机删掉 20% 顶点），看算子能否补回来。
    """
    cases = []
    for deg, drop in ((12, 0.0), (40, 0.0), (40, 0.2)):
        adj = _random_graph(200, deg, 0)
        rng = random.Random(0)
        x = _greedy_complete(adj, set(), rng)
        if drop:
            keep = rng.sample(sorted(x), int(len(x) * (1 - drop)))
            x = _greedy_complete(adj, set(keep), rng)
        cases.append((f"deg={deg} drop={int(drop * 100)}%", adj, x))

    for name, adj, x0 in cases:
        line = [f"{name}: |x0| = {len(x0)}"]
        for beta, alpha, ns, rounds in ((1, 1, 0, 1), (1, 4, 3, 1), (4, 5, 4, 2)):
            y = beam_improve(
                adj, x0, beta=beta, alpha=alpha, ns=ns, rounds=rounds, rng=random.Random(0)
            )
            line.append(f"b{beta}a{alpha}n{ns}r{rounds} -> {len(y)} ({len(y) - len(x0):+d})")
            assert _is_independent(adj, y), "输出不是独立集"
        print(", ".join(line))


if __name__ == "__main__":
    _selftest()
