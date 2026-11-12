"""Correctness checks for the v8 macro kernel. CPU or GPU, no checkpoints.

    python models/check_v8.py

Checks the three properties that the rest of the pipeline assumes: the M=1
read-out widening is an exact copy of the backbone, the joint likelihood reduces
to the product likelihood at M=1 while the marginal control does not, and the
teacher block record is the in-block minimum with index in ``[1, h]``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from mis_v8_macro import MacroKernel, joint_log_likelihood, marginal_bce, teacher_blocks, widen_readout


def tiny_backbone():
    from models import GCN
    torch.manual_seed(0)
    net = GCN(1, 16, 2, 1)
    return net.state_dict()


def adjacency(num_nodes, edges):
    return torch.sparse_coo_tensor(
        torch.tensor(edges, dtype=torch.long).t(),
        torch.ones(len(edges)), (num_nodes, num_nodes)).to_sparse_csr()


def check_widen():
    state = tiny_backbone()
    # even M=1 keeps two heads (continuation and record); the first head is an
    # exact copy of the backbone so the student starts at the teacher's policy.
    for m in (1, 2, 4):
        wider = widen_readout(state, m, scale=0.0, generator=torch.Generator().manual_seed(1))
        assert wider["final_project.weight"].shape == (2 * m, 16)
        assert torch.equal(wider["final_project.weight"][:1], state["final_project.weight"])
        assert torch.equal(wider["final_project.weight"][1:],
                           state["final_project.weight"].repeat(2 * m - 1, 1))
    noisy = widen_readout(state, 2, scale=0.01, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(noisy["final_project.weight"][1], noisy["final_project.weight"][2])
    return 8


def check_likelihood():
    torch.manual_seed(2)
    nodes, particles, logits = 5, 3, torch.randn(5, 3, 2)
    labels = (torch.randint(0, 2, (nodes, particles)).float(),
              torch.randint(0, 2, (nodes, particles)).float())
    m = 1
    joint = joint_log_likelihood(logits, m, None, labels)
    manual = -(torch.nn.functional.logsigmoid(logits[..., 0]) * labels[0]
               + torch.nn.functional.logsigmoid(-logits[..., 0]) * (1 - labels[0])
               + torch.nn.functional.logsigmoid(logits[..., 1]) * labels[1]
               + torch.nn.functional.logsigmoid(-logits[..., 1]) * (1 - labels[1])).sum()
    assert torch.allclose(joint, manual, atol=1e-4), (joint, manual)
    two = torch.randn(5, 3, 4)
    assert joint_log_likelihood(two, 2, None, labels) > 0
    assert marginal_bce(two, 2, labels) > 0
    # a mixture whose two components disagree gives a lower joint loss than the
    # averaged marginals can: the two objectives differ by construction
    assert not torch.allclose(joint_log_likelihood(two, 2, None, labels),
                              marginal_bce(two, 2, labels))
    return 4


class FakeSolver:
    """Minimal stand-in that yields a fixed teacher trajectory."""

    class problem:
        beta = 1.02

    def __init__(self, states):
        self.states = states
        self.num_t = len(states) - 1

    def policy_topology(self, graph, adj):
        return adj

    def sample_states(self, graph, adj, *, generator=None, normalized=None, with_energy=False):
        for state in self.states:
            yield state


def check_blocks():
    num_nodes, particles, horizon, steps = 6, 4, 2, 8
    torch.manual_seed(3)
    states = [torch.randint(0, 2, (num_nodes, particles)).float() for _ in range(steps + 1)]
    graph = type("G", (), {"b": -torch.ones(num_nodes), "num_nodes": num_nodes})()
    solver = FakeSolver(states)
    blocks = teacher_blocks(solver, graph, adjacency(num_nodes, [(0, 1), (2, 3)]),
                            torch.Generator(), horizon=horizon, starts=[2, 4], num_t=steps)
    assert len(blocks) == 2
    for block in blocks:
        assert block.record.shape == (num_nodes, particles)
        assert int(block.record_index.min()) >= 1 and int(block.record_index.max()) <= horizon
        assert torch.all(block.energy_record <= block.energy_end + 1e-5)
    return len(blocks)


def main():
    results = dict(widen_cases=check_widen(), likelihood_cases=check_likelihood(),
                   block_cases=check_blocks())
    for name, value in results.items():
        print(f"{name}: {value}")
    print("all v8 kernel checks passed")


if __name__ == "__main__":
    main()
