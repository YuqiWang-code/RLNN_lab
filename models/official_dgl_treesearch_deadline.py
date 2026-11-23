"""Deadline evaluation of the upstream DGL-TreeSearch GCN and queue."""
import logging
from time import perf_counter, process_time

import dgl
from logzero import logger
import numpy as np
import torch
from dgl_treesearch.treesearch import _TreeSearch
from dgl_treesearch.utils import _load_model


PROBABILITY_MAPS = 32
BRANCH_LIMITS = (32, 16, 8, 4, 2, 1)
REVISION = "976f53ed126ea64c56ab0aa4563135abae49338b"


class DeadlineTreeSearch(_TreeSearch):
    def __init__(self, graph, neighbors, model, checkpoint, tick, budget, max_maps, step_estimate=0.0):
        super().__init__(pid=0, num_threads=1, queue=None, lock=None, weight_file=checkpoint,
                         pickle_path=None, g=graph, time_budget=max(0.0, tick + budget - perf_counter()),
                         solution_budget=None, self_loop=True, max_prob_maps=max_maps,
                         model_prob_maps=PROBABILITY_MAPS, cuda_devs=["cuda:0"], reduction=False,
                         local_search=False, queue_pruning=False, noise_as_prob_maps=False,
                         weighted_queue_pop=False, optimum_found=False)
        self.model = model
        self.neighbor_map = neighbors
        self.tick = tick
        self.deadline = tick + budget
        self.step_estimate = step_estimate
        self.last_status_report = float("inf")
        self.updates = []

    def load_model(self):
        self.model.eval()

    def generate_neighborhood_map(self):
        pass

    def search_step(self):
        start = perf_counter()
        if start + self.step_estimate >= self.deadline:
            return True
        done = super().search_step()
        self.step_estimate = perf_counter() - start
        return done

    def record(self, nodes):
        ready = perf_counter()
        if len(nodes) and ready <= self.deadline and (not self.updates or len(nodes) > len(self.updates[-1][1])):
            self.updates.append((ready - self.tick, nodes.copy()))

    def record_partial(self, solution):
        selected = solution.ndata["ts_label"].flatten() == 1
        self.record(solution.ndata["id_map"][selected].numpy().flatten())

    def update_best_solution(self, labeled_graph):
        previous = self.best_solution_weight
        super().update_best_solution(labeled_graph)
        if self.best_solution_weight != previous:
            self.record(self.best_solution.flatten())


class OfficialDGLTreeSearch:
    def __init__(self, checkpoint):
        logger.setLevel(logging.WARNING)
        self.checkpoint = checkpoint
        self.model = _load_model(PROBABILITY_MAPS, checkpoint, "cuda:0").eval()
        self.max_maps = PROBABILITY_MAPS
        self.step_estimate = 0.0
        self.steps = 0

    def prepare(self, graph):
        prepared = dgl.graph(tuple(graph.edge_index), num_nodes=graph.num_nodes)
        prepared.ndata["weight"] = torch.ones((graph.num_nodes, 1))
        incoming = [[] for _ in range(graph.num_nodes)]
        outgoing = [[] for _ in range(graph.num_nodes)]
        for source, target in graph.edge_index.T.tolist():
            if source != target:
                incoming[target].append(source)
                outgoing[source].append(target)
        neighbors = {node: incoming[node] + [node] + outgoing[node] + [node] for node in range(graph.num_nodes)}
        return prepared, neighbors

    def warmup(self, prepared):
        list(self.solve(prepared, 1017, perf_counter(), 0.2))
        graph, neighbors = prepared
        with torch.no_grad():
            for limit in BRANCH_LIMITS:
                start = perf_counter()
                search = DeadlineTreeSearch(graph, neighbors, self.model, self.checkpoint, start, 1.0, limit)
                search.start_time = start
                search.start_process_time = process_time()
                search.prepare_queue()
                search.search_step()
                elapsed = perf_counter() - start
                print(f"DGL calibration: {limit} maps in {elapsed * 1000:.2f} ms", flush=True)
                self.max_maps = limit
                self.step_estimate = search.step_estimate
                if elapsed <= 0.065:
                    break

    def solve(self, prepared, seed, tick, clock):
        np.random.seed(seed)
        torch.manual_seed(seed)
        graph, neighbors = prepared
        search = DeadlineTreeSearch(graph, neighbors, self.model, self.checkpoint, tick, clock,
                                    self.max_maps, self.step_estimate)
        search.run()
        self.steps = search.total_solutions
        yield from search.updates

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), pretrained="locally supervised RB-small",
                    upstream_revision=REVISION, dgl=dgl.__version__, probability_maps=PROBABILITY_MAPS,
                    max_maps_per_expansion=self.max_maps, calibration_candidates=BRANCH_LIMITS,
                    calibration_target_ms=65,
                    self_loops=True, workers=1, reduction=False, local_search=False,
                    queue_pruning=False, weighted_queue_pop=False, device="cuda:0",
                    delivery="largest legal complete or partial result after each probability map")
