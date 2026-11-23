"""Deadline evaluation of the released JAX DiffUCO policy and MIS decoder."""
import pickle
from time import perf_counter

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from EnergyFunctions.MISEnergy import MISEnergyClass
from Networks.DiffModel import DiffModel


SAMPLES = 8
FACTOR = 3


class OfficialDiffUCO:
    def __init__(self, graphs, checkpoint):
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"DiffUCO requires a JAX GPU backend, found {jax.devices()}")
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if isinstance(saved, dict):
            params, cfg = saved["params"], saved["config"]
        else:
            params, cfg = saved[:2]
        self.params = jax.device_put(jax.tree.map(lambda x: x[0], params))
        self.width = max(graph.num_nodes for graph in graphs) + 1
        self.edges = max(graph.edge_index.size(1) for graph in graphs)
        self.diffusion_steps = cfg["n_diffusion_steps"]
        keys = ("n_features_list_prob", "n_features_list_nodes", "n_features_list_edges",
                "n_features_list_messages", "n_features_list_encode", "n_features_list_decode",
                "n_diffusion_steps", "n_message_passes", "message_passing_weight_tied",
                "linear_message_passing", "edge_updates", "n_bernoulli_features", "mean_aggr",
                "n_random_node_features", "train_mode", "graph_norm", "bfloat16", "dataset_name")
        net = {key: cfg[key] for key in keys}
        self.model = DiffModel(**net, n_diff_steps=self.diffusion_steps, problem_type=cfg["problem_name"],
                               EncoderModel=cfg["graph_mode"], time_encoding=cfg.get("time_encoding", "one_hot"))
        self.step = jax.vmap(self.model.make_one_step, in_axes=(None, None, 1, None, 0), out_axes=(1, 0))
        energy = MISEnergyClass(cfg)
        self.decode = jax.vmap(energy.calculate_Energy_CE_p_values, in_axes=(None, 1), out_axes=1)
        self.run = jax.jit(self._run, static_argnums=(2,))
        self.runtime = {}
        self.checkpoint = checkpoint
        self.device = str(jax.devices()[0])

    def prepare(self, graph):
        senders, receivers = graph.edge_index.numpy().astype(np.int32)
        n = graph.num_nodes

        def pad_edges(src, dst):
            e = len(src)
            return jraph.GraphsTuple(nodes=np.zeros((self.width, 1), dtype=np.float32),
                                     edges=(np.arange(self.edges) < e).astype(np.float32)[:, None],
                                     senders=np.pad(src, (0, self.edges - e), constant_values=n),
                                     receivers=np.pad(dst, (0, self.edges - e), constant_values=n),
                                     globals=np.array([n, 0], dtype=np.int32),
                                     n_node=np.array([n, self.width - n], dtype=np.int32),
                                     n_edge=np.array([e, self.edges - e], dtype=np.int32))

        upper = senders < receivers
        return pad_edges(senders, receivers), pad_edges(senders[upper], receivers[upper]), n

    def _run(self, graphs, seed, factor):
        graph, energy_graph = graphs
        key = jax.random.PRNGKey(seed)
        states, _, _, key = self.model.sample_prior(graph, SAMPLES, key)

        def advance(index, state):
            states, _, key = state
            key, batch_key = jax.random.split(key)
            time = jnp.full((self.width, 1), index // factor, dtype=jnp.int16)
            values, _ = self.step(self.params, {"graphs": [graph]}, states, time,
                                  jax.random.split(batch_key, SAMPLES))
            return values["X_next"], values["spin_logits"][..., 1], key

        log_prob = jnp.zeros((self.width, SAMPLES, 1), dtype=jnp.float32)
        _, logits, _ = jax.lax.fori_loop(0, self.diffusion_steps * factor, advance, (states, log_prob, key))
        masks, _, _ = self.decode(energy_graph, jnp.exp(logits))
        return masks

    def warmup(self, prepared):
        graph, energy, _ = prepared
        for factor in (1, FACTOR):
            self.run((graph, energy), np.int32(1017), factor).block_until_ready()
            start = perf_counter()
            self.run((graph, energy), np.int32(1017), factor).block_until_ready()
            self.runtime[factor] = perf_counter() - start
        print(f"DiffUCO warm solve estimate (seconds): {self.runtime}", flush=True)

    def solve(self, prepared, seed, tick, clock):
        graph, energy, n = prepared
        self.steps = 0
        best = -1
        for factor in (1, FACTOR):
            if perf_counter() - tick + self.runtime[factor] >= clock:
                break
            start = perf_counter()
            masks = np.asarray(self.run((graph, energy), np.int32(seed), factor))[:n, :, 0].astype(bool)
            self.steps += self.diffusion_steps * factor
            e = energy.n_edge[0]
            conflicts = (masks[energy.senders[:e]] & masks[energy.receivers[:e]]).any(axis=0)
            sizes = np.where(conflicts, -1, masks.sum(axis=0))
            top = int(sizes.argmax())
            nodes = np.flatnonzero(masks[:, top])
            ready = perf_counter() - tick
            self.runtime[factor] = ready - (start - tick)
            if ready <= clock and sizes[top] > best:
                best = sizes[top]
                yield ready, nodes

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), run_id=self.checkpoint.stem.removeprefix("best_"), samples=SAMPLES,
                    diffusion_steps=self.diffusion_steps, factors=[1, FACTOR],
                    accelerator=self.device, jax=jax.__version__,
                    decoding="official MIS Conditional Expectation")
