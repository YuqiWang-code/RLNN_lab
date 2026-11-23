"""Run the official DISCS MIS model and sampler on the author graph tensors."""
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from discs.experiment.configs.mis import ertest
from discs.models import mis
from discs.models.configs import mis_config
from discs.samplers import path_auxiliary
from discs.samplers.configs import path_auxiliary_config
from ml_collections import ConfigDict


class TensorMIS(mis.MIS):
    def __init__(self, config):
        self.config = config
        self.max_num_nodes = config.max_num_nodes
        self.penalty_coeff = config.penalty


class OfficialISCO:
    block_steps = 10

    def __init__(self, graphs, horizon=300, particles=200, max_steps=None):
        self.max_steps = max_steps
        self.particles = particles
        self.width = max(graph.num_nodes for graph in graphs)
        self.edges = max(graph.edge_index.shape[1] // 2 for graph in graphs)
        config = mis_config.get_config()
        config.shape = (self.width,)
        config.max_num_nodes = self.width
        config.num_categories = 2
        self.model = TensorMIS(config)
        self.sampler = path_auxiliary.build_sampler(ConfigDict(dict(
            model=config, sampler=path_auxiliary_config.get_config())))
        temperature = ertest.get_config().experiment
        self.schedule = optax.exponential_decay(temperature.init_temperature, horizon,
                                                temperature.decay_rate, end_value=temperature.final_temperature)
        self.begin = jax.jit(self._begin)
        self.block = jax.jit(self._block)

    def prepare(self, graph):
        u, v = graph.edge_index.numpy()
        keep = u < v
        n_edges = int(keep.sum())
        return dict(edge_from=np.pad(u[keep].astype(np.int32), (0, self.edges - n_edges)),
                    edge_to=np.pad(v[keep].astype(np.int32), (0, self.edges - n_edges)),
                    edge_mask=(np.arange(self.edges) < n_edges).astype(np.float32),
                    mask=(np.arange(self.width) < graph.num_nodes).astype(np.float32))

    def _begin(self, seed):
        key, init, sampler_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        return key, self.model.get_init_samples(init, self.particles), self.sampler.make_init_state(sampler_key)

    def _block(self, position, graph, chain):
        def step(index, state):
            (key, x, sampler_state), incumbent, best_size = state
            key = jax.random.fold_in(key, index + 1)
            params = {**graph, "temperature": self.schedule(index + 1)}
            x, sampler_state, _ = self.sampler.step(self.model, key, x, params, sampler_state, graph["mask"])
            sizes = self.model.objective(params, x)
            feasible_sizes = jnp.where(self.model.penalty(params, x) == 0, sizes, -1)
            top = jnp.argmax(feasible_sizes)
            size = feasible_sizes[top]
            return (key, x, sampler_state), jnp.where(size > best_size, x[top], incumbent), jnp.maximum(size, best_size)

        initial = chain, jnp.zeros(self.width, dtype=jnp.int32), jnp.float32(-1)
        return jax.lax.fori_loop(position, position + self.block_steps, step, initial)

    def solve(self, graph, seed, tick, clock):
        graph_mask = graph['mask'].astype(bool)
        graph = jax.device_put(graph)
        chain = self.begin(np.int32(seed))
        elapsed = 0.0
        steps = 0
        best = -1
        while (self.max_steps is None or steps < self.max_steps) and perf_counter() - tick + elapsed < clock:
            launched = perf_counter()
            chain, chosen, size = self.block(np.int32(steps), graph, chain)
            mask = np.asarray(chosen, dtype=bool)
            count = int(size)
            nodes = np.flatnonzero(mask & graph_mask)
            finished = perf_counter()
            elapsed = finished - launched
            steps += self.block_steps
            if count >= 0 and count > best and finished - tick <= clock:
                best = count
                yield finished - tick, nodes
        self.steps = steps

    def warmup(self, graph):
        params = jax.device_put(graph)
        chain = self.begin(np.int32(1017))
        jax.block_until_ready(self.block(np.int32(0), params, chain))
