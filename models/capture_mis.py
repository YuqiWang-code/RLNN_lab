"""Replay one MIS sampling step on a fixed graph with the same seeded RNG."""

import torch


@torch.no_grad()
def captured_states(solver, graph, adjacency, topology, generator):
    device = graph.b.device
    initial = torch.randint(0, 2, (graph.num_nodes, solver.num_k), device=device,
                            dtype=solver.dtype, generator=generator)
    state = initial.clone()
    product = adjacency @ state
    beta = solver.problem.beta

    def energy(x, ax):
        return (x * (beta * ax + graph.b)).sum(dim=0)

    def step():
        if solver.v6_frequency:
            logits = solver.model(state, graph.edge_index, adjacency, topology, adjacency_product=product)
        else:
            logits = solver.flip_logits(state, graph, adjacency, topology, adjacency_product=product)
        noise = torch.rand(state.shape, device=device, dtype=state.dtype, generator=generator)
        x = torch.where(noise < logits.sigmoid(), 1 - state, state)
        ax = adjacency @ x
        state.copy_(x)
        product.copy_(ax)
        return x, energy(x, ax)

    # Warm up cuBLAS and cuSPARSE on the capture stream before recording.
    rng_state = generator.get_state()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream(device).wait_stream(stream)
    generator.set_state(rng_state)
    state.copy_(initial)
    product.copy_(adjacency @ initial)
    captured = torch.cuda.CUDAGraph()
    captured.register_generator_state(generator)
    with torch.cuda.graph(captured, stream=stream):
        output, score = step()
    yield initial, energy(initial, product)
    for _ in range(solver.num_t):
        captured.replay()
        # Callers consume the buffers on the current stream before the next replay.
        yield output, score
