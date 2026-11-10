"""Pure spectral and fixed-weight intervention helpers for frequency diagnostics."""
import torch
from torch import nn
from torch.nn import functional as F


BANDS = ((0.0, 0.5, "low"), (0.5, 1.0, "mid"),
         (1.0, 1.5, "higher"), (1.5, 2.000001, "extreme"))
BAND_BOUNDARY_TOLERANCE = 1e-10


@torch.no_grad()
def trace_sequential(block, x, *, zero_bias=(), activation=None, skip_norm=False):
    """Trace Sequential layers, optionally simulating changes without mutation."""
    if activation not in (None, "identity"):
        raise ValueError("activation must be None or 'identity'")
    stages = {"input": x.detach()}
    out = x
    for index, layer in enumerate(block):
        if isinstance(layer, nn.Linear) and index in zero_bias:
            out = F.linear(out, layer.weight, None)
        elif isinstance(layer, nn.SiLU) and activation == "identity":
            pass
        elif isinstance(layer, nn.LayerNorm) and skip_norm:
            pass
        else:
            out = layer(out)
        stages[f"stage{index}_{type(layer).__name__}"] = out.detach()
    return stages


@torch.no_grad()
def spectral_basis(edge_index, size):
    """L_sym basis with zero isolated rows and sqrt(max(degree, 1)) weights."""
    adjacency = torch.zeros((size, size), dtype=torch.float64)
    edges = edge_index.detach().cpu()
    adjacency[edges[0], edges[1]] = 1
    degree = adjacency.sum(1)
    inverse = degree.clamp_min(1).rsqrt()
    laplacian = torch.diag((degree > 0).double()) - inverse[:, None] * adjacency * inverse[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvalues.clamp(0, 2), eigenvectors, degree.clamp_min(1).sqrt()


def _spectral_statistics(values, coefficients, basis, eps, relative_floor):
    eigenvalues, _, weight = basis
    energy = coefficients.square().sum(1)
    zero = eigenvalues.abs() <= eps
    nonstationary = energy * (~zero)
    total = float(energy.sum())
    remaining = float(nonstationary.sum())
    valid_total = total > 1e-30
    valid_centered = remaining > max(1e-30, total * relative_floor)
    unweighted = float(values.square().sum())
    constant_energy = float(values.mean(0).square().sum() * values.shape[0])
    weighted_mean = (weight.square()[:, None] * values).sum(0) / weight.square().sum()
    weighted_constant = float(weighted_mean.square().sum() * weight.square().sum())
    result = dict(
        feature_columns=values.shape[1], total_energy=total,
        energy_per_column=total / values.shape[1],
        stationary_energy=float(energy[zero].sum()),
        stationary_fraction=float(energy[zero].sum()) / total if valid_total else None,
        nonstationary_energy=remaining, valid_total=valid_total, valid_centered=valid_centered,
        zero_modes=int(zero.sum()), eigenvalue_max=float(eigenvalues.max()),
        unweighted_total_energy=unweighted, node_constant_energy=constant_energy,
        node_constant_fraction=constant_energy / unweighted if unweighted > 1e-30 else None,
        weighted_global_constant_energy=weighted_constant,
        weighted_global_constant_fraction=weighted_constant / total if valid_total else None,
        raw_spectral_centroid=float((energy * eigenvalues).sum()) / total if valid_total else None,
        centered_spectral_centroid=float((nonstationary * eigenvalues).sum()) / remaining if valid_centered else None,
    )
    band_values = eigenvalues.clone()
    for boundary in (0.5, 1.0, 1.5):
        band_values[(band_values - boundary).abs() <= BAND_BOUNDARY_TOLERANCE] = boundary
    for low, high, name in BANDS:
        mask = (band_values >= low) & (band_values < high)
        raw = float(energy[mask].sum())
        centered = float(nonstationary[mask].sum())
        result.update({f"support_{name}": int(mask.sum()), f"raw_{name}_energy": raw,
                       f"raw_{name}_fraction": raw / total if valid_total else None,
                       f"centered_{name}_energy": centered,
                       f"centered_{name}_fraction": centered / remaining if valid_centered else None})
    return result


@torch.no_grad()
def spectral_energy_many(signals, basis, eps=1e-7, relative_floor=1e-12):
    """Project a dictionary of signals once; aggregate only feature columns, not graphs.

    Stationary removal excludes the entire numerical zero eigenspace. Global
    node-constant statistics use ordinary node means; the weighted counterpart
    uses degree-weighted means. Neither operation changes any model input.
    """
    if not signals:
        return {}
    eigenvalues, eigenvectors, weight = basis
    basis = (eigenvalues.double().cpu(), eigenvectors.double().cpu(), weight.double().cpu())
    parts = [value.detach().to(device="cpu", dtype=torch.float64).reshape(value.shape[0], -1)
             for value in signals.values()]
    sizes = [part.shape[1] for part in parts]
    values = torch.cat(parts, dim=1)
    coefficients = basis[1].T @ (basis[2][:, None] * values)
    output = {}
    offset = 0
    for name, part, size in zip(signals, parts, sizes):
        output[name] = _spectral_statistics(part, coefficients[:, offset:offset + size],
                                           basis, eps, relative_floor)
        offset += size
    return output


def spectral_energy_extended(signal, basis, eps=1e-7, relative_floor=1e-12):
    return spectral_energy_many({"signal": signal}, basis, eps, relative_floor)["signal"]


@torch.no_grad()
def remove_stationary(signal, basis, eps=1e-7):
    """Return a CPU float64 diagnostic signal with every zero mode removed."""
    eigenvalues, eigenvectors, weight = basis
    values = signal.detach().to(device="cpu", dtype=torch.float64).reshape(signal.shape[0], -1)
    coefficients = eigenvectors.T @ (weight[:, None] * values)
    coefficients[eigenvalues.abs() <= eps] = 0
    centered = (eigenvectors @ coefficients) / weight[:, None]
    return centered.reshape(signal.shape)


def node_groups(x, c, r=None):
    groups = dict(all=torch.ones_like(x, dtype=torch.bool),
                  selected_conflicted=(x == 1) & (c > 0),
                  selected_clear=(x == 1) & (c == 0),
                  unselected_free=(x == 0) & (c == 0),
                  unselected_one=(x == 0) & (c == 1),
                  unselected_many=(x == 0) & (c >= 2))
    if r is not None:
        groups["release_positive"] = r > 0
    return groups


@torch.no_grad()
def intervention_details(adapter, x, adj, topology, *, low_scale=1.0, high_scale=1.0,
                         gate=None, zero_bias=None, activation=None, skip_norm=False):
    """Evaluate fixed-weight adapter interventions without editing parameters.

    zero_bias maps branch names to Sequential layer indices, e.g.
    {'high': (0,)}. activation can be 'identity' or a branch mapping.
    skip_norm can be True or a tuple of branch names.
    """
    details = adapter.inspect(x, adj, topology)
    gates = details["gates"] if gate is None else torch.full_like(details["gates"], gate)
    details["gates"] = gates
    for branch, column, scale in (("low", 0, low_scale), ("high", 1, high_scale)):
        biases = (zero_bias or {}).get(branch, ())
        act = activation.get(branch) if isinstance(activation, dict) else activation
        norm = skip_norm if isinstance(skip_norm, bool) else branch in skip_norm
        if biases or act is not None or norm:
            stages = trace_sequential(getattr(adapter, branch), details[f"{branch}_input"],
                                      zero_bias=biases, activation=act, skip_norm=norm)
            details[f"{branch}_trace"] = stages
            details[f"u_{branch}"] = next(reversed(stages.values()))
        enabled = adapter.mode not in (("v4a_high", "v4a_near") if branch == "low" else ("v4a_low",))
        override = getattr(adapter, f"{branch}_override")
        if override is not None:
            enabled = override
        residual = getattr(adapter, f"alpha_{branch}") * gates[..., column:column + 1] * details[f"u_{branch}"]
        details[f"delta_{branch}"] = residual * scale if enabled else torch.zeros_like(residual)
    return details


@torch.no_grad()
def override_logits(model, x, edge_index, adj, topology, **intervention):
    details = intervention_details(model.adapter, x, adj, topology, **intervention)
    hidden = model.initial_project(x.unsqueeze(-1))
    hidden = hidden + details["delta_low"].to(hidden.dtype) + details["delta_high"].to(hidden.dtype)
    return model.project_hidden(hidden, topology.normalized).squeeze(-1)


@torch.no_grad()
def trace_model(model, x, edge_index, adj, topology, **intervention):
    """Return adapter details and encoder/GCN stages for one unchanged state."""
    details = intervention_details(model.adapter, x, adj, topology, **intervention)
    signals = {name: details[name] for name in ("z", "pz", "ppz", "B0", "B1")}
    for branch in ("low", "high"):
        stages = details.get(f"{branch}_trace")
        if stages is None:
            stages = trace_sequential(getattr(model.adapter, branch), details[f"{branch}_input"])
        signals.update({f"{branch}_{name}": value for name, value in stages.items()})
    hidden = model.initial_project(x.unsqueeze(-1))
    signals["raw_hidden"] = hidden
    signals["delta_low"] = details["delta_low"]
    signals["delta_high"] = details["delta_high"]
    hidden = hidden + details["delta_low"].to(hidden.dtype) + details["delta_high"].to(hidden.dtype)
    signals["fused_hidden"] = hidden
    for index, (conv, linear) in enumerate(zip(model.convs, model.linears)):
        hidden = model.activation(conv(hidden, topology.normalized) + linear(hidden)) + hidden
        signals[f"gcn_{index + 1}"] = hidden
    signals["logits"] = model.final_project(hidden).squeeze(-1)
    return details, signals
