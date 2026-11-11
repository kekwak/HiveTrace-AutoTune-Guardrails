"""Experiment inventory and explicit ablations; no automatic GPU launches."""

from .variants import spec_for


def matrix(model):
    shared = ["P1", "C1", "C2_reduce", "C2_graph"]
    specific = (["H1", "H2_linear", "H2_conv", "H2_norm", "H2_all", "H4_block"]
                if model == "halo" else ["G1", "G2"])
    rows = []
    for name in shared + specific:
        control = "P1" if name.startswith("H2") else "C1" if name == "C2_reduce" else "B0"
        rows.append({"candidate": name, "control": control, "spec": spec_for(name, model),
                     "status": "prepared_not_gpu_validated"})
    rows.append({"candidate": "H3" if model == "halo" else "G3",
                 "status": "feasibility_probe_only" if model == "halo" else "already_in_baseline",
                 "reason": "vLLM paired adapter awaits T4 feasibility" if model == "halo" else
                           "GLiNER schema object and library word cache already reused"})
    return rows


def ablations(model, combined):
    """Explicit remove-one variants for a user-selected compatible combination."""
    spec_for("X1", model, combined)
    rows = [{"label": "combined", "spec": dict(combined)}]
    for key in combined:
        if key == "dtype":
            if combined[key] == "float32":
                continue
            changed = {**combined, "dtype": "float32"}
        else:
            changed = {k: v for k, v in combined.items() if k != key}
        spec_for("X1", model, changed)
        rows.append({"label": "without_" + key, "spec": changed})
    return rows
