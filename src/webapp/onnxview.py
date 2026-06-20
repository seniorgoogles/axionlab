"""Inspect ONNX graphs and diff them, to see how a transformation changes the net.

Each FINN step is an ONNX->ONNX transformation; if every step's output is saved
(see transforms.snapshot), this module turns those snapshots into something the
GUI can show:

  - summary(path): the graph as nodes + initializers + IO, with tensor dtypes
  - diff(a, b):    what one step changed -- nodes added/removed, op-type deltas

The heavy `onnx` package is imported lazily, so importing this module is cheap and
the rest of the webapp works even where onnx isn't installed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List


def _safe(repo_root: str, rel_path: str) -> Path:
    p = (Path(repo_root) / rel_path).resolve()
    root = Path(repo_root).resolve()
    if root not in p.parents and p != root:
        raise ValueError("path escapes the repository")
    if not p.exists():
        raise FileNotFoundError(rel_path)
    return p


def _elem_type_name(t: int) -> str:
    from onnx import TensorProto
    return TensorProto.DataType.Name(t) if t else "UNDEFINED"


def summary(repo_root: str, rel_path: str) -> Dict:
    """Parse an ONNX file into a JSON-able summary of its graph."""
    import onnx

    p = _safe(repo_root, rel_path)
    model = onnx.load(str(p))
    g = model.graph

    nodes = [{
        "name": n.name or f"{n.op_type}_{i}",
        "op_type": n.op_type,
        "inputs": list(n.input),
        "outputs": list(n.output),
    } for i, n in enumerate(g.node)]

    def _io(vis):
        out = []
        for v in vis:
            tt = v.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else "?" for d in tt.shape.dim]
            out.append({"name": v.name, "dtype": _elem_type_name(tt.elem_type), "shape": dims})
        return out

    inits = [{"name": init.name, "dtype": _elem_type_name(init.data_type),
              "shape": list(init.dims)} for init in g.initializer]

    return {
        "path": rel_path,
        "op_counts": dict(Counter(n["op_type"] for n in nodes)),
        "n_nodes": len(nodes),
        "n_initializers": len(inits),
        "inputs": _io(g.input),
        "outputs": _io(g.output),
        "nodes": nodes,
        "initializers": inits,
    }


def diff(repo_root: str, rel_a: str, rel_b: str) -> Dict:
    """What changed going from graph A to graph B (before -> after a step)."""
    sa, sb = summary(repo_root, rel_a), summary(repo_root, rel_b)

    a_nodes = {n["name"]: n for n in sa["nodes"]}
    b_nodes = {n["name"]: n for n in sb["nodes"]}
    added = [n for name, n in b_nodes.items() if name not in a_nodes]
    removed = [n for name, n in a_nodes.items() if name not in b_nodes]

    ca, cb = Counter(sa["op_counts"]), Counter(sb["op_counts"])
    op_delta = {}
    for op in set(ca) | set(cb):
        d = cb[op] - ca[op]
        if d:
            op_delta[op] = d

    return {
        "a": rel_a, "b": rel_b,
        "added": added, "removed": removed,
        "n_added": len(added), "n_removed": len(removed),
        "op_delta": op_delta,
        "n_nodes_before": sa["n_nodes"], "n_nodes_after": sb["n_nodes"],
    }
