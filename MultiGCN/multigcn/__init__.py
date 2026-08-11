"""
MultiGCN - GCN graph-autoencoder ensemble for gene regulatory network inference.

Torch is imported lazily. The registry, loaders, topology, fusion and metrics
modules all work without it, so the algorithm catalogue can be inspected and the
data layer tested on machines with no PyTorch/PyG install.
"""
from .config import Config
from .registry import ALGORITHMS, AlgorithmSpec, Runtime, by_family, by_runtime

__version__ = "0.1.0"
__all__ = ["MultiGCN", "Config", "ALGORITHMS", "AlgorithmSpec",
           "Runtime", "by_family", "by_runtime"]


def __getattr__(name):
    if name == "MultiGCN":
        from .pipeline import MultiGCN as _M
        return _M
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
