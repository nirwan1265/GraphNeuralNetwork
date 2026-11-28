"""Run configuration. Defaults are the validated production settings."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import json


@dataclass
class Config:
    # --- topology ---
    top_x: float = 0.10
    keep_weights: bool = False
    symmetric: bool = True          # rank the upper triangle, mirror edges
    laplacian: bool = True          # feed the normalised Laplacian to GCNConv

    # --- evaluation ---
    undirected_eval: bool = True    # score the upper triangle only

    # --- architecture ---
    # "joint": one multi-relational GAE over all algorithms (R-GCN encoder,
    #          DistMult decoder per algorithm) - the default since it beat the
    #          per-algorithm model on all five DREAM4 networks, held-out
    #          networks included.
    # "separate": the original one-GAE-per-algorithm model.
    degree: Optional[float] = None  # average neighbours per gene in each input
                                    # graph; overrides top_x when set
                                    # (top_x = degree / (n - 1)). Use for large
                                    # networks: 10% of 4,511 genes is 1M edges.
    model: str = "joint"
    joint_bases: int = 8            # relations share this many weight bases
    hidden: int = 16
    embedding_dim: int = 16
    dropout: float = 0.01
    decoders: List[str] = field(default_factory=lambda: ["bilinear", "distance"])
                                    # separate model only

    # --- fusion ---
    fusion_weights: str = "agreement"   # "agreement" or "flat"
    agreement_power: float = 2.0
    input_blend: float = 0.5        # share of the rank-average of the raw
                                    # scores in the final output; 0 = model only

    # --- training ---
    epochs: int = 300
    iterations: int = 3
    lr: float = 0.01
    weight_decay: float = 1e-8
    seed: Optional[int] = 0

    # --- execution ---
    device: Optional[str] = None
    prefer_docker: bool = False
    docker_platform: str = "linux/amd64"   # GENECI images are amd64-only;
                                           # required (and slow) on Apple Silicon
    n_jobs: int = 1
    overwrite_inference: bool = False

    # --- output ---
    output_dir: str = "multigcn_output"
    verbose: bool = True

    @classmethod
    def from_file(cls, path) -> "Config":
        """Load from JSON or YAML."""
        text = open(path).read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import yaml
            data = yaml.safe_load(text)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
