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
    hidden: int = 16
    embedding_dim: int = 16
    dropout: float = 0.01
    decoders: List[str] = field(default_factory=lambda: ["bilinear", "distance"])

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
