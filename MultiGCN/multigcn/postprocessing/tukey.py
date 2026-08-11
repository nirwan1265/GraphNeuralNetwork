"""Optional final network extraction via Tukey's IQR outlier rule (Tukey, 1977)."""
from __future__ import annotations

import numpy as np

# k is not universal across decoders - it depends on the shape of that decoder's
# score distribution. Validated: plain inner-product k=3.0, bilinear k=1.5.
# NOT yet validated for the bilinear+distance ensemble output.
RECOMMENDED_K = {"inner_product": 3.0, "bilinear": 1.5}


def tukey_cutoff(scores: np.ndarray, k: float = 1.5,
                 per_gene: bool = True) -> np.ndarray:
    """
    Keep scores above Q3 + k*IQR. Standard boxplot-whisker convention, so no
    tuning against ground truth is required: k=1.5 (mild) and k=3.0 (extreme)
    are both citable defaults.

    Applied per-gene by default, so each gene is judged against its own score
    distribution rather than a global one.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if not per_gene:
        q1, q3 = np.percentile(scores, [25, 75])
        return (scores > q3 + k * (q3 - q1)).astype(np.uint8)

    q1 = np.percentile(scores, 25, axis=1, keepdims=True)
    q3 = np.percentile(scores, 75, axis=1, keepdims=True)
    return (scores > q3 + k * (q3 - q1)).astype(np.uint8)
