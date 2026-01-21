# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KernelDensity
# import time
# from scipy.special import lambertw

import numpy as np
from scipy.special import lambertw



def sample_laplace_radius(epsilon: float) -> float:
    """PLM radius sampler: returns a single float."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    u = np.random.rand()
    w = lambertw((u - 1.0) / np.e, k=-1).real
    r = -(w + 1.0) / epsilon
    return float(r)

def sample_psm_radius(epsilon: float, delta: float = 1.0) -> float:
    """PSM radius sampler (annulus-uniform): returns a single float."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0")

    q = np.exp(-epsilon * delta)
    p = 1.0 - q

    I = np.random.geometric(p)          # {1,2,...}
    a = (I - 1) * delta
    b = I * delta

    u = np.random.rand()
    r = np.sqrt(a*a + u*(b*b - a*a))
    return float(r)    





def sample_psm_radius_bounded(epsilon: float, L: float, delta: float = 1.0) -> float:
    """
    Exact bounded PSM radius sampler for the annulus-uniform PSM.
    Samples r ~ PSM conditioned on r <= L.

    - Annulus i has a=(i-1)Δ, b=iΔ, and base prob (1-q)q^{i-1}.
    - Within annulus, r^2 ~ Unif(a^2, b^2).
    - Truncation keeps full mass for i < K, partial mass for i = K, where K = ceil(L/Δ).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")

    q = np.exp(-epsilon * delta)

    # K is the first annulus whose outer radius reaches/exceeds L
    K = int(np.ceil(L / delta))
    aK = (K - 1) * delta
    bK = K * delta  # >= L by construction

    # Fraction of annulus K that lies within radius L (in area measure)
    # because r^2 is uniform in [a^2, b^2]
    fracK = (L*L - aK*aK) / (bK*bK - aK*aK) if L < bK else 1.0
    fracK = float(np.clip(fracK, 0.0, 1.0))

    # weights for annuli 1..K-1 are q^{i-1} (since (1-q) cancels in normalization)
    # weight for annulus K is q^{K-1} * fracK
    if K == 1:
        weights = np.array([fracK], dtype=float)
    else:
        weights = np.concatenate([
            q ** np.arange(0, K-1, dtype=float),          # i = 1..K-1
            np.array([q ** (K-1) * fracK], dtype=float)   # i = K
        ])

    # Normalize and sample I in {1..K}
    weights_sum = weights.sum()
    if weights_sum <= 0:
        # Shouldn't happen unless L is numerically ~0
        raise RuntimeError("No mass under the bound L; check L and delta.")
    probs = weights / weights_sum

    I = 1 + np.searchsorted(np.cumsum(probs), np.random.rand())  # 1..K
    a = (I - 1) * delta
    b = I * delta

    # Sample r inside the chosen annulus, but if it's the last annulus, truncate at L
    if I == K and L < b:
        u = np.random.rand()
        r = np.sqrt(a*a + u*(L*L - a*a))
    else:
        u = np.random.rand()
        r = np.sqrt(a*a + u*(b*b - a*a))

    return float(r)

    





# def sample_staircase_radius_one(epsilon, delta=1.0, L=None, size=1):
#     """
#     Sample the radius r from the infinite (or truncated) planar staircase PDF:
#       f_i ∝ e^{-(i-1)ε}, on interval [(i-1)Δ, iΔ].
#     If L is None, treat as infinite. Otherwise truncate at L and renormalize.
#     """
#     if L is None:
#         p = 1 - np.exp(-epsilon)            # success prob
#         I = np.random.geometric(p, size=size) - 1  # 0-based bin index
#     else:
#         intervals = int(np.ceil(L / delta))
#         k = np.arange(intervals)
#         weights = np.exp(-epsilon * k)
#         weights /= weights.sum()
#         cdf = np.cumsum(weights)
#         u = np.random.rand(size)
#         I = np.searchsorted(cdf, u)
#     r = (I + np.random.rand(size)) * delta
#     return r    

# def sample_staircase_radius(epsilon, delta=1.0, L=None, size=1):
#     """
#     Sample the radius r from the infinite (or truncated) planar staircase PDF:
#       f_i ∝ e^{-(i-1)ε}, on interval [(i-1)Δ, iΔ].
#     If L is None, treat as infinite. Otherwise truncate at L and renormalize.
#     """
#     if L is None:
#         p = 1 - np.exp(-epsilon)                  # success prob
#         I = np.random.geometric(p, size=size) - 1 # 0-based bin index
#     else:
#         intervals = int(np.ceil(L / delta))
#         k = np.arange(intervals)
#         weights = np.exp(-epsilon * k)
#         weights /= weights.sum()
#         cdf = np.cumsum(weights)
#         u = np.random.rand(size)
#         I = np.searchsorted(cdf, u)
    
#     r = (I + np.random.rand(size)) * delta
#     return r.item() if size == 1 else r   
    