import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import time
from scipy.special import lambertw



def sample_laplace_radius(epsilon):
    u = np.random.rand()
    w = lambertw((u - 1) / np.e, k=-1).real
    r = -1.0 / epsilon * (w + 1)
    return r




def sample_staircase_radius(epsilon, delta=1.0, L=None, size=1):
    """
    Sample the radius r from the infinite (or truncated) planar staircase PDF:
      f_i ∝ e^{-(i-1)ε}, on interval [(i-1)Δ, iΔ].
    If L is None, treat as infinite. Otherwise truncate at L and renormalize.
    """
    if L is None:
        p = 1 - np.exp(-epsilon)            # success prob
        I = np.random.geometric(p, size=size) - 1  # 0-based bin index
    else:
        intervals = int(np.ceil(L / delta))
        k = np.arange(intervals)
        weights = np.exp(-epsilon * k)
        weights /= weights.sum()
        cdf = np.cumsum(weights)
        u = np.random.rand(size)
        I = np.searchsorted(cdf, u)
    r = (I + np.random.rand(size)) * delta
    return r    