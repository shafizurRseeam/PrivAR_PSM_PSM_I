

import pandas as pd
import numpy as np


# -----------------------------
# Great-circle utilities for PSM-I
# -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6_371_000):
    """Vectorized distance in meters between (lat1, lon1) and (lat2, lon2)."""
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    return earth_radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def initial_bearing(lat1, lon1, lat2, lon2):
    """Bearing (radians) from (lat1,lon1) to (lat2,lon2) on a sphere."""
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)
    return np.arctan2(y, x)  # radians    


def destination_point(lat1, lon1, distance_m, bearing_rad, earth_radius=6_371_000):
    """Move from (lat1,lon1) by great-circle distance and bearing; returns (lat2,lon2) in degrees."""
    φ1 = np.radians(lat1)
    λ1 = np.radians(lon1)
    δ  = distance_m / earth_radius
    θ  = bearing_rad

    sinφ2 = np.sin(φ1) * np.cos(δ) + np.cos(φ1) * np.sin(δ) * np.cos(θ)
    φ2 = np.arcsin(np.clip(sinφ2, -1.0, 1.0))

    y = np.sin(θ) * np.sin(δ) * np.cos(φ1)
    x = np.cos(δ) - np.sin(φ1) * np.sin(φ2)
    λ2 = λ1 + np.arctan2(y, x)

    # normalize lon to [-pi, pi]
    λ2 = (λ2 + np.pi) % (2*np.pi) - np.pi

    return np.degrees(φ2), np.degrees(λ2)    