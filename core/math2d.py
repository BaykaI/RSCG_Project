from __future__ import annotations
import math
import numpy as np

EPS = 1e-9

def vec(x: float, z: float) -> np.ndarray:
    return np.array([float(x), float(z)], dtype=float)

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < EPS:
        return np.zeros(2, dtype=float)
    return v / n

def perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)

def from_angle_deg(deg: float) -> np.ndarray:
    a = math.radians(float(deg))
    return np.array([math.cos(a), math.sin(a)], dtype=float)

def angle_deg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))
