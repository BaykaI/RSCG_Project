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
        return np.array([1.0, 0.0], dtype=float)
    return v / n

def from_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    return np.array([math.cos(a), math.sin(a)], dtype=float)

def signed_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = normalize(a)
    b_n = normalize(b)
    dot = max(-1.0, min(1.0, float(np.dot(a_n, b_n))))
    ang = math.degrees(math.acos(dot))
    cross = a_n[0] * b_n[1] - a_n[1] * b_n[0]
    return ang if cross >= 0.0 else -ang
