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
    return ang if cross >= 0 else -ang

def resample_polyline(points: np.ndarray, count: int):
    if len(points) < 2:
        return points.copy(), np.zeros(len(points), dtype=int)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= EPS:
        return np.repeat(points[:1], count, axis=0), np.zeros(count, dtype=int)
    targets = np.linspace(0.0, total, count)
    out, out_idx = [], []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and s[j + 1] < t:
            j += 1
        ds = s[j + 1] - s[j]
        alpha = 0.0 if ds <= EPS else (t - s[j]) / ds
        out.append(points[j] * (1.0 - alpha) + points[j + 1] * alpha)
        out_idx.append(j)
    return np.array(out, dtype=float), np.array(out_idx, dtype=int)
