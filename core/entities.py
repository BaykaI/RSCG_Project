from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BodyState:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
