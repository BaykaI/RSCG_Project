from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .math2d import normalize, norm, resample_polyline

@dataclass
class State:
    missile_pos: np.ndarray
    missile_air_vel: np.ndarray
    missile_ground_vel: np.ndarray
    target_pos: np.ndarray
    target_ground_vel: np.ndarray
    distance: float
    time: float
    theta_deg: float
    eps_deg: float
    jc_deg: float

@dataclass
class Result:
    states: list
    hit: bool
    hit_index: int
    hit_time: float

def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = float(np.arccos(dot))
    cross = a[0] * b[1] - a[1] * b[0]
    return ang if cross >= 0.0 else -ang

def simulate_direct(missile_pos, missile_speed, missile_course_dir, target_pos, target_speed, target_course_dir,
                    wind, dt, hit_threshold, t_max, max_normal_acc):
    states = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()

    mdir = normalize(missile_course_dir)
    tdir = normalize(target_course_dir)

    tv_air = tdir * target_speed
    hit = False
    hit_idx = -1
    hit_time = t_max

    while t <= t_max:
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        # Ограничение скорости поворота по максимальной перегрузке:
        # a_n = V * omega -> omega_max = a_n_max / V
        if missile_speed > 1e-9 and max_normal_acc > 0.0:
            ang_to_los = _signed_angle(mdir, los_dir)
            omega_max = max_normal_acc / missile_speed
            max_turn = omega_max * dt
            if abs(ang_to_los) <= max_turn:
                mdir = los_dir
            else:
                turn = np.sign(ang_to_los) * max_turn
                c = float(np.cos(turn))
                s = float(np.sin(turn))
                rot = np.array([[c, -s], [s, c]], dtype=float)
                mdir = normalize(rot @ mdir)
        else:
            mdir = los_dir

        missile_air = mdir * missile_speed
        missile_ground = missile_air + wind
        target_ground = tv_air + wind

        horiz = np.array([1.0, 0.0], dtype=float)
        eps_deg = np.degrees(_signed_angle(horiz, los_dir))
        theta_deg = np.degrees(_signed_angle(horiz, mdir))
        jc_deg = theta_deg - eps_deg

        states.append(State(
            mp.copy(), missile_air.copy(), missile_ground.copy(),
            tp.copy(), target_ground.copy(), d, t,
            theta_deg, eps_deg, jc_deg
        ))

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            break

        mp = mp + missile_ground * dt
        tp = tp + target_ground * dt
        t += dt

    if not states:
        states.append(State(
            missile_pos.copy(), np.zeros(2), np.zeros(2),
            target_pos.copy(), tv_air.copy() + wind, norm(target_pos - missile_pos), 0.0,
            0.0, 0.0, 0.0
        ))
    return Result(states, hit, hit_idx, hit_time)

def make_equal_parts(result: Result, parts: int):
    missile_points = np.array([s.missile_pos for s in result.states], dtype=float)
    sampled_missile, idx = resample_polyline(missile_points, parts + 1)
    sampled_target = np.array([result.states[min(i, len(result.states)-1)].target_pos for i in idx], dtype=float)
    sampled_times = np.array([result.states[min(i, len(result.states)-1)].time for i in idx], dtype=float)
    sampled_state_idx = np.array([min(i, len(result.states)-1) for i in idx], dtype=int)
    return sampled_missile, sampled_target, sampled_times, sampled_state_idx