from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .math2d import normalize, norm, signed_angle_deg

PROPORTIONAL_METHOD = "Пропорциональное наведение"
DIRECT_LEAD_METHOD = "Прямой с постоянным углом упреждения"
PURSUIT_METHOD = "Метод погони"
PD_METHOD = "Пропорционально-дифференциальный"

@dataclass
class State:
    missile_pos: np.ndarray
    missile_air_vel: np.ndarray
    missile_ground_vel: np.ndarray
    target_pos: np.ndarray
    target_ground_vel: np.ndarray
    distance: float
    time: float
    params: dict = field(default_factory=dict)

@dataclass
class Result:
    states: list[State]
    hit: bool
    hit_index: int
    hit_time: float
    stopped_by_divergence: bool = False
    stop_reason: str = ""
    method: str = "Прямой метод"

def _rotate(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)

def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])

def _miss_params(los: np.ndarray, relative_vel: np.ndarray, missile_speed: float) -> dict:
    d = norm(los)
    if d <= 1e-9:
        los_rate = 0.0
    else:
        los_rate = _cross2(los, relative_vel) / (d * d)
    miss = 0.0 if missile_speed <= 1e-9 else d * d * los_rate / missile_speed
    return {
        "los_rate_rad_s": los_rate,
        "los_rate_deg_s": np.rad2deg(los_rate),
        "miss_value": miss,
        "miss_abs": abs(miss),
    }

def _limit_turn(prev_dir: np.ndarray, desired_dir: np.ndarray, speed: float, dt: float, max_normal_acc: float) -> np.ndarray:
    prev_dir = normalize(prev_dir)
    desired_dir = normalize(desired_dir)
    if speed <= 1e-9 or max_normal_acc <= 0.0:
        return desired_dir
    ang_deg = signed_angle_deg(prev_dir, desired_dir)
    ang = np.deg2rad(ang_deg)
    omega_max = max_normal_acc / speed
    max_turn = omega_max * dt
    if abs(ang) <= max_turn:
        return desired_dir
    return normalize(_rotate(prev_dir, np.sign(ang) * max_turn))

def _append_state(states: list[State], mp: np.ndarray, missile_air: np.ndarray, missile_ground: np.ndarray,
                  tp: np.ndarray, target_ground: np.ndarray, d: float, t: float, params: dict) -> None:
    states.append(
        State(
            missile_pos=mp.copy(),
            missile_air_vel=missile_air.copy(),
            missile_ground_vel=missile_ground.copy(),
            target_pos=tp.copy(),
            target_ground_vel=target_ground.copy(),
            distance=float(d),
            time=float(t),
            params=params,
        )
    )

def simulate_direct_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    mdir_prev = normalize(missile_course_dir)
    tdir = normalize(target_course_dir)

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        desired_dir = los_dir
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        target_air = tdir * target_speed
        missile_ground = missile_air + wind
        target_ground = target_air + wind

        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        theta = signed_angle_deg(np.array([1.0, 0.0]), mdir)
        jc = theta - eps
        q = signed_angle_deg(los_dir, mdir)
        q_c = signed_angle_deg(los_dir, normalize(target_ground))
        miss = _miss_params(los, target_ground - missile_ground, missile_speed)

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(eps_deg=eps, theta_deg=theta, jc_deg=jc, q_deg=q, q_c_deg=q_c, delta=jc, **miss)
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, "Прямой метод")

def simulate_parallel_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    tdir = normalize(target_course_dir)
    mdir_prev = normalize(missile_course_dir)

    eps0 = signed_angle_deg(np.array([1.0, 0.0]), normalize(tp - mp))

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        target_air = tdir * target_speed
        target_ground = target_air + wind
        vt_dir = normalize(target_ground)

        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        q_c = signed_angle_deg(los_dir, vt_dir)
        ratio = 0.0 if missile_speed <= 1e-9 else np.clip(
            norm(target_ground) / missile_speed * np.sin(np.deg2rad(q_c)),
            -1.0,
            1.0,
        )
        q_p = np.rad2deg(np.arcsin(ratio))
        theta = eps0 + q_p

        desired_dir = normalize(np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))], dtype=float))
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        missile_ground = missile_air + wind

        delta = q_p - np.rad2deg(np.arcsin(ratio))
        miss = _miss_params(los, target_ground - missile_ground, missile_speed)

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(eps_deg=eps, eps0_deg=eps0, q_p_deg=q_p, q_c_deg=q_c, delta=delta, theta_deg=theta, **miss)
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, "Параллельное сближение")

def simulate_proportional_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
    navigation_constant: float = 3.0,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    tdir = normalize(target_course_dir)
    mdir_prev = normalize(missile_course_dir)
    nav_const = max(0.0, float(navigation_constant))

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        target_air = tdir * target_speed
        target_ground = target_air + wind
        missile_ground_prev = mdir_prev * missile_speed + wind
        relative_vel = target_ground - missile_ground_prev

        if d <= 1e-9:
            los_rate = 0.0
        else:
            los_rate = _cross2(los, relative_vel) / (d * d)
        closing_speed = -float(np.dot(los_dir, relative_vel))
        effective_closing_speed = max(0.0, closing_speed)
        normal_acc_cmd = nav_const * effective_closing_speed * los_rate
        omega_cmd = 0.0 if missile_speed <= 1e-9 else normal_acc_cmd / missile_speed

        desired_dir = normalize(_rotate(mdir_prev, omega_cmd * dt))
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        missile_ground = missile_air + wind
        rel_after = target_ground - missile_ground

        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        theta = signed_angle_deg(np.array([1.0, 0.0]), mdir)
        q = signed_angle_deg(los_dir, mdir)
        q_c = signed_angle_deg(-los_dir, normalize(target_ground))
        miss = _miss_params(los, rel_after, missile_speed)
        closing_after = -float(np.dot(los_dir, rel_after))
        normal_acc_real = missile_speed * signed_angle_deg(mdir_prev, mdir) * np.pi / 180.0 / dt if dt > 1e-9 else 0.0

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(
                eps_deg=eps,
                theta_deg=theta,
                q_deg=q,
                q_c_deg=q_c,
                closing_speed=closing_after,
                nav_const=nav_const,
                normal_acc_cmd=normal_acc_cmd,
                normal_acc_real=normal_acc_real,
                delta=normal_acc_cmd - normal_acc_real,
                **miss,
            )
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, PROPORTIONAL_METHOD)

def simulate_direct_lead_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
    lead_angle_deg: float = 0.0,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    mdir_prev = normalize(missile_course_dir)
    tdir = normalize(target_course_dir)
    lead_rad = float(np.deg2rad(lead_angle_deg))

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        desired_dir = normalize(_rotate(los_dir, lead_rad))
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        target_air = tdir * target_speed
        missile_ground = missile_air + wind
        target_ground = target_air + wind

        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        theta = signed_angle_deg(np.array([1.0, 0.0]), mdir)
        jc = theta - eps
        q = signed_angle_deg(los_dir, mdir)
        q_c = signed_angle_deg(los_dir, normalize(target_ground))
        miss = _miss_params(los, target_ground - missile_ground, missile_speed)
        normal_acc_real = missile_speed * signed_angle_deg(mdir_prev, mdir) * np.pi / 180.0 / dt if dt > 1e-9 else 0.0

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(
                eps_deg=eps,
                theta_deg=theta,
                jc_deg=jc,
                q_deg=q,
                q_c_deg=q_c,
                lead_angle_deg=float(lead_angle_deg),
                delta=jc - float(lead_angle_deg),
                normal_acc_real=normal_acc_real,
                **miss,
            )
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, DIRECT_LEAD_METHOD)


def _solve_pursuit_course(los_dir: np.ndarray, wind: np.ndarray, missile_speed: float) -> np.ndarray:
    # Найти такой курс ОУ, что путевая скорость направлена вдоль ЛВ:
    # missile_air = k * los_dir - wind, |missile_air| = missile_speed, k > 0.
    # k^2 - 2*(wind·los_dir)*k + (|wind|^2 - V^2) = 0
    b = float(np.dot(wind, los_dir))
    c = float(np.dot(wind, wind)) - float(missile_speed) ** 2
    disc = b * b - c
    if disc < 0.0:
        # Ветер сильнее ОУ — погоня вдоль ЛВ невозможна. Откатываемся на прямой метод.
        return normalize(los_dir)
    root = b + float(np.sqrt(disc))
    if root <= 1e-9:
        root = b - float(np.sqrt(disc))
    if root <= 1e-9:
        return normalize(los_dir)
    air_vel = root * los_dir - wind
    return normalize(air_vel)


def simulate_pursuit_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    mdir_prev = normalize(missile_course_dir)
    tdir = normalize(target_course_dir)

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        desired_dir = _solve_pursuit_course(los_dir, wind, missile_speed)
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        target_air = tdir * target_speed
        missile_ground = missile_air + wind
        target_ground = target_air + wind

        vg_dir = normalize(missile_ground)
        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        theta = signed_angle_deg(np.array([1.0, 0.0]), mdir)
        gamma = signed_angle_deg(np.array([1.0, 0.0]), vg_dir)
        q = signed_angle_deg(los_dir, vg_dir)
        q_c = signed_angle_deg(los_dir, normalize(target_ground))
        miss = _miss_params(los, target_ground - missile_ground, missile_speed)
        normal_acc_real = missile_speed * signed_angle_deg(mdir_prev, mdir) * np.pi / 180.0 / dt if dt > 1e-9 else 0.0

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(
                eps_deg=eps,
                theta_deg=theta,
                gamma_deg=gamma,
                q_deg=q,
                q_c_deg=q_c,
                delta=gamma - eps,
                normal_acc_real=normal_acc_real,
                **miss,
            )
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, PURSUIT_METHOD)


def simulate_pd_discrete(
    missile_pos: np.ndarray,
    missile_speed: float,
    missile_course_dir: np.ndarray,
    target_pos: np.ndarray,
    target_speed: float,
    target_course_dir: np.ndarray,
    wind: np.ndarray,
    dt: float,
    t_max: float,
    hit_threshold: float,
    max_normal_acc: float,
    nav_const_p: float = 3.0,
    nav_const_d: float = 0.5,
) -> Result:
    states: list[State] = []
    t = 0.0
    mp = missile_pos.copy()
    tp = target_pos.copy()
    tdir = normalize(target_course_dir)
    mdir_prev = normalize(missile_course_dir)
    np_const = max(0.0, float(nav_const_p))
    nd_const = max(0.0, float(nav_const_d))

    hit = False
    hit_idx = -1
    hit_time = t_max
    stopped_by_divergence = False
    stop_reason = "Достигнуто t_max"

    steps = int(np.floor(t_max / dt)) + 1
    prev_d: float | None = None
    was_closing = False
    prev_los_rate: float | None = None

    for _ in range(steps):
        los = tp - mp
        d = norm(los)
        los_dir = normalize(los)

        target_air = tdir * target_speed
        target_ground = target_air + wind
        missile_ground_prev = mdir_prev * missile_speed + wind
        relative_vel = target_ground - missile_ground_prev

        if d <= 1e-9:
            los_rate = 0.0
        else:
            los_rate = _cross2(los, relative_vel) / (d * d)
        closing_speed = -float(np.dot(los_dir, relative_vel))
        effective_closing_speed = max(0.0, closing_speed)

        if prev_los_rate is None or dt <= 1e-9:
            los_rate_dot = 0.0
        else:
            los_rate_dot = (los_rate - prev_los_rate) / dt

        normal_acc_cmd = effective_closing_speed * (np_const * los_rate + nd_const * los_rate_dot)
        omega_cmd = 0.0 if missile_speed <= 1e-9 else normal_acc_cmd / missile_speed

        desired_dir = normalize(_rotate(mdir_prev, omega_cmd * dt))
        mdir = _limit_turn(mdir_prev, desired_dir, missile_speed, dt, max_normal_acc)

        missile_air = mdir * missile_speed
        missile_ground = missile_air + wind
        rel_after = target_ground - missile_ground

        eps = signed_angle_deg(np.array([1.0, 0.0]), los_dir)
        theta = signed_angle_deg(np.array([1.0, 0.0]), mdir)
        q = signed_angle_deg(los_dir, mdir)
        q_c = signed_angle_deg(-los_dir, normalize(target_ground))
        miss = _miss_params(los, rel_after, missile_speed)
        closing_after = -float(np.dot(los_dir, rel_after))
        normal_acc_real = missile_speed * signed_angle_deg(mdir_prev, mdir) * np.pi / 180.0 / dt if dt > 1e-9 else 0.0

        _append_state(
            states, mp, missile_air, missile_ground, tp, target_ground, d, t,
            dict(
                eps_deg=eps,
                theta_deg=theta,
                q_deg=q,
                q_c_deg=q_c,
                closing_speed=closing_after,
                nav_const_p=np_const,
                nav_const_d=nd_const,
                lambda_dot_rad_s=los_rate,
                lambda_ddot_rad_s2=los_rate_dot,
                normal_acc_cmd=normal_acc_cmd,
                normal_acc_real=normal_acc_real,
                delta=normal_acc_cmd - normal_acc_real,
                **miss,
            )
        )

        if d <= hit_threshold:
            hit = True
            hit_idx = len(states) - 1
            hit_time = t
            stop_reason = "Условный контакт"
            break

        if prev_d is not None:
            if d < prev_d - 1e-9:
                was_closing = True
            elif was_closing and d > prev_d + 1e-9:
                stopped_by_divergence = True
                stop_reason = "ОУ начал удаляться от ОС"
                break

        tp = tp + target_ground * dt
        mp = mp + missile_ground * dt
        mdir_prev = mdir.copy()
        prev_d = d
        prev_los_rate = los_rate
        t += dt

    return Result(states, hit, hit_idx, hit_time, stopped_by_divergence, stop_reason, PD_METHOD)


def make_step_arrays(result: Result):
    idx = np.arange(len(result.states), dtype=int)
    sampled_missile = np.array([result.states[i].missile_pos for i in idx], dtype=float)
    sampled_target = np.array([result.states[i].target_pos for i in idx], dtype=float)
    sampled_times = np.array([result.states[i].time for i in idx], dtype=float)
    return sampled_missile, sampled_target, sampled_times, idx
