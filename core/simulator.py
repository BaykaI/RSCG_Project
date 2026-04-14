from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math2d import angle_deg, norm, normalize


@dataclass
class EventState:
    happened: bool = False
    hit: bool = False
    miss: bool = False
    time: float | None = None
    distance: float | None = None
    reason: str = ""


@dataclass
class SimHistory:
    t: list[float] = field(default_factory=list)
    missile_pos: list[np.ndarray] = field(default_factory=list)
    target_pos: list[np.ndarray] = field(default_factory=list)
    missile_air_vel: list[np.ndarray] = field(default_factory=list)
    missile_ground_vel: list[np.ndarray] = field(default_factory=list)
    target_vel: list[np.ndarray] = field(default_factory=list)
    distance: list[float] = field(default_factory=list)
    los_rate: list[float] = field(default_factory=list)
    missile_turn_rate: list[float] = field(default_factory=list)
    proportional_turn_rate: list[float] = field(default_factory=list)
    min_distance: float = float("inf")
    min_distance_time: float | None = None
    min_distance_missile_pos: np.ndarray | None = None
    final_time: float = 0.0
    missile_final_course_deg: float = 0.0
    event: EventState = field(default_factory=EventState)


class DirectGuidanceSimulator:
    def __init__(self, missile, target, params: dict):
        self.missile = missile
        self.target = target
        self.params = dict(params)
        self.time = 0.0
        self.finished = False
        self.history = SimHistory()

        self.target_ref_pos = target.pos.copy()
        self.target_base_speed = norm(target.vel)
        if self.target_base_speed > 1e-9:
            self.target_longitudinal = normalize(target.vel)
        else:
            self.target_longitudinal = np.array([1.0, 0.0], dtype=float)
        self.target_normal = np.array(
            [-self.target_longitudinal[1], self.target_longitudinal[0]],
            dtype=float,
        )

        self.target_u = 0.0
        self.current_los_rate = 0.0
        self.current_missile_turn_rate = 0.0
        self.current_proportional_turn_rate = 0.0

        self._record()

    def replace_runtime_parameters(self, params: dict) -> None:
        self.params.update(params)

    def _guidance_method(self) -> str:
        return str(self.params.get("guidance_method", "Прямой метод"))

    def _wind_vector(self) -> np.ndarray:
        return np.array(self.params["wind"], dtype=float)

    def _ground_velocity(self, vel: np.ndarray) -> np.ndarray:
        return vel + self._wind_vector()

    def _relative_kinematics(self, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        missile_ground_vel = self._ground_velocity(self.missile.vel)
        target_ground_vel = self._ground_velocity(self.target.vel)
        rel_vel = target_ground_vel - missile_ground_vel
        rel_norm_sq = float(np.dot(rel, rel))
        if rel_norm_sq < 1e-12:
            return missile_ground_vel, rel_vel, 0.0, 0.0

        los_rate = float((rel[0] * rel_vel[1] - rel[1] * rel_vel[0]) / rel_norm_sq)
        rel_norm = float(np.sqrt(rel_norm_sq))
        closing_speed = float(-np.dot(rel, rel_vel) / max(rel_norm, 1e-9))
        return missile_ground_vel, rel_vel, los_rate, closing_speed

    def _direct_guidance_direction(self, rel: np.ndarray, current_dir: np.ndarray, dt: float) -> np.ndarray:
        missile_speed = float(self.params["missile_speed"])
        a_n_max = float(self.params.get("missile_max_normal_acc", 1e9))
        los_hat = normalize(rel)
        desired_dir = los_hat
        dot_cd = float(np.clip(np.dot(current_dir, desired_dir), -1.0, 1.0))
        ang = float(np.arccos(dot_cd))
        cross_z = current_dir[0] * desired_dir[1] - current_dir[1] * desired_dir[0]
        if cross_z > 0.0:
            sign = 1.0
        elif cross_z < 0.0:
            sign = -1.0
        else:
            sign = 0.0

        omega_max = a_n_max / max(missile_speed, 1e-9)
        max_turn = omega_max * dt
        if ang > max_turn and max_turn > 0.0:
            turn = sign * max_turn
            c = float(np.cos(turn))
            s = float(np.sin(turn))
            rot = np.array([[c, -s], [s, c]], dtype=float)
            return normalize(rot @ current_dir)
        return desired_dir

    def _proportional_navigation_direction(self, rel: np.ndarray, current_dir: np.ndarray, dt: float) -> np.ndarray:
        a_n_max = float(self.params.get("missile_max_normal_acc", 1e9))
        nav_const = float(self.params.get("navigation_constant", 3.0))

        missile_ground_vel, _, los_rate, closing_speed = self._relative_kinematics(rel)
        missile_ground_speed = norm(missile_ground_vel)
        if missile_ground_speed < 1e-12:
            self.current_los_rate = 0.0
            self.current_missile_turn_rate = 0.0
            self.current_proportional_turn_rate = 0.0
            return current_dir

        # According to the PDF:
        # 1) angular rate of the velocity vector is proportional to LOS rate
        # 2) equivalently, commanded normal acceleration is proportional to
        #    LOS rate in a fixed frame and closing speed.
        commanded_acc = nav_const * closing_speed * los_rate
        commanded_acc = float(np.clip(commanded_acc, -a_n_max, a_n_max))
        missile_turn_rate = commanded_acc / max(missile_ground_speed, 1e-9)
        turn = missile_turn_rate * dt

        self.current_los_rate = float(los_rate)
        self.current_missile_turn_rate = float(missile_turn_rate)
        self.current_proportional_turn_rate = float(nav_const * los_rate)

        c = float(np.cos(turn))
        s = float(np.sin(turn))
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return normalize(rot @ current_dir)

    def _next_missile_direction(self, rel: np.ndarray, current_dir: np.ndarray, dt: float) -> np.ndarray:
        if self._guidance_method() == "Пропорциональное наведение":
            return self._proportional_navigation_direction(rel, current_dir, dt)
        return self._direct_guidance_direction(rel, current_dir, dt)

    def _record(self) -> None:
        rel = self.target.pos - self.missile.pos
        d = norm(rel)
        wind = self._wind_vector()
        if self._guidance_method() == "Пропорциональное наведение":
            _, _, los_rate, _ = self._relative_kinematics(rel)
            self.current_los_rate = float(los_rate)
            self.current_proportional_turn_rate = float(self.params.get("navigation_constant", 3.0)) * self.current_los_rate
        else:
            self.current_los_rate = 0.0
            self.current_missile_turn_rate = 0.0
            self.current_proportional_turn_rate = 0.0

        self.history.t.append(self.time)
        self.history.missile_pos.append(self.missile.pos.copy())
        self.history.target_pos.append(self.target.pos.copy())
        self.history.missile_air_vel.append(self.missile.vel.copy())
        self.history.missile_ground_vel.append((self.missile.vel + wind).copy())
        self.history.target_vel.append(self.target.vel.copy())
        self.history.distance.append(d)
        self.history.los_rate.append(float(self.current_los_rate))
        self.history.missile_turn_rate.append(float(self.current_missile_turn_rate))
        self.history.proportional_turn_rate.append(float(self.current_proportional_turn_rate))
        if d < self.history.min_distance:
            self.history.min_distance = d
            self.history.min_distance_time = self.time
            self.history.min_distance_missile_pos = self.missile.pos.copy()

    def _target_motion_update(self, dt: float):
        mode = self.params.get("target_motion_mode", "stationary")

        if mode == "stationary":
            self.target.vel = np.array([0.0, 0.0], dtype=float)
            self.target.acc = np.array([0.0, 0.0], dtype=float)
            return False

        if mode == "uniform":
            if not self.params.get("target_override_acc", False):
                self.target.acc = np.array([0.0, 0.0], dtype=float)
            return False

        if mode == "accelerated":
            if not self.params.get("target_override_acc", False):
                if np.any(self.target.acc):
                    self.target.vel = self.target.vel + self.target.acc * dt
            return False

        if mode == "sinusoidal":
            amplitude = float(self.params.get("target_sine_amplitude", 1500.0))
            frequency = float(self.params.get("target_sine_frequency", 0.0002))
            wave_number = 2.0 * np.pi * frequency
            phase = float(self.params.get("target_sine_phase", 0.0))
            vmag = max(self.target_base_speed, 1e-9)

            arg = wave_number * self.target_u + phase
            slope = amplitude * wave_number * np.cos(arg)
            ds_du = np.sqrt(1.0 + slope * slope)
            du_dt = vmag / ds_du

            self.target_u += du_dt * dt
            arg = wave_number * self.target_u + phase
            w = amplitude * np.sin(arg)
            slope = amplitude * wave_number * np.cos(arg)
            curv2 = -amplitude * wave_number * wave_number * np.sin(arg)

            self.target.pos = self.target_ref_pos + self.target_longitudinal * self.target_u + self.target_normal * w

            vel = (self.target_longitudinal + self.target_normal * slope) * du_dt
            self.target.vel = vel

            d_du_dt = -(vmag * curv2 * slope) / ((1.0 + slope * slope) ** 2)
            acc = (self.target_longitudinal + self.target_normal * slope) * d_du_dt + self.target_normal * curv2 * (
                du_dt ** 2
            )
            self.target.acc = acc
            return True

        return False

    def step(self) -> None:
        if self.finished:
            return

        dt = float(self.params["dt"])
        hit_threshold = float(self.params["hit_threshold"])
        missile_speed = float(self.params["missile_speed"])

        rel = self.target.pos - self.missile.pos
        d = norm(rel)

        if d <= hit_threshold:
            self.finished = True
            self.history.event = EventState(True, True, False, self.time, d, "Попадание")
            self.history.final_time = self.time
            self.history.missile_final_course_deg = angle_deg(self.missile.vel)
            return

        current_dir = normalize(self.missile.vel)
        if norm(current_dir) < 1e-12:
            current_dir = normalize(rel)
            if norm(current_dir) < 1e-12:
                current_dir = np.array([1.0, 0.0], dtype=float)

        new_dir = self._next_missile_direction(rel, current_dir, dt)
        self.missile.vel = new_dir * missile_speed

        wind = self._wind_vector()
        missile_ground_vel = self.missile.vel + wind

        target_position_already_set = self._target_motion_update(dt)
        target_ground_vel = self.target.vel + wind

        if target_position_already_set:
            self.target.pos = self.target.pos + wind * dt
        else:
            self.target.pos = self.target.pos + target_ground_vel * dt

        self.missile.pos = self.missile.pos + missile_ground_vel * dt

        self.time += dt
        self._record()

        d_new = self.history.distance[-1]
        if d_new <= hit_threshold:
            self.finished = True
            self.history.event = EventState(True, True, False, self.time, d_new, "Попадание")
            self.history.final_time = self.time
            self.history.missile_final_course_deg = angle_deg(self.missile.vel)
            return

        if self.time >= float(self.params["t_max"]) and not self.finished:
            self.finished = True
            self.history.event = EventState(False, False, False, self.time, d_new, "Достигнуто t_max")

        self.history.final_time = self.time
        self.history.missile_final_course_deg = angle_deg(self.missile.vel)
