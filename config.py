DEFAULT_VALUES = {
    "missile_x": -12000.0,
    "missile_z": -4000.0,
    "missile_speed": 450.0,
    "missile_course_deg": 20.0,
    "missile_max_normal_acc": 80.0,
    "target_x": 0.0,
    "target_z": 0.0,
    "target_vx": 0.0,
    "target_vz": 0.0,
    "target_course_deg": 0.0,
    "target_ax": 0.0,
    "target_az": 0.0,
    "wind_x": 0.0,
    "wind_z": 0.0,
    "dt": 0.05,
    "t_max": 60.0,
    "hit_threshold": 25.0,
    "steps_per_frame": 3.0,
    "frame_delay_ms": 40.0,
    "target_sine_radius": 1500.0,
}
GUIDANCE_METHODS = ["Прямой метод"]
MODES = ["Полный расчет", "Пошаговая анимация"]
SCENARIOS = [
    "Цель стоит",
    "Цель движется равномерно",
    "Цель движется равноускоренно",
    "Цель маневрирует по синусу",
]