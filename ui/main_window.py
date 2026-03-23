from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import DEFAULT_VALUES, GUIDANCE_METHODS, MODES, SCENARIOS
from core.entities import BodyState
from core.math2d import vec, from_angle_deg, normalize, perp, norm
from core.simulator import DirectGuidanceSimulator

BG_COLOR = "#313338"
PANEL_BG = "#2B2D31"
SURFACE_BG = "#1E1F22"
TEXT_COLOR = "#F2F3F5"
MUTED_TEXT = "#B5BAC1"
GRID_COLOR = "#4E5058"
ACCENT_COLOR = "#5865F2"
ENTRY_BG = "#1E1F22"
ENTRY_FG = "#F2F3F5"
REPLACE_BG = "#3F4258"
REPLACE_FG = "#F8E58C"

TRAJ_MISSILE_COLOR = "#4EA8FF"
TRAJ_TARGET_COLOR = "#ED4245"
VECTOR_COLOR = "#57F287"
HIT_COLOR = "#A3E635"

class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)
        self.vars = {}
        self.replace_vars = {}
        self.info_var = tk.StringVar(value="Готов к расчету")
        self.pause_info_var = tk.StringVar(value="")
        self.sim = None
        self.paused = False
        self.after_id = None
        self.locked_widgets = []
        self.start_button = None
        self.started = False
        self.slider_vars = {}
        self.slider_value_labels = {}
        self.slider_scales = {}
        self.current_target_motion_mode = "stationary"
        self.sine_params = None
        self.sine_controls_frame = None
        self.sine_amplitude_info_var = tk.StringVar(value="R = -")
        self.sine_frequency_info_var = tk.StringVar(value="f = -")
        self._configure_theme()
        self._build_ui()
        self._apply_values(DEFAULT_VALUES)
        self._apply_scenario_values("Цель стоит")
        self._apply_replace_defaults()
        self._draw_empty()

    def _configure_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.master.configure(bg=BG_COLOR)
        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("Muted.TLabel", background=BG_COLOR, foreground=MUTED_TEXT)
        style.configure("TSeparator", background=GRID_COLOR)
        style.configure("TPanedwindow", background=BG_COLOR)
        style.configure("TLabelframe", background=BG_COLOR, foreground=TEXT_COLOR, borderwidth=1)
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure(
            "TButton",
            background=PANEL_BG,
            foreground=TEXT_COLOR,
            borderwidth=0,
            padding=(8, 5),
        )
        style.map(
            "TButton",
            background=[("active", ACCENT_COLOR), ("pressed", "#4752C4"), ("disabled", "#232428")],
            foreground=[("disabled", "#7A7E87")],
        )
        style.configure(
            "TEntry",
            fieldbackground=ENTRY_BG,
            foreground=ENTRY_FG,
            insertcolor=ENTRY_FG,
            borderwidth=0,
            padding=4,
        )
        style.configure(
            "TCombobox",
            fieldbackground=ENTRY_BG,
            background=PANEL_BG,
            foreground=ENTRY_FG,
            arrowcolor=TEXT_COLOR,
            borderwidth=0,
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", ENTRY_BG)],
            foreground=[("readonly", ENTRY_FG)],
            selectbackground=[("readonly", ACCENT_COLOR)],
            selectforeground=[("readonly", TEXT_COLOR)],
        )

    def _add_live_slider(self, parent, row, key, label, from_, to_, resolution=0.1):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = tk.DoubleVar(value=0.0)
        self.slider_vars[key] = var
        scale = tk.Scale(
            parent,
            variable=var,
            from_=from_,
            to=to_,
            orient="horizontal",
            resolution=resolution,
            showvalue=False,
            length=155,
            command=lambda value, k=key: self._on_slider_change(k, float(value)),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            troughcolor=SURFACE_BG,
            activebackground=ACCENT_COLOR,
            highlightthickness=0,
        )
        scale.grid(row=row, column=1, columnspan=2, sticky="ew", pady=1)
        lbl = ttk.Label(parent, text="0.0")
        lbl.grid(row=row, column=3, sticky="w", padx=(6, 0))
        self.slider_value_labels[key] = lbl
        self.slider_scales[key] = scale
        return row + 1

    def _set_slider_value(self, key, value):
        if key in self.slider_vars:
            self.slider_vars[key].set(float(value))
        if key in self.slider_value_labels:
            fmt = "{:.2f}"
            if key == "target_sine_frequency":
                fmt = "{:.4f}"
            self.slider_value_labels[key].configure(text=fmt.format(float(value)))

    def _sync_sliders_from_model(self):
        if self.sim is None:
            return
        wind = np.array(self.sim.params["wind"], dtype=float)
        self._set_slider_value("missile_speed", self.sim.params.get("missile_speed", norm(self.sim.missile.vel)))
        self._set_slider_value("target_vx", self.sim.target.vel[0])
        self._set_slider_value("target_vz", self.sim.target.vel[1])
        self._set_slider_value("target_ax", self.sim.target.acc[0])
        self._set_slider_value("target_az", self.sim.target.acc[1])
        self._set_slider_value("wind_x", wind[0])
        self._set_slider_value("wind_z", wind[1])
        self._update_sine_output_labels()

    def _target_speed_for_sine(self) -> float:
        return norm(vec(self._f("target_vx"), self._f("target_vz")))

    def _auto_sine_frequency(self, speed: float, target_a_n_max: float) -> float:
        if speed <= 1e-9 or target_a_n_max <= 1e-9:
            return 0.0002
        turn_radius = speed * speed / target_a_n_max
        wavelength = max(200.0, 4.0 * turn_radius)
        return float(np.clip(1.0 / wavelength, 0.0001, 0.0050))

    def _max_sine_normal_acc(self, amplitude: float, frequency: float, speed: float) -> float:
        if amplitude <= 0.0 or frequency <= 0.0 or speed <= 0.0:
            return 0.0
        wave_number = 2.0 * math.pi * frequency
        phi = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
        slope = amplitude * wave_number * np.cos(phi)
        curv2 = -amplitude * wave_number * wave_number * np.sin(phi)
        curvature = np.abs(curv2) / np.power(1.0 + slope * slope, 1.5)
        return float(speed * speed * np.max(curvature))

    def _update_sine_output_labels(self):
        try:
            amplitude = self._f("target_sine_amplitude")
            frequency = self._f("target_sine_frequency")
            self.sine_amplitude_info_var.set(f"R = {amplitude:.1f} м")
            self.sine_frequency_info_var.set(f"f = {frequency:.4f} 1/м")
        except Exception:
            self.sine_amplitude_info_var.set("R = -")
            self.sine_frequency_info_var.set("f = -")

    def _pick_sine_parameters_from_target_acc(self):
        try:
            target_speed = self._target_speed_for_sine()
            target_a_n_max = self._f("target_max_normal_acc")

            if target_speed <= 1e-9:
                raise ValueError("Для расчета параметров скорость ОС должна быть больше нуля.")
            if target_a_n_max <= 0.0:
                raise ValueError("a_n,max ОС должно быть больше нуля.")

            frequency = self._auto_sine_frequency(target_speed, target_a_n_max)
            lo = 0.0
            hi = 100.0
            while self._max_sine_normal_acc(hi, frequency, target_speed) < target_a_n_max and hi < 1e6:
                hi *= 2.0

            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self._max_sine_normal_acc(mid, frequency, target_speed) <= target_a_n_max:
                    lo = mid
                else:
                    hi = mid

            amplitude = lo
            self.vars["target_sine_amplitude"].set(f"{amplitude:.3f}")
            self.vars["target_sine_frequency"].set(f"{frequency:.6f}")
            self._update_sine_output_labels()

            if self.sim is not None and self.paused and self.current_target_motion_mode == "sinusoidal":
                self.sim.params["target_sine_amplitude"] = amplitude
                self.sim.params["target_sine_frequency"] = frequency
                self._render(True)
        except Exception as exc:
            messagebox.showerror("Ошибка расчета синусоиды", str(exc))


    def _on_slider_change(self, key, value):
        self._set_slider_value(key, value)
        if self.sim is None:
            return

        if key == "missile_speed":
            cur = normalize(self.sim.missile.vel)
            if norm(cur) < 1e-9:
                cur = from_angle_deg(self._f("missile_course_deg"))
            self.sim.missile.vel = cur * value
            self.sim.params["missile_speed"] = value

        elif key == "target_vx":
            self.sim.target.vel[0] = value
            self.sim.params["target_override_vel"] = True
        elif key == "target_vz":
            self.sim.target.vel[1] = value
            self.sim.params["target_override_vel"] = True
        elif key == "target_ax":
            self.sim.target.acc[0] = value
            self.sim.params["target_override_acc"] = True
        elif key == "target_az":
            self.sim.target.acc[1] = value
            self.sim.params["target_override_acc"] = True
        elif key == "wind_x":
            self.sim.params["wind"][0] = value
        elif key == "wind_z":
            self.sim.params["wind"][1] = value
        self._render(True)

    def _build_ui(self):
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        # Левая область со скроллингом
        self.left_container = ttk.Frame(paned, style="TFrame")
        self.right_panel = ttk.Frame(paned, padding=8, style="TFrame")
        paned.add(self.left_container, weight=0)
        paned.add(self.right_panel, weight=1)

        self.left_container.configure(width=650)
        self.left_container.pack_propagate(False)

        self.left_canvas = tk.Canvas(self.left_container, highlightthickness=0, bg=BG_COLOR)
        self.left_scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_scrollbar.pack(side="right", fill="y")
        self.left_canvas.pack(side="left", fill="both", expand=True)

        self.left_panel = ttk.Frame(self.left_canvas, padding=8, style="TFrame")
        self.left_window_id = self.left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        def _on_left_configure(event=None):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self.left_canvas.itemconfigure(self.left_window_id, width=event.width)

        self.left_panel.bind("<Configure>", _on_left_configure)
        self.left_canvas.bind("<Configure>", _on_canvas_configure)

        # Колесо мыши над левой панелью
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_left()
        self._build_right()
    def _build_left(self):
        p = self.left_panel
        p.columnconfigure(1, weight=1)
        p.columnconfigure(2, weight=1)
        p.columnconfigure(3, weight=0)
        row = 0

        def combo(label, values, key, default):
            nonlocal row
            ttk.Label(p, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            self.vars[key] = var
            cb = ttk.Combobox(p, textvariable=var, values=values, state="readonly", width=24)
            cb.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
            self.locked_widgets.append(cb)
            row += 1
            return cb

        def sep(title):
            nonlocal row
            ttk.Separator(p, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 6))
            row += 1
            ttk.Label(p, text=title).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1

        def entry(key, label, replace_key=None, parent=None):
            nonlocal row
            grid_parent = p if parent is None else parent
            ttk.Label(grid_parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            var = tk.StringVar()
            self.vars[key] = var
            e = ttk.Entry(grid_parent, textvariable=var, width=14)
            e.grid(row=row, column=1, sticky="ew", pady=1)
            self.locked_widgets.append(e)

            if replace_key is not None:
                rvar = tk.StringVar()
                self.replace_vars[replace_key] = rvar
                re = tk.Entry(
                    grid_parent,
                    textvariable=rvar,
                    width=14,
                    bg=REPLACE_BG,
                    fg=REPLACE_FG,
                    insertbackground=REPLACE_FG,
                    relief="flat",
                    highlightthickness=1,
                    highlightbackground="#7c651d",
                    highlightcolor=ACCENT_COLOR,
                )
                re.grid(row=row, column=2, sticky="ew", pady=1, padx=(8,0))
            else:
                ttk.Label(grid_parent, text="").grid(row=row, column=2, sticky="ew")
            row += 1

        combo("Метод наведения", GUIDANCE_METHODS, "guidance_method", GUIDANCE_METHODS[0])
        combo("Режим", MODES, "mode", MODES[1])
        cb = combo("Сценарий", SCENARIOS, "scenario", SCENARIOS[0])
        cb.bind("<<ComboboxSelected>>", self._on_scenario)

        # Внутренние параметры синусоидального движения храним без прямого редактирования в UI.
        for hidden_key in ("target_sine_amplitude", "target_sine_frequency", "target_sine_phase"):
            if hidden_key not in self.vars:
                self.vars[hidden_key] = tk.StringVar()

        sep("ОУ")
        entry("missile_x", "x, м")
        entry("missile_z", "z, м")
        entry("missile_speed", "скорость, м/с", "missile_speed_new")
        entry("missile_course_deg", "курс, град")
        entry("missile_max_normal_acc", "a_n max, м/с²")
        entry("navigation_constant", "N")

        sep("ОС")
        entry("target_x", "x, м")
        entry("target_z", "z, м")
        entry("target_vx", "Vx, м/с", "target_vx_new")
        entry("target_vz", "Vz, м/с", "target_vz_new")
        entry("target_course_deg", "курс ОС, град")
        entry("target_ax", "ax, м/с²", "target_ax_new")
        entry("target_az", "az, м/с²", "target_az_new")

        sep("Среда и расчет")
        entry("wind_x", "ветер x, м/с", "wind_x_new")
        entry("wind_z", "ветер z, м/с", "wind_z_new")
        entry("dt", "шаг dt, с")
        entry("t_max", "t_max, с")
        entry("hit_threshold", "порог, м")
        entry("steps_per_frame", "шагов за кадр")
        entry("frame_delay_ms", "задержка кадра, мс")

        ttk.Label(p, text="Желтые поля справа: можно менять в паузе скорости, ускорения, ветер и параметры синусоиды цели", style="Muted.TLabel").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        row += 1

        self.sine_controls_frame = ttk.LabelFrame(
            p,
            text="Параметры синусоидального движения",
            padding=6,
        )
        self.sine_controls_frame.columnconfigure(1, weight=1)
        self.sine_controls_frame.columnconfigure(2, weight=1)
        self.sine_controls_frame.columnconfigure(3, weight=0)
        self.sine_controls_frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        sine_row_start = row
        row = 0
        entry("target_max_normal_acc", "a_n,max ОС, м/с²", parent=self.sine_controls_frame)
        entry("target_sine_phase", "фаза синуса, рад", "target_sine_phase_new", parent=self.sine_controls_frame)
        ttk.Button(
            self.sine_controls_frame,
            text="Рассчитать R и f",
            command=self._pick_sine_parameters_from_target_acc,
        ).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 6))
        row += 1
        ttk.Label(self.sine_controls_frame, textvariable=self.sine_amplitude_info_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        row += 1
        ttk.Label(self.sine_controls_frame, textvariable=self.sine_frequency_info_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        row += 1
        row = self._add_live_slider(self.sine_controls_frame, row, "target_sine_phase", "фаза синуса, рад", 0.0, 6.3, 0.01)
        row = sine_row_start + 1

        ttk.Separator(p, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 4))
        row += 1
        ttk.Label(p, text="Слайдеры для изменения в паузе").grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        row = self._add_live_slider(p, row, "missile_speed", "V ОУ, м/с", 50.0, 1500.0)
        row = self._add_live_slider(p, row, "target_vx", "Vx ОС, м/с", -500.0, 500.0)
        row = self._add_live_slider(p, row, "target_vz", "Vz ОС, м/с", -500.0, 500.0)
        row = self._add_live_slider(p, row, "target_ax", "ax ОС, м/с²", -30.0, 30.0)
        row = self._add_live_slider(p, row, "target_az", "az ОС, м/с²", -30.0, 30.0)
        row = self._add_live_slider(p, row, "wind_x", "ветер x, м/с", -150.0, 150.0)
        row = self._add_live_slider(p, row, "wind_z", "ветер z, м/с", -150.0, 150.0)
        
        btn = ttk.Frame(p)
        btn.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for i in range(5):
            btn.columnconfigure(i, weight=1)
        self.start_button = ttk.Button(btn, text="Старт", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(btn, text="Пауза", command=self.pause).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(btn, text="Продолжить", command=self.resume).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(btn, text="Заменить", command=self.replace_parameters).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(btn, text="Очистить", command=self.clear_plots).grid(row=0, column=4, sticky="ew", padx=2)
        row += 1

        ttk.Label(p, textvariable=self.info_var, justify="left", wraplength=500).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

    def _build_right(self):
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.columnconfigure(1, weight=0)
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor=PANEL_BG)
        self.ax_traj = self.figure.add_subplot(211)
        self.ax_dist = self.figure.add_subplot(212)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.38)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        ttk.Label(self.right_panel, textvariable=self.pause_info_var, justify="left", wraplength=220).grid(
            row=0, column=1, sticky="ne", padx=(10,0), pady=(10,0)
        )

    def _random_realistic_velocity(self):
        speed = random.uniform(120.0, 320.0)
        angle = random.uniform(-35.0, 35.0)
        v = from_angle_deg(angle) * speed
        return float(v[0]), float(v[1]), float(angle)

    def _random_realistic_acceleration(self):
        return random.uniform(-12.0, 12.0), random.uniform(-12.0, 12.0)


    def _apply_scenario_values(self, scenario_name: str):
        vals = dict(DEFAULT_VALUES)
        vals["target_x"] = 0.0
        vals["target_z"] = 0.0
        vals["target_course_deg"] = 0.0
        vals["target_sine_amplitude"] = 1500.0
        vals["target_sine_frequency"] = 0.0002
        vals["target_sine_phase"] = 0.0
        vals["target_max_normal_acc"] = 10.0
        self.current_target_motion_mode = "stationary"
        self.sine_params = None

        if scenario_name == "Цель стоит":
            vals["target_vx"] = 0.0
            vals["target_vz"] = 0.0
            vals["target_ax"] = 0.0
            vals["target_az"] = 0.0
            vals["t_max"] = 60.0
            self.current_target_motion_mode = "stationary"

        elif scenario_name == "Цель движется равномерно":
            vx, vz, course = self._random_realistic_velocity()
            vals["target_vx"] = vx
            vals["target_vz"] = vz
            vals["target_course_deg"] = course
            vals["target_ax"] = 0.0
            vals["target_az"] = 0.0
            vals["t_max"] = 80.0
            self.current_target_motion_mode = "uniform"

        elif scenario_name == "Цель движется равноускоренно":
            vx, vz, course = self._random_realistic_velocity()
            ax, az = self._random_realistic_acceleration()
            vals["target_vx"] = vx
            vals["target_vz"] = vz
            vals["target_course_deg"] = course
            vals["target_ax"] = ax
            vals["target_az"] = az
            vals["t_max"] = 80.0
            self.current_target_motion_mode = "accelerated"

        elif scenario_name == "Цель маневрирует по синусу":
            vx, vz, course = self._random_realistic_velocity()
            amplitude = random.uniform(800.0, 3000.0)
            frequency = random.uniform(0.0001, 0.0003)
            vals["target_vx"] = vx
            vals["target_vz"] = vz
            vals["target_course_deg"] = course
            vals["target_ax"] = 0.0
            vals["target_az"] = 0.0
            vals["target_sine_amplitude"] = amplitude
            vals["target_sine_frequency"] = frequency
            vals["target_sine_phase"] = random.uniform(0.0, 2.0 * math.pi)
            vals["target_max_normal_acc"] = self._max_sine_normal_acc(amplitude, frequency, norm(vec(vx, vz)))
            vals["t_max"] = 80.0
            self.current_target_motion_mode = "sinusoidal"
            self.sine_params = {
                "target_sine_frequency": frequency,
                "target_sine_phase": vals["target_sine_phase"],
            }

        self._apply_values(vals)
        self._update_sine_output_labels()
        self._update_sine_controls_visibility()
    def _set_initial_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        combo_state = "disabled" if locked else "readonly"
        for w in self.locked_widgets:
            if isinstance(w, ttk.Combobox):
                w.configure(state=combo_state)
            else:
                w.configure(state=state)

    def _on_scenario(self, event=None):
        self._apply_scenario_values(self.vars["scenario"].get())
        self._apply_replace_defaults()

    def _update_sine_controls_visibility(self):
        if self.sine_controls_frame is None:
            return
        if self.current_target_motion_mode == "sinusoidal":
            self.sine_controls_frame.grid()
        else:
            self.sine_controls_frame.grid_remove()

    def _apply_values(self, values):
        for k, v in values.items():
            if k in self.vars:
                self.vars[k].set(str(round(v, 3) if isinstance(v, float) else v))


    def _apply_replace_defaults(self):
        mapping = {
            "missile_speed_new": "missile_speed",
            "target_vx_new": "target_vx",
            "target_vz_new": "target_vz",
            "target_ax_new": "target_ax",
            "target_az_new": "target_az",
            "wind_x_new": "wind_x",
            "wind_z_new": "wind_z",
        }
        for rk, bk in mapping.items():
            if rk in self.replace_vars and bk in self.vars:
                self.replace_vars[rk].set(self.vars[bk].get())
    def _f(self, key):
        return float(self.vars[key].get().strip().replace(",", "."))

    def _fr(self, key):
        return float(self.replace_vars[key].get().strip().replace(",", "."))


    def _runtime_params(self):
        params = {
            "guidance_method": self.vars["guidance_method"].get(),
            "missile_speed": self._f("missile_speed"),
            "missile_max_normal_acc": self._f("missile_max_normal_acc"),
            "navigation_constant": self._f("navigation_constant"),
            "wind": vec(self._f("wind_x"), self._f("wind_z")),
            "dt": self._f("dt"),
            "t_max": self._f("t_max"),
            "hit_threshold": self._f("hit_threshold"),
            "target_motion_mode": self.current_target_motion_mode,
        }
        if self.sine_params is not None:
            amplitude = self._f("target_sine_amplitude")
            frequency = self._f("target_sine_frequency")
            phase = self._f("target_sine_phase")
            self.sine_params["target_sine_frequency"] = frequency
            self.sine_params["target_sine_phase"] = phase
            params.update({
                "target_sine_frequency": frequency,
                "target_sine_amplitude": amplitude,
                "target_sine_phase": phase,
            })
        return params

    def _build_states(self):
        missile = BodyState(
            pos=vec(self._f("missile_x"), self._f("missile_z")),
            vel=from_angle_deg(self._f("missile_course_deg")) * self._f("missile_speed"),
            acc=vec(0.0, 0.0),
        )

        tv_raw = vec(self._f("target_vx"), self._f("target_vz"))
        tv_mag = norm(tv_raw)
        tcourse = self._f("target_course_deg")
        if tv_mag > 1e-9:
            tv = from_angle_deg(tcourse) * tv_mag
        else:
            tv = vec(0.0, 0.0)

        target = BodyState(
            pos=vec(self._f("target_x"), self._f("target_z")),
            vel=tv,
            acc=vec(self._f("target_ax"), self._f("target_az")),
        )
        return missile, target, self._runtime_params()
    def _cancel_after(self):
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def start(self):
        try:
            self._cancel_after()
            self.paused = False
            missile, target, params = self._build_states()
            self.sim = DirectGuidanceSimulator(missile, target, params)
            self._set_initial_locked(True)
            self.started = True
            if self.start_button is not None:
                self.start_button.configure(state="disabled")
            self._sync_sliders_from_model()
            if self.vars["mode"].get() == "Полный расчет":
                while self.sim is not None and not self.sim.finished:
                    self.sim.step()
                self._render(False)
            else:
                self._animate_tick()
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def pause(self):
        self.paused = True
        self._cancel_after()
        self._sync_sliders_from_model()
        self._render(True)

    def resume(self):
        if self.sim is None:
            return
        if self.sim.finished:
            self._render(False)
            return
        self.paused = False
        self._animate_tick()


    def replace_parameters(self):
        if self.sim is None:
            return
        try:
            if not self.paused:
                self.pause()

            new_speed = self._fr("missile_speed_new")
            current_dir = normalize(self.sim.missile.vel)
            if norm(current_dir) < 1e-9:
                current_dir = from_angle_deg(self._f("missile_course_deg"))
            self.sim.missile.vel = current_dir * new_speed

            self.sim.target.vel = vec(self._fr("target_vx_new"), self._fr("target_vz_new"))
            self.sim.target.acc = vec(self._fr("target_ax_new"), self._fr("target_az_new"))

            new_params = {
                "guidance_method": self.vars["guidance_method"].get(),
                "missile_speed": new_speed,
                "missile_max_normal_acc": self._f("missile_max_normal_acc"),
                "navigation_constant": self._f("navigation_constant"),
                "wind": vec(self._fr("wind_x_new"), self._fr("wind_z_new")),
                "dt": self._f("dt"),
                "t_max": self._f("t_max"),
                "hit_threshold": self._f("hit_threshold"),
            }
            if self.current_target_motion_mode == "sinusoidal":
                new_params.update({
                    "target_sine_amplitude": self._f("target_sine_amplitude"),
                    "target_sine_frequency": self._f("target_sine_frequency"),
                    "target_sine_phase": self._f("target_sine_phase"),
                })
            self.sim.replace_runtime_parameters(new_params)
            self.info_var.set("Заменены скорости, ускорения, ветер и параметры синусоидального движения. Курс, dt, t_max, порог и координаты не изменялись.")
            self._render(True)
        except Exception as exc:
            messagebox.showerror("Ошибка замены параметров", str(exc))
    def clear_plots(self):
        self._cancel_after()
        self.sim = None
        self.paused = False
        self.started = False
        self.pause_info_var.set("")
        self._set_initial_locked(False)
        if self.start_button is not None:
            self.start_button.configure(state="normal")
        self._draw_empty()
        for key in list(self.slider_vars.keys()):
            self._set_slider_value(key, 0.0)
        self.info_var.set("Очищено")

    def _animate_tick(self):
        if self.sim is None or self.paused:
            return
        steps = max(1, int(round(self._f("steps_per_frame"))))
        for _ in range(steps):
            if self.sim.finished:
                break
            self.sim.step()
        self._render(False)
        if self.sim is not None and not self.sim.finished and not self.paused:
            self.after_id = self.after(max(1, int(round(self._f("frame_delay_ms")))), self._animate_tick)

    def _missile_patch(self, pos: np.ndarray, vel: np.ndarray, size: float) -> patches.Polygon:
        direction = vel / max(norm(vel), 1e-9)
        left = np.array([-direction[1], direction[0]])
        nose = pos + direction * size
        tail = pos - direction * 0.7 * size
        p1 = tail + left * 0.25 * size
        p2 = tail - left * 0.25 * size
        return patches.Polygon([nose, p1, p2], closed=True, facecolor=TRAJ_MISSILE_COLOR, edgecolor=TRAJ_MISSILE_COLOR, zorder=7)

    def _aircraft_patch(self, pos: np.ndarray, vel: np.ndarray, size: float) -> patches.Polygon:
        sp = norm(vel)
        if sp < 1e-9:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = vel / sp
        left = np.array([-direction[1], direction[0]])
        nose = pos + direction * size
        tail = pos - direction * 0.7 * size
        wing_l = pos + left * 0.65 * size
        wing_r = pos - left * 0.65 * size
        tail_l = tail + left * 0.25 * size
        tail_r = tail - left * 0.25 * size
        pts = [nose, wing_l, pos + direction * 0.05 * size, tail_l, tail, tail_r, pos + direction * 0.05 * size, wing_r]
        return patches.Polygon(pts, closed=True, facecolor="white", edgecolor="black", linewidth=1.2, zorder=7)

    def _draw_empty(self):
        self.figure.clear()
        self.figure.set_facecolor(PANEL_BG)
        self.ax_traj = self.figure.add_subplot(211)
        self.ax_dist = self.figure.add_subplot(212)
        self._style_axes(self.ax_traj)
        self._style_axes(self.ax_dist)
        self.ax_traj.set_title("Траектории в плоскости OXZ")
        self.ax_traj.set_xlabel("x, м")
        self.ax_traj.set_ylabel("z, м")
        self.ax_traj.grid(True)
        self.ax_traj.set_aspect("auto")
        self.ax_traj.set_xlim(-1000.0, 1000.0)
        self.ax_traj.set_ylim(-1000.0, 1000.0)
        self.ax_dist.set_title("Дальность во времени")
        self.ax_dist.set_xlabel("t, с")
        self.ax_dist.set_ylabel("d, м")
        self.ax_dist.grid(True)
        self.ax_dist.set_xlim(0.0, 1.0)
        self.ax_dist.set_ylim(0.0, 1.0)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.38)
        self.canvas.draw()

    def _style_axes(self, ax):
        ax.set_facecolor(SURFACE_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.65)

    def _draw_explosion(self, x, z):
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        x0, x1 = self.ax_traj.get_xlim()
        z0, z1 = self.ax_traj.get_ylim()
        span = max(abs(x1 - x0), abs(z1 - z0))
        scale = max(0.012 * span, 90.0)
        radii = np.where(np.arange(12) % 2 == 0, scale, 0.45 * scale)
        points = np.column_stack((x + radii * np.cos(angles), z + radii * np.sin(angles)))
        patch = patches.Polygon(points, closed=True, facecolor="none", edgecolor=HIT_COLOR, linewidth=1.5, zorder=8)
        self.ax_traj.add_patch(patch)
        self.ax_traj.plot([x], [z], marker="o", markersize=7, color=HIT_COLOR, linestyle="None", zorder=9, label="Точка встречи")

    def _draw_icons(self, missile_xy, target_xy):
        missile_xy = np.array(missile_xy, dtype=float)
        target_xy = np.array(target_xy, dtype=float)

        x0, x1 = self.ax_traj.get_xlim()
        z0, z1 = self.ax_traj.get_ylim()
        span = max(abs(x1 - x0), abs(z1 - z0))
        scale = max(0.012 * span, 90.0)

        if self.sim is not None:
            mvel = self.sim.missile.vel + np.array(self.sim.params["wind"], dtype=float)
            tvel = self.sim.target.vel + np.array(self.sim.params["wind"], dtype=float)
        else:
            mvel = np.array([1.0, 0.0], dtype=float)
            tvel = np.array([1.0, 0.0], dtype=float)

        self.ax_traj.add_patch(self._missile_patch(missile_xy, mvel, scale))
        self.ax_traj.add_patch(self._aircraft_patch(target_xy, tvel, 1.15 * scale))

    def _draw_pause_vectors(self):
        if self.sim is None:
            return
        mp = self.sim.missile.pos.copy()
        tp = self.sim.target.pos.copy()
        vg_m = self.sim.missile.vel + np.array(self.sim.params["wind"], dtype=float)
        vg_t = self.sim.target.vel + np.array(self.sim.params["wind"], dtype=float)
        mt = normalize(vg_m)
        mn = perp(mt)
        tt = normalize(vg_t)
        tn = perp(tt)

        scale_m = max(200.0, norm(vg_m) * 2.0)
        scale_t = max(200.0, norm(vg_t) * 2.0)

        self.ax_traj.arrow(mp[0], mp[1], mt[0]*scale_m, mt[1]*scale_m, color=VECTOR_COLOR,
                           width=12.0, head_width=120.0, length_includes_head=True, zorder=6)
        self.ax_traj.arrow(mp[0], mp[1], mn[0]*scale_m*0.6, mn[1]*scale_m*0.6, color=VECTOR_COLOR,
                           width=8.0, head_width=90.0, length_includes_head=True, zorder=6)

        tau_m = mp + mt * scale_m * 1.08
        nrm_m = mp + mn * scale_m * 0.72
        self.ax_traj.text(tau_m[0], tau_m[1], "τ_ОУ", color=VECTOR_COLOR, fontsize=10, zorder=9)
        self.ax_traj.text(nrm_m[0], nrm_m[1], "n_ОУ", color=VECTOR_COLOR, fontsize=10, zorder=9)

        self.ax_traj.arrow(tp[0], tp[1], tt[0]*scale_t, tt[1]*scale_t, color=VECTOR_COLOR,
                           width=12.0, head_width=120.0, length_includes_head=True, zorder=6)
        self.ax_traj.arrow(tp[0], tp[1], tn[0]*scale_t*0.6, tn[1]*scale_t*0.6, color=VECTOR_COLOR,
                           width=8.0, head_width=90.0, length_includes_head=True, zorder=6)

        tau_t = tp + tt * scale_t * 1.08
        nrm_t = tp + tn * scale_t * 0.72
        self.ax_traj.text(tau_t[0], tau_t[1], "τ_ОС", color=VECTOR_COLOR, fontsize=10, zorder=9)
        self.ax_traj.text(nrm_t[0], nrm_t[1], "n_ОС", color=VECTOR_COLOR, fontsize=10, zorder=9)

        ang = math.degrees(math.acos(float(np.clip(np.dot(mt, tt), -1.0, 1.0))))
        self.pause_info_var.set(
            f"Пауза\n\nУгол между векторами\nскорости ОУ и ОС:\n{ang:.2f} град"
        )

    def _render(self, show_vectors=False):
        self.figure.clear()
        self.figure.set_facecolor(PANEL_BG)
        self.ax_traj = self.figure.add_subplot(211)
        self.ax_dist = self.figure.add_subplot(212)
        self._style_axes(self.ax_traj)
        self._style_axes(self.ax_dist)
        if self.sim is None:
            self._draw_empty()
            return

        h = self.sim.history
        missile = np.array(h.missile_pos)
        target = np.array(h.target_pos)
        time = np.array(h.t)
        distance = np.array(h.distance)

        self.ax_traj.plot(missile[:, 0], missile[:, 1], color=TRAJ_MISSILE_COLOR, linewidth=1.8, label="Траектория ОУ")
        self.ax_traj.plot(target[:, 0], target[:, 1], color=TRAJ_TARGET_COLOR, linewidth=1.8, label="Траектория ОС")

        if self.sim.finished and h.event.hit:
            x, z = target[-1]
            self._draw_explosion(x, z)
        else:
            self._draw_icons(missile[-1], target[-1])

        if show_vectors:
            self._draw_pause_vectors()
        else:
            self.pause_info_var.set("")

        self.ax_traj.set_title("Траектории в плоскости OXZ")
        self.ax_traj.set_xlabel("x, м")
        self.ax_traj.set_ylabel("z, м")
        self.ax_traj.set_aspect("equal", adjustable="box")
        self.ax_traj.margins(x=0.08, y=0.12)
        legend_traj = self.ax_traj.legend(loc="best")
        legend_traj.get_frame().set_facecolor(PANEL_BG)
        legend_traj.get_frame().set_edgecolor(GRID_COLOR)
        for text in legend_traj.get_texts():
            text.set_color(TEXT_COLOR)

        self.ax_dist.plot(time, distance, color=TEXT_COLOR, linewidth=1.8, label="Дальность")
        if h.min_distance_time is not None:
            self.ax_dist.plot([h.min_distance_time], [h.min_distance], marker="o", color=TEXT_COLOR, linestyle="None", label="d_min")
        self.ax_dist.set_title("Дальность во времени")
        self.ax_dist.set_xlabel("t, с")
        self.ax_dist.set_ylabel("d, м")
        legend_dist = self.ax_dist.legend(loc="best")
        legend_dist.get_frame().set_facecolor(PANEL_BG)
        legend_dist.get_frame().set_edgecolor(GRID_COLOR)
        for text in legend_dist.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.38)
        self.canvas.draw()
        self._update_info(show_vectors)


    def _update_info(self, show_vectors):
        if self.sim is None:
            self.info_var.set("Готов к расчету")
            return
        h = self.sim.history
        wind = np.array(self.sim.params["wind"], dtype=float)
        vg_m = self.sim.missile.vel + wind
        vg_t = self.sim.target.vel + wind
        lines = []
        lines.append("Пауза" if self.paused else ("Расчет завершен" if self.sim.finished else "Анимация"))
        lines.append(f"Метод: {self.sim.params.get('guidance_method', 'Прямой метод')}")
        lines.append(f"t = {self.sim.time:.2f} с")
        lines.append(f"d = {h.distance[-1]:.2f} м")
        lines.append(f"d_min = {h.min_distance:.2f} м")
        lines.append(f"V_ОУ = ({vg_m[0]:.2f}, {vg_m[1]:.2f}) м/с")
        lines.append(f"V_ОС = ({vg_t[0]:.2f}, {vg_t[1]:.2f}) м/с")
        lines.append(f"a_n,max = {self.sim.params.get('missile_max_normal_acc', 0.0):.2f} м/с²")
        if self.sim.params.get("guidance_method") == "Пропорциональное наведение":
            lines.append(f"N = {self.sim.params.get('navigation_constant', 3.0):.2f}")
        if self.sim.params.get("target_motion_mode") == "sinusoidal":
            lines.append(f"R = {self.sim.params.get('target_sine_amplitude', 0.0):.1f} м")
            lines.append(f"f = {self.sim.params.get('target_sine_frequency', 0.0):.4f} 1/м")
            lines.append(f"phase = {self.sim.params.get('target_sine_phase', 0.0):.2f} рад")
            lines.append(f"a_n,max ОС = {self._f('target_max_normal_acc'):.2f} м/с²")
        lines.append(f"dt = {self.sim.params['dt']:.3f} с")
        if self.sim.finished:
            lines.append(f"Событие: {h.event.reason}")
        self.info_var.set("\n".join(lines))

