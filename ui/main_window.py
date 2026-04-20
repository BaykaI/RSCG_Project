from __future__ import annotations
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import DEFAULT_VALUES, GUIDANCE_METHODS, MODES, PN_PRESETS, SCENARIOS
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
SAVED_TRAJECTORY_COLORS = ["#4EA8FF", "#57F287", "#FEE75C", "#FF9F1C", "#C084FC"]


@dataclass
class SavedTrajectory:
    label: str
    missile: np.ndarray
    target: np.ndarray
    event_reason: str
    hit: bool

class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)
        self.vars = {}
        self.replace_vars = {}
        self.info_var = tk.StringVar(value="Готов к расчету")
        self.pause_info_var = tk.StringVar(value="")
        self.guidance_info_var = tk.StringVar(value="")
        self.sim = None
        self.paused = False
        self.after_id = None
        self.locked_widgets = []
        self.start_button = None
        self.traj_window = None
        self.traj_window_figure = None
        self.traj_window_canvas = None
        self.traj_window_ax = None
        self.traj_drag_state = None
        self.saved_trajectories = []
        self.saved_traj_window = None
        self.saved_traj_window_figure = None
        self.saved_traj_window_canvas = None
        self.saved_traj_window_ax = None
        self.saved_traj_drag_state = None
        self.saved_menu_window = None
        self.saved_menu_listbox = None
        self.started = False
        self.slider_vars = {}
        self.slider_value_labels = {}
        self.slider_scales = {}
        self.current_target_motion_mode = "stationary"
        self.sine_params = None
        self.sine_controls_frame = None
        self.pn_controls_frame = None
        self.sine_amplitude_info_var = tk.StringVar(value="R = -")
        self.sine_frequency_info_var = tk.StringVar(value="f = -")
        self.pn_hint_var = tk.StringVar(value="")
        self._configure_theme()
        self._build_ui()
        self._apply_values(DEFAULT_VALUES)
        self._apply_scenario_values("Цель стоит")
        self._apply_replace_defaults()
        self._set_slider_value("navigation_constant", DEFAULT_VALUES["navigation_constant"])
        self._update_pn_controls_visibility()
        self._update_pn_hint()
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
        self._set_slider_value("navigation_constant", self.sim.params.get("navigation_constant", 3.0))
        self._set_slider_value("wind_x", wind[0])
        self._set_slider_value("wind_z", wind[1])
        self._update_sine_output_labels()
        self._update_pn_hint()

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
        elif key == "navigation_constant":
            self.sim.params["navigation_constant"] = value
            if "navigation_constant" in self.vars:
                self.vars["navigation_constant"].set(f"{value:.2f}")
            if "pn_preset" in self.vars:
                self.vars["pn_preset"].set("Пользовательское N")
            self._update_pn_hint()
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

        method_cb = combo("Метод наведения", GUIDANCE_METHODS, "guidance_method", GUIDANCE_METHODS[0])
        method_cb.bind("<<ComboboxSelected>>", self._on_guidance_method_changed)
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
        entry("navigation_constant", "N", "navigation_constant_new")

        self.pn_controls_frame = ttk.LabelFrame(
            p,
            text="Параметры пропорционального наведения",
            padding=6,
        )
        self.pn_controls_frame.columnconfigure(1, weight=1)
        self.pn_controls_frame.columnconfigure(2, weight=1)
        self.pn_controls_frame.columnconfigure(3, weight=0)
        self.pn_controls_frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        pn_row_start = row
        row = 0
        ttk.Label(self.pn_controls_frame, text="Пресет ПН").grid(row=row, column=0, sticky="w", pady=1)
        self.vars["pn_preset"] = tk.StringVar(value=PN_PRESETS[2])
        pn_cb = ttk.Combobox(
            self.pn_controls_frame,
            textvariable=self.vars["pn_preset"],
            values=PN_PRESETS,
            state="readonly",
            width=28,
        )
        pn_cb.grid(row=row, column=1, columnspan=2, sticky="ew", pady=1)
        pn_cb.bind("<<ComboboxSelected>>", self._on_pn_preset_changed)
        row += 1
        ttk.Label(self.pn_controls_frame, textvariable=self.pn_hint_var, style="Muted.TLabel", wraplength=420, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 4)
        )
        row += 1
        row = self._add_live_slider(self.pn_controls_frame, row, "navigation_constant", "Коэффициент N", 1.0, 20.0, 0.1)
        row = pn_row_start + 1

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
        for i in range(8):
            btn.columnconfigure(i, weight=1)
        self.start_button = ttk.Button(btn, text="Старт", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(btn, text="Пауза", command=self.pause).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(btn, text="Продолжить", command=self.resume).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(btn, text="Заменить", command=self.replace_parameters).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(btn, text="Траектория", command=self.open_trajectory_window).grid(row=0, column=4, sticky="ew", padx=2)
        ttk.Button(btn, text="Запомнить", command=self.save_current_trajectory).grid(row=0, column=5, sticky="ew", padx=2)
        ttk.Button(btn, text="Сохранённые", command=self.open_saved_trajectories_menu).grid(row=0, column=6, sticky="ew", padx=2)
        ttk.Button(btn, text="Очистить", command=self.clear_plots).grid(row=0, column=7, sticky="ew", padx=2)
        row += 1

        ttk.Label(p, textvariable=self.info_var, justify="left", wraplength=500).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

    def _build_right(self):
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=0)
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.columnconfigure(1, weight=0)
        self.right_panel.columnconfigure(2, weight=0)
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor=PANEL_BG)
        self.ax_guidance = None
        self.ax_traj = None
        self.ax_dist = None
        self.traj_default_limits = None
        self._layout_standard_figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.toolbar_frame = tk.Frame(self.right_panel, bg=BG_COLOR)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        ttk.Label(self.right_panel, textvariable=self.pause_info_var, justify="left", wraplength=220).grid(
            row=0, column=1, sticky="ne", padx=(10, 0), pady=(10, 0)
        )
        self.guidance_info_panel = tk.Frame(self.right_panel, bg=BG_COLOR)
        self.guidance_info_panel.grid(row=0, column=2, sticky="nw", padx=(6, 0), pady=(10, 0))
        self.guidance_info_panel.grid_remove()

    def _on_plot_scroll(self, event):
        if self.sim is None or self.ax_traj is None or event.inaxes != self.ax_traj:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale = 0.85
        elif event.button == "down":
            scale = 1.18
        else:
            return

        x0, x1 = self.ax_traj.get_xlim()
        y0, y1 = self.ax_traj.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        new_x0 = x - (x - x0) * scale
        new_x1 = x + (x1 - x) * scale
        new_y0 = y - (y - y0) * scale
        new_y1 = y + (y1 - y) * scale

        self.ax_traj.set_xlim(new_x0, new_x1)
        self.ax_traj.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if self.sim is None or self.ax_traj is None or event.inaxes != self.ax_traj:
            return
        if not getattr(event, "dblclick", False):
            return
        if self.traj_default_limits is None:
            return

        (x0, x1), (y0, y1) = self.traj_default_limits
        self.ax_traj.set_xlim(x0, x1)
        self.ax_traj.set_ylim(y0, y1)
        self.canvas.draw_idle()

    def _close_trajectory_window(self):
        if self.traj_window is not None:
            try:
                self.traj_window.destroy()
            except Exception:
                pass
        self.traj_window = None
        self.traj_window_figure = None
        self.traj_window_canvas = None
        self.traj_window_ax = None
        self.traj_drag_state = None

    def _on_traj_window_scroll(self, event):
        if self.traj_window_ax is None or event.inaxes != self.traj_window_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale = 0.85
        elif event.button == "down":
            scale = 1.18
        else:
            return

        x0, x1 = self.traj_window_ax.get_xlim()
        y0, y1 = self.traj_window_ax.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)
        self.traj_window_ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        self.traj_window_ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        if self.traj_window_canvas is not None:
            self.traj_window_canvas.draw_idle()

    def _on_traj_window_press(self, event):
        if self.traj_window_ax is None or event.inaxes != self.traj_window_ax or event.button != 1:
            return
        if event.x is None or event.y is None:
            return
        bbox = self.traj_window_ax.bbox
        self.traj_drag_state = {
            "x_px": float(event.x),
            "y_px": float(event.y),
            "xlim": self.traj_window_ax.get_xlim(),
            "ylim": self.traj_window_ax.get_ylim(),
            "bbox_width": max(float(bbox.width), 1.0),
            "bbox_height": max(float(bbox.height), 1.0),
        }

    def _on_traj_window_motion(self, event):
        if self.traj_drag_state is None or self.traj_window_ax is None or event.inaxes != self.traj_window_ax:
            return
        if event.x is None or event.y is None:
            return

        dx_px = float(event.x) - self.traj_drag_state["x_px"]
        dy_px = float(event.y) - self.traj_drag_state["y_px"]
        xlim = self.traj_drag_state["xlim"]
        ylim = self.traj_drag_state["ylim"]
        x_span = float(xlim[1] - xlim[0])
        y_span = float(ylim[1] - ylim[0])
        dx = dx_px * x_span / self.traj_drag_state["bbox_width"]
        dy = dy_px * y_span / self.traj_drag_state["bbox_height"]
        self.traj_window_ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.traj_window_ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        if self.traj_window_canvas is not None:
            self.traj_window_canvas.draw_idle()

    def _on_traj_window_release(self, event):
        self.traj_drag_state = None

    def _draw_trajectory_axes(self, ax, show_legend: bool = True):
        if self.sim is None:
            return

        h = self.sim.history
        missile = np.array(h.missile_pos)
        target = np.array(h.target_pos)
        time = np.array(h.t)
        self._style_axes(ax)
        ax.plot(missile[:, 0], missile[:, 1], color=TRAJ_MISSILE_COLOR, linewidth=2.0, label="Траектория ОУ")
        ax.plot(target[:, 0], target[:, 1], color=TRAJ_TARGET_COLOR, linewidth=2.0, label="Траектория ОС")
        if self.sim.params.get("guidance_method") == "Пропорциональное наведение":
            current_ax = self.ax_traj
            self.ax_traj = ax
            try:
                self._draw_traj_snapshots(missile, target, time, np.array(h.missile_ground_vel))
            finally:
                self.ax_traj = current_ax
        if self.sim.finished and h.event.hit:
            x, z = target[-1]
            current_ax = self.ax_traj
            self.ax_traj = ax
            try:
                self._draw_explosion(x, z)
            finally:
                self.ax_traj = current_ax
        else:
            current_ax = self.ax_traj
            self.ax_traj = ax
            try:
                self._draw_icons(missile[-1], target[-1])
            finally:
                self.ax_traj = current_ax
        ax.set_title("Траектории в плоскости OXZ")
        ax.set_xlabel("x, м")
        ax.set_ylabel("z, м")
        ax.set_aspect("equal", adjustable="box")
        ax.margins(x=0.08, y=0.12)
        if show_legend:
            legend = ax.legend(loc="best")
            legend.get_frame().set_facecolor(PANEL_BG)
            legend.get_frame().set_edgecolor(GRID_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)

    def open_trajectory_window(self):
        if self.sim is None:
            messagebox.showinfo("Траектория", "Сначала выполните расчёт, затем можно открыть траекторию в отдельном окне.")
            return

        if self.traj_window is not None:
            try:
                self.traj_window.lift()
                self.traj_window.focus_force()
                return
            except Exception:
                self._close_trajectory_window()

        win = tk.Toplevel(self)
        win.title("Траектория в отдельном окне")
        win.geometry("1200x760")
        try:
            win.state("zoomed")
        except Exception:
            try:
                win.attributes("-zoomed", True)
            except Exception:
                pass
        win.configure(bg=BG_COLOR)
        win.protocol("WM_DELETE_WINDOW", self._close_trajectory_window)

        fig = Figure(figsize=(10, 6), dpi=100, facecolor=PANEL_BG)
        ax = fig.add_subplot(111)
        self._draw_trajectory_axes(ax, show_legend=True)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.mpl_connect("scroll_event", self._on_traj_window_scroll)
        canvas.mpl_connect("button_press_event", self._on_traj_window_press)
        canvas.mpl_connect("motion_notify_event", self._on_traj_window_motion)
        canvas.mpl_connect("button_release_event", self._on_traj_window_release)
        toolbar_frame = tk.Frame(win, bg=BG_COLOR)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left", fill="x")
        canvas.draw()

        self.traj_window = win
        self.traj_window_figure = fig
        self.traj_window_canvas = canvas
        self.traj_window_ax = ax
        self.traj_drag_state = None

    def _selected_guidance_method(self) -> str:
        if "guidance_method" not in self.vars:
            return GUIDANCE_METHODS[0]
        return self.vars["guidance_method"].get()

    def _saved_trajectory_label(self) -> str:
        if self.sim is None:
            return "Траектория"
        method = self.sim.params.get("guidance_method", "Прямой метод")
        short_method = "ПН" if method == "Пропорциональное наведение" else "ПМ"
        run_number = len(self.saved_trajectories) + 1
        state = "попадание" if self.sim.history.event.hit else "без попадания"
        return f"{run_number}. {short_method}, t={self.sim.time:.1f} c, {state}"

    def _is_proportional_mode(self) -> bool:
        return self._selected_guidance_method() == "Пропорциональное наведение"

    def _build_saved_trajectory(self) -> SavedTrajectory | None:
        if self.sim is None:
            return None
        history = self.sim.history
        if not history.missile_pos or not history.target_pos:
            return None
        return SavedTrajectory(
            label=self._saved_trajectory_label(),
            missile=np.array(history.missile_pos, dtype=float),
            target=np.array(history.target_pos, dtype=float),
            event_reason=history.event.reason,
            hit=history.event.hit,
        )

    def _close_saved_trajectories_window(self):
        if self.saved_traj_window is not None:
            try:
                self.saved_traj_window.destroy()
            except Exception:
                pass
        self.saved_traj_window = None
        self.saved_traj_window_figure = None
        self.saved_traj_window_canvas = None
        self.saved_traj_window_ax = None
        self.saved_traj_drag_state = None

    def _on_saved_traj_window_scroll(self, event):
        if self.saved_traj_window_ax is None or event.inaxes != self.saved_traj_window_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale = 0.85
        elif event.button == "down":
            scale = 1.18
        else:
            return

        x0, x1 = self.saved_traj_window_ax.get_xlim()
        y0, y1 = self.saved_traj_window_ax.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)
        self.saved_traj_window_ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        self.saved_traj_window_ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        if self.saved_traj_window_canvas is not None:
            self.saved_traj_window_canvas.draw_idle()

    def _on_saved_traj_window_press(self, event):
        if self.saved_traj_window_ax is None or event.inaxes != self.saved_traj_window_ax or event.button != 1:
            return
        if event.x is None or event.y is None:
            return
        bbox = self.saved_traj_window_ax.bbox
        self.saved_traj_drag_state = {
            "x_px": float(event.x),
            "y_px": float(event.y),
            "xlim": self.saved_traj_window_ax.get_xlim(),
            "ylim": self.saved_traj_window_ax.get_ylim(),
            "bbox_width": max(float(bbox.width), 1.0),
            "bbox_height": max(float(bbox.height), 1.0),
        }

    def _on_saved_traj_window_motion(self, event):
        if self.saved_traj_drag_state is None or self.saved_traj_window_ax is None or event.inaxes != self.saved_traj_window_ax:
            return
        if event.x is None or event.y is None:
            return

        dx_px = float(event.x) - self.saved_traj_drag_state["x_px"]
        dy_px = float(event.y) - self.saved_traj_drag_state["y_px"]
        xlim = self.saved_traj_drag_state["xlim"]
        ylim = self.saved_traj_drag_state["ylim"]
        x_span = float(xlim[1] - xlim[0])
        y_span = float(ylim[1] - ylim[0])
        dx = dx_px * x_span / self.saved_traj_drag_state["bbox_width"]
        dy = dy_px * y_span / self.saved_traj_drag_state["bbox_height"]
        self.saved_traj_window_ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.saved_traj_window_ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        if self.saved_traj_window_canvas is not None:
            self.saved_traj_window_canvas.draw_idle()

    def _on_saved_traj_window_release(self, event):
        self.saved_traj_drag_state = None

    def _close_saved_menu_window(self):
        if self.saved_menu_window is not None:
            try:
                self.saved_menu_window.destroy()
            except Exception:
                pass
        self.saved_menu_window = None
        self.saved_menu_listbox = None

    def _refresh_saved_menu(self):
        if self.saved_menu_listbox is None:
            return
        self.saved_menu_listbox.delete(0, tk.END)
        for index, saved in enumerate(self.saved_trajectories, start=1):
            suffix = "попадание" if saved.hit else saved.event_reason
            self.saved_menu_listbox.insert(tk.END, f"{index}. {saved.label} | {suffix}")

    def _selected_saved_trajectory_index(self) -> int | None:
        if self.saved_menu_listbox is None:
            return None
        selection = self.saved_menu_listbox.curselection()
        if not selection:
            messagebox.showinfo("Сохранённые траектории", "Сначала выберите траекторию в списке.")
            return None
        return int(selection[0])

    def delete_selected_saved_trajectory(self):
        index = self._selected_saved_trajectory_index()
        if index is None:
            return
        removed = self.saved_trajectories.pop(index)
        self.info_var.set(f"Удалена сохранённая траектория: {removed.label}.")
        self._refresh_saved_menu()
        if self.saved_traj_window is not None:
            if self.saved_trajectories:
                self._refresh_saved_trajectories_window()
            else:
                self._close_saved_trajectories_window()

    def clear_saved_trajectories(self):
        if not self.saved_trajectories:
            messagebox.showinfo("Сохранённые траектории", "Список сохранённых траекторий уже пуст.")
            return
        self.saved_trajectories.clear()
        self.info_var.set("Сохранённые траектории удалены.")
        self._refresh_saved_menu()
        self._close_saved_trajectories_window()

    def open_saved_trajectories_menu(self):
        if self.saved_menu_window is not None:
            try:
                self.saved_menu_window.lift()
                self.saved_menu_window.focus_force()
                self._refresh_saved_menu()
                return
            except Exception:
                self._close_saved_menu_window()

        win = tk.Toplevel(self)
        win.title("Сохранённые траектории")
        win.geometry("720x340")
        win.configure(bg=BG_COLOR)
        win.protocol("WM_DELETE_WINDOW", self._close_saved_menu_window)

        container = ttk.Frame(win, padding=10, style="TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        ttk.Label(
            container,
            text="Здесь можно удалить отдельные сохранённые траектории или открыть их общий график.",
            style="Muted.TLabel",
            wraplength=660,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        listbox = tk.Listbox(
            container,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            selectbackground=ACCENT_COLOR,
            selectforeground=TEXT_COLOR,
            activestyle="none",
            relief="flat",
            highlightthickness=1,
            highlightbackground=GRID_COLOR,
            highlightcolor=ACCENT_COLOR,
        )
        listbox.grid(row=1, column=0, sticky="nsew")

        btn_frame = ttk.Frame(container, style="TFrame")
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)
        ttk.Button(btn_frame, text="Показать вместе", command=self.open_saved_trajectories_window).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(btn_frame, text="Удалить выбранную", command=self.delete_selected_saved_trajectory).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(btn_frame, text="Очистить список", command=self.clear_saved_trajectories).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(btn_frame, text="Закрыть", command=self._close_saved_menu_window).grid(row=0, column=3, sticky="ew", padx=2)

        self.saved_menu_window = win
        self.saved_menu_listbox = listbox
        self._refresh_saved_menu()

    def _draw_saved_trajectories_axes(self, ax):
        self._style_axes(ax)
        ax.set_title("Сохраненные траектории")
        ax.set_xlabel("x, м")
        ax.set_ylabel("z, м")
        ax.set_aspect("equal", adjustable="box")

        for index, saved in enumerate(self.saved_trajectories):
            color = SAVED_TRAJECTORY_COLORS[index % len(SAVED_TRAJECTORY_COLORS)]
            ax.plot(
                saved.missile[:, 0],
                saved.missile[:, 1],
                color=color,
                linewidth=2.0,
                label=saved.label,
            )
            ax.plot(
                saved.target[:, 0],
                saved.target[:, 1],
                color=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.8,
                label="_nolegend_",
            )

        ax.margins(x=0.08, y=0.12)
        legend = ax.legend(loc="best")
        legend.get_frame().set_facecolor(PANEL_BG)
        legend.get_frame().set_edgecolor(GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    def _refresh_saved_trajectories_window(self):
        if self.saved_traj_window_ax is None or self.saved_traj_window_canvas is None:
            return
        self.saved_traj_window_ax.clear()
        self._draw_saved_trajectories_axes(self.saved_traj_window_ax)
        self.saved_traj_window_canvas.draw_idle()

    def save_current_trajectory(self):
        if self.sim is None:
            messagebox.showinfo("Сохранение траектории", "Сначала выполните расчёт, затем можно сохранить траекторию.")
            return

        saved = self._build_saved_trajectory()
        if saved is None:
            messagebox.showerror("Сохранение траектории", "Не удалось сохранить текущую траекторию.")
            return

        if len(self.saved_trajectories) >= 5:
            self.saved_trajectories.pop(0)

        self.saved_trajectories.append(saved)
        self.info_var.set(
            f"Сохранено траекторий: {len(self.saved_trajectories)} из 5. Последняя: {saved.label}."
        )
        self._refresh_saved_menu()
        if self.saved_traj_window is not None:
            self._refresh_saved_trajectories_window()

    def open_saved_trajectories_window(self):
        if not self.saved_trajectories:
            messagebox.showinfo("Сравнение траекторий", "Пока нет сохранённых траекторий. Нажмите «Запомнить» после расчёта.")
            return

        if self.saved_traj_window is not None:
            try:
                self.saved_traj_window.lift()
                self.saved_traj_window.focus_force()
                self._refresh_saved_trajectories_window()
                return
            except Exception:
                self._close_saved_trajectories_window()

        win = tk.Toplevel(self)
        win.title("Сравнение сохраненных траекторий")
        win.geometry("1200x760")
        win.configure(bg=BG_COLOR)
        win.protocol("WM_DELETE_WINDOW", self._close_saved_trajectories_window)

        fig = Figure(figsize=(10, 6), dpi=100, facecolor=PANEL_BG)
        ax = fig.add_subplot(111)
        self.saved_traj_window_ax = ax
        self._draw_saved_trajectories_axes(ax)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.mpl_connect("scroll_event", self._on_saved_traj_window_scroll)
        canvas.mpl_connect("button_press_event", self._on_saved_traj_window_press)
        canvas.mpl_connect("motion_notify_event", self._on_saved_traj_window_motion)
        canvas.mpl_connect("button_release_event", self._on_saved_traj_window_release)
        toolbar_frame = tk.Frame(win, bg=BG_COLOR)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left", fill="x")
        canvas.draw()

        self.saved_traj_window = win
        self.saved_traj_window_figure = fig
        self.saved_traj_window_canvas = canvas
        self.saved_traj_window_ax = ax
        self.saved_traj_drag_state = None

    def _layout_standard_figure(self):
        self.ax_guidance = None
        self.ax_traj = self.figure.add_subplot(211)
        self.ax_dist = self.figure.add_subplot(212)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.38)

    def _layout_proportional_figure(self):
        grid = self.figure.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 0.8], hspace=0.30)
        self.ax_guidance = self.figure.add_subplot(grid[0])
        self.ax_traj = self.figure.add_subplot(grid[1])
        self.ax_dist = self.figure.add_subplot(grid[2])
        self.figure.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.07)

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
        self._update_pn_controls_visibility()
        self._update_pn_hint()
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

    def _on_guidance_method_changed(self, event=None):
        self._update_pn_controls_visibility()
        self._update_pn_hint()
        if self.sim is None:
            self._draw_empty()
            return
        self._render(self.paused)

    def _pn_preset_value(self, preset_name: str) -> float | None:
        mapping = {
            "Погоня (N = 1)": 1.0,
            "Классическое ПН (N = 3)": 3.0,
            "Близко к идеальному (N = 4)": 4.0,
            "Близко к идеальному (N = 6)": 6.0,
            "Почти параллельное сближение (N = 12)": 12.0,
            "Почти параллельное сближение (N = 20)": 20.0,
        }
        return mapping.get(preset_name)

    def _update_pn_hint(self):
        try:
            n_value = self._f("navigation_constant")
        except Exception:
            self.pn_hint_var.set("")
            return

        if not self._is_proportional_mode():
            self.pn_hint_var.set("Для перехода к параллельному сближению увеличивайте коэффициент N.")
            return

        if n_value <= 1.2:
            text = "N≈1: метод вырождается в погоню."
        elif n_value < 4.0:
            text = "N≈3: классическое пропорциональное наведение."
        elif n_value <= 6.5:
            text = "N=4..6: траектория близка к идеальной по конспекту."
        elif n_value < 12.0:
            text = "Большие N уменьшают кривизну и приближают метод к параллельному сближению."
        else:
            text = "Очень большие N: режим близок к параллельному сближению."
        self.pn_hint_var.set(text)

    def _on_pn_preset_changed(self, event=None):
        preset_name = self.vars["pn_preset"].get()
        value = self._pn_preset_value(preset_name)
        if value is None:
            self._update_pn_hint()
            return

        self.vars["navigation_constant"].set(f"{value:.2f}")
        self._set_slider_value("navigation_constant", value)
        self._update_pn_hint()
        if self.sim is not None:
            self.sim.params["navigation_constant"] = value
            self._render(self.paused)

    def _update_sine_controls_visibility(self):
        if self.sine_controls_frame is None:
            return
        if self.current_target_motion_mode == "sinusoidal":
            self.sine_controls_frame.grid()
        else:
            self.sine_controls_frame.grid_remove()

    def _update_pn_controls_visibility(self):
        if self.pn_controls_frame is None:
            return
        if self._is_proportional_mode():
            self.pn_controls_frame.grid()
        else:
            self.pn_controls_frame.grid_remove()

    def _apply_values(self, values):
        for k, v in values.items():
            if k in self.vars:
                self.vars[k].set(str(round(v, 3) if isinstance(v, float) else v))


    def _apply_replace_defaults(self):
        mapping = {
            "missile_speed_new": "missile_speed",
            "navigation_constant_new": "navigation_constant",
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
                "navigation_constant": self._fr("navigation_constant_new"),
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
            self.vars["navigation_constant"].set(f"{new_params['navigation_constant']:.2f}")
            self._set_slider_value("navigation_constant", new_params["navigation_constant"])
            self.vars["pn_preset"].set("Пользовательское N")
            self._update_pn_hint()
            self.info_var.set("Заменены скорости, ускорения, ветер и параметры синусоидального движения. Курс, dt, t_max, порог и координаты не изменялись.")
            self._render(True)
        except Exception as exc:
            messagebox.showerror("Ошибка замены параметров", str(exc))
    def clear_plots(self):
        self._cancel_after()
        self._close_trajectory_window()
        self.sim = None
        self.paused = False
        self.started = False
        self.pause_info_var.set("")
        self.guidance_info_var.set("")
        self._set_initial_locked(False)
        if self.start_button is not None:
            self.start_button.configure(state="normal")
        self._draw_empty()
        for key in list(self.slider_vars.keys()):
            self._set_slider_value(key, 0.0)
        self.info_var.set(
            f"Очищен текущий расчёт. Сохранённые траектории оставлены: {len(self.saved_trajectories)}."
        )

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
        self.guidance_info_var.set("")
        if self._is_proportional_mode():
            self._layout_proportional_figure()
            self._style_axes(self.ax_guidance)
            self.ax_guidance.set_title("Пропорциональное наведение: вращение векторов относительно центра масс ОУ")
            self.ax_guidance.set_xlabel("ось x")
            self.ax_guidance.set_ylabel("ось z")
            self.ax_guidance.set_xlim(-1000.0, 1000.0)
            self.ax_guidance.set_ylim(-1000.0, 1000.0)
            self.ax_guidance.set_aspect("equal", adjustable="box")
        else:
            self._layout_standard_figure()
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
        if not self._is_proportional_mode():
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

    def _style_info_axes(self, ax):
        ax.set_facecolor(SURFACE_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

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

    def _vector_arrow(self, ax, origin: np.ndarray, vector: np.ndarray, color: str, label: str, linewidth: float = 2.4, zorder: int = 6):
        vec_norm = norm(vector)
        if vec_norm < 1e-9:
            return
        ax.annotate(
            "",
            xy=(origin[0] + vector[0], origin[1] + vector[1]),
            xytext=(origin[0], origin[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth, shrinkA=0.0, shrinkB=0.0),
            zorder=zorder,
        )
        if label:
            text_pos = origin + vector * 1.05
            ax.text(text_pos[0], text_pos[1], label, color=color, fontsize=11, weight="bold", zorder=zorder + 1)

    def _update_proportional_side_info(self, rel_norm: float):
        if self.sim is None or self.sim.params.get("guidance_method") != "Пропорциональное наведение":
            self.guidance_info_var.set("")
            if hasattr(self, "guidance_info_panel"):
                self.guidance_info_panel.grid_remove()
            return

        h = self.sim.history
        los_rate = h.los_rate[-1] if h.los_rate else 0.0
        missile_turn_rate = h.missile_turn_rate[-1] if h.missile_turn_rate else 0.0
        proportional_turn_rate = h.proportional_turn_rate[-1] if h.proportional_turn_rate else 0.0
        ratio_text = "не определяется"
        if abs(proportional_turn_rate) > 1e-9:
            ratio_text = f"{missile_turn_rate / proportional_turn_rate:.2f}"
        self._render_guidance_info_panel(
            [
                (
                    "Верхний монитор",
                    [
                        ("Линия визирования", "#FEE75C"),
                        ("Направление скорости ОУ", VECTOR_COLOR),
                        ("Нормаль к линии визирования", "#FF9F1C"),
                        ("Поворот скорости ОУ", "#C084FC"),
                    ],
                ),
                (
                    "Средний монитор",
                    [
                        ("Траектория ОУ", TRAJ_MISSILE_COLOR),
                        ("Траектория цели", TRAJ_TARGET_COLOR),
                        ("Линия визирования каждые 6 с", "#FEE75C"),
                        ("Точка встречи", HIT_COLOR),
                    ],
                ),
            ],
            [
                f"Угловая скорость линии визирования: {los_rate:.4f} рад/с",
                f"Угловая скорость поворота скорости ОУ: {missile_turn_rate:.4f} рад/с",
                f"Расчетное значение N·ω_ЛВ: {proportional_turn_rate:.4f} рад/с",
                f"Отношение ω_ОУ / (N·ω_ЛВ): {ratio_text}",
                f"Дальность до цели: {rel_norm:.1f} м",
            ],
        )

    def _render_guidance_info_panel(self, sections, stats_lines):
        panel = self.guidance_info_panel
        for child in panel.winfo_children():
            child.destroy()

        panel.configure(bg=BG_COLOR)
        row = 0
        for title, items in sections:
            ttk.Label(panel, text=title, style="TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
            row += 1
            for text, color in items:
                sample = tk.Canvas(panel, width=34, height=12, bg=BG_COLOR, highlightthickness=0)
                sample.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
                sample.create_line(2, 6, 32, 6, fill=color, width=3)
                ttk.Label(panel, text=text, style="Muted.TLabel").grid(row=row, column=1, sticky="w", pady=1)
                row += 1
            tk.Frame(panel, height=8, bg=BG_COLOR).grid(row=row, column=0, columnspan=2, sticky="ew")
            row += 1

        for line in stats_lines:
            ttk.Label(panel, text=line, style="Muted.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
            row += 1

        panel.grid()

    def _draw_proportional_guidance_view(self):
        if self.sim is None or self.ax_guidance is None:
            return

        ax = self.ax_guidance
        h = self.sim.history
        missile_pos = np.array(h.missile_pos)
        target_pos = np.array(h.target_pos)
        missile_vel = np.array(h.missile_ground_vel)
        rel = target_pos - missile_pos

        los_dirs = []
        vel_dirs = []
        for rel_vec, vel_vec in zip(rel, missile_vel):
            if norm(rel_vec) > 1e-9:
                los_dirs.append(normalize(rel_vec))
            else:
                los_dirs.append(np.array([1.0, 0.0], dtype=float))
            if norm(vel_vec) > 1e-9:
                vel_dirs.append(normalize(vel_vec))
            else:
                vel_dirs.append(np.array([1.0, 0.0], dtype=float))

        los_dirs = np.array(los_dirs)
        vel_dirs = np.array(vel_dirs)
        radius = 1.0
        los_tip = los_dirs * radius
        vel_tip = vel_dirs * radius * 0.82
        los_rate = h.los_rate[-1] if h.los_rate else 0.0
        missile_turn_rate = h.missile_turn_rate[-1] if h.missile_turn_rate else 0.0
        proportional_turn_rate = h.proportional_turn_rate[-1] if h.proportional_turn_rate else 0.0
        nav_const = float(self.sim.params.get("navigation_constant", 3.0))

        theta = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.0, color=GRID_COLOR, alpha=0.7, zorder=1)
        ax.axhline(0.0, color=GRID_COLOR, linewidth=0.8, alpha=0.7, zorder=1)
        ax.axvline(0.0, color=GRID_COLOR, linewidth=0.8, alpha=0.7, zorder=1)
        ax.plot(los_tip[:, 0], los_tip[:, 1], color="#FEE75C", linewidth=1.8, alpha=0.9, zorder=3)
        ax.plot(vel_tip[:, 0], vel_tip[:, 1], color=VECTOR_COLOR, linewidth=1.8, alpha=0.9, zorder=3)

        ax.plot([0.0], [0.0], marker="o", markersize=7, color=TRAJ_MISSILE_COLOR, zorder=7)

        self._vector_arrow(ax, np.array([0.0, 0.0]), los_tip[-1], "#FEE75C", "", linewidth=2.8, zorder=6)
        self._vector_arrow(ax, np.array([0.0, 0.0]), vel_tip[-1], VECTOR_COLOR, "", linewidth=2.8, zorder=6)

        current_missile_pos = missile_pos[-1]
        current_target_pos = target_pos[-1]
        current_rel = current_target_pos - current_missile_pos
        rel_norm = max(norm(current_rel), 1e-9)
        los_normal = perp(current_rel / rel_norm)
        self._vector_arrow(ax, np.array([0.0, 0.0]), los_normal * 0.58, "#FF9F1C", "", linewidth=2.2, zorder=5)

        turn_sign = 1.0 if missile_turn_rate >= 0.0 else -1.0
        accel_axis = turn_sign * perp(vel_dirs[-1]) * 0.52
        self._vector_arrow(ax, np.array([0.0, 0.0]), accel_axis, "#C084FC", "", linewidth=2.2, zorder=5)

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("W")
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        self._update_proportional_side_info(rel_norm)

    def _draw_traj_snapshots(self, missile: np.ndarray, target: np.ndarray, time: np.ndarray, missile_ground_vel: np.ndarray):
        if len(time) == 0:
            return

        snapshot_times = np.arange(0.0, float(time[-1]) + 1e-9, 6.0)
        if len(snapshot_times) == 0:
            return

        x0, x1 = self.ax_traj.get_xlim()
        z0, z1 = self.ax_traj.get_ylim()
        span = max(abs(x1 - x0), abs(z1 - z0), 1.0)
        vel_scale = max(1200.0, 0.08 * span)

        used_los_label = False
        used_vel_label = False
        for snapshot_time in snapshot_times:
            idx = int(np.argmin(np.abs(time - snapshot_time)))
            mp = missile[idx]
            tp = target[idx]
            vg = missile_ground_vel[idx]
            rel = tp - mp

            los_kwargs = {
                "color": "#FEE75C",
                "linewidth": 1.1,
                "alpha": 0.45,
                "zorder": 4,
            }
            vel_kwargs = {
                "color": VECTOR_COLOR,
                "linewidth": 1.2,
                "alpha": 0.55,
                "zorder": 5,
            }

            if not used_los_label:
                los_kwargs["label"] = "Линия визирования каждые 6 с"
                used_los_label = True
            if not used_vel_label:
                vel_kwargs["label"] = "Вектор скорости ОУ каждые 6 с"
                used_vel_label = True

            self.ax_traj.plot([mp[0], tp[0]], [mp[1], tp[1]], **los_kwargs)

            if norm(vg) > 1e-9:
                vel_dir = normalize(vg)
                vel_vec = vel_dir * vel_scale
                self.ax_traj.annotate(
                    "",
                    xy=(mp[0] + vel_vec[0], mp[1] + vel_vec[1]),
                    xytext=(mp[0], mp[1]),
                    arrowprops=dict(arrowstyle="-|>", color=VECTOR_COLOR, lw=1.2, alpha=0.55, shrinkA=0.0, shrinkB=0.0),
                    zorder=5,
                )

    def _render(self, show_vectors=False):
        self.figure.clear()
        self.figure.set_facecolor(PANEL_BG)
        if self.sim is not None and self.sim.params.get("guidance_method") == "Пропорциональное наведение":
            self._layout_proportional_figure()
            self._style_axes(self.ax_guidance)
        else:
            self._layout_standard_figure()
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

        if self.sim.params.get("guidance_method") == "Пропорциональное наведение":
            self._draw_traj_snapshots(missile, target, time, np.array(h.missile_ground_vel))

        if self.sim.finished and h.event.hit:
            x, z = target[-1]
            self._draw_explosion(x, z)
        else:
            self._draw_icons(missile[-1], target[-1])

        if show_vectors:
            self._draw_pause_vectors()
        else:
            self.pause_info_var.set("")

        if self.sim.params.get("guidance_method") == "Пропорциональное наведение":
            self._draw_proportional_guidance_view()

        self.ax_traj.set_title("Траектории в плоскости OXZ")
        self.ax_traj.set_xlabel("x, м")
        self.ax_traj.set_ylabel("z, м")
        self.ax_traj.set_aspect("equal", adjustable="box")
        self.ax_traj.set_anchor("W")
        self.ax_traj.margins(x=0.08, y=0.12)
        self.traj_default_limits = (self.ax_traj.get_xlim(), self.ax_traj.get_ylim())
        if self.sim.params.get("guidance_method") != "Пропорциональное наведение":
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

        if self.sim.params.get("guidance_method") != "Пропорциональное наведение":
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
            n_value = float(self.sim.params.get("navigation_constant", 3.0))
            if n_value <= 1.2:
                lines.append("Режим ПН близок к погоне")
            elif n_value >= 12.0:
                lines.append("Режим ПН близок к параллельному сближению")
            if h.los_rate:
                lines.append(f"ω_ЛВ = {h.los_rate[-1]:.4f} рад/с")
            if h.missile_turn_rate:
                lines.append(f"ω_ОУ = {h.missile_turn_rate[-1]:.4f} рад/с")
        if self.sim.params.get("target_motion_mode") == "sinusoidal":
            lines.append(f"R = {self.sim.params.get('target_sine_amplitude', 0.0):.1f} м")
            lines.append(f"f = {self.sim.params.get('target_sine_frequency', 0.0):.4f} 1/м")
            lines.append(f"phase = {self.sim.params.get('target_sine_phase', 0.0):.2f} рад")
            lines.append(f"a_n,max ОС = {self._f('target_max_normal_acc'):.2f} м/с²")
        lines.append(f"dt = {self.sim.params['dt']:.3f} с")
        if self.sim.finished:
            lines.append(f"Событие: {h.event.reason}")
        self.info_var.set("\n".join(lines))

