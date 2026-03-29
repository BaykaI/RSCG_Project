from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, RegularPolygon

from core.math2d import vec, normalize, norm, signed_angle_deg, from_deg
from core.simulator import simulate_direct, make_equal_parts

class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self.result = None
        self.sampled_missile = None
        self.sampled_target = None
        self.sampled_times = None
        self.sampled_state_idx = None
        self.current_step = 0
        self.zoom_mode = False
        self.zoom_factor_var = tk.DoubleVar(value=2.0)
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(1, weight=1)

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        row = 0
        def entry(key, label, value):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            self.vars[key] = tk.StringVar(value=value)
            ttk.Entry(left, textvariable=self.vars[key], width=18).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1

        ttk.Label(left, text="Метод прямого наведения", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="ОУ").grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        entry("mx", "x ОУ, м", "-12000")
        entry("mz", "z ОУ, м", "-4000")
        entry("mspeed", "V ОУ, м/с", "450")
        entry("mcourse", "курс ОУ, град", "20")
        entry("man", "a_n max, м/с²", "80")

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6); row += 1
        ttk.Label(left, text="ОС").grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        entry("tx", "x ОС, м", "0")
        entry("tz", "z ОС, м", "0")
        entry("tspeed", "V ОС, м/с", "120")
        entry("tcourse", "курс ОС, град", "0")

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6); row += 1
        ttk.Label(left, text="Параметры расчета").grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        entry("windx", "ветер x, м/с", "0")
        entry("windz", "ветер z, м/с", "0")
        entry("dt", "dt моделирования, с", "0.05")
        entry("tmax", "t_max, с", "120")
        entry("hit", "порог контакта, м", "25")
        entry("parts", "число равных частей", "12")

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for i in range(5):
            btns.columnconfigure(i, weight=1)
        ttk.Button(btns, text="Построить траекторию", command=self.build_trajectory).grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_prev = ttk.Button(btns, text="Назад", command=self.prev_step, state="disabled")
        self.btn_prev.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_next = ttk.Button(btns, text="Вперед", command=self.next_step, state="disabled")
        self.btn_next.grid(row=0, column=2, sticky="ew", padx=2)
        self.btn_zoom = ttk.Button(btns, text="Приблизить", command=self.toggle_zoom, state="disabled")
        self.btn_zoom.grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(btns, text="Очистить", command=self.clear_all).grid(row=0, column=4, sticky="ew", padx=2)
        row += 1

        self.info = tk.StringVar(
            value="Введите начальные координаты, скорости, курсы и максимальную перегрузку. "
                  "После построения кнопки «Назад/Вперед» показывают один шаг графоаналитического построения."
        )
        ttk.Label(left, textvariable=self.info, justify="left", wraplength=380).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Пошаговое графическое построение метода прямого наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        zoom_box = ttk.LabelFrame(right, text="Приближение", padding=8)
        zoom_box.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        ttk.Label(zoom_box, text="Масштаб").grid(row=0, column=0, sticky="w")
        self.zoom_scale = tk.Scale(
            zoom_box,
            from_=1.0,
            to=15.0,
            resolution=0.1,
            orient="vertical",
            variable=self.zoom_factor_var,
            command=lambda _v: self._on_zoom_change(),
            length=240,
        )
        self.zoom_scale.grid(row=1, column=0, sticky="ns")


    def _on_zoom_change(self):
        if self.result is not None and self.zoom_mode:
            self._draw_step(self.current_step)

    def _f(self, key):
        return float(self.vars[key].get().replace(",", ".").strip())

    def _update_nav_buttons(self):
        if self.result is None:
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            self.btn_zoom.configure(state="disabled")
            return
        last_idx = len(self.sampled_missile) - 1
        self.btn_prev.configure(state=("normal" if self.current_step > 0 else "disabled"))
        self.btn_next.configure(state=("normal" if self.current_step < last_idx else "disabled"))
        self.btn_zoom.configure(state="normal")

    def build_trajectory(self):
        try:
            missile_pos = vec(self._f("mx"), self._f("mz"))
            missile_speed = self._f("mspeed")
            missile_course = from_deg(self._f("mcourse"))
            max_normal_acc = self._f("man")

            target_pos = vec(self._f("tx"), self._f("tz"))
            target_speed = self._f("tspeed")
            target_course = from_deg(self._f("tcourse"))

            wind = vec(self._f("windx"), self._f("windz"))
            dt = self._f("dt")
            tmax = self._f("tmax")
            hit = self._f("hit")
            parts = int(round(self._f("parts")))
            if parts < 1:
                raise ValueError("Число равных частей должно быть не меньше 1.")

            self.result = simulate_direct(
                missile_pos=missile_pos,
                missile_speed=missile_speed,
                missile_course_dir=missile_course,
                target_pos=target_pos,
                target_speed=target_speed,
                target_course_dir=target_course,
                wind=wind,
                dt=dt,
                hit_threshold=hit,
                t_max=tmax,
                max_normal_acc=max_normal_acc,
            )
            self.sampled_missile, self.sampled_target, self.sampled_times, self.sampled_state_idx = make_equal_parts(self.result, parts)
            self.current_step = 0
            self.zoom_mode = False
            self.btn_zoom.configure(text="Приблизить")
            self._draw_step(self.current_step)
            self._update_nav_buttons()

            status = f"Контакт достигнут, t = {self.result.hit_time:.2f} с." if self.result.hit else "Контакт не достигнут на интервале моделирования."
            self.info.set(
                f"Траектория построена и разбита на {parts} равных частей. {status} "
                f"Различение линий усилено: траектория ОУ — черная сплошная, ОС — красная сплошная, "
                f"ЛВ — оранжевая, OX1 — черная утолщенная, Vп — синяя."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _set_view_limits(self, state, p, c):
        if not self.zoom_mode:
            missile_path = np.array([s.missile_pos for s in self.result.states], dtype=float)
            target_path = np.array([s.target_pos for s in self.result.states], dtype=float)
            allp = np.vstack([missile_path, target_path])
            xmin, ymin = np.min(allp, axis=0)
            xmax, ymax = np.max(allp, axis=0)
            dx = max(xmax - xmin, 1.0)
            dy = max(ymax - ymin, 1.0)
            self.ax.set_xlim(xmin - 0.08 * dx, xmax + 0.08 * dx)
            self.ax.set_ylim(ymin - 0.12 * dy, ymax + 0.12 * dy)
            return

        # Приближенный режим: окно центрируется на ОУ
        vp = state.missile_ground_vel
        vt = state.target_ground_vel
        los = c - p

        # В режиме приближения масштаб определяется только бегунком.
        # Никакого дополнительного автоматического приближения на конечном этапе нет.
        base_window = 6000.0
        zoom_factor = max(1.0, float(self.zoom_factor_var.get()))
        half_window = base_window / zoom_factor

        cx = p[0]
        cy = p[1]
        self.ax.set_xlim(cx - half_window, cx + half_window)
        self.ax.set_ylim(cy - half_window, cy + half_window)

    def _scaled_length(self, v, base_len=900.0, min_len=420.0, max_len=1150.0):
        vnorm = norm(v)
        if vnorm < 1e-9:
            return min_len
        ref = 250.0
        scaled = base_len * (vnorm / ref)
        return max(min_len, min(max_len, scaled))

    def _draw_base(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.45)
        self.ax.set_title("Пошаговое графическое построение метода прямого наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        if self.result is None:
            self.canvas.draw()
            return

        missile_path = np.array([s.missile_pos for s in self.result.states], dtype=float)
        target_path = np.array([s.target_pos for s in self.result.states], dtype=float)

        self.ax.plot(missile_path[:, 0], missile_path[:, 1], color="black", linewidth=2.8, label="Траектория ОУ")
        self.ax.plot(target_path[:, 0], target_path[:, 1], color="red", linewidth=2.4, linestyle="-", label="Траектория ОС")
        self.ax.plot(self.sampled_missile[:, 0], self.sampled_missile[:, 1], linestyle="None", marker="o", color="black", ms=3, alpha=0.35)

        if self.result.hit:
            hit_pt = self.result.states[self.result.hit_index].target_pos
            self.ax.add_patch(RegularPolygon(
                (hit_pt[0], hit_pt[1]), numVertices=8, radius=180,
                orientation=0.0, fill=False, edgecolor="black", linewidth=1.8
            ))
        self.ax.legend(loc="best")
        self.ax.set_aspect("equal", adjustable="box")

    def _draw_step(self, step_idx):
        self._draw_base()
        if self.result is None:
            return

        step_idx = max(0, min(step_idx, len(self.sampled_missile) - 1))
        self.current_step = step_idx
        state = self.result.states[self.sampled_state_idx[step_idx]]
        p = self.sampled_missile[step_idx]
        c = self.sampled_target[step_idx]

        los = c - p
        los_dir = normalize(los)
        ox1_dir = normalize(state.missile_air_vel)
        vt = state.target_ground_vel.copy()

        self._set_view_limits(state, p, c)

        self.ax.plot([p[0]], [p[1]], "ko", ms=8, zorder=6)
        self.ax.text(p[0] + 80, p[1] - 120, f"ОУ{step_idx}", color="black", fontsize=11)
        self.ax.plot([c[0]], [c[1]], "o", color="red", ms=8, zorder=6)
        self.ax.text(c[0] + 80, c[1] + 80, f"ОС{step_idx}", color="red", fontsize=11)

        # Горизонт OXg
        self.ax.plot([p[0] - 1600, p[0] + 2600], [p[1], p[1]], color="0.45", linestyle="--", linewidth=1.1)
        self.ax.text(p[0] + 2650, p[1] + 20, "OXg", color="0.35", fontsize=10)

        # Линия визирования
        self.ax.plot([p[0], c[0]], [p[1], c[1]], color="#ff8c00", linewidth=2.8, linestyle="-", zorder=4)
        self.ax.text((p[0] + c[0]) * 0.5, (p[1] + c[1]) * 0.5 + 120, "ЛВ", color="#ff8c00", fontsize=11)

        # Продольная ось OX1
        axis_len = self._scaled_length(state.missile_air_vel, base_len=950.0, min_len=520.0, max_len=1200.0)
        ox1_end = p + ox1_dir * axis_len
        self.ax.arrow(p[0], p[1], ox1_end[0] - p[0], ox1_end[1] - p[1],
                      color="black", width=6.0, head_width=70.0, length_includes_head=True, zorder=5)
        self.ax.text(ox1_end[0] + 50, ox1_end[1] + 35, "OX1", color="black", fontsize=11)

        # Скорость цели
        vt_end = c + normalize(vt) * self._scaled_length(vt, base_len=850.0, min_len=430.0, max_len=1100.0)
        self.ax.arrow(c[0], c[1], vt_end[0] - c[0], vt_end[1] - c[1],
                      color="red", width=4.0, head_width=55.0, length_includes_head=True, zorder=5)
        self.ax.text(vt_end[0] + 40, vt_end[1] + 35, "Vц", color="red", fontsize=11)

        eps = state.eps_deg
        theta = state.theta_deg
        jc = state.jc_deg
        delta = jc

        # Обозначения углов рисуются после векторов, чтобы ничто их не закрывало
        self.ax.add_patch(Arc((p[0], p[1]), 700, 700, angle=0.0,
                              theta1=min(0, eps), theta2=max(0, eps),
                              color="#ff8c00", linewidth=1.2, zorder=9))
        self.ax.text(p[0] + 340, p[1] + 105, "ε", color="#ff8c00", fontsize=12, zorder=10)

        self.ax.add_patch(Arc((p[0], p[1]), 500, 500, angle=0.0,
                              theta1=min(0, theta), theta2=max(0, theta),
                              color="black", linewidth=1.1, zorder=9))
        self.ax.text(p[0] + 235, p[1] + 15, "ϑ", color="black", fontsize=12, zorder=10)

        self.ax.text(
            0.02, 0.98,
            f"Шаг {step_idx + 1} / {len(self.sampled_missile)}\n"
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"ε = {eps:.2f}°\n"
            f"ϑ = {theta:.2f}°\n"
            f"jц = ϑ - ε = {jc:.2f}°\n"
            f"Δ = jц = {delta:.2f}°\n"
            f"|Vц| = {norm(vt):.2f} м/с\n"
            f"d = {state.distance:.2f} м",
            transform=self.ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.93)
        )
        self.canvas.draw()

    def next_step(self):
        if self.result is None:
            return
        self._draw_step(self.current_step + 1)
        self._update_nav_buttons()

    def prev_step(self):
        if self.result is None:
            return
        self._draw_step(self.current_step - 1)
        self._update_nav_buttons()

    def toggle_zoom(self):
        if self.result is None:
            return
        self.zoom_mode = not self.zoom_mode
        self.btn_zoom.configure(text=("Общий вид" if self.zoom_mode else "Приблизить"))
        self._draw_step(self.current_step)

    def clear_all(self):
        self.result = None
        self.sampled_missile = None
        self.sampled_target = None
        self.sampled_times = None
        self.sampled_state_idx = None
        self.current_step = 0
        self.zoom_mode = False
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title("Пошаговое графическое построение метода прямого наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.canvas.draw()
        self._update_nav_buttons()
        self.info.set("Очищено. Введите данные и снова постройте траекторию.")