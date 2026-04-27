from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, RegularPolygon

from core.math2d import vec, normalize, norm, from_deg, signed_angle_deg
from core.simulator import (
    simulate_direct_discrete,
    simulate_parallel_discrete,
    make_step_arrays,
)

COLOR_MAP = {
    "серый": "0.45",
    "синий": "#1f77b4",
    "зеленый": "#2ca02c",
    "фиолетовый": "#9467bd",
    "коричневый": "#8c564b",
    "черный": "black",
}
STYLE_MAP = {
    "сплошная": "-",
    "штриховая": "--",
    "штрихпунктирная": "-.",
    "пунктирная": ":",
}

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
        self.history = []
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

        ttk.Label(left, text="Методы наведения", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(left, text="Метод").grid(row=row, column=0, sticky="w", pady=2)
        self.method_var = tk.StringVar(value="Прямой метод")
        ttk.Combobox(
            left,
            textvariable=self.method_var,
            values=["Прямой метод", "Параллельное сближение"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="ОУ").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        entry("mx", "x ОУ, м", "12000")
        entry("mz", "z ОУ, м", "-8000")
        entry("mspeed", "V ОУ, м/с", "450")
        entry("mcourse", "курс ОУ, град", "120")
        entry("man", "a_n max, м/с²", "80")

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="ОС").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        entry("tx", "x ОС, м", "0")
        entry("tz", "z ОС, м", "0")
        entry("tspeed", "V ОС, м/с", "120")
        entry("tcourse", "курс ОС, град", "0")

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="Параметры расчета").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        entry("windx", "ветер x, м/с", "0")
        entry("windz", "ветер z, м/с", "0")
        entry("dt", "dt моделирования, с", "5")
        entry("tmax", "t_max, с", "120")
        entry("hit", "радиус условного контакта, м", "25")

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="Память траекторий").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.save_history_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Сохранять предыдущую траекторию", variable=self.save_history_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.show_history_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Показывать сохраненные траектории", variable=self.show_history_var, command=self._redraw_if_ready).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(left, text="Сохраненные траектории").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.history_listbox = tk.Listbox(left, selectmode=tk.EXTENDED, height=6, exportselection=False)
        self.history_listbox.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.history_listbox.bind("<<ListboxSelect>>", lambda event: self._redraw_if_ready())
        row += 1

        history_btns = ttk.Frame(left)
        history_btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        history_btns.columnconfigure(0, weight=1)
        history_btns.columnconfigure(1, weight=1)
        ttk.Button(history_btns, text="Удалить выбранные", command=self.delete_selected_history).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(history_btns, text="Удалить все", command=self.clear_history).grid(row=0, column=1, sticky="ew", padx=(2, 0))
        row += 1

        ttk.Label(left, text="Цвет сохраненной").grid(row=row, column=0, sticky="w", pady=2)
        self.history_color_var = tk.StringVar(value="серый")
        ttk.Combobox(left, textvariable=self.history_color_var, values=list(COLOR_MAP.keys()), state="readonly").grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(left, text="Тип линии").grid(row=row, column=0, sticky="w", pady=2)
        self.history_style_var = tk.StringVar(value="штриховая")
        ttk.Combobox(left, textvariable=self.history_style_var, values=list(STYLE_MAP.keys()), state="readonly").grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(left, text="Отображение на шаге").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.show_xg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Показывать Xg", variable=self.show_xg_var, command=self._redraw_if_ready).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.show_epsilon_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Показывать угол ε", variable=self.show_epsilon_var, command=self._redraw_if_ready).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for i in range(4):
            btns.columnconfigure(i, weight=1)
        ttk.Button(btns, text="Построить", command=self.build_trajectory).grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_prev = ttk.Button(btns, text="Назад", command=self.prev_step, state="disabled")
        self.btn_prev.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_next = ttk.Button(btns, text="Вперед", command=self.next_step, state="disabled")
        self.btn_next.grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(btns, text="Очистить", command=self.clear_all).grid(row=0, column=3, sticky="ew", padx=2)
        row += 1

        self.info = tk.StringVar(value="Выберите метод, задайте исходные данные и постройте дискретную траекторию.")
        ttk.Label(left, textvariable=self.info, justify="left", wraplength=400).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(left, text="Параметры текущего шага").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.step_info = tk.StringVar(value="Нет данных")
        ttk.Label(left, textvariable=self.step_info, justify="left", wraplength=400, relief="solid", padding=8).grid(row=row, column=0, columnspan=2, sticky="ew")

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(left=0.045, right=0.988, bottom=0.075, top=0.945)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Дискретное графическое построение методов наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.ax.grid(True, alpha=0.45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _f(self, key):
        return float(self.vars[key].get().replace(",", ".").strip())

    def _redraw_if_ready(self):
        if self.result is not None:
            self._draw_step(self.current_step)

    def _update_nav_buttons(self):
        if self.result is None:
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            return
        last_idx = len(self.sampled_missile) - 1
        self.btn_prev.configure(state=("normal" if self.current_step > 0 else "disabled"))
        self.btn_next.configure(state=("normal" if self.current_step < last_idx else "disabled"))

    def _refresh_history_listbox(self):
        if not hasattr(self, "history_listbox"):
            return
        selected_indices = set(self.history_listbox.curselection())
        self.history_listbox.delete(0, tk.END)
        for i, h in enumerate(self.history, start=1):
            self.history_listbox.insert(tk.END, f"{i}. {h['method']}")
        if self.history:
            if selected_indices:
                for idx in sorted(selected_indices):
                    if 0 <= idx < len(self.history):
                        self.history_listbox.selection_set(idx)
            else:
                self.history_listbox.selection_set(0, tk.END)

    def _selected_history_indices(self):
        if not hasattr(self, "history_listbox"):
            return []
        return list(map(int, self.history_listbox.curselection()))

    def _store_current_history(self):
        if not self.save_history_var.get():
            return
        if self.result is None or self.sampled_missile is None:
            return
        self.history.append({
            "method": self.result.method,
            "missile": self.sampled_missile.copy(),
            "target": self.sampled_target.copy(),
            "color": COLOR_MAP[self.history_color_var.get()],
            "style": STYLE_MAP[self.history_style_var.get()],
        })
        self._refresh_history_listbox()

    def build_trajectory(self):
        try:
            self._store_current_history()

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

            if self.method_var.get() == "Прямой метод":
                self.result = simulate_direct_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc
                )
            else:
                self.result = simulate_parallel_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc
                )

            self.sampled_missile, self.sampled_target, self.sampled_times, self.sampled_state_idx = make_step_arrays(self.result)
            self.current_step = 0
            self._draw_step(self.current_step)
            self._update_nav_buttons()

            if self.result.hit:
                status = f"Останов: условный контакт, t = {self.result.hit_time:.2f} с."
            elif self.result.stopped_by_divergence:
                status = "Останов: ОУ начал удаляться от ОС."
            else:
                status = f"Останов: {self.result.stop_reason}."
            self.info.set(f"{self.result.method}. Число дискретных точек = {len(self.sampled_missile)}. {status}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _set_view_limits(self):
        if self.sampled_missile is None or self.sampled_target is None:
            return
        groups = [self.sampled_missile, self.sampled_target]
        if self.show_history_var.get():
            selected = set(self._selected_history_indices())
            for i, h in enumerate(self.history):
                if selected and i not in selected:
                    continue
                groups += [h["missile"], h["target"]]
        allp = np.vstack(groups)
        xmin, ymin = np.min(allp, axis=0)
        xmax, ymax = np.max(allp, axis=0)
        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)
        self.ax.set_xlim(xmin - 0.08 * dx, xmax + 0.08 * dx)
        self.ax.set_ylim(ymin - 0.12 * dy, ymax + 0.12 * dy)

    def _draw_history(self):
        if not self.show_history_var.get():
            return
        selected = set(self._selected_history_indices())
        for i, h in enumerate(self.history, start=1):
            if selected and (i - 1) not in selected:
                continue
            self.ax.plot(h["missile"][:, 0], h["missile"][:, 1], color=h["color"], linestyle=h["style"], linewidth=1.2, alpha=0.9)
            self.ax.plot(h["target"][:, 0], h["target"][:, 1], color=h["color"], linestyle=h["style"], linewidth=1.0, alpha=0.7)
            self.ax.text(h["missile"][0, 0], h["missile"][0, 1], f"H{i}", color=h["color"], fontsize=8)

    def _draw_base(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.45)
        self.ax.set_title("Дискретное графическое построение методов наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")

        self._draw_history()

        if self.result is None:
            self.canvas.draw()
            return

        mp = self.sampled_missile
        tp = self.sampled_target

        self.ax.plot(mp[:, 0], mp[:, 1], color="black", linewidth=1.8, label="Траектория ОУ")
        self.ax.plot(tp[:, 0], tp[:, 1], color="red", linewidth=1.6, linestyle="-", label="Траектория ОС")
        self.ax.plot(mp[:, 0], mp[:, 1], linestyle="None", marker="o", color="black", ms=3, alpha=0.7)
        self.ax.plot(tp[:, 0], tp[:, 1], linestyle="None", marker="o", color="red", ms=3, alpha=0.7)

        for i in range(len(mp)):
            p_i = mp[i]
            c_i = tp[i]
            self.ax.plot([p_i[0], c_i[0]], [p_i[1], c_i[1]], color="0.45", linestyle=(0, (4, 4)), linewidth=0.9, alpha=0.9, zorder=1)
            self.ax.text(p_i[0] + 55, p_i[1] - 85, f"Ос{i}", color="black", fontsize=9, alpha=0.95)
            self.ax.text(c_i[0] + 55, c_i[1] + 55, f"Оц{i}", color="red", fontsize=9, alpha=0.95)

        if self.result.hit:
            hit_pt = self.result.states[self.result.hit_index].target_pos
            self.ax.add_patch(
                RegularPolygon(
                    (hit_pt[0], hit_pt[1]),
                    numVertices=8,
                    radius=180,
                    orientation=0.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                )
            )

        self._set_view_limits()
        self.ax.legend(loc="best")



    @staticmethod
    def _ang_deg(v):
        return float(np.rad2deg(np.arctan2(v[1], v[0])))

    @staticmethod
    def _wrap_deg(angle):
        return ((float(angle) + 180.0) % 360.0) - 180.0

    def _draw_angle_arc(self, center, from_dir, to_dir, radius, color, label, text_radius=None, text_shift=(0.0, 0.0), linewidth=2.2, fontsize=12, min_visual_angle_deg=None):
        a1 = self._ang_deg(normalize(from_dir))
        a2 = self._ang_deg(normalize(to_dir))
        delta_real = self._wrap_deg(a2 - a1)
        if abs(delta_real) < 1e-6:
            return
        delta_draw = delta_real
        if min_visual_angle_deg is not None and abs(delta_draw) < float(min_visual_angle_deg):
            delta_draw = np.sign(delta_draw) * float(min_visual_angle_deg)
        if delta_draw >= 0.0:
            theta1 = a1
            theta2 = a1 + delta_draw
        else:
            theta1 = a1 + delta_draw
            theta2 = a1
        self.ax.add_patch(Arc((center[0], center[1]), 2*radius, 2*radius, angle=0.0, theta1=theta1, theta2=theta2, color=color, linewidth=linewidth, zorder=9))
        mid = a1 + 0.5 * delta_draw
        tr = radius + (70.0 if text_radius is None else text_radius)
        tx = center[0] + tr * np.cos(np.deg2rad(mid)) + text_shift[0]
        ty = center[1] + tr * np.sin(np.deg2rad(mid)) + text_shift[1]
        self.ax.text(tx, ty, label, color=color, fontsize=fontsize, fontweight="bold", zorder=10)

    def _draw_direct_step(self, state, step_idx, p, c):
        los = c - p
        los_dir = normalize(los)
        vr_dir = normalize(state.missile_ground_vel)
        vt_dir = normalize(state.target_ground_vel)

        self.ax.plot([p[0]], [p[1]], "ko", ms=7, zorder=7)
        self.ax.plot([c[0]], [c[1]], "o", color="red", ms=7, zorder=7)

        if self.show_xg_var.get():
            self.ax.plot([p[0] - 1600, p[0] + 2600], [p[1], p[1]], color="0.45", linestyle="--", linewidth=1.1)
            self.ax.text(p[0] + 2650, p[1] + 20, "OXg", color="0.35", fontsize=10)

        self.ax.plot([p[0], c[0]], [p[1], c[1]], color="#ff8c00", linewidth=1.6, linestyle="-", zorder=6)
        self.ax.text((p[0] + c[0]) * 0.5, (p[1] + c[1]) * 0.5 + 120, "ЛВ", color="#ff8c00", fontsize=11)

        vr_end = p + vr_dir * 900.0
        self.ax.arrow(p[0], p[1], vr_end[0] - p[0], vr_end[1] - p[1], color="#1f77b4", width=2.8, head_width=42.0, length_includes_head=True, zorder=5)
        self.ax.text(vr_end[0] + 35, vr_end[1] + 25, "Vр", color="#1f77b4", fontsize=11)

        vt_end = c + vt_dir * 850.0
        self.ax.arrow(c[0], c[1], vt_end[0] - c[0], vt_end[1] - c[1], color="red", width=2.6, head_width=40.0, length_includes_head=True, zorder=5)
        self.ax.text(vt_end[0] + 40, vt_end[1] + 35, "Vц", color="red", fontsize=11)

        eps = state.params["eps_deg"]
        theta = state.params["theta_deg"]
        jc = state.params["jc_deg"]
        delta = state.params["delta"]
        if self.show_epsilon_var.get():
            self._draw_angle_arc(p, np.array([1.0, 0.0]), los_dir, 380.0, "#ff8c00", "ε", text_shift=(18.0, 12.0), linewidth=2.0)

        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"ε = {eps:.2f}°\n"
            f"ϑ = {theta:.2f}°\n"
            f"jц = {jc:.2f}°\n"
            f"Δ = {delta:.2f}°"
        )
        self.step_info.set(panel_text)
        self.ax.text(
            0.02, 0.02, panel_text,
            transform=self.ax.transAxes,
            ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.9),
            zorder=20,
        )

    def _draw_parallel_step(self, state, step_idx, p, c):
        los = c - p
        los_dir = normalize(los)
        vr = state.missile_ground_vel.copy()
        vc = state.target_ground_vel.copy()

        self.ax.plot([p[0]], [p[1]], "ko", ms=7, zorder=7)
        self.ax.plot([c[0]], [c[1]], "o", color="red", ms=7, zorder=7)

        if self.show_xg_var.get():
            self.ax.plot([p[0] - 1400, p[0] + 2600], [p[1], p[1]], color="0.45", linestyle="--", linewidth=1.0)
            self.ax.text(p[0] + 2650, p[1] + 20, "Xg", color="0.35", fontsize=10)

        self.ax.plot([p[0], c[0]], [p[1], c[1]], color="#ff8c00", linewidth=1.4, zorder=6)
        self.ax.text((p[0] + c[0]) * 0.5, (p[1] + c[1]) * 0.5 + 110, "R", color="#ff8c00", fontsize=11)

        vr_end = p + normalize(vr) * 900.0
        vc_end = c + normalize(vc) * 850.0
        self.ax.arrow(p[0], p[1], vr_end[0] - p[0], vr_end[1] - p[1], color="#1f77b4", width=2.8, head_width=42.0, length_includes_head=True, zorder=5)
        self.ax.text(vr_end[0] + 35, vr_end[1] + 25, "Vр", color="#1f77b4", fontsize=11)
        self.ax.arrow(c[0], c[1], vc_end[0] - c[0], vc_end[1] - c[1], color="red", width=2.6, head_width=40.0, length_includes_head=True, zorder=5)
        self.ax.text(vc_end[0] + 35, vc_end[1] + 25, "Vц", color="red", fontsize=11)

        q_p = signed_angle_deg(los_dir, normalize(vr))
        q_c = signed_angle_deg(-los_dir, normalize(vc))
        eps0 = state.params["eps0_deg"]
        eps = state.params.get("eps_deg", eps0)
        theta = self._ang_deg(normalize(vr))

        self._draw_angle_arc(p, los_dir, normalize(vr), 560.0, "#0057b8", "qр", text_shift=(24.0, 20.0), linewidth=3.4, fontsize=14, min_visual_angle_deg=4.0)
        self._draw_angle_arc(c, -los_dir, normalize(vc), 460.0, "#d00000", "qц", text_shift=(18.0, 14.0), linewidth=3.2, fontsize=14, min_visual_angle_deg=3.0)
        if self.show_epsilon_var.get():
            self._draw_angle_arc(p, np.array([1.0, 0.0]), los_dir, 460.0, "#ff8c00", "ε", text_shift=(18.0, 12.0), linewidth=2.0)

        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"ε = {eps:.2f}°\n"
            f"ε₀ = {eps0:.2f}°\n"
            f"ϑр = {theta:.2f}°\n"
            f"qр = {q_p:.2f}°\n"
            f"qц = {q_c:.2f}°\n"
            f"Δ = {state.params['delta']:.2f}°"
        )
        self.step_info.set(panel_text)
        self.ax.text(
            0.02, 0.02, panel_text,
            transform=self.ax.transAxes,
            ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.9),
            zorder=20,
        )

    def _draw_step(self, step_idx):
        self._draw_base()
        if self.result is None:
            return

        step_idx = max(0, min(step_idx, len(self.sampled_missile) - 1))
        self.current_step = step_idx
        state = self.result.states[self.sampled_state_idx[step_idx]]
        p = self.sampled_missile[step_idx]
        c = self.sampled_target[step_idx]

        if self.result.method == "Прямой метод":
            self._draw_direct_step(state, step_idx, p, c)
        else:
            self._draw_parallel_step(state, step_idx, p, c)

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


    def delete_selected_history(self):
        selected = self._selected_history_indices()
        if not selected:
            return
        for idx in sorted(selected, reverse=True):
            if 0 <= idx < len(self.history):
                del self.history[idx]
        self._refresh_history_listbox()
        self._redraw_if_ready()

    def clear_history(self):
        self.history.clear()
        self._refresh_history_listbox()
        self._redraw_if_ready()

    def clear_all(self):
        self.result = None
        self.sampled_missile = None
        self.sampled_target = None
        self.sampled_times = None
        self.sampled_state_idx = None
        self.current_step = 0
        self.ax.clear()
        self.ax.grid(True, alpha=0.45)
        self.ax.set_title("Дискретное графическое построение методов наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.canvas.draw()
        self._update_nav_buttons()
        self.info.set("Очищено. Введите данные и постройте новую траекторию.")
        self.step_info.set("Нет данных")
