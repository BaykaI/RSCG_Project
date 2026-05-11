from __future__ import annotations

import tkinter as tk
import time
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, RegularPolygon

from core.math2d import vec, normalize, norm, from_deg, signed_angle_deg
from core.simulator import (
    PROPORTIONAL_METHOD,
    simulate_direct_discrete,
    simulate_parallel_discrete,
    simulate_proportional_discrete,
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
MAX_STEP_LABELS = 60

class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self.result = None
        self.sampled_missile = None
        self.sampled_target = None
        self.sampled_times = None
        self.sampled_state_idx = None
        self.efficiency = None
        self.current_step = 0
        self._pan_start = None
        self._last_pan_draw = 0.0
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
        self.method_combo = ttk.Combobox(
            left,
            textvariable=self.method_var,
            values=["Прямой метод", "Параллельное сближение", PROPORTIONAL_METHOD],
            state="readonly",
        )
        self.method_combo.grid(row=row, column=1, sticky="ew", pady=2)
        self.method_combo.bind("<<ComboboxSelected>>", lambda event: self._sync_method_controls())
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

        self.navconst_label = ttk.Label(left, text="N наведения")
        self.navconst_label.grid(row=row, column=0, sticky="w", pady=2)
        self.navconst_frame = ttk.Frame(left)
        self.navconst_frame.grid(row=row, column=1, sticky="ew", pady=2)
        self.navconst_frame.columnconfigure(0, weight=1)
        self.navconst_var = tk.DoubleVar(value=3.0)
        self.navconst_label_var = tk.StringVar(value="3.00")

        def update_navconst_label(value):
            self.navconst_label_var.set(f"{float(value):.2f}")

        ttk.Scale(
            self.navconst_frame,
            from_=3.0,
            to=20.0,
            variable=self.navconst_var,
            command=update_navconst_label,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(self.navconst_frame, textvariable=self.navconst_label_var, width=5).grid(row=0, column=1, sticky="e")
        row += 1

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
        self.show_miss_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Показывать промах h", variable=self.show_miss_var, command=self._redraw_if_ready).grid(row=row, column=0, columnspan=2, sticky="w")
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
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(left, text="Оценка эффективности").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.efficiency_info = tk.StringVar(value="Нет данных")
        ttk.Label(left, textvariable=self.efficiency_info, justify="left", wraplength=400, relief="solid", padding=8).grid(row=row, column=0, columnspan=2, sticky="ew")

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(left=0.045, right=0.988, bottom=0.075, top=0.945)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Дискретное графическое построение методов наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.ax.grid(True, alpha=0.45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self._sync_method_controls()

    def _f(self, key):
        return float(self.vars[key].get().replace(",", ".").strip())

    def _sync_method_controls(self):
        if self.method_var.get() == PROPORTIONAL_METHOD:
            self.navconst_label.grid()
            self.navconst_frame.grid()
        else:
            self.navconst_label.grid_remove()
            self.navconst_frame.grid_remove()

    def _result_display_name(self):
        if self.result is None:
            return ""
        if self.result.method == PROPORTIONAL_METHOD:
            return f"{self.result.method}, N={self.navconst_var.get():.2f}"
        return self.result.method

    def _compute_efficiency(self):
        if self.result is None or not self.result.states:
            self.efficiency = None
            self.efficiency_info.set("Нет данных")
            return

        distances = np.array([state.distance for state in self.result.states], dtype=float)
        miss_values = np.array([state.params.get("miss_abs", np.nan) for state in self.result.states], dtype=float)
        acc_values = np.array([abs(state.params.get("normal_acc_real", np.nan)) for state in self.result.states], dtype=float)

        min_distance_idx = int(np.nanargmin(distances))
        finite_miss = miss_values[np.isfinite(miss_values)]
        finite_acc = acc_values[np.isfinite(acc_values)]

        h_min = float(np.nanmin(finite_miss)) if finite_miss.size else None
        h_avg = float(np.nanmean(finite_miss)) if finite_miss.size else None
        a_max = float(np.nanmax(finite_acc)) if finite_acc.size else None

        if self.result.hit:
            rating = "высокая"
            outcome = f"контакт, t = {self.result.hit_time:.2f} с"
        elif self.result.stopped_by_divergence:
            rating = "низкая"
            outcome = "расхождение"
        else:
            rating = "не достигнута"
            outcome = self.result.stop_reason

        self.efficiency = {
            "min_distance_idx": min_distance_idx,
            "min_distance": float(distances[min_distance_idx]),
            "min_distance_time": float(self.result.states[min_distance_idx].time),
            "h_min": h_min,
            "h_avg": h_avg,
            "a_max": a_max,
            "rating": rating,
            "outcome": outcome,
        }

        a_line = "a_n max = нет данных" if a_max is None else f"a_n max = {a_max:.2f} м/с²"
        h_min_line = "h_min = нет данных" if h_min is None else f"h_min = {h_min:.2f} м"
        h_avg_line = "h_avg = нет данных" if h_avg is None else f"h_avg = {h_avg:.2f} м"
        self.efficiency_info.set(
            f"Эффективность: {rating}\n"
            f"Итог: {outcome}\n"
            f"R_min = {distances[min_distance_idx]:.2f} м при t = {self.result.states[min_distance_idx].time:.2f} с\n"
            f"{h_min_line}\n"
            f"{h_avg_line}\n"
            f"{a_line}"
        )

    def _redraw_if_ready(self):
        if self.result is not None:
            self._draw_step(self.current_step)

    def _on_plot_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_span = (xlim[1] - xlim[0]) * scale
        y_span = (ylim[1] - ylim[0]) * scale
        rel_x = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])
        self.ax.set_xlim(event.xdata - x_span * rel_x, event.xdata + x_span * (1.0 - rel_x))
        self.ax.set_ylim(event.ydata - y_span * rel_y, event.ydata + y_span * (1.0 - rel_y))
        self.canvas.draw_idle()

    def _on_plot_button_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self._pan_start = {
            "x": event.xdata,
            "y": event.ydata,
            "xlim": self.ax.get_xlim(),
            "ylim": self.ax.get_ylim(),
        }

    def _on_plot_button_release(self, event):
        if event.button == 1:
            self._pan_start = None

    def _on_plot_motion(self, event):
        if self._pan_start is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._pan_start["x"]
        dy = event.ydata - self._pan_start["y"]
        xlim = self._pan_start["xlim"]
        ylim = self._pan_start["ylim"]
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        now = time.monotonic()
        if now - self._last_pan_draw >= 1.0 / 45.0:
            self._last_pan_draw = now
            self.canvas.draw_idle()

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
            "method": self._result_display_name(),
            "missile": self.sampled_missile.copy(),
            "target": self.sampled_target.copy(),
            "color": COLOR_MAP[self.history_color_var.get()],
            "style": STYLE_MAP[self.history_style_var.get()],
        })
        self._refresh_history_listbox()

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

            method = self.method_var.get()
            if method == "Прямой метод":
                self.result = simulate_direct_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc
                )
            elif method == "Параллельное сближение":
                self.result = simulate_parallel_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc
                )
            else:
                nav_const = self.navconst_var.get()
                if nav_const >= 19.999:
                    self.result = simulate_parallel_discrete(
                        missile_pos, missile_speed, missile_course,
                        target_pos, target_speed, target_course,
                        wind, dt, tmax, hit, max_normal_acc
                    )
                else:
                    self.result = simulate_proportional_discrete(
                        missile_pos, missile_speed, missile_course,
                        target_pos, target_speed, target_course,
                        wind, dt, tmax, hit, max_normal_acc,
                        nav_const
                    )

            self.sampled_missile, self.sampled_target, self.sampled_times, self.sampled_state_idx = make_step_arrays(self.result)
            self._compute_efficiency()
            self.current_step = 0
            self._draw_step(self.current_step)
            self._update_nav_buttons()

            if self.result.hit:
                status = f"Останов: условный контакт, t = {self.result.hit_time:.2f} с."
            elif self.result.stopped_by_divergence:
                status = "Останов: ОУ начал удаляться от ОС."
            else:
                status = f"Останов: {self.result.stop_reason}."
            self.info.set(f"{self._result_display_name()}. Число дискретных точек = {len(self.sampled_missile)}. {status}")
            self._store_current_history()
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
        self.ax.scatter(mp[:, 0], mp[:, 1], marker="o", color="black", s=9, alpha=0.7)
        self.ax.scatter(tp[:, 0], tp[:, 1], marker="o", color="red", s=9, alpha=0.7)

        los_segments = np.stack((mp, tp), axis=1)
        self.ax.add_collection(
            LineCollection(
                los_segments,
                colors="0.45",
                linestyles=(0, (4, 4)),
                linewidths=0.9,
                alpha=0.9,
                zorder=1,
            )
        )

        label_stride = max(1, int(np.ceil(len(mp) / MAX_STEP_LABELS)))
        for i in range(0, len(mp), label_stride):
            p_i = mp[i]
            c_i = tp[i]
            self.ax.text(p_i[0] + 55, p_i[1] - 85, f"Ос{i}", color="black", fontsize=9, alpha=0.95)
            self.ax.text(c_i[0] + 55, c_i[1] + 55, f"Оц{i}", color="red", fontsize=9, alpha=0.95)
            if self.show_miss_var.get() and i < len(self.sampled_state_idx):
                state_i = self.result.states[self.sampled_state_idx[i]]
                miss = state_i.params.get("miss_abs")
                if miss is not None:
                    mid = (p_i + c_i) * 0.5
                    self.ax.text(
                        mid[0] + 80,
                        mid[1] - 120,
                        f"h{i}={miss:.1f} м",
                        color="#2ca02c",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#2ca02c", alpha=0.82),
                        zorder=8,
                    )

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

        if self.efficiency is not None:
            min_idx = self.efficiency["min_distance_idx"]
            if 0 <= min_idx < len(mp):
                p_min = mp[min_idx]
                c_min = tp[min_idx]
                mid = (p_min + c_min) * 0.5
                self.ax.plot(
                    [p_min[0], c_min[0]],
                    [p_min[1], c_min[1]],
                    color="#2ca02c",
                    linewidth=2.6,
                    linestyle="-",
                    zorder=9,
                    label="Минимальная дистанция",
                )
                self.ax.plot([p_min[0], c_min[0]], [p_min[1], c_min[1]], linestyle="None", marker="o", color="#2ca02c", ms=7, zorder=10)
                self.ax.text(
                    mid[0] + 120,
                    mid[1] + 120,
                    f"R_min={self.efficiency['min_distance']:.1f} м",
                    color="#2ca02c",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2ca02c", alpha=0.9),
                    zorder=11,
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
            f"ωε = {state.params['los_rate_deg_s']:.3f}°/с\n"
            f"h = {state.params['miss_abs']:.2f} м\n"
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
            f"ωε = {state.params['los_rate_deg_s']:.3f}°/с\n"
            f"h = {state.params['miss_abs']:.2f} м\n"
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

    def _draw_proportional_step(self, state, step_idx, p, c):
        los = c - p
        los_dir = normalize(los)
        vr = state.missile_ground_vel.copy()
        vc = state.target_ground_vel.copy()

        self.ax.plot([p[0]], [p[1]], "ko", ms=7, zorder=7)
        self.ax.plot([c[0]], [c[1]], "o", color="red", ms=7, zorder=7)

        if self.show_xg_var.get():
            self.ax.plot([p[0] - 1400, p[0] + 2600], [p[1], p[1]], color="0.45", linestyle="--", linewidth=1.0)
            self.ax.text(p[0] + 2650, p[1] + 20, "Xg", color="0.35", fontsize=10)

        self.ax.plot([p[0], c[0]], [p[1], c[1]], color="#ff8c00", linewidth=1.5, zorder=6)
        self.ax.text((p[0] + c[0]) * 0.5, (p[1] + c[1]) * 0.5 + 110, "ЛВ", color="#ff8c00", fontsize=11)

        vr_end = p + normalize(vr) * 900.0
        vc_end = c + normalize(vc) * 850.0
        normal_dir = normalize(np.array([-vr[1], vr[0]], dtype=float))
        if state.params.get("normal_acc_real", 0.0) < 0.0:
            normal_dir = -normal_dir
        acc_end = p + normal_dir * 700.0

        self.ax.arrow(p[0], p[1], vr_end[0] - p[0], vr_end[1] - p[1], color="#1f77b4", width=2.8, head_width=42.0, length_includes_head=True, zorder=5)
        self.ax.text(vr_end[0] + 35, vr_end[1] + 25, "Vр", color="#1f77b4", fontsize=11)
        self.ax.arrow(c[0], c[1], vc_end[0] - c[0], vc_end[1] - c[1], color="red", width=2.6, head_width=40.0, length_includes_head=True, zorder=5)
        self.ax.text(vc_end[0] + 35, vc_end[1] + 25, "Vц", color="red", fontsize=11)
        self.ax.arrow(p[0], p[1], acc_end[0] - p[0], acc_end[1] - p[1], color="#2ca02c", width=2.4, head_width=38.0, length_includes_head=True, zorder=5)
        self.ax.text(acc_end[0] + 35, acc_end[1] + 25, "a_n", color="#2ca02c", fontsize=11)

        if self.show_epsilon_var.get():
            self._draw_angle_arc(p, np.array([1.0, 0.0]), los_dir, 440.0, "#ff8c00", "ε", text_shift=(18.0, 12.0), linewidth=2.0)
        self._draw_angle_arc(p, los_dir, normalize(vr), 560.0, "#0057b8", "qр", text_shift=(24.0, 20.0), linewidth=3.0, fontsize=13, min_visual_angle_deg=4.0)

        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"N = {state.params['nav_const']:.2f}\n"
            f"ε = {state.params['eps_deg']:.2f}°\n"
            f"ϑр = {state.params['theta_deg']:.2f}°\n"
            f"qр = {state.params['q_deg']:.2f}°\n"
            f"λ_dot = {state.params['los_rate_deg_s']:.3f}°/с\n"
            f"h = {state.params['miss_abs']:.2f} м\n"
            f"Vсбл = {state.params['closing_speed']:.2f} м/с\n"
            f"a_n = {state.params['normal_acc_real']:.2f} м/с²"
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
        elif self.result.method == "Параллельное сближение":
            self._draw_parallel_step(state, step_idx, p, c)
        else:
            self._draw_proportional_step(state, step_idx, p, c)

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
        self.efficiency = None
        self.ax.clear()
        self.ax.grid(True, alpha=0.45)
        self.ax.set_title("Дискретное графическое построение методов наведения")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("z, м")
        self.canvas.draw()
        self._update_nav_buttons()
        self.info.set("Очищено. Введите данные и постройте новую траекторию.")
        self.step_info.set("Нет данных")
        self.efficiency_info.set("Нет данных")
