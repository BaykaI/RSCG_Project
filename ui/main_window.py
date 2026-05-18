from __future__ import annotations

import base64
import io
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
    DIRECT_LEAD_METHOD,
    PURSUIT_METHOD,
    PD_METHOD,
    simulate_direct_discrete,
    simulate_direct_lead_discrete,
    simulate_pursuit_discrete,
    simulate_parallel_discrete,
    simulate_proportional_discrete,
    simulate_pd_discrete,
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

METHOD_INFO = {
    "Прямой метод": {
        "section": "4.1",
        "title": "Метод прямого наведения",
        "definition": (
            "Продольная ось ОУ O_PX_1 = OX_СВ связанной СК в каждый момент "
            "времени должна совмещаться с направлением на цель; требуемый "
            "угол q_T должен быть равен нулю."
        ),
        "formulas": [
            (r"$\varphi_{\mathrm{Ц}} = \vartheta - \varepsilon$",
             "(4) Взаимная связь углов"),
            (r"$\varphi_{\mathrm{Ц}} = 0,\quad \vartheta = \varepsilon$",
             "(5)–(6) Уравнение идеальной связи"),
            (r"$\Delta = \varphi_{\mathrm{Ц}} = \vartheta - \varepsilon$",
             "(7)–(8) Параметр рассогласования (алгоритм траекторного управления)"),
        ],
        "advantages": [
            "инвариантность к дальности наведения и высоте полёта цели и ОУ",
            "относительная простота КБА: измеряются непосредственно бортовые "
            "пеленги φ_г и φ_в (БРЛС, РЛС, ТГС) или углы ϑ и ε с их "
            "последующим вычитанием",
            "возможность применения неподвижного координатора при совпадении "
            "осей ОУ и координатора",
        ],
        "disadvantages": [
            "ограниченность применения — только по неподвижным целям "
            "(V_цели << V_ОУ)",
            "низкая точность наведения — большие поперечные (нормальные) "
            "перегрузки ОУ на конечном этапе наведения (даже по неподвижным "
            "целям)",
            "влияние ветра — искривление траектории за счёт сноса (кривизна "
            "тем больше, чем меньше скорость ОУ и больше скорость ветра в "
            "поперечном направлении)",
        ],
    },
    DIRECT_LEAD_METHOD: {
        "section": "4.2",
        "title": "Метод прямого наведения с постоянным углом упреждения",
        "definition": (
            "В течение всего времени полёта ОУ угол между продольной осью и "
            "линией визирования остаётся постоянным (развитие метода прямого "
            "наведения)."
        ),
        "formulas": [
            (r"$\varphi_{\mathrm{Ц}} = \varphi_0,\quad "
             r"\vartheta - \varepsilon = \varphi_0$",
             "(9)–(10) Уравнение идеальной связи"),
            (r"$\Delta = \varphi_{\mathrm{Ц}} - \varphi_0 = "
             r"\vartheta - \varepsilon - \varphi_0$",
             "(11)–(12) Параметр рассогласования"),
        ],
        "notes": (
            "Метод обеспечивает меньшую кривизну траектории. Реализация — "
            "аналогична прямому методу."
        ),
    },
    PURSUIT_METHOD: {
        "section": "4.3",
        "title": "Метод погони (флюгерный метод)",
        "definition": (
            "Метод погони — с ЛВ «ОУ–цель» непрерывно совмещается вектор "
            "скорости ОУ. Флюгерный метод — с ЛВ непрерывно совмещается "
            "вектор воздушной скорости ОУ. При движении ОУ в невозмущённой "
            "атмосфере оба метода идентичны."
        ),
        "formulas": [
            (r"$q = 0,\quad \theta = \varepsilon,\quad "
             r"\varphi_{\mathrm{Ц}} = \alpha$",
             "(13)–(15) Уравнение идеальной связи"),
            (r"$\Delta = q = \theta - \varepsilon = "
             r"\varphi_{\mathrm{Ц}} - \alpha$",
             "(16)–(18) Параметр рассогласования"),
        ],
        "notes": (
            "Особенности: ОУ независимо от своего начального положения "
            "стремится выйти строго «в хвост» цели; при α = 0 (β = 0) метод "
            "вырождается в метод прямого наведения."
        ),
        "advantages": [
            "инвариантность к дальности наведения и высоте полёта цели и ОУ",
            "компенсирует наличие угла атаки и скольжения",
        ],
        "disadvantages": [
            "ограниченность применения — только по неподвижным целям и при "
            "отсутствии ветра",
            "искривление траектории за счёт движения цели или бокового ветра",
            "несовпадение мгновенного направления взаимного перемещения с "
            "направлением на цель — необходимость угла упреждения "
            "(треугольник скоростей)",
            "кривизна траектории тем больше, чем больше скорость ОУ и "
            "скорость цели (ветра) в поперечном направлении",
            "увеличение ошибок и времени наведения, уменьшение дальности "
            "действия, сложность маневра в диапазоне допустимых перегрузок "
            "на среднем и конечном этапах наведения",
            "усложнение КБА по сравнению с реализацией прямого метода: "
            "измерение φ_Ц и угла атаки/скольжения (углометр БРЛС, РЛС, "
            "ТГС; ФД — флюгерный датчик)",
        ],
    },
    "Параллельное сближение": {
        "section": "4.4",
        "title": "Метод параллельного сближения (в мгновенную точку встречи)",
        "definition": (
            "В любой момент времени вектор скорости ОУ направлен в "
            "упреждённую точку (линия визирования перемещается параллельно "
            "сама себе)."
        ),
        "formulas": [
            (r"$\omega = \dot{\varepsilon} = "
             r"\dfrac{V_P \sin q_P - V_{\mathrm{Ц}} \sin q_{\mathrm{Ц}}}{R} "
             r"\approx 0$",
             "(19) Условие выполнения"),
            (r"$\dot{\varepsilon} = 0,\quad \varepsilon = \varepsilon_{R_0}$",
             "(20)–(21) Уравнение метода (вариант)"),
            (r"$q_{P\mathrm{T}} \approx "
             r"\dfrac{V_{\mathrm{Ц}}}{V_P}\,\sin q_{\mathrm{Ц}}$",
             "(22) Уравнение метода (вариант)"),
            (r"$\Delta = \dot{\varepsilon}$",
             "(23) Параметр рассогласования (вариант)"),
            (r"$\Delta = \varepsilon - \varepsilon_{D_0}$",
             "(24) Параметр рассогласования (вариант)"),
            (r"$V_{\mathrm{Ц}} \sin q_{\mathrm{Ц}} = V_P \sin q_P$",
             "(25)"),
            (r"$\Delta = q_P - "
             r"\dfrac{V_{\mathrm{Ц}}}{V_P}\,\sin q_{\mathrm{Ц}}$",
             "(26) Параметр рассогласования (вариант)"),
        ],
        "notes": (
            "При наведении на неманеврирующую цель траектория ОУ "
            "прямолинейна. Для реализации метода необходим состав "
            "измерителей (датчиков), аналогичный реализации метода погони."
        ),
    },
    PROPORTIONAL_METHOD: {
        "section": "4.5",
        "title": "Метод пропорционального наведения",
        "definition": (
            "Вариант 1: в любой момент времени угловая скорость вращения "
            "вектора скорости ОУ в плоскости управления должна быть "
            "пропорциональна угловой скорости линии визирования ОУ–Ц.\n"
            "Вариант 2: в любой момент времени нормальное (боковое) "
            "ускорение ОУ в плоскости управления должно быть пропорционально "
            "угловой скорости линии визирования ОУ–Ц и скорости ОУ."
        ),
        "formulas": [
            (r"$J^{\mathrm{Треб}}_{\mathrm{УР}} = "
             r"N_0\,|\dot{r}|\,\omega + 1{,}5\,J_{\mathrm{Ц}}$",
             "(27) Оптимальное уравнение (согласно ТОУ)"),
            (r"$^{*}J^{\mathrm{Треб}}_{\mathrm{УР}} = "
             r"N_0\,|\dot{r}|\,\omega$",
             "(28) Упрощённое уравнение (цель равномерно прямолинейно)"),
            (r"$\Delta = N_0\,|\dot{r}|\,\omega + 1{,}5\,J_{\mathrm{Ц}} "
             r"- J_{\mathrm{УР}}$",
             "(29) Параметр рассогласования (оптимальный)"),
            (r"$^{*}\Delta = N_0\,|\dot{r}|\,\omega - J_{\mathrm{УР}}$",
             "(30) Параметр рассогласования (упрощённый)"),
            (r"$\dot{\theta} = "
             r"\dfrac{N_0\,|\dot{r}|}{V_{\mathrm{УР}}}\,\omega = C\,\omega$",
             "(31) Формулировка через угловую скорость (вариант 1)"),
            (r"$\Delta = \dot{\theta} - C\,\omega$",
             "(33) Параметр рассогласования (вариант 1)"),
            (r"$J^{\mathrm{Треб}}_{\mathrm{УР}} = "
             r"C\,V_{\mathrm{УР}}\,\omega = N_0\,|\dot{r}|\,\omega$",
             "(35) Формулировка через нормальное ускорение (вариант 2)"),
            (r"$\Delta = C\,V_{\mathrm{УР}}\,\omega - J_{\mathrm{УР}}$",
             "(36) Параметр рассогласования (вариант 2)"),
        ],
        "notes": (
            "N_0 = 3 — навигационная постоянная; ṙ — скорость сближения; "
            "J_Ц — нормальное (боковое) ускорение цели. При пропорциональном "
            "наведении управление безынерционного объекта формируется "
            "пропорционально ускорению цели J_Ц."
        ),
    },
    PD_METHOD: {
        "section": "4.5",
        "title": "Пропорционально-дифференциальный метод (расширение 4.5)",
        "definition": (
            "Расширение пропорционального наведения дифференциальной "
            "составляющей: к слагаемому N_0·|ṙ|·ω добавляется численная "
            "оценка λ̈ — производной угловой скорости ЛВ, упреждающая "
            "ускорение цели (по смыслу аналогично слагаемому 1,5·J_Ц "
            "в оптимальном уравнении 4.5)."
        ),
        "formulas": [
            (r"$J^{\mathrm{Треб}}_{\mathrm{УР}} = "
             r"N_0\,|\dot{r}|\,\omega + 1{,}5\,J_{\mathrm{Ц}}$",
             "(27) Оптимальное уравнение пропорционального наведения"),
            (r"$^{*}J^{\mathrm{Треб}}_{\mathrm{УР}} = "
             r"N_0\,|\dot{r}|\,\omega$",
             "(28) Упрощённое уравнение пропорционального наведения"),
        ],
        "notes": (
            "Дифференциальная часть упреждает ускорение цели и уменьшает "
            "промах при манёврах (раздел 4.5 семинара рассматривает "
            "оптимальный и упрощённый варианты)."
        ),
    },
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
        self.efficiency = None
        self.current_step = 0
        self._pan_start = None
        self._last_pan_draw = 0.0
        self.history = []
        self._formula_cache: dict = {}
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

        def entry(key, label, value, help_text="", parent=None):
            nonlocal row
            target = parent if parent is not None else left
            if parent is None:
                r = row
                row += 1
            else:
                r = target.grid_size()[1]
            lbl = ttk.Label(target, text=label)
            lbl.grid(row=r, column=0, sticky="w", pady=2)
            self.vars[key] = tk.StringVar(value=value)
            ent = ttk.Entry(target, textvariable=self.vars[key], width=18)
            ent.grid(row=r, column=1, sticky="ew", pady=2)
            if help_text:
                self._bind_help(lbl, help_text)
                self._bind_help(ent, help_text)

        ttk.Label(left, text="Методы наведения", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(left, text="Подсказка: ПКМ по любому элементу — описание", foreground="#666", font=("TkDefaultFont", 8, "italic")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        method_label = ttk.Label(left, text="Метод")
        method_label.grid(row=row, column=0, sticky="w", pady=2)
        self.method_var = tk.StringVar(value="Прямой метод")
        self.method_combo = ttk.Combobox(
            left,
            textvariable=self.method_var,
            values=[
                "Прямой метод",
                DIRECT_LEAD_METHOD,
                PURSUIT_METHOD,
                "Параллельное сближение",
                PROPORTIONAL_METHOD,
                PD_METHOD,
            ],
            state="readonly",
        )
        self.method_combo.grid(row=row, column=1, sticky="ew", pady=2)
        self.method_combo.bind("<<ComboboxSelected>>", lambda event: self._sync_method_controls())
        method_combo_help = (
            "Выбор закона наведения. Влияет на формулу расчёта требуемого курса "
            "ракеты на каждом шаге. Краткое описание текущего метода — в рамке "
            "ниже."
        )
        self._bind_help(method_label, method_combo_help)
        self._bind_help(self.method_combo, method_combo_help)
        row += 1

        desc_container, desc_body, desc_header = self._make_collapsible(left, "Описание метода")
        desc_container.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        self._bind_help(desc_header, "Описание выбранного метода с формулами (по материалам семинара 3). ЛКМ по заголовку — свернуть/развернуть.")

        desc_frame = tk.Frame(desc_body, relief="solid", borderwidth=1, bg="#bda842")
        desc_frame.grid(row=0, column=0, sticky="ew")
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)

        self.method_description_text = tk.Text(
            desc_frame,
            width=48,
            height=18,
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
            font=("TkDefaultFont", 9),
            bg="#fffbf2",
            fg="#222",
            cursor="arrow",
            spacing1=2,
            spacing3=2,
        )
        self.method_description_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        desc_scroll = ttk.Scrollbar(desc_frame, orient="vertical", command=self.method_description_text.yview)
        desc_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 1), pady=1)
        self.method_description_text.configure(yscrollcommand=desc_scroll.set)
        self.method_description_text.tag_configure("title", font=("TkDefaultFont", 10, "bold"), foreground="#1a4a8a", spacing3=4)
        self.method_description_text.tag_configure("subhead", font=("TkDefaultFont", 9, "bold"), foreground="#444", spacing1=4, spacing3=2)
        self.method_description_text.tag_configure("caption", foreground="#555", font=("TkDefaultFont", 8, "italic"), lmargin1=12, lmargin2=12)
        self.method_description_text.tag_configure("body", foreground="#222", lmargin1=0, lmargin2=0)
        self.method_description_text.tag_configure("bullet", foreground="#222", lmargin1=12, lmargin2=24)
        self.method_description_text.configure(state="disabled")
        self._bind_help(self.method_description_text, "Описание выбранного метода: определение, формулы, достоинства и недостатки (по семинару 3).")
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        mu_container, mu_body, mu_header = self._make_collapsible(left, "ОУ — объект управления (ракета)")
        mu_container.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_help(mu_header, "Блок начальных параметров ракеты (объекта управления). ЛКМ по заголовку — свернуть/развернуть.")
        row += 1
        entry("mx", "x ОУ, м", "12000",
              "Начальная координата X ракеты (объекта управления) в метрах. Задаёт стартовую точку дискретной траектории ОУ.",
              parent=mu_body)
        entry("mz", "z ОУ, м", "-8000",
              "Начальная координата Z ракеты (ОУ) в метрах. Задаёт стартовую точку траектории.",
              parent=mu_body)
        entry("mspeed", "V ОУ, м/с", "450",
              "Модуль воздушной скорости ракеты V_р, м/с. Из него формируется вектор курса: V_возд = V_р · e_курс. Путевая скорость: V_пут = V_возд + W.",
              parent=mu_body)
        entry("mcourse", "курс ОУ, град", "120",
              "Начальный курс ракеты — угол вектора V_возд от оси OX, град. Используется на первом шаге; далее курс пересчитывается законом наведения.",
              parent=mu_body)
        entry("man", "a_n max, м/с²", "80",
              "Максимальная нормальная перегрузка ракеты a_n_max, м/с². Ограничивает скорость разворота: |Δϑ| / dt ≤ a_n_max / V_р. Если требуемый поворот превышает лимит — обрезается.",
              parent=mu_body)

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        os_container, os_body, os_header = self._make_collapsible(left, "ОС — объект сопровождения (цель)")
        os_container.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_help(os_header, "Блок начальных параметров цели (объекта сопровождения). ЛКМ по заголовку — свернуть/развернуть.")
        row += 1
        entry("tx", "x ОС, м", "0",
              "Начальная координата X цели (объекта сопровождения), м.",
              parent=os_body)
        entry("tz", "z ОС, м", "0",
              "Начальная координата Z цели (ОС), м.",
              parent=os_body)
        entry("tspeed", "V ОС, м/с", "120",
              "Модуль воздушной скорости цели V_ц, м/с.",
              parent=os_body)
        entry("tcourse", "курс ОС, град", "0",
              "Курс цели (угол вектора скорости от оси OX), град. Цель движется равномерно прямолинейно (в текущей модели — без манёвров).",
              parent=os_body)

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        params_container, params_body, params_header = self._make_collapsible(left, "Параметры расчета")
        params_container.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_help(params_header, "Численные параметры моделирования (ветер, dt, t_max, R_контакта) и коэффициенты выбранного метода. ЛКМ по заголовку — свернуть/развернуть.")
        row += 1

        entry("windx", "ветер x, м/с", "0",
              "Компонента вектора ветра W по X, м/с. Прибавляется к воздушной скорости: V_пут = V_возд + W. Именно ветер делает поведение метода погони отличным от прямого метода.",
              parent=params_body)
        entry("windz", "ветер z, м/с", "0",
              "Компонента вектора ветра W по Z, м/с.",
              parent=params_body)
        entry("dt", "dt моделирования, с", "5",
              "Шаг дискретного моделирования, с. Уменьшение dt повышает точность интегрирования кинематики и плотность точек траектории, но увеличивает время счёта.",
              parent=params_body)
        entry("tmax", "t_max, с", "120",
              "Максимальное модельное время симуляции, с. Симуляция останавливается по достижении t_max, условного контакта или начала расхождения.",
              parent=params_body)
        entry("hit", "радиус условного контакта, м", "25",
              "Радиус условного контакта R_к, м. При d ≤ R_к фиксируется попадание (hit=True).",
              parent=params_body)

        nr = params_body.grid_size()[1]
        self.navconst_label = ttk.Label(params_body, text="N наведения")
        self.navconst_label.grid(row=nr, column=0, sticky="w", pady=2)
        self.navconst_frame = ttk.Frame(params_body)
        self.navconst_frame.grid(row=nr, column=1, sticky="ew", pady=2)
        self.navconst_frame.columnconfigure(0, weight=1)
        self.navconst_var = tk.DoubleVar(value=3.0)
        self.navconst_label_var = tk.StringVar(value="3.00")

        def update_navconst_label(value):
            self.navconst_label_var.set(f"{float(value):.2f}")

        navconst_scale = ttk.Scale(
            self.navconst_frame,
            from_=3.0,
            to=20.0,
            variable=self.navconst_var,
            command=update_navconst_label,
        )
        navconst_scale.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(self.navconst_frame, textvariable=self.navconst_label_var, width=5).grid(row=0, column=1, sticky="e")
        navconst_help = (
            "Навигационная константа N (для пропорционального и ПД методов).\n"
            "Формула пропорционального метода:\n"
            "    a_n_cmd = N · V_сбл · λ̇.\n"
            "Чем больше N, тем агрессивнее реакция на скорость вращения ЛВ. "
            "Типичный диапазон 3…5; при N → ∞ метод сходится к параллельному "
            "сближению. В ПД методе это коэффициент Nп."
        )
        self._bind_help(self.navconst_label, navconst_help)
        self._bind_help(navconst_scale, navconst_help)

        lr = params_body.grid_size()[1]
        self.lead_angle_label = ttk.Label(params_body, text="ψ упреждения, град")
        self.lead_angle_label.grid(row=lr, column=0, sticky="w", pady=2)
        self.lead_angle_var = tk.StringVar(value="10")
        self.lead_angle_entry = ttk.Entry(params_body, textvariable=self.lead_angle_var, width=18)
        self.lead_angle_entry.grid(row=lr, column=1, sticky="ew", pady=2)
        lead_help = (
            "Постоянный угол упреждения ψ, град. Применяется только в методе "
            "«Прямой с постоянным углом упреждения». Курс задаётся как "
            "ϑ = ε + ψ, где ε — угол ЛВ. При ψ = 0 совпадает с прямым методом."
        )
        self._bind_help(self.lead_angle_label, lead_help)
        self._bind_help(self.lead_angle_entry, lead_help)

        dr = params_body.grid_size()[1]
        self.navconst_d_label = ttk.Label(params_body, text="Nд (дифф.)")
        self.navconst_d_label.grid(row=dr, column=0, sticky="w", pady=2)
        self.navconst_d_frame = ttk.Frame(params_body)
        self.navconst_d_frame.grid(row=dr, column=1, sticky="ew", pady=2)
        self.navconst_d_frame.columnconfigure(0, weight=1)
        self.navconst_d_var = tk.DoubleVar(value=0.5)
        self.navconst_d_label_var = tk.StringVar(value="0.50")

        def update_navconst_d_label(value):
            self.navconst_d_label_var.set(f"{float(value):.2f}")

        navconst_d_scale = ttk.Scale(
            self.navconst_d_frame,
            from_=0.0,
            to=5.0,
            variable=self.navconst_d_var,
            command=update_navconst_d_label,
        )
        navconst_d_scale.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(self.navconst_d_frame, textvariable=self.navconst_d_label_var, width=5).grid(row=0, column=1, sticky="e")
        navconst_d_help = (
            "Дифференциальный коэффициент Nд для пропорционально-"
            "дифференциального метода.\n"
            "Формула: a_n_cmd = V_сбл · (Nп · λ̇ + Nд · λ̈),\n"
            "где λ̈ ≈ (λ̇_t − λ̇_{t−dt}) / dt — численная оценка углового "
            "ускорения ЛВ. Дифференциальная часть упреждает ускорение цели "
            "и уменьшает промах при манёврах."
        )
        self._bind_help(self.navconst_d_label, navconst_d_help)
        self._bind_help(navconst_d_scale, navconst_d_help)

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="Память траекторий").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.save_history_var = tk.BooleanVar(value=True)
        save_history_cb = ttk.Checkbutton(left, text="Сохранять предыдущую траекторию", variable=self.save_history_var)
        save_history_cb.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bind_help(save_history_cb, "При включении после каждого «Построить» текущая траектория добавляется в список сохранённых для сравнения с новыми расчётами.")
        row += 1

        self.show_history_var = tk.BooleanVar(value=True)
        show_history_cb = ttk.Checkbutton(left, text="Показывать сохраненные траектории", variable=self.show_history_var, command=self._redraw_if_ready)
        show_history_cb.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bind_help(show_history_cb, "Показывать сохранённые ранее траектории на графике поверх текущей. Не влияет на сам список — только на отрисовку.")
        row += 1

        ttk.Label(left, text="Сохраненные траектории").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.history_listbox = tk.Listbox(left, selectmode=tk.EXTENDED, height=6, exportselection=False)
        self.history_listbox.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.history_listbox.bind("<<ListboxSelect>>", lambda event: self._redraw_if_ready())
        self._bind_help(self.history_listbox, "Список сохранённых траекторий. Выбор подмножества фильтрует, какие из них рисовать (если ни одна не выбрана — рисуются все).")
        row += 1

        history_btns = ttk.Frame(left)
        history_btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        history_btns.columnconfigure(0, weight=1)
        history_btns.columnconfigure(1, weight=1)
        btn_del_sel = ttk.Button(history_btns, text="Удалить выбранные", command=self.delete_selected_history)
        btn_del_sel.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        btn_del_all = ttk.Button(history_btns, text="Удалить все", command=self.clear_history)
        btn_del_all.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self._bind_help(btn_del_sel, "Удалить выбранные в списке сохранённые траектории.")
        self._bind_help(btn_del_all, "Удалить все сохранённые траектории.")
        row += 1

        history_color_label = ttk.Label(left, text="Цвет сохраненной")
        history_color_label.grid(row=row, column=0, sticky="w", pady=2)
        self.history_color_var = tk.StringVar(value="серый")
        history_color_combo = ttk.Combobox(left, textvariable=self.history_color_var, values=list(COLOR_MAP.keys()), state="readonly")
        history_color_combo.grid(row=row, column=1, sticky="ew", pady=2)
        color_help = "Цвет, которым будет нарисована следующая сохраняемая траектория (для отличия от текущей чёрно-красной)."
        self._bind_help(history_color_label, color_help)
        self._bind_help(history_color_combo, color_help)
        row += 1

        history_style_label = ttk.Label(left, text="Тип линии")
        history_style_label.grid(row=row, column=0, sticky="w", pady=2)
        self.history_style_var = tk.StringVar(value="штриховая")
        history_style_combo = ttk.Combobox(left, textvariable=self.history_style_var, values=list(STYLE_MAP.keys()), state="readonly")
        history_style_combo.grid(row=row, column=1, sticky="ew", pady=2)
        style_help = "Тип линии (сплошная, штриховая, штрихпунктирная, пунктирная) для следующей сохраняемой траектории."
        self._bind_help(history_style_label, style_help)
        self._bind_help(history_style_combo, style_help)
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(left, text="Отображение на шаге").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.show_xg_var = tk.BooleanVar(value=True)
        cb_xg = ttk.Checkbutton(left, text="Показывать Xg", variable=self.show_xg_var, command=self._redraw_if_ready)
        cb_xg.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bind_help(cb_xg, "Показывать на каждом шаге горизонтальную ось OXg — пунктирную опорную линию через текущую точку ОУ. Нужна для визуальной привязки углов ε и ϑ.")
        row += 1
        self.show_epsilon_var = tk.BooleanVar(value=True)
        cb_eps = ttk.Checkbutton(left, text="Показывать угол ε", variable=self.show_epsilon_var, command=self._redraw_if_ready)
        cb_eps.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bind_help(cb_eps, "Показывать дугу угла линии визирования ε (от оси OXg до направления на цель).")
        row += 1
        self.show_miss_var = tk.BooleanVar(value=True)
        cb_miss = ttk.Checkbutton(left, text="Показывать промах h", variable=self.show_miss_var, command=self._redraw_if_ready)
        cb_miss.grid(row=row, column=0, columnspan=2, sticky="w")
        self._bind_help(cb_miss, "Показывать мгновенный промах h на каждом шаге траектории. Формула:\n    h = d² · λ̇ / V_р,\nгде d — текущая дистанция, λ̇ — угловая скорость ЛВ, V_р — скорость ракеты.")
        row += 1

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for i in range(4):
            btns.columnconfigure(i, weight=1)
        btn_build = ttk.Button(btns, text="Построить", command=self.build_trajectory)
        btn_build.grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_prev = ttk.Button(btns, text="Назад", command=self.prev_step, state="disabled")
        self.btn_prev.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_next = ttk.Button(btns, text="Вперед", command=self.next_step, state="disabled")
        self.btn_next.grid(row=0, column=2, sticky="ew", padx=2)
        btn_clear = ttk.Button(btns, text="Очистить", command=self.clear_all)
        btn_clear.grid(row=0, column=3, sticky="ew", padx=2)
        self._bind_help(btn_build, "Запускает симуляцию по введённым параметрам и рисует траекторию (с подсветкой текущего шага).")
        self._bind_help(self.btn_prev, "Перейти на один дискретный шаг назад. Параметры шага и графика обновляются.")
        self._bind_help(self.btn_next, "Перейти на один дискретный шаг вперёд.")
        self._bind_help(btn_clear, "Очистить текущий результат и график. Сохранённые траектории при этом остаются — их удаляют отдельными кнопками.")
        row += 1

        self.info = tk.StringVar(value="Выберите метод, задайте исходные данные и постройте дискретную траекторию.")
        info_label = ttk.Label(left, textvariable=self.info, justify="left", wraplength=400)
        info_label.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._bind_help(info_label, "Краткий итог последней симуляции: использованный метод, число дискретных точек, причина останова (контакт / расхождение / достигнут t_max).")
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(left, text="Параметры текущего шага").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.step_info = tk.StringVar(value="Нет данных")
        step_info_label = ttk.Label(left, textvariable=self.step_info, justify="left", wraplength=400, relief="solid", padding=8)
        step_info_label.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_help(step_info_label, (
            "Параметры текущего дискретного шага:\n"
            "  t — модельное время; d — дистанция ОУ-цель;\n"
            "  ε — угол ЛВ от OXg; ϑ — курс ОУ; jц = ϑ − ε;\n"
            "  qр / qц — углы вектора скорости ОУ/цели относительно ЛВ;\n"
            "  ωε = λ̇ — угловая скорость ЛВ; λ̈ — её производная (только ПД);\n"
            "  h — мгновенный промах: h = d²·λ̇ / V_р;\n"
            "  V_сбл = −e_ЛВ · (V_ц − V_ОУ) — скорость сближения;\n"
            "  a_n — реализованная нормальная перегрузка;\n"
            "  Δ — невязка между требуемым и реализованным управлением."
        ))
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(left, text="Оценка эффективности").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.efficiency_info = tk.StringVar(value="Нет данных")
        efficiency_label = ttk.Label(left, textvariable=self.efficiency_info, justify="left", wraplength=400, relief="solid", padding=8)
        efficiency_label.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._bind_help(efficiency_label, (
            "Сводка эффективности по всей траектории:\n"
            "  Рейтинг: высокая (контакт) / низкая (расхождение) / не достигнута.\n"
            "  R_min — минимальная дистанция между ОУ и целью за весь полёт и время её достижения.\n"
            "  h_min, h_avg — минимальный и средний промах h по всем шагам.\n"
            "  a_n max — максимум реализованной нормальной перегрузки."
        ))

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

    def _bind_help(self, widget, text):
        widget.bind("<Button-3>", lambda event, t=text: self._show_help_popup(event, t))

    def _make_collapsible(self, parent, title, expanded=True):
        container = ttk.Frame(parent)
        container.columnconfigure(0, weight=1)
        body = ttk.Frame(container)
        body.columnconfigure(1, weight=1)
        arrow_var = tk.StringVar()
        state = {"expanded": bool(expanded)}

        def update_header():
            arrow = "▼" if state["expanded"] else "▶"
            arrow_var.set(f"{arrow}  {title}")

        def toggle(_event=None):
            state["expanded"] = not state["expanded"]
            if state["expanded"]:
                body.grid()
            else:
                body.grid_remove()
            update_header()

        header = ttk.Label(
            container,
            textvariable=arrow_var,
            cursor="hand2",
            font=("TkDefaultFont", 10, "bold"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(2, 2))
        header.bind("<Button-1>", toggle)
        body.grid(row=1, column=0, sticky="ew")
        update_header()
        if not state["expanded"]:
            body.grid_remove()
        return container, body, header

    def _show_help_popup(self, event, text):
        existing = getattr(self, "_help_popup", None)
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass
            self._help_popup = None

        top = tk.Toplevel(self.winfo_toplevel())
        top.wm_overrideredirect(True)
        top.attributes("-topmost", True)
        border = tk.Frame(top, bg="#bda842")
        border.pack()
        lbl = tk.Label(
            border,
            text=text,
            justify="left",
            wraplength=460,
            bg="#fff8c8",
            fg="#222",
            padx=10,
            pady=8,
            font=("TkDefaultFont", 9),
            anchor="w",
        )
        lbl.pack(fill="both", padx=1, pady=1)
        top.update_idletasks()
        x = int(event.x_root) + 12
        y = int(event.y_root) + 12
        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()
        win_w = top.winfo_reqwidth()
        win_h = top.winfo_reqheight()
        if x + win_w > screen_w:
            x = max(0, screen_w - win_w - 4)
        if y + win_h > screen_h:
            y = max(0, screen_h - win_h - 4)
        top.geometry(f"+{x}+{y}")

        def dismiss(_event=None):
            try:
                top.destroy()
            except tk.TclError:
                pass

        top.bind("<Button-1>", dismiss)
        top.bind("<Button-3>", dismiss)
        border.bind("<Button-1>", dismiss)
        lbl.bind("<Button-1>", dismiss)
        top.bind_all("<Escape>", dismiss, add="+")
        self._help_popup = top

    def _sync_method_controls(self):
        method = self.method_var.get()
        if method in (PROPORTIONAL_METHOD, PD_METHOD):
            self.navconst_label.grid()
            self.navconst_frame.grid()
        else:
            self.navconst_label.grid_remove()
            self.navconst_frame.grid_remove()
        if method == PD_METHOD:
            self.navconst_d_label.grid()
            self.navconst_d_frame.grid()
        else:
            self.navconst_d_label.grid_remove()
            self.navconst_d_frame.grid_remove()
        if method == DIRECT_LEAD_METHOD:
            self.lead_angle_label.grid()
            self.lead_angle_entry.grid()
        else:
            self.lead_angle_label.grid_remove()
            self.lead_angle_entry.grid_remove()
        if hasattr(self, "method_description_text"):
            self._populate_description(method)

    def _render_formula(self, latex: str, fontsize: int = 12,
                        color: str = "#222", bg: str = "#fffbf2") -> tk.PhotoImage:
        cache_key = (latex, fontsize, color, bg)
        cached = self._formula_cache.get(cache_key)
        if cached is not None:
            return cached
        fig = Figure(figsize=(8.0, 1.0), dpi=130)
        fig.patch.set_facecolor(bg)
        fig.text(0.02, 0.5, latex, fontsize=fontsize, color=color,
                 va="center", ha="left")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0.06, facecolor=bg)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        photo = tk.PhotoImage(data=data)
        self._formula_cache[cache_key] = photo
        return photo

    def _populate_description(self, method: str) -> None:
        info = METHOD_INFO.get(method)
        txt = self.method_description_text
        txt.configure(state="normal")
        txt.delete("1.0", "end")

        if info is None:
            txt.insert("end", "Описание для метода не задано.", "body")
            txt.configure(state="disabled")
            return

        section = info.get("section", "")
        title = info.get("title", method)
        header = f"{section}. {title}" if section else title
        txt.insert("end", f"{header}\n", "title")

        definition = info.get("definition")
        if definition:
            txt.insert("end", "\nОпределение.\n", "subhead")
            txt.insert("end", f"{definition}\n", "body")

        formulas = info.get("formulas") or []
        if formulas:
            txt.insert("end", "\nФормулы:\n", "subhead")
            for latex, caption in formulas:
                try:
                    img = self._render_formula(latex)
                    txt.image_create("end", image=img, padx=6, pady=2)
                    txt.insert("end", "\n")
                except Exception:
                    txt.insert("end", f"{latex}\n", "body")
                if caption:
                    txt.insert("end", f"{caption}\n", "caption")

        notes = info.get("notes")
        if notes:
            txt.insert("end", "\nПримечания.\n", "subhead")
            txt.insert("end", f"{notes}\n", "body")

        advantages = info.get("advantages") or []
        if advantages:
            txt.insert("end", "\nДостоинства:\n", "subhead")
            for item in advantages:
                txt.insert("end", f"• {item}\n", "bullet")

        disadvantages = info.get("disadvantages") or []
        if disadvantages:
            txt.insert("end", "\nНедостатки:\n", "subhead")
            for item in disadvantages:
                txt.insert("end", f"• {item}\n", "bullet")

        txt.configure(state="disabled")

    def _result_display_name(self):
        if self.result is None:
            return ""
        if self.result.method == PROPORTIONAL_METHOD:
            return f"{self.result.method}, N={self.navconst_var.get():.2f}"
        if self.result.method == PD_METHOD:
            return f"{self.result.method}, Nп={self.navconst_var.get():.2f}, Nд={self.navconst_d_var.get():.2f}"
        if self.result.method == DIRECT_LEAD_METHOD:
            try:
                lead = float(self.lead_angle_var.get().replace(",", ".").strip())
            except ValueError:
                lead = 0.0
            return f"{self.result.method}, ψ={lead:.2f}°"
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
            elif method == DIRECT_LEAD_METHOD:
                lead_angle = float(self.lead_angle_var.get().replace(",", ".").strip())
                self.result = simulate_direct_lead_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc,
                    lead_angle
                )
            elif method == PURSUIT_METHOD:
                self.result = simulate_pursuit_discrete(
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
            elif method == PD_METHOD:
                self.result = simulate_pd_discrete(
                    missile_pos, missile_speed, missile_course,
                    target_pos, target_speed, target_course,
                    wind, dt, tmax, hit, max_normal_acc,
                    self.navconst_var.get(),
                    self.navconst_d_var.get()
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

    def _draw_direct_lead_step(self, state, step_idx, p, c):
        los = c - p
        los_dir = normalize(los)
        vr_dir = normalize(state.missile_air_vel)
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
        lead = state.params.get("lead_angle_deg", 0.0)
        delta = state.params["delta"]
        if self.show_epsilon_var.get():
            self._draw_angle_arc(p, np.array([1.0, 0.0]), los_dir, 380.0, "#ff8c00", "ε", text_shift=(18.0, 12.0), linewidth=2.0)
        self._draw_angle_arc(p, los_dir, vr_dir, 560.0, "#0057b8", "ψ", text_shift=(24.0, 20.0), linewidth=3.0, fontsize=13, min_visual_angle_deg=4.0)

        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"ψ = {lead:.2f}°\n"
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

    def _draw_pursuit_step(self, state, step_idx, p, c):
        los = c - p
        los_dir = normalize(los)
        vr_air_dir = normalize(state.missile_air_vel)
        vr_ground_dir = normalize(state.missile_ground_vel)
        vt_dir = normalize(state.target_ground_vel)

        self.ax.plot([p[0]], [p[1]], "ko", ms=7, zorder=7)
        self.ax.plot([c[0]], [c[1]], "o", color="red", ms=7, zorder=7)

        if self.show_xg_var.get():
            self.ax.plot([p[0] - 1600, p[0] + 2600], [p[1], p[1]], color="0.45", linestyle="--", linewidth=1.1)
            self.ax.text(p[0] + 2650, p[1] + 20, "OXg", color="0.35", fontsize=10)

        self.ax.plot([p[0], c[0]], [p[1], c[1]], color="#ff8c00", linewidth=1.6, linestyle="-", zorder=6)
        self.ax.text((p[0] + c[0]) * 0.5, (p[1] + c[1]) * 0.5 + 120, "ЛВ", color="#ff8c00", fontsize=11)

        vr_end = p + vr_air_dir * 900.0
        vg_end = p + vr_ground_dir * 950.0
        self.ax.arrow(p[0], p[1], vr_end[0] - p[0], vr_end[1] - p[1], color="#1f77b4", width=2.8, head_width=42.0, length_includes_head=True, zorder=5)
        self.ax.text(vr_end[0] + 35, vr_end[1] + 25, "Vр (курс)", color="#1f77b4", fontsize=11)
        self.ax.arrow(p[0], p[1], vg_end[0] - p[0], vg_end[1] - p[1], color="#2ca02c", width=2.2, head_width=38.0, length_includes_head=True, zorder=5, alpha=0.9)
        self.ax.text(vg_end[0] + 35, vg_end[1] - 60, "Vпут", color="#2ca02c", fontsize=11)

        vt_end = c + vt_dir * 850.0
        self.ax.arrow(c[0], c[1], vt_end[0] - c[0], vt_end[1] - c[1], color="red", width=2.6, head_width=40.0, length_includes_head=True, zorder=5)
        self.ax.text(vt_end[0] + 40, vt_end[1] + 35, "Vц", color="red", fontsize=11)

        eps = state.params["eps_deg"]
        theta = state.params["theta_deg"]
        gamma = state.params.get("gamma_deg", theta)
        q = state.params["q_deg"]
        delta = state.params["delta"]
        if self.show_epsilon_var.get():
            self._draw_angle_arc(p, np.array([1.0, 0.0]), los_dir, 380.0, "#ff8c00", "ε", text_shift=(18.0, 12.0), linewidth=2.0)
        self._draw_angle_arc(p, los_dir, vr_air_dir, 560.0, "#0057b8", "ϑ−ε", text_shift=(24.0, 20.0), linewidth=2.4, fontsize=12, min_visual_angle_deg=4.0)

        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"ε = {eps:.2f}°\n"
            f"ϑ = {theta:.2f}°\n"
            f"γ (путевой) = {gamma:.2f}°\n"
            f"q (Vпут к ЛВ) = {q:.2f}°\n"
            f"ωε = {state.params['los_rate_deg_s']:.3f}°/с\n"
            f"h = {state.params['miss_abs']:.2f} м\n"
            f"Δ = γ − ε = {delta:.2f}°"
        )
        self.step_info.set(panel_text)
        self.ax.text(
            0.02, 0.02, panel_text,
            transform=self.ax.transAxes,
            ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.9),
            zorder=20,
        )

    def _draw_pd_step(self, state, step_idx, p, c):
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

        lambda_dot_deg = float(np.rad2deg(state.params.get("lambda_dot_rad_s", 0.0)))
        lambda_ddot_deg = float(np.rad2deg(state.params.get("lambda_ddot_rad_s2", 0.0)))
        panel_text = (
            f"t = {self.sampled_times[step_idx]:.2f} с\n"
            f"d = {state.distance:.2f} м\n"
            f"Nп = {state.params['nav_const_p']:.2f}, Nд = {state.params['nav_const_d']:.2f}\n"
            f"ε = {state.params['eps_deg']:.2f}°\n"
            f"ϑр = {state.params['theta_deg']:.2f}°\n"
            f"qр = {state.params['q_deg']:.2f}°\n"
            f"λ̇ = {lambda_dot_deg:.3f}°/с\n"
            f"λ̈ = {lambda_ddot_deg:.3f}°/с²\n"
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

        method = self.result.method
        if method == "Прямой метод":
            self._draw_direct_step(state, step_idx, p, c)
        elif method == DIRECT_LEAD_METHOD:
            self._draw_direct_lead_step(state, step_idx, p, c)
        elif method == PURSUIT_METHOD:
            self._draw_pursuit_step(state, step_idx, p, c)
        elif method == "Параллельное сближение":
            self._draw_parallel_step(state, step_idx, p, c)
        elif method == PD_METHOD:
            self._draw_pd_step(state, step_idx, p, c)
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
