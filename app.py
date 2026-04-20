import os
import sys
import tempfile
from pathlib import Path

def _configure_frozen_runtime() -> None:
    if not getattr(sys, "frozen", False):
        return

    base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    tcl_candidates = [
        base_dir / "lib" / "tcl8.6",
        base_dir / "_internal" / "lib" / "tcl8.6",
        base_dir / "_tcl_data",
        base_dir / "_internal" / "_tcl_data",
    ]
    tk_candidates = [
        base_dir / "lib" / "tk8.6",
        base_dir / "_internal" / "lib" / "tk8.6",
        base_dir / "_tk_data",
        base_dir / "_internal" / "_tk_data",
    ]

    for tcl_dir in tcl_candidates:
        if tcl_dir.is_dir():
            os.environ["TCL_LIBRARY"] = str(tcl_dir)
            break

    for tk_dir in tk_candidates:
        if tk_dir.is_dir():
            os.environ["TK_LIBRARY"] = str(tk_dir)
            break

    mpl_config_dir = Path(tempfile.gettempdir()) / "rscg_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def main() -> None:
    _configure_frozen_runtime()
    import tkinter as tk
    from ui.main_window import MainWindow

    root = tk.Tk()
    root.title("Демонстрация методов наведения")
    root.geometry("1720x980")
    root.minsize(1450, 880)
    app = MainWindow(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()
