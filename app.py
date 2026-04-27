import tkinter as tk
from ui.main_window import MainWindow

def main() -> None:
    root = tk.Tk()
    root.title("Методы наведения — дискретное графическое построение")
    root.geometry("1650x950")
    root.minsize(1400, 860)
    MainWindow(root).pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()
