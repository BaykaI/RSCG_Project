import tkinter as tk
from ui.main_window import MainWindow

def main():
    root = tk.Tk()
    root.title("Прямое наведение — пошаговое графическое построение")
    root.geometry("1550x920")
    root.minsize(1320, 800)
    MainWindow(root).pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()
