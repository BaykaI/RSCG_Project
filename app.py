import tkinter as tk
from ui.main_window import MainWindow

def main() -> None:
    root = tk.Tk()
    root.title("Демонстрация методов наведения")
    root.geometry("1720x980")
    root.minsize(1450, 880)
    app = MainWindow(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()
