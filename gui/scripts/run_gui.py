from pathlib import Path
import sys
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.fl_pso_gui import FLPSO_GUI


def main():
    root = tk.Tk()
    app = FLPSO_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
