import tkinter as tk
from tkinter import messagebox

from glm import ivec2

import time

previous_time = time.perf_counter()
delta_time = 0
def update_delta_time() -> float:
  global previous_time, delta_time

  cur_time = time.perf_counter()
  delta_time = cur_time - previous_time
  previous_time = cur_time

  return delta_time

root: tk.Tk = None
canvas: tk.Canvas = None

window_size: ivec2 = None


def init_window(win_size=ivec2(800,600), win_name="Window") -> None:
  global window_size, root
  window_size = win_size

  root = tk.Tk()
  root.title(win_name)
  root.geometry(str(window_size.x) + "x" + str(window_size.y) + "+0+0")

  init_canvas(root)

def init_canvas(root) -> None:
  global canvas
  canvas = tk.Canvas(root, width=window_size.x, height=window_size.y, bg="black")

  canvas.pack()

def end_window(event) -> None:
  response = messagebox.askyesno("Confirm Close", "Are you sure you want to close Tetris?")

  if response == True:
    root.destroy()