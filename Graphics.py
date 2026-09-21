import tkinter as tk
from tkinter import messagebox

from glm import vec2, ivec2
import random
import time


root: tk.Tk = None
canvas: tk.Canvas = None

window_size: ivec2 = None


delta_time = 0
previous_time = time.perf_counter()
def update_delta_time() -> float:
  global delta_time, previous_time
  current_time = time.perf_counter()
  delta_time = current_time - previous_time
  previous_time = current_time
  return delta_time


POINT_RADIUS = 15
def create_circle(x, y, r, **kwargs):
  return canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)

model_instances = []
def init_model_instances(count: int) -> None:
  for i in range(count):
    position = vec2(random.uniform(0,1), random.uniform(0,1))
    screen_position = vec2(position * vec2(window_size)) - vec2(1,1) * POINT_RADIUS

    model_instances.append([
      create_circle(screen_position.x,screen_position.y, POINT_RADIUS, fill="#00FF00", width=0),
      position
    ])
def update_model_instances() -> None:
  for instance in model_instances:
    screen_position = vec2(instance[1] * vec2(window_size)) - vec2(1,1) * POINT_RADIUS
    canvas.moveto(instance[0], x=screen_position.x, y=screen_position.y)


def init_window(win_size: ivec2, win_name = "Window") -> None:
  global window_size, root
  window_size = win_size

  root = tk.Tk()
  root.title(win_name)
  root.geometry(str(window_size.x) + "x" + str(window_size.y) + "+0+0")

  init_canvas(root)

def init_canvas(root) -> None:
  global canvas
  canvas = tk.Canvas(root, width=window_size.x, height=window_size.y)

  canvas.pack()

def end_window(event) -> None:
  response = messagebox.askyesno("Confirm Close", "Are you sure you want to close Tetris?")

  if response == True:
    root.destroy()