import os
import random
import time
from glm import vec2

import VisualTest

from NeuralNet.FNN import *


training_data = []
for i in range(20):
  a = vec2(random.uniform(0,1), random.uniform(0,1))
  b = vec2(random.uniform(0,1), random.uniform(0,1))

  training_data.append(DataPoint(
    Tensor(1,4).load_list([a.x,a.y, b.x,b.y]),
    Tensor(1,2).load_list([b.x-a.x, b.y-a.y])
  ))

fnn = FNN()
fnn.add_layer(Layer.Dense(4,8))
fnn.add_layer(Layer.Dense(8,2))


epoch_count = int(input("Enter Epoch Count: "))
epoch_sample_rate = epoch_count // 10
os.system("cls")

pre_loss = fnn.loss(training_data[0], fnn.forward(training_data[0].inputs))

print(f"training for {epoch_count} epochs...")
for i in range(10):
  prev_time = time.perf_counter()
  fnn.learn(epoch_sample_rate, .01, training_data)

  sample_time = int((time.perf_counter() - prev_time) * 1000)
  loss = fnn.loss(training_data[0], fnn.forward(training_data[0].inputs))
  print(f"{sample_time}ms | loss: {loss} | epochs: {epoch_sample_rate * (i + 1)}")
print("training complete")

print(f"\npre training loss: {pre_loss}")
print(f"post training loss: {fnn.loss(training_data[0], fnn.forward(training_data[0].inputs))}")

input("Press Enter to Continue...")


VisualTest.init_window(VisualTest.ivec2(800,600), "Neural Network")

def create_circle(x, y, r, **kwargs):
  return VisualTest.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)


POINT_RADIUS = 25
mouse_shape = create_circle(0,0, POINT_RADIUS, fill="red", width=0)
model_positions = []
for i in range(10):
  pos = vec2(random.uniform(0,1), random.uniform(0,1))
  model_positions.append([create_circle(0,0, POINT_RADIUS, fill="green", width=0), vec2(pos.x,pos.y)])

def update() -> None:
  window_pos = vec2(VisualTest.root.winfo_x(), VisualTest.root.winfo_y())
  window_size = vec2(VisualTest.root.winfo_width(), VisualTest.root.winfo_height())
  mouse_pos = vec2(VisualTest.root.winfo_pointerxy()) - window_pos - vec2(9,31)

  delta_time = VisualTest.update_delta_time()

  norm_mouse = vec2(mouse_pos / window_size)
  mouse_shape_pos = norm_mouse * window_size - vec2(POINT_RADIUS,POINT_RADIUS)
  VisualTest.canvas.moveto(mouse_shape, mouse_shape_pos.x, mouse_shape_pos.y)

  for i in range(len(model_positions)):
    shape = model_positions[i][0]
    position = model_positions[i][1]

    output = fnn.forward(Tensor(1,4).load_list([position.x,position.y, norm_mouse.x,norm_mouse.y]))
    new_position = position + vec2(output.m_data[0], output.m_data[1]) * delta_time
    model_positions[i][1] = vec2(min(1, max(0, new_position.x)), min(1, max(0, new_position.y)))

    model_shape_pos = position * window_size - vec2(POINT_RADIUS,POINT_RADIUS)

    VisualTest.canvas.moveto(shape, model_shape_pos.x, model_shape_pos.y)


  VisualTest.root.after(10, update)

update()
VisualTest.root.mainloop()