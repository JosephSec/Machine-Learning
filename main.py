import os
import random
from glm import vec2

import VisualTest

from NeuralNet.FNN import *


training_data = []
for i in range(100):
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
os.system("cls")

pre_loss = fnn.loss(training_data[0], fnn.forward(training_data[0].inputs))

print(f"training for {epoch_count} epochs...")
fnn.learn(epoch_count, .01, training_data)

print(f"\npre training loss: {pre_loss}")
print(f"post training loss: {fnn.loss(training_data[0], fnn.forward(training_data[0].inputs))}")

input("Press Enter to Continue...")


VisualTest.init_window()

def create_circle(x, y, r, **kwargs):
  return VisualTest.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)
create_circle(VisualTest.window_size.x // 2, VisualTest.window_size.y // 2, 25, fill="green", width=0)

def update() -> None:
  delta_time = VisualTest.update_delta_time()
  mouse_pos = vec2(VisualTest.root.winfo_pointerxy()) - window_pos - vec2(9,31)
  window_pos = vec2(VisualTest.root.winfo_x(), VisualTest.root.winfo_y())
  norm_mouse = vec2(mouse_pos.x / window_size)


  print(norm_mouse)

  VisualTest.root.after(1, update)

update()
VisualTest.root.mainloop()