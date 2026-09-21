import os
import time
import random

from glm import vec2, ivec2

from NeuralNet.FNN import *
import Graphics


def sample_learning(nn: FNN, learn_rate, data_set, epoch_count, sample_rate):  
  print(f"training for {epoch_count} epochs...")
  for i in range(10):
    prev_time = time.perf_counter()
    nn.learn(sample_rate, learn_rate, data_set)
    sample_time = int((time.perf_counter() - prev_time) * 1000)
    loss = nn.loss(data_set[0], nn.forward(data_set[0].inputs))
    print(f"{sample_time}ms | loss: {loss} | epochs: {sample_rate * (i + 1)}")
  print("training complete")


data_set = []
for i in range(20):
  start = vec2(random.uniform(0,1), random.uniform(0,1))
  end = vec2(random.uniform(0,1), random.uniform(0,1))
  output = end - start

  data_set.append(DataPoint(numpy.array([start.x,start.y, end.x,end.y]), numpy.array([output.x,output.y])))

fnn = FNN()
fnn.add_layer(Dense(4,8))
fnn.add_layer(Dense(8,2))


epoch_count = int(input("Enter Epoch Count: "))
os.system("cls")

sample_learning(fnn, .01, data_set, epoch_count, epoch_count // 10)


Graphics.init_window(ivec2(800,600), "Machine Learning -- Python")
Graphics.canvas.config(background="black")

Graphics.init_model_instances(50)

def update() -> None:
  delta_time = Graphics.update_delta_time()
  window_pos = vec2(Graphics.root.winfo_x(), Graphics.root.winfo_y())
  mouse_pos = vec2(Graphics.root.winfo_pointerxy()) - window_pos - vec2(9,31)
  norm_mouse = vec2(mouse_pos / vec2(Graphics.window_size))

  for i in range(len(Graphics.model_instances)):
    start = Graphics.model_instances[i][1]

    output = fnn.forward(numpy.array([start.x,start.y, norm_mouse.x,norm_mouse.y])).flatten()
    Graphics.model_instances[i][1] += vec2(output[0], output[1]) * delta_time * 2

  Graphics.update_model_instances()
  Graphics.root.after(1, update)

Graphics.root.after(1, update)
Graphics.root.mainloop()