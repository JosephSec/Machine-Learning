import time
import glm
import random

from NeuralNet.FNN import *

import Graphics


fnn = FNN()
data_set = []

def load_model(model_name: str) -> None:
  global fnn
  
  if fnn.load_file(f"Tests/{model_name}/post_trained.txt") == True:
    fnn.save_file(f"Tests/{model_name}/post_load_file.txt")
  else:
    match(model_name):
      case "FollowPoint":
        fnn.add_layer(Dense(4,6))
        fnn.add_layer(Dense(6,2))
      case "Driver":
        fnn.add_layer(Dense(8,16))
        fnn.add_layer(Dense(16,2))

  fnn.save_file(f"Tests/{model_name}/pre_trained.txt")

def load_data_set(model_name: str) -> None:
  global data_set

  match(model_name):
    case "FollowPoint":
      for i in range(20):
        start = glm.vec2(random.uniform(0,1), random.uniform(0,1))
        end = glm.vec2(random.uniform(0,1), random.uniform(0,1))
        output = end - start

        data_set.append(DataPoint(numpy.array([start.x,start.y, end.x,end.y]), numpy.array([output.x,output.y])))


def update_graphics(model_name: str) -> None:
  match(model_name):
    case "FollowPoint":
      delta_time = Graphics.update_delta_time()
      window_pos = glm.vec2(Graphics.root.winfo_x(), Graphics.root.winfo_y())
      mouse_pos = glm.vec2(Graphics.root.winfo_pointerxy()) - window_pos - glm.vec2(9,31)
      norm_mouse = glm.vec2(mouse_pos / glm.vec2(Graphics.window_size))

      for i in range(len(Graphics.model_instances)):
        start = Graphics.model_instances[i][1]

        output = fnn.forward(numpy.array([start.x,start.y, norm_mouse.x,norm_mouse.y])).flatten()
        Graphics.model_instances[i][1] += glm.vec2(output[0], output[1]) * delta_time * 2

      Graphics.update_model_instances()



def sample_learning(nn: FNN, learn_rate, data_set, epoch_count, sample_rate):  
  print(f"training for {epoch_count} epochs...")
  for i in range(10):
    prev_time = time.perf_counter()
    nn.learn(sample_rate, learn_rate, data_set)
    sample_time = int((time.perf_counter() - prev_time) * 1000)
    loss = nn.loss(data_set[0], nn.forward(data_set[0].inputs))
    print(f"{sample_time}ms | loss: {loss} | epochs: {sample_rate * (i + 1)}")
  print("training complete")