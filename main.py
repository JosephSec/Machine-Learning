import os

import NeuralNetTest

from NeuralNet.FNN import *
import Graphics


model_name = "FollowPoint"
NeuralNetTest.load_model(model_name)
NeuralNetTest.load_data_set(model_name)

epoch_count = int(input("Enter Epoch Count: "))
os.system("cls")

NeuralNetTest.sample_learning(NeuralNetTest.fnn, .01, NeuralNetTest.data_set, epoch_count, epoch_count // 10)
NeuralNetTest.fnn.save_file(f"Tests/{model_name}/post_trained.txt")


Graphics.init_window(Graphics.ivec2(800,600), "Machine Learning -- Python")
Graphics.canvas.config(background="black")

Graphics.init_model_instances(50)

def update() -> None:
  NeuralNetTest.update_graphics(model_name)
  Graphics.root.after(1, update)

Graphics.root.after(1, update)
Graphics.root.mainloop()