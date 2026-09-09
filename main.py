import os

from FNN import *


training_data = [
  DataPoint(
    Matrix.init_from_data([[1,0]]),
    Matrix.init_from_data([[1,0]])
  )
]

neural_network = FNN([2,2])


os.system("cls")

#Neural Network Layers
for layer in neural_network.layers:
  print(f"weights: {layer.weights.data}")
  print(f"biases: {layer.biases.data}\n")
#Neural Network Layers

#Pre Training Results
for data_point in training_data:
  output = neural_network.calculate_outputs(data_point.inputs)

  print(f"Input: {data_point.inputs.data[0]}")
  print(f"Output: {output.data[0]}")
  print(f"Loss: {neural_network.calculate_loss(data_point, output)}\n")

input()
os.system("cls")
#Pre Training Results

#Training
LEARN_RATE = .1
NUDGE = .001

iteration_count = int(input("Enter Iteration Count: "))
os.system("cls")

print(f"training for {iteration_count} iterations...")

for iteration in range(iteration_count):
  for data_point in training_data:
    pre_loss = neural_network.calculate_loss(data_point, neural_network.calculate_outputs(data_point.inputs))

    output_layer = neural_network.layers[-1]

    for i in range(output_layer.output_count):
      for j in range(output_layer.input_count):
        output_layer.weights[j][i] += NUDGE
        post_loss = neural_network.calculate_loss(data_point, neural_network.calculate_outputs(data_point.inputs))
        output_layer.weights[j][i] -= NUDGE

        output_layer.weights[j][i] -= LEARN_RATE * ((post_loss - pre_loss) / NUDGE)
    
    for i in range(output_layer.output_count):
      output_layer.biases[0][i] += NUDGE;
      post_loss = neural_network.calculate_loss(data_point, neural_network.calculate_outputs(data_point.inputs))
      output_layer.biases[0][i] -= NUDGE;

      output_layer.biases[0][i] -= LEARN_RATE * ((post_loss - pre_loss) / NUDGE)

print("training finished.")
input()
#Training

#Post Training Results
for data_point in training_data:
  output = neural_network.calculate_outputs(data_point.inputs)

  print(f"Input: {data_point.inputs.data[0]}")
  print(f"Output: {output.data[0]}")
  print(f"Loss: {neural_network.calculate_loss(data_point, output)}\n")

input()
#Post Training Results

#Neural Network Layers
for layer in neural_network.layers:
  print(f"weights: {layer.weights.data}")
  print(f"biases: {layer.biases.data}\n")
#Neural Network Layers