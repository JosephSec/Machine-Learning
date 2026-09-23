import ast
import numpy

from NeuralNet.Dense import *

__all__ = ["FNN", "DataPoint", "Dense", "numpy"]


NUDGE = 1e-20

def MSE(output: float, expected_output: float):
  error = output - expected_output
  return (error * error) * .5


class DataPoint:
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs

class FNN:
  def __init__(self):
    self.layers: list[Dense] = []

  def save_file(self, file_name: str) -> bool:
    try:
      with open(file_name, "w") as file:
        for layer in self.layers:
          file.write(f"\nDense Layer ({layer.inputs}, {layer.outputs}) [\n")

          file.write(f"\tweights: [\n")
          for row in layer.weights:
            row_str = numpy.array2string(row, max_line_width=numpy.inf)
            file.write(f"\t\t{row_str}\n")
          file.write(f"\t]\n")

          file.write(f"\tbiases: [\n")
          for row in layer.biases:
            row_str = numpy.array2string(row, max_line_width=numpy.inf)
            file.write(f"\t\t{row_str}\n")
          file.write(f"\t]\n")
          file.write("]\n")

        return True

    except FileNotFoundError:
      print(f"File was not found or could not be opened ({file_name})")

    return False
  def load_file(self, file_name: str) -> bool:
    try:
      with open(file_name, "r") as file:
        lines = file.readlines()

        i = 0
        while i < len(lines):
          line = lines[i]

          if "Layer" in line:
            shape_str = line[line.find('('):-3]
            layer_shape = ast.literal_eval(shape_str)
            self.layers.append(Dense(layer_shape[0], layer_shape[1]))

          elif "weights" in line:
            matrix_lines = []
            
            for j in range(self.layers[-1].inputs):
              i += 1
              matrix_lines.append(f"[{", ".join(lines[i].strip("\n\t[]").split())}]")

            lst = ast.literal_eval(f"[{",".join(matrix_lines)}]")
            self.layers[-1].weights = numpy.array(lst).reshape(self.layers[-1].inputs,self.layers[-1].outputs)
            
          elif "biases" in line:
            i += 1
            matrix_line = f"[{", ".join(lines[i].strip("\n\t[]").split())}]"

            lst = ast.literal_eval(matrix_line)
            self.layers[-1].biases = numpy.array(lst).reshape(1,self.layers[-1].outputs)

          i += 1

        return True

    except FileNotFoundError:
      print(f"File was not found or could not be opened ({file_name})")

    return False

  def add_layer(self, layer: Dense) -> None:
    self.layers.append(layer)

  def forward(self, input: numpy.ndarray) -> numpy.ndarray:
    output = input
    for layer in self.layers: output = layer.forward(output)
    return output
  def loss(self, data_point: DataPoint, outputs) -> float:
    error = outputs - data_point.outputs
    return float(numpy.mean(error ** 2))


  def update_gradients(self, data_point: DataPoint) -> None:
    self.forward(data_point.inputs)

    output_layer = self.layers[-1]
    node_values = output_layer.calculate_output_layer_node_values(data_point.outputs)
    output_layer.update_gradients(node_values)

    for i in range(len(self.layers) - 2, -1, -1):
      hidden_layer = self.layers[i]
      node_values = hidden_layer.calculate_hidden_layer_node_values(self.layers[i + 1], node_values)
      hidden_layer.update_gradients(node_values)
  def apply_gradients(self, learn_rate) -> None:
    for layer in self.layers: layer.apply_gradients(learn_rate)
  def clear_gradients(self) -> None:
    for layer in self.layers: layer.clear_gradients()

  def learn(self, epochs, learn_rate, data_set: list[DataPoint]) -> None:
    for epoch in range(epochs):
      for data_point in data_set:
        self.update_gradients(data_point)
        self.apply_gradients(learn_rate)
        self.clear_gradients()

  def learn_old(self, epochs, learn_rate, data_set: list[DataPoint]) -> None:
    for _ in range(epochs):
      for data_point in data_set:
        for layer in self.layers:
          for o in range(layer.outputs):
            pre_loss = self.loss(data_point, self.forward(data_point.inputs))

            layer.biases[0,o] += NUDGE
            post_loss = self.loss(data_point, self.forward(data_point.inputs))
            layer.biases[0,o] -= NUDGE

            layer.biases[0,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)

            for i in range(layer.inputs):
              pre_loss = self.loss(data_point, self.forward(data_point.inputs))

              layer.weights[i,o] += NUDGE
              post_loss = self.loss(data_point, self.forward(data_point.inputs))
              layer.weights[i,o] -= NUDGE

              layer.weights[i,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)