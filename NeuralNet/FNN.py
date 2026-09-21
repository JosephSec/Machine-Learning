import numpy

from NeuralNet.Dense import *

__all__ = ["FNN", "DataPoint", "Dense", "numpy"]


NUDGE = .00001

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

  def add_layer(self, layer: Dense) -> None:
    self.layers.append(layer)

  def forward(self, input: numpy.ndarray) -> numpy.ndarray:
    output = input
    for layer in self.layers: output = layer.forward(output)
    return output

  def loss(self, data_point: DataPoint, outputs) -> float:
    error = outputs - data_point.outputs
    return float(numpy.mean(error ** 2))
  
  def learn(self, epochs, learn_rate, data_set: list[DataPoint]) -> None:
    for _ in range(epochs):
      for data_point in data_set:
        output_layer = self.layers[-1]

        for o in range(output_layer.outputs):
          for i in range(output_layer.inputs):
            pre_loss = self.loss(data_point, self.forward(data_point.inputs))

            output_layer.weights[i,o] += NUDGE
            post_loss = self.loss(data_point, self.forward(data_point.inputs))
            output_layer.weights[i,o] -= NUDGE

            output_layer.weights[i,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)
        
        for o in range(output_layer.outputs):
          pre_loss = self.loss(data_point, self.forward(data_point.inputs))

          output_layer.biases[0,o] += NUDGE
          post_loss = self.loss(data_point, self.forward(data_point.inputs))
          output_layer.biases[0,o] -= NUDGE

          output_layer.biases[0,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)