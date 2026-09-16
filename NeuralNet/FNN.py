from NeuralNet.Tensor import *
from NeuralNet.DataPoint import *
import NeuralNet.Layer as Layer
import NeuralNet.Loss as Loss

__all__ = ["FNN", "Tensor", "DataPoint", "Layer", "Loss"]

NUDGE = .0001

class FNN:
  def __init__(self):
    self.m_layers = []

  def learn(self, epochs, learn_rate, data_set) -> None:
    for epoch in range(epochs):
      for data_point in data_set:
        output_layer = self.m_layers[-1]

        for o in range(output_layer.out_count):
          for i in range(output_layer.in_count):
            pre_loss = self.loss(data_point, self.forward(data_point.inputs))

            output_layer.m_weights[o,i] += NUDGE
            post_loss = self.loss(data_point, self.forward(data_point.inputs))
            output_layer.m_weights[o,i] -= NUDGE

            output_layer.m_weights[o,i] -= learn_rate * ((post_loss - pre_loss) / NUDGE)
        
        for o in range(output_layer.out_count):
          pre_loss = self.loss(data_point, self.forward(data_point.inputs))

          output_layer.m_biases[0,o] += NUDGE
          post_loss = self.loss(data_point, self.forward(data_point.inputs))
          output_layer.m_biases[0,o] -= NUDGE

          output_layer.m_biases[0,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)


  def loss(self, data_point, outputs) -> float:
    totalLoss = 0

    for i in range(self.m_layers[-1].out_count):
      totalLoss += Loss.MSE(outputs[0,i], data_point.outputs[0,i])

    return totalLoss / self.m_layers[-1].out_count

  def forward(self, input: Tensor) -> Tensor:
    output = input
    for layer in self.m_layers: output = layer.forward(output)

    return output


  def add_layer(self, layer) -> None:
    self.m_layers.append(layer)