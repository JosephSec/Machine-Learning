from WeightLayer import *
import math
from enum import Enum
from dataclasses import dataclass

__all__ = ["FNN", "DataPoint", "WeightLayer", "Matrix"]


@dataclass
class DataPoint:
  inputs: Matrix
  expected_outputs: Matrix

class FNN:
  layers: list[WeightLayer] = []

  def __init__(self, layers):
    for i in range(1, len(layers)):
      self.layers.append(WeightLayer(layers[i-1], layers[i]))

  def calculate_outputs(self, inputs: Matrix) -> Matrix:
    output = inputs
    for layer in self.layers:
      output = layer.calculate_outputs(output)

    return output

  def calculate_loss(self, data_point: DataPoint, outputs: Matrix) -> float:
    totalLoss = 0

    for i in range(self.layers[-1].output_count):
      totalLoss += WeightLayer.MSENodeLoss(outputs[0][i], data_point.expected_outputs[0][i])

    return totalLoss / self.layers[-1].output_count