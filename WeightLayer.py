from Matrix import *
from enum import Enum
import math

__all__ = ["WeightLayer", "Matrix"]


class ActivationType(Enum):
  Linear = 0
  Sigmoid = 1
  Tanh = 2
  ReLU = 3
  LeakyReLU = 4
  SiLU = 5
  GELU = 6

class WeightLayer:
  input_count = 0
  output_count = 0

  weights: Matrix
  biases: Matrix

  activation: ActivationType = ActivationType.Sigmoid


  def __init__(self, input_count, output_count):
    self.input_count  = input_count
    self.output_count = output_count

    self.weights = Matrix(glm.ivec2(self.input_count, self.output_count))
    self.biases  = Matrix(glm.ivec2(1, self.output_count), 0)

    self.weights.randomize_data(-.5, .5)

  def calculate_outputs(self, inputs: Matrix) -> Matrix:
    output = Matrix(glm.ivec2(1, self.output_count), 0)

    for o in range(self.output_count):
      weightedInput = self.biases[0][o]
      for i in range(self.input_count):
        weightedInput += self.weights[i][o] * inputs[0][i]

      output[0][o] = weightedInput

    return output

  


  def MSENodeLoss(output: float, expected_output: float):
    error = output - expected_output
    return (error * error) * .5
  def CCENodeLoss(output: float, expected_output: float):
    if expected_output <= 0: return 0
    return -(expected_output * math.log(max(output, 1e-15)))
  def MSENodeLossDerivative(output: float, expected_output: float):
    return output - expected_output;
  def CCENodeLossDerivative(output: float, expected_output: float):
    if expected_output <= 0: return 0;
    return -expected_output / max(output, 1e-15);

  def Activation(self, weighted_input):
    match self.activation:
      case ActivationType.Linear: return weighted_input
      case ActivationType.Sigmoid: return 1.0 / (1.0 + math.exp(-weighted_input))
      case ActivationType.Tanh: return math.tanh(weighted_input)
      case ActivationType.ReLU: return max(0, weighted_input);   
      case ActivationType.LeakyReLU: return weighted_input if (weighted_input > 0) else weighted_input * .01
      case ActivationType.SiLU: weighted_input / (1 + math.exp(-weighted_input))
      case ActivationType.GELU: return 0.5 * weighted_input * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (weighted_input + 0.044715 * pow(weighted_input, 3))))
  def ActivationDerivative(self, weighted_input):
    match self.activation:
      case ActivationType.Linear: return 1
      case ActivationType.Sigmoid:
        val = self.Activation(weighted_input)
        return val * (1 - val)
      case ActivationType.Tanh:
        val = self.Activation(weighted_input)
        return 1 - (val * val)
      case ActivationType.ReLU: return float(weighted_input > 0)
      case ActivationType.LeakyReLU:
        val = self.Activation(weighted_input)
        return 1 if (val > 0) else .01
      case ActivationType.SiLU:
        sig = 1.0 / (1.0 + math.exp(-weighted_input))
        return sig * (1.0 + weighted_input * (1.0 - sig))
      case ActivationType.GELU:
        x3 = math.pow(weighted_input, 3)
        inner = math.sqrt(2 / math.pi) * (weighted_input + 0.044715 * x3)
        tanh_inner = math.tanh(inner)
        sech2_inner = 1.0 - (tanh_inner * tanh_inner)     
        return 0.5 * (1.0 + tanh_inner) + (0.5 * weighted_input * sech2_inner * math.sqrt(2 / math.pi) * (1.0 + 3.0 * 0.044715 * weighted_input * weighted_input))
