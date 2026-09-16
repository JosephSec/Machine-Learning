from NeuralNet.Tensor import *

__all__ = ["Dense"]

class Dense:
  def __init__(self, in_count, out_count):
    self.in_count = in_count
    self.out_count = out_count

    self.m_weights = Tensor(out_count, in_count).randomize(-.5,.5)
    self.m_biases = Tensor(1,out_count, 0)

  def forward(self, input: Tensor) -> Tensor:
    output = Tensor(1, self.out_count, 0)

    for o in range(self.out_count):
      weightedInput = self.m_biases[0,o]
      for i in range(self.in_count):
        weightedInput += self.m_weights[o,i] * input[0,i]

      output[0,o] = weightedInput

    return output