from NeuralNet.Tensor import *
import ctypes
import numpy
import NeuralNet.c_bridge as c_bridge

__all__ = ["Dense"]

class Dense:
  def __init__(self, in_count, out_count):
    self.in_count = in_count
    self.out_count = out_count

    self.m_weights = Tensor(out_count, in_count).randomize(-.5,.5)
    self.m_biases = Tensor(1,out_count, 0)

  def forward(self, input: Tensor) -> Tensor:
    output = numpy.empty(self.out_count, dtype=numpy.float32)

    input = numpy.array(input.m_data, dtype=numpy.float32)
    weights = numpy.array(self.m_weights.m_data, dtype=numpy.float32)
    biases = numpy.array(self.m_biases.m_data, dtype=numpy.float32)

    c_bridge.lib.ForwardTensor(
      output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      ctypes.c_uint(self.in_count),
      ctypes.c_uint(self.out_count)
    )
    return Tensor(1, self.out_count).load_list(list(output))

    # python
    output = Tensor(1, self.out_count, 0)

    for o in range(self.out_count):
      weightedInput = self.m_biases[0,o]
      for i in range(self.in_count):
        weightedInput += self.m_weights[o,i] * input[0,i]

      output[0,o] = weightedInput

    return output
    # python