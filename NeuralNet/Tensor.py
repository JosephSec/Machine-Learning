import random
import ctypes
import NeuralNet.c_bridge as c_bridge

__all__ = ["Tensor"]


class Tensor:
  def __init__(self, rows=1, cols=1, value=0):
    self.rows = rows
    self.cols = cols
    self.m_data = [value for _ in range(rows * cols)]

  def __getitem__(self, index):
    if isinstance(index, tuple):
      return self.m_data[index[1] + index[0] * self.cols]
    else: return self.m_data[index]

  def __setitem__(self, index, value) -> None:
    if isinstance(index, tuple):
      self.m_data[index[1] + index[0] * self.cols] = value
    else: self.m_data[index] = value


  def load_list(self, lst: list) -> Tensor:
    output = Tensor(self.rows, self.cols)
    for i in range(min(len(self.m_data), len(lst))): output[i] = lst[i]
    return output

  def randomize(self, min=-.5, max=.5) -> Tensor:
    count = self.rows * self.cols
    InputArrayType = ctypes.c_float * count
    output = InputArrayType()

    c_bridge.lib.InitRandomFloat(output, ctypes.c_uint(count), ctypes.c_float(min), ctypes.c_float(max))
    return Tensor(self.rows, self.cols).load_list(list(output))

    # random_list = [(random.uniform(min,max)) for _ in self.m_data]
    # return Tensor(self.rows, self.cols).load_list(random_list)