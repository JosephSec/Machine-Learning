import glm
import random
from dataclasses import dataclass

__all__ = ["Matrix", "glm"]



type MatrixData = list[list[float]]

def init_matrix_data(size: glm.ivec2, value=0) -> MatrixData:
  return [[value for _ in range(size.y)] for _ in range(size.x)]

@dataclass
class Matrix:
  size = glm.ivec2(0)
  data: MatrixData


  def __init__(self, size=glm.ivec2(0), value=0):
    self.size = size
    self.data = init_matrix_data(self.size, value)
  def __getitem__(self, key) -> list[float]:
    return self.data[key]

  def init_from_data(data: MatrixData) -> Matrix:
    if (not data) or (len(data) == 0 and len(data[0]) == 0):
      return None
    
    output = Matrix(glm.ivec2(len(data), len(data[0])))
    output.data = data
    return output

  def randomize_data(self, min, max) -> None:
    for i in range(self.size.x):
      for j in range(self.size.y):
        self.data[i][j] = random.uniform(min, max)