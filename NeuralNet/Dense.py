import numpy

class Dense:
  def __init__(self, inputs=1, outputs=1):
    self.inputs = inputs
    self.outputs = outputs

    self.biases = numpy.full((1,outputs), 0, dtype=numpy.float32)

    rng = numpy.random.default_rng()
    self.weights = rng.uniform(low=-.5, high=.5, size=(inputs,outputs)).astype(numpy.float32)

  def forward(self, input_arr: numpy.ndarray) -> numpy.ndarray:
    return numpy.dot(input_arr, self.weights) + self.biases