import os
import time
import numpy

NUDGE = .00001

def MSE(output: float, expected_output: float):
  error = output - expected_output
  return (error * error) * .5

class DataPoint:
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs

class Dense:
  def __init__(self, inputs=1, outputs=1):
    self.inputs = inputs
    self.outputs = outputs

    self.biases = numpy.full((1,outputs), 0, dtype=numpy.float32)

    rng = numpy.random.default_rng()
    self.weights = rng.uniform(low=-.5, high=.5, size=(inputs,outputs)).astype(numpy.float32)

  def forward(self, input_arr: numpy.ndarray) -> numpy.ndarray:
    return numpy.dot(input_arr, self.weights) + self.biases

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

            output_layer.weights[o,i] += NUDGE
            post_loss = self.loss(data_point, self.forward(data_point.inputs))
            output_layer.weights[o,i] -= NUDGE

            output_layer.weights[o,i] -= learn_rate * ((post_loss - pre_loss) / NUDGE)
        
        for o in range(output_layer.outputs):
          pre_loss = self.loss(data_point, self.forward(data_point.inputs))

          output_layer.biases[0,o] += NUDGE
          post_loss = self.loss(data_point, self.forward(data_point.inputs))
          output_layer.biases[0,o] -= NUDGE

          output_layer.biases[0,o] -= learn_rate * ((post_loss - pre_loss) / NUDGE)


data_set = [
  DataPoint(numpy.array([0,0]).astype(numpy.float32), numpy.array([0,0]).astype(numpy.float32)),
  DataPoint(numpy.array([0,1]).astype(numpy.float32), numpy.array([0,1]).astype(numpy.float32)),
  DataPoint(numpy.array([1,0]).astype(numpy.float32), numpy.array([1,0]).astype(numpy.float32)),
  DataPoint(numpy.array([1,1]).astype(numpy.float32), numpy.array([1,1]).astype(numpy.float32)),
]

fnn = FNN()
fnn.add_layer(Dense(2,2))


epoch_count = int(input("Enter Epoch Count: "))
epoch_sample_rate = epoch_count // 10
os.system("cls")

pre_loss = fnn.loss(data_set[0], fnn.forward(data_set[0].inputs))

print(f"training for {epoch_count} epochs...")
for i in range(10):
  prev_time = time.perf_counter()
  fnn.learn(epoch_sample_rate, .01, data_set)
  sample_time = int((time.perf_counter() - prev_time) * 1000)
  loss = fnn.loss(data_set[0], fnn.forward(data_set[0].inputs))
  print(f"{sample_time}ms | loss: {loss} | epochs: {epoch_sample_rate * (i + 1)}")
print("training complete")

post_loss = fnn.loss(data_set[0], fnn.forward(data_set[0].inputs))