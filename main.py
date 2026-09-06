class WeightLayer:
  def __init__(self, input_count, output_count):
    self.input_count = input_count
    self.output_count = output_count

    self.weights = [[0] * output_count] * input_count
    self.biases = [[0] * output_count] * input_count

  def Learn(self, learn_rate, cost) -> None:
    for i in range(self.input_count):
      for j in range(self.output_count):
        self.weights[i][j] -= cost[j] * learn_rate
        self.biases[i][j] -= cost[j] * learn_rate

  def CalculateOutputs(self, inputs) -> list[float]:
    outputs = [0] * self.output_count

    for i in range(self.input_count):
      for j in range(self.output_count):
        outputs[j] += inputs[i] * self.weights[i][j] + self.biases[i][j]

    return outputs
  def CalculateCost(self, data_point) -> list[float]:
    outputs = self.CalculateOutputs(data_point[0])
    cost = [0] * self.output_count

    for i in range(self.output_count):
      cost[i] = outputs[i] - data_point[1][i]

    return cost

output_layer = WeightLayer(1, 2)

training_data = [
  [[  1], [  1, -1]],
  [[ -1], [ -1,  1]],
  # [[ .5], [ .5,-.5]],
  # [[-.5], [-.5, .5]],
  # [[  0], [  0,  0]],
]

test_data = training_data[0]

print("-- PRE TRAINING ITERATIONS --")
print(f"{test_data[0]} -> {output_layer.CalculateOutputs(test_data[0])}")
print(f"Cost: {output_layer.CalculateCost(test_data)}")
print("-- PRE TRAINING ITERATIONS --\n")

training_iterations = 10000
print(f"Training Iterations: {training_iterations}\n")
for i in range(10000):
  for data_point in training_data:
    output_layer.Learn(.01, output_layer.CalculateCost(data_point))

print("-- POST TRAINING ITERATIONS --")
print(f"{test_data[0]} -> {output_layer.CalculateOutputs(test_data[0])}")
print(f"Cost: {output_layer.CalculateCost(test_data)}")
print("-- POST TRAINING ITERATIONS --\n")

print(output_layer.weights)
print(output_layer.biases)