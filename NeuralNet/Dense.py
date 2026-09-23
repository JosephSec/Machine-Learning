import numpy

class Dense:
  def __init__(self, inputs=1, outputs=1):
    self.inputs = inputs
    self.outputs = outputs

    rng = numpy.random.default_rng()
    self.weights = rng.uniform(low=-.5, high=.5, size=(inputs,outputs)).astype(numpy.float32)

    self.biases = numpy.full((1,outputs), 0, dtype=numpy.float32)

    self.last_inputs = None
    self.weighted_inputs = None
    self.activations = None

    self.cost_gradient_w = numpy.full((self.inputs,self.outputs), 0, dtype=numpy.float32)
    self.cost_gradient_b = numpy.full((1,self.outputs), 0, dtype=numpy.float32)


  def activation(self, x: numpy.ndarray) -> numpy.ndarray:
    return 1 / (1 + numpy.exp(-x))
  def activation_derivative(self, x: numpy.ndarray) -> numpy.ndarray:
    x = self.activation(x)
    return x * (1 - x)

  def node_cost_derivative(self, output_activation: numpy.ndarray, expected_activation: numpy.ndarray) -> numpy.ndarray:
    return 2 * (output_activation - expected_activation)
  

  def forward(self, input_arr: numpy.ndarray) -> numpy.ndarray:
    self.last_inputs = input_arr
    self.weighted_inputs = numpy.dot(input_arr, self.weights) + self.biases
    self.activations = self.activation(self.weighted_inputs)

    return self.activations


  def calculate_output_layer_node_values(self, expected_outputs: numpy.ndaray) -> numpy.ndarray:
    return self.activation_derivative(self.weighted_inputs) * self.node_cost_derivative(self.activations, expected_outputs)
  def calculate_hidden_layer_node_values(self, old_layer: Dense, old_node_values: numpy.ndarray) -> numpy.ndarray:
    return numpy.dot(old_node_values, old_layer.weights.T) * self.activation_derivative(self.weighted_inputs)

  def update_gradients(self, node_values: numpy.ndarray) -> None:
    self.cost_gradient_w += numpy.dot(self.inputs, node_values)
    self.cost_gradient_b += numpy.sum(node_values, axis=0, keepdims=True)
  def apply_gradients(self, learn_rate) -> None:
    self.weights -= self.cost_gradient_w * learn_rate
    self.biases -= self.cost_gradient_b * learn_rate
  def clear_gradients(self) -> None:
    self.cost_gradient_w.fill(0)
    self.cost_gradient_b.fill(0)