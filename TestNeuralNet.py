import unittest

from NeuralNet.FNN import *


class TestNeuralNet(unittest.TestCase):
  def setUp(self):
    self.fnn = FNN()
    self.fnn.add_layer(Layer.Dense(2,3))
    self.fnn.add_layer(Layer.Dense(3,2))

    self.training_data = [
      DataPoint(Tensor(1,2).load_list([0,0]), Tensor(1,2).load_list([0,0])),
      DataPoint(Tensor(1,2).load_list([1,0]), Tensor(1,2).load_list([1,0])),
      DataPoint(Tensor(1,2).load_list([0,1]), Tensor(1,2).load_list([0,1])),
      DataPoint(Tensor(1,2).load_list([1,1]), Tensor(1,2).load_list([1,1])),
    ]

  def test_layer_init(self):
    test_layer = self.fnn.m_layers[0]
    self.assertEqual(test_layer.m_weights.rows, test_layer.m_biases.cols, "Layer weight.rows should equal bias.cols")

  def test_learn(self):
    data_point = self.training_data[0]
    pre_loss = self.fnn.loss(data_point, self.fnn.forward(data_point.inputs))
    self.fnn.learn(1, .1, self.training_data)
    post_loss = self.fnn.loss(data_point, self.fnn.forward(data_point.inputs))

    self.assertNotEqual(pre_loss, post_loss, "Loss should return a new value after training")


if __name__ == "__main__":
  unittest.main()