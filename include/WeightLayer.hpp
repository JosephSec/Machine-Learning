#pragma once

#include <Matrix.hpp>


enum class ActivationType {
  Linear,
  Sigmoid,
  Tanh,
  ReLU,
  LeakyReLU,
  SiLU,
  GELU,
};
class WeightLayer {
public:
  static NNValueType MSENodeLoss(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType CCENodeLoss(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType MSENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType CCENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput);


  NNValueType Activation(NNValueType _weightedInput) const;
  NNValueType ActivationDerivative(NNValueType _weightedInput) const;  

  Matrix CalculateOutputs(const Matrix &_inputs);


  WeightLayer(uint32_t _inputCount, uint32_t _outputCount);


  ActivationType m_activation = ActivationType::Linear;

  uint32_t m_inputCount = 0;
  uint32_t m_outputCount = 0;

  Matrix m_weights;
  Matrix m_biases;
};