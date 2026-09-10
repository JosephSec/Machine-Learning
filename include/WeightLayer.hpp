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
enum class LossType {
  MeanSquaredError,
  CategoricalCrossEntropy
};

class WeightLayer {
public:
  WeightLayer(uint32_t _inputCount, uint32_t _outputCount);


  NNValueType Activation(NNValueType _weightedInput) const;
  NNValueType ActivationDerivative(NNValueType _weightedInput) const;

  Matrix CalculateOutputs(const Matrix &_inputs);

  Matrix CalculateOutputLayerNodeValues(const Matrix &_expectedOutputs, LossType _lossType) const;
  Matrix CalculateHiddenLayerNodeValues(const WeightLayer &_oldLayer, const Matrix &_oldNodeValues) const;

  void UpdateGradients(const Matrix &_nodeValues);
  void ApplyGradients(NNValueType _learnRate);
  void ClearGradients();


  ActivationType m_activation = ActivationType::Linear;

  uint32_t m_inputCount = 0;
  uint32_t m_outputCount = 0;

  Matrix m_weights;
  Matrix m_biases;

  Matrix m_costGradientW;
  Matrix m_costGradientB;

  Matrix m_weightedInputs;
  Matrix m_activations;
  Matrix m_inputs;
};