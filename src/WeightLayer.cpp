#include <WeightLayer.hpp>

#include <NeuralNetwork.hpp>

#include <cmath>
#include <stdexcept>
#include <glm/gtc/constants.hpp>


NNValueType WeightLayer::Activation(NNValueType _weightedInput) const {
  switch(m_activation) {
    default:
    case ActivationType::Linear: return _weightedInput;

    case ActivationType::Sigmoid: return 1.0 / (1.0 + std::exp(-_weightedInput));
    case ActivationType::Tanh: return std::tanh(_weightedInput);
    case ActivationType::ReLU: return std::max<NNValueType>(0.0, _weightedInput);    
    case ActivationType::LeakyReLU: return _weightedInput > 0.0? _weightedInput : _weightedInput * .01;
    case ActivationType::SiLU: _weightedInput / (1 + std::exp(-_weightedInput));
    case ActivationType::GELU: return 0.5 * _weightedInput * (1.0 + std::tanh(std::sqrt(glm::two_over_pi<NNValueType>()) * (_weightedInput + 0.044715 * std::pow(_weightedInput, 3))));
  }
}
NNValueType WeightLayer::ActivationDerivative(NNValueType _weightedInput) const {
  switch(m_activation) {
    default:
    case ActivationType::Linear: return 1;

    case ActivationType::Sigmoid:
      {
        NNValueType activation = Activation(_weightedInput);
        return activation * (1 - activation);
      }
    case ActivationType::Tanh:
      {
        NNValueType activation = Activation(_weightedInput);
        return 1 - (activation * activation);
      }
    case ActivationType::ReLU: return static_cast<NNValueType>(_weightedInput > 0);
    case ActivationType::LeakyReLU:
      {
        NNValueType activation = Activation(_weightedInput);
        return activation > 0 ? 1 : .01;
      }
    case ActivationType::SiLU:
      { 
        NNValueType sig = 1.0 / (1.0 + std::exp(-_weightedInput));
        return sig * (1.0 + _weightedInput * (1.0 - sig));
      }
    case ActivationType::GELU:
      {
        NNValueType x3 = std::pow(_weightedInput, 3);
        NNValueType inner = std::sqrt(glm::two_over_pi<NNValueType>()) * (_weightedInput + 0.044715 * x3);
        NNValueType tanh_inner = std::tanh(inner);
        NNValueType sech2_inner = 1.0 - (tanh_inner * tanh_inner);
        
        return 0.5 * (1.0 + tanh_inner) + (0.5 * _weightedInput * sech2_inner * std::sqrt(glm::two_over_pi<NNValueType>()) * (1.0 + 3.0 * 0.044715 * _weightedInput * _weightedInput));
      }
  }
}


Matrix WeightLayer::CalculateOutputs(const Matrix &_inputs) {
  Matrix output(glm::ivec2(1, m_outputCount));

  for(int out = 0; out < m_outputCount; out++) {
    NNValueType weightedInput = m_biases[0][out];
    for(int in = 0; in < m_inputCount; in++) {
      weightedInput += _inputs[0][in] * m_weights[in][out];
    }

    m_weightedInputs[0][out] = weightedInput;
    output[0][out] = Activation(weightedInput);
  }

  m_inputs = _inputs;
  m_activations = output;

  return output;
}

Matrix WeightLayer::CalculateOutputLayerNodeValues(const Matrix &_expectedOutputs, LossType _lossType) const {
  Matrix nodeValues(_expectedOutputs.m_size);

  //THIS IS DISGUSTING !! CHANGE IT
  //THIS IS DISGUSTING !! CHANGE IT
  //THIS IS DISGUSTING !! CHANGE IT
  switch(_lossType) {
    default:
    case LossType::MeanSquaredError:
      for(int i = 0; i < nodeValues.m_size.y; i++) {
        NNValueType costDerivative = FNN::MSENodeLossDerivative(m_activations[0][i], _expectedOutputs[0][i]);
        NNValueType activationDerivative = ActivationDerivative(m_weightedInputs[0][i]);
        nodeValues[0][i] = activationDerivative * costDerivative;
      }
      break;
    case LossType::CategoricalCrossEntropy:
      for(int i = 0; i < nodeValues.m_size.y; i++) {
        NNValueType costDerivative = FNN::CCENodeLossDerivative(m_activations[0][i], _expectedOutputs[0][i]);
        NNValueType activationDerivative = ActivationDerivative(m_weightedInputs[0][i]);
        nodeValues[0][i] = activationDerivative * costDerivative;
      }
      break;
  }
  //THIS IS DISGUSTING !! CHANGE IT
  //THIS IS DISGUSTING !! CHANGE IT
  //THIS IS DISGUSTING !! CHANGE IT

  return nodeValues;
}
Matrix WeightLayer::CalculateHiddenLayerNodeValues(const WeightLayer &_oldLayer, const Matrix &_oldNodeValues) const {
  Matrix newNodeValues(glm::ivec2(1,m_outputCount));

  for(int newNodeIndex = 0; newNodeIndex < newNodeValues.m_size.y; newNodeIndex++) {
    NNValueType newNodeValue = 0;
    for(int oldNodeIndex = 0; oldNodeIndex < _oldNodeValues.m_size.y; oldNodeIndex++) {
      NNValueType weightedInputDerivative = _oldLayer.m_weights[newNodeIndex][oldNodeIndex];
      newNodeValue += weightedInputDerivative * _oldNodeValues[0][oldNodeIndex];
    }

    newNodeValue *= ActivationDerivative(m_weightedInputs[0][newNodeIndex]);
    newNodeValues[0][newNodeIndex] = newNodeValue;
  }

  return newNodeValues;
}

void WeightLayer::UpdateGradients(const Matrix &_nodeValues) {
  for(int out = 0; out < m_outputCount; out++) {
    for(int in = 0; in < m_inputCount; in++) {
      NNValueType derivativeCostWrtWeight = m_inputs[0][in] * _nodeValues[0][out];
      m_costGradientW[in][out] += derivativeCostWrtWeight;
    }

    NNValueType derivateCostWrtBias = _nodeValues[0][out];
    m_costGradientB[0][out] += derivateCostWrtBias;
  }
}
void WeightLayer::ApplyGradients(NNValueType learnRate) {
  for(int out = 0; out < m_outputCount; out++) {
    m_biases[0][out] -= m_costGradientB[0][out] * learnRate;

  #if !defined(GPU_MODE) || defined(CPU_MODE)
    for(int in = 0; in < m_inputCount; in++) {
      m_weights[in][out] -= m_costGradientW[in][out] * learnRate;
    }
  #endif
  }

#ifdef GPU_MODE
  weights = GPUMath::ASubtractBMulScalar(weights, costGradientW, learnRate);
#endif
}
void WeightLayer::ClearGradients() {
  for(int out = 0; out < m_outputCount; out++) {
    for(int in = 0; in < m_inputCount; in++) {
      m_costGradientW[in][out] = 0;
    }
    m_costGradientB[0][out] = 0;
  }
}


WeightLayer::WeightLayer(uint32_t _inputCount, uint32_t _outputCount) {
  m_inputCount = _inputCount;
  m_outputCount = _outputCount;

  m_weights = Matrix(glm::ivec2(m_inputCount, m_outputCount), -.5, .5);
  m_biases = Matrix(glm::ivec2(1, m_outputCount));

  m_costGradientW = Matrix(glm::ivec2(m_inputCount, m_outputCount));
  m_costGradientB = Matrix(glm::ivec2(1, m_outputCount));

  m_weightedInputs = Matrix(glm::ivec2(1, m_outputCount));
  m_activations = Matrix(glm::ivec2(1, m_outputCount));
  m_inputs = Matrix(glm::ivec2(1, m_inputCount));
}