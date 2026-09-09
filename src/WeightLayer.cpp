#include <WeightLayer.hpp>

#include <cmath>
#include <stdexcept>
#include <glm/gtc/constants.hpp>


NNValueType WeightLayer::MSENodeLoss(NNValueType _output, NNValueType _expectedOutput) {
  NNValueType error = _output - _expectedOutput;
  return (error * error) * .5;
}
NNValueType WeightLayer::CCENodeLoss(NNValueType _output, NNValueType _expectedOutput) {
  if(_expectedOutput <= 0) return 0;
  return -(_expectedOutput * std::log(std::max<NNValueType>(_output, 1e-15)));
}
NNValueType WeightLayer::MSENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput) {
  return _output - _expectedOutput;
}
NNValueType WeightLayer::CCENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput) {
  if(_expectedOutput <= 0) return 0.0;
  return -_expectedOutput / std::max<NNValueType>(_output, 1e-15);
}


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
        float activation = Activation(_weightedInput);
        return activation * (1 - activation);
      }
    case ActivationType::Tanh:
      {
        float activation = Activation(_weightedInput);
        return 1 - (activation * activation);
      }
    case ActivationType::ReLU: return static_cast<NNValueType>(_weightedInput > 0);
    case ActivationType::LeakyReLU:
      {
        float activation = Activation(_weightedInput);
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
    for(int in = 0; in < m_inputCount; in++) weightedInput += m_weights[in][out] * _inputs[0][in];

    output[0][out] = Activation(weightedInput);
  }

  return output;
}


WeightLayer::WeightLayer(uint32_t _inputCount, uint32_t _outputCount) {
  m_inputCount = _inputCount;
  m_outputCount = _outputCount;

  m_weights = Matrix(glm::ivec2(m_inputCount, m_outputCount), -.5, .5);
  m_biases = Matrix(glm::ivec2(1, m_outputCount), -.5, .5);
}