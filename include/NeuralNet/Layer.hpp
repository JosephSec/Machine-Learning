#pragma once

#include <NeuralNet/API.hpp>
#include <NeuralNet/Tensor.hpp>


namespace NEURALNET_API NeuralNetwork {
  enum class TensorInit {
    Constant,
    Random
  };
  enum class LayerType {
    Dense
  };

  class Layer {
  public:
    explicit Layer(uint32_t _in, uint32_t _out)
    : m_inCount(_in), m_outCount(_out) {}

    virtual LayerType RetreiveType() const noexcept = 0;
    virtual std::vector<Tensor<float>*> RetreiveTrainables() = 0;

    virtual Tensor<float> Forward(const Tensor<float> &_input) = 0;


    uint32_t m_inCount;
    uint32_t m_outCount;
  };

  template <TensorInit WeightInit = TensorInit::Random, TensorInit BiasInit = TensorInit::Constant>
  class Dense : public Layer {
  public:
    Dense(uint32_t _in, uint32_t _out, const std::vector<float> &_params) : Layer(_in, _out) {
      const float *paramPtr = _params.data();

      if constexpr(WeightInit == TensorInit::Constant) {
        m_weights = Tensor<float>(_in, _out, *paramPtr);
        paramPtr += 1;
      }
      else if constexpr(WeightInit == TensorInit::Random) {
        m_weights = Tensor<float>::Random(_in, _out, *paramPtr, *(paramPtr + 1));
        paramPtr += 2;
      }

      if constexpr(BiasInit == TensorInit::Constant) {
        m_biases = Tensor<float>(1, _out, *paramPtr);
        paramPtr += 1;
      }
      else if constexpr(BiasInit == TensorInit::Random) {
        m_biases = Tensor<float>::Random(_in, _out, *paramPtr, *(paramPtr + 1));
        paramPtr += 2;
      }
    }

    LayerType RetreiveType() const noexcept override {
      return LayerType::Dense;
    }
    std::vector<Tensor<float>*> RetreiveTrainables() override {
      return {&m_weights, &m_biases};
    }

    Tensor<float> Forward(const Tensor<float> &_input) override {
      return m_biases.add(_input.dot(m_weights));
    }


    Tensor<float> m_weights;
    Tensor<float> m_biases;
  };
};