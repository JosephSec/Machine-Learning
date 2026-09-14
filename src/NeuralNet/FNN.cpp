#include <NeuralNet/FNN.hpp>

#include <stdexcept>
#include <cmath>


namespace NeuralNetwork {
  FNN::~FNN() {
    for(Layer *layer : m_layers) delete layer;
  }

  Tensor<float> FNN::Forward(Tensor<float> _input) {
    for(Layer *layer : m_layers) {
      _input = layer->Forward(_input);
    }
    return _input;
  }

  float FNN::MSELoss(const DataPoint &_dataPoint) {
    const Tensor<float> output = Forward(Tensor<float>::Copy(_dataPoint.input));

    float loss = 0;
    for(int i = 0; i < output.m_width; i++) {    
      const float error = *(output.m_data + i) - *(_dataPoint.output.m_data + i);
      loss += .5 * error * error;
    }

    return loss / static_cast<float>(output.m_width);
  }
  float FNN::CCELoss(const DataPoint &_dataPoint) {
    const Tensor<float> output = Forward(Tensor<float>::Copy(_dataPoint.input));

    float loss = 0;
    for(int i = 0; i < output.m_width; i++) {    
      const float expectedOutput = *(_dataPoint.output.m_data + i);

      if(expectedOutput == 0) continue;
      loss += -(expectedOutput * std::log(std::max<float>(*(output.m_data + i), 1e-15)));
    }

    return loss / static_cast<float>(output.m_width);
  }

  void FNN::AddLayer(Layer *_layer) {
    if(m_layers.empty()) {
      m_layers.push_back(_layer);
      return;
    }

    const Layer *layerA = m_layers.back();
    m_layers.push_back(_layer);
    const Layer *layerB = m_layers.back();

    if(layerA->m_outCount != layerB->m_inCount) {
      throw std::invalid_argument("new layer input must match the previous layers output");
    }
  }
};