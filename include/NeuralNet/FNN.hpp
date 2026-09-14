#pragma once

#include <vector>

#include <NeuralNet/API.hpp>
#include <NeuralNet/Layer.hpp>
#include <NeuralNet/DataPoint.hpp>


namespace NEURALNET_API NeuralNetwork {
  class FNN {
  public:
    ~FNN();

    Tensor<float> Forward(Tensor<float> _input);

    float MSELoss(const DataPoint &_dataPoint);
    float CCELoss(const DataPoint &_dataPoint);


    void AddLayer(Layer *_layer);


    std::vector<Layer*> m_layers;
  };
};