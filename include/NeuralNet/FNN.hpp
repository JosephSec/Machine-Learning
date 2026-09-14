#pragma once

#include <vector>
#include <filesystem>

#include <NeuralNet/API.hpp>
#include <NeuralNet/Layer.hpp>
#include <NeuralNet/DataPoint.hpp>


namespace NEURALNET_API NeuralNetwork {
  class FNN {
  public:
    ~FNN();

    bool SaveToFile(const std::filesystem::path &_filePath);

    Tensor<float> Forward(Tensor<float> _input);

    float MSELoss(const DataPoint &_dataPoint);
    float CCELoss(const DataPoint &_dataPoint);


    void AddLayer(Layer *_layer);


    std::vector<Layer*> m_layers;
  };
};