#pragma once

#include <vector>
#include <filesystem>

#include <NeuralNet/API.hpp>
#include <NeuralNet/Layer.hpp>
#include <NeuralNet/DataPoint.hpp>


namespace NEURALNET_API NeuralNetwork {
<<<<<<< Updated upstream
=======
  struct LearnData {
  public:
    float time;
    float loss;
    uint64_t epochs;
  };

>>>>>>> Stashed changes
  class FNN {
  public:
    ~FNN();

    bool SaveToFile(const std::filesystem::path &_filePath);

<<<<<<< Updated upstream
=======
    LearnData Learn(uint64_t _epochs, float _learnRate, const std::vector<DataPoint> &_dataSet);
    
>>>>>>> Stashed changes
    Tensor<float> Forward(Tensor<float> _input);

    float MSELoss(const DataPoint &_dataPoint);
    float CCELoss(const DataPoint &_dataPoint);


    void AddLayer(Layer *_layer);


    std::vector<Layer*> m_layers;
  };
};