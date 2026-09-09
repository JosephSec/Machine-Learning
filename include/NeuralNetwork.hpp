#pragma once

#include <filesystem>

#include <WeightLayer.hpp>


struct DataPoint {
public:
  Matrix inputs;
  Matrix expectedOutputs;
};

class FNN {
public:
  FNN(const std::vector<uint32_t> &_layers);

  bool SaveToFile(const std::filesystem::path &_path, const std::vector<DataPoint> &_trainingData);

  Matrix CalculateOutputs(const DataPoint &_dataPoint);
  NNValueType CalculateLoss(const DataPoint &_dataPoint);


  std::vector<WeightLayer> m_layers;
};