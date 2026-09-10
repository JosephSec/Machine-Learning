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
  static NNValueType MSENodeLoss(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType MSENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType CCENodeLoss(NNValueType _output, NNValueType _expectedOutput);
  static NNValueType CCENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput);


  FNN(const std::vector<uint32_t> &_layers);

  bool SaveToFile(const std::filesystem::path &_path, const std::vector<DataPoint> &_trainingData);

  void Learn(const std::vector<DataPoint> &_trainingData);

  void UpdateAllGradients(const DataPoint &_dataPoint);
  void ApplyAllGradients(NNValueType _learnRate);
  void ClearAllGradients();

  NNValueType Loss(NNValueType _output, NNValueType _expectedOutput) const;
  NNValueType LossDerivative(NNValueType _output, NNValueType _expectedOutput) const;

  Matrix CalculateOutputs(const Matrix &_inputs);

  NNValueType CalculateLoss(const DataPoint &_dataPoint);
  NNValueType CalculateLoss(const std::vector<DataPoint> &_dataPoints);


  std::vector<WeightLayer> m_layers;

  LossType m_loss = LossType::MeanSquaredError;
  NNValueType m_learnRate = .01;
  uint32_t m_trainingIterations = 0;
};