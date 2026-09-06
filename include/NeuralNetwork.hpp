#pragma once

#define MAX_FLOAT 3.402823466e+38F

#include <vector>
#include <string>
#include <System/Matrix.hpp>


struct DataPoint {
public:
  std::vector<float> inputs;
  std::vector<float> expectedOutputs;
};

struct WeightLayer {
public:
  static float ActivationFunction(float _x);
  static float ActivationDerivative(float _x);

  static float NodeCost(float outputActivation, float expectedActivation);
  static float NodeCostDerivative(float outputActivation, float expectedActivation);


  WeightLayer(unsigned int _in = 0, unsigned int _out = 0) {
    Resize(_in, _out);
  }


  std::vector<float> CalculateOutputs(const std::vector<float>& _inputs);
  std::vector<float> CalculateOutputLayerNodeValues(const std::vector<float>& expectedOutputs) const;
  std::vector<float> CalculateHiddenLayerNodeValues(const WeightLayer& oldLayer, const std::vector<float>& oldNodeValues) const;

  void ApplyGradients(float learnRate);
  void UpdateGradients(const std::vector<float>& nodeValues);
  void ClearGradients();

  void InitializeRandomWeights();
  void Resize(unsigned int _in, unsigned int _out);


  unsigned int inputCount;
  unsigned int outputCount;
  Matrix weights;
  Matrix costGradientW;
  std::vector<float> biases;
  std::vector<float> costGradientB;

  std::vector<float> weightedInputs;
  std::vector<float> activations;

  std::vector<float> inputs;
};

class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<unsigned int>& layerSizes);


  void SaveToFile(const std::string& _path) const;
  [[nodiscard]] bool LoadFromFile(const std::string& _path);

  std::vector<float> CalculateOutputs(std::vector<float> inputs);

  void ApplyAllGradients(float learnRate);
  void UpdateAllGradients(const DataPoint& dataPoint);
  void ClearAllGradients();

  void LoadNextTrainingBatch();
  float Learn(); //@returns how long it took to train

  float Cost(DataPoint dataPoint);
  float Cost(const std::vector<DataPoint>& data);

  int Classify(const std::vector<float>& inputs);


  std::vector<WeightLayer> layers;

  float learnRate = .1f;
  float prevCost = 0;
  unsigned int trainingIterations = 0;
  std::vector<DataPoint> trainingData;
  unsigned int trainingBatchSize;
  std::vector<DataPoint> trainingBatch;


private:
  static int IndexOfMaxValue(const std::vector<float>& outputs);
};