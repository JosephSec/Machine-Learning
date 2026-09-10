#include <NeuralNetwork.hpp>

#include <cmath>
#include <fstream>


NNValueType FNN::MSENodeLoss(NNValueType _output, NNValueType _expectedOutput) {
  NNValueType error = _output - _expectedOutput;
  return .5 * error * error;
}
NNValueType FNN::MSENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput) {
  return _output - _expectedOutput;
}
NNValueType FNN::CCENodeLoss(NNValueType _output, NNValueType _expectedOutput) {
  if(_expectedOutput == 0) return 0;
  return -(_expectedOutput * std::log(std::max<NNValueType>(_output, 1e-15)));
}
NNValueType FNN::CCENodeLossDerivative(NNValueType _output, NNValueType _expectedOutput) {
  if(_expectedOutput <= 0) return 0.0;
  return -_expectedOutput / std::max<NNValueType>(_output, 1e-15);
}


FNN::FNN(const std::vector<uint32_t> &_layers) {
  for(int i = 1; i < _layers.size(); i++) {
    m_layers.push_back(WeightLayer(_layers[i-1], _layers[i]));
    m_layers.back().m_activation = ActivationType::Sigmoid;
  }
}

bool FNN::SaveToFile(const std::filesystem::path &_path, const std::vector<DataPoint> &_trainingData) {
  std::ofstream file(_path);
  if(file.is_open() == false) return false;
  
  for(const WeightLayer &_layer : m_layers) {
    file << "weights: " << std::string(_layer.m_weights) << '\n';
    file << "biases: " << std::string(_layer.m_biases) << '\n';
  }

  float totalLoss = 0;
  for(const DataPoint &_dataPoint : _trainingData) totalLoss += CalculateLoss(_dataPoint);
  file << "average loss: " << (totalLoss / static_cast<NNValueType>(_trainingData.size())) << '\n';

  file.close();
  return true;
}


void FNN::Learn(const std::vector<DataPoint> &_trainingData) {
  for(const DataPoint &_dataPoint : _trainingData) UpdateAllGradients(_dataPoint);

  ApplyAllGradients(m_learnRate / static_cast<NNValueType>(_trainingData.size()));
  ClearAllGradients();

  m_trainingIterations += 1;
}

void FNN::UpdateAllGradients(const DataPoint &_dataPoint) {
  CalculateOutputs(_dataPoint.inputs);

  WeightLayer& outputLayer = m_layers.back();
  Matrix nodeValues = outputLayer.CalculateOutputLayerNodeValues(_dataPoint.expectedOutputs, m_loss);
  outputLayer.UpdateGradients(nodeValues);

  for(int hiddenLayerIndex = m_layers.size() - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
    WeightLayer& hiddenLayer = m_layers[hiddenLayerIndex];
    nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(m_layers[hiddenLayerIndex + 1], nodeValues);
    hiddenLayer.UpdateGradients(nodeValues);
  }
}
void FNN::ApplyAllGradients(NNValueType _learnRate) {
  for(WeightLayer &_layer : m_layers) _layer.ApplyGradients(_learnRate);
}
void FNN::ClearAllGradients() {
  for(WeightLayer &_layer : m_layers) _layer.ClearGradients();
}

NNValueType FNN::Loss(NNValueType _output, NNValueType _expectedOutput) const {
  switch(m_loss) {
    default:
    case LossType::MeanSquaredError: return MSENodeLoss(_output, _expectedOutput);
    case LossType::CategoricalCrossEntropy: return CCENodeLoss(_output, _expectedOutput);
  }
}
NNValueType FNN::LossDerivative(NNValueType _output, NNValueType _expectedOutput) const {
  switch(m_loss) {
    default:
    case LossType::MeanSquaredError: return MSENodeLossDerivative(_output, _expectedOutput);
    case LossType::CategoricalCrossEntropy: return CCENodeLossDerivative(_output, _expectedOutput);
  }
}

Matrix FNN::CalculateOutputs(const Matrix &_inputs) {
  Matrix output = _inputs;
  for(WeightLayer &_layer : m_layers) output = _layer.CalculateOutputs(output);
  return output;
}

NNValueType FNN::CalculateLoss(const DataPoint &_dataPoint) {
  Matrix output = CalculateOutputs(_dataPoint.inputs);
  
  NNValueType totalLoss = 0;
  for(int i = 0; i < output.m_size.y; i++) {
    totalLoss += Loss(output[0][i], _dataPoint.expectedOutputs[0][i]);
  }

  return totalLoss / static_cast<NNValueType>(output.m_size.y);
}
NNValueType FNN::CalculateLoss(const std::vector<DataPoint> &_dataPoints) {
  NNValueType totalLoss = 0;
  for(const DataPoint &_dataPoint : _dataPoints) totalLoss += CalculateLoss(_dataPoint);

  return totalLoss / static_cast<NNValueType>(_dataPoints.size());
}