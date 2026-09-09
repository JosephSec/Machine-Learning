#include <NeuralNetwork.hpp>

#include <fstream>


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

Matrix FNN::CalculateOutputs(const DataPoint &_dataPoint) {
  Matrix output = _dataPoint.inputs;
  for(WeightLayer &_layer : m_layers) output = _layer.CalculateOutputs(output);

  return output;
}
NNValueType FNN::CalculateLoss(const DataPoint &_dataPoint) {
  Matrix output = CalculateOutputs(_dataPoint);

  NNValueType totalLoss = 0;

  for(int i = 0; i < output.m_size.y; i++) {
    totalLoss += WeightLayer::MSENodeLoss(output[0][i], _dataPoint.expectedOutputs[0][i]);
  }

  return totalLoss / static_cast<NNValueType>(output.m_size.y);
}