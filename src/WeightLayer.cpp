#include <WeightLayer.hpp>

#include <chrono>
#include <random>


Matrix::Matrix(const glm::ivec2 _size) : m_size(_size) {
  m_data = std::vector<std::vector<NNValueType>>(_size.x, std::vector<NNValueType>(_size.y));
}
Matrix::Matrix(const glm::ivec2 _size, NNValueType _min, NNValueType _max) : m_size(_size) {
  m_data = std::vector<std::vector<NNValueType>>(_size.x, std::vector<NNValueType>(_size.y));

  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_real_distribution<NNValueType> rand(_min, _max);

  for(int x = 0; x < _size.x; x++) {
    for(int y = 0; y < _size.y; y++) m_data[x][y] = rand(gen);
  }
}

Matrix::operator std::string() const {
  std::stringstream ss;

  ss << "{\n";
  for(int x = 0; x < m_size.x - 1; x++) {
    ss << "\t{" << join_string<NNValueType>(m_data[x], ", ") << "}\n";
  }
  ss << "\t{" << join_string<NNValueType>(m_data.back(), ", ") << "}\n}";

  return ss.str();
}


static NNValueType ActivationFunction(NNValueType _x) {
  return 1.0f / (1.0f + std::exp(-_x));
}
Matrix WeightLayer::CalculateOutputs(const std::vector<NNValueType> &_inputs) {
  std::vector<NNValueType> output(outputCount);

  for(int out = 0; out < outputCount; out++) {
    NNValueType weightedInput = biases[0][out];
    for(int in = 0; in < inputCount; in++) weightedInput += weights[in][out];

    output[out] = ActivationFunction(weightedInput);
  }

  return outputs;
}

WeightLayer::WeightLayer(uint32_t _inputCount, uint32_t _outputCount) {
  inputCount = _inputCount;
  outputCount = _outputCount;
}