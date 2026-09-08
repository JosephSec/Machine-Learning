#pragma once

#include <cstdint>
#include <vector>
#include <glm/vec2.hpp>

#include <string>
#include <sstream>
template <typename T>
static std::string join_string(const std::vector<T> &_vec, const std::string &_delim) {
  std::stringstream ss;

  ss << _vec[0];
  for(int i = 1; i < _vec.size(); i++) ss << _delim << _vec[i];

  return ss.str();
}


typedef float NNValueType;


struct Matrix {
public:
  Matrix(const glm::ivec2 _size);
  Matrix(const glm::ivec2 _size, NNValueType _min, NNValueType _max); //Init with random values (_min >= x <= _max)

  operator std::string() const;

  inline std::vector<NNValueType>& operator[](size_t _index) {
    return m_data[_index];
  }
  inline const std::vector<NNValueType>& operator[](size_t _index) const {
    return m_data[_index];
  }


  glm::ivec2 m_size;
  std::vector<std::vector<NNValueType>> m_data;
};
class WeightLayer {
public:
  Matrix CalculateOutputs(const std::vector<NNValueType> &_inputs);


  WeightLayer(uint32_t _inputCount, uint32_t _outputCount);
  

  uint32_t inputCount = 0;
  uint32_t outputCount = 0;

  Matrix weights;
  Matrix biases;
};