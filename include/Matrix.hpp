#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>

#include <glm/vec2.hpp>


template <typename T>
static std::string join_string(const std::vector<T> &_vec, const std::string &_delim = ", ") {
  std::stringstream ss;

  ss << _vec[0];
  for(int i = 1; i < _vec.size(); i++) ss << _delim << _vec[i];

  return ss.str();
}


typedef float NNValueType;
typedef std::vector<std::vector<NNValueType>> MatrixData;

struct Matrix {
public:
  Matrix(const glm::ivec2 _size = glm::ivec2(0), NNValueType _value = 0); //Init with all values = _value
  Matrix(const glm::ivec2 _size, NNValueType _min, NNValueType _max); //Init with random values (_min >= x <= _max)
  Matrix(const MatrixData &_vec); //Init with nested vector

  operator std::string() const;

  inline std::vector<NNValueType>& operator[](size_t _index) {
    return m_data[_index];
  }
  inline const std::vector<NNValueType>& operator[](size_t _index) const {
    return m_data[_index];
  }


  glm::ivec2 m_size;
  MatrixData m_data;
};