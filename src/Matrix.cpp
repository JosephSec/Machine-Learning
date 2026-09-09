#include <Matrix.hpp>

#include <random>
#include <chrono>


Matrix::Matrix(const glm::ivec2 _size, NNValueType _value) : m_size(_size) {
  m_data = MatrixData(_size.x, std::vector<NNValueType>(_size.y, _value));
}
Matrix::Matrix(const glm::ivec2 _size, NNValueType _min, NNValueType _max) : m_size(_size) {
  m_data = MatrixData(_size.x, std::vector<NNValueType>(_size.y));

  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_real_distribution<NNValueType> rand(_min, _max);

  for(int x = 0; x < _size.x; x++) {
    for(int y = 0; y < _size.y; y++) m_data[x][y] = rand(gen);
  }
}
Matrix::Matrix(const MatrixData &_vec) {
  if(_vec.empty() || _vec[0].empty()) {
    throw std::invalid_argument("Error: The input vector cannot be empty!");
  }

  m_size = glm::ivec2(_vec.size(), _vec[0].size());
  m_data = _vec;
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
