#pragma once

#include <vector>
#include <ostream>


class Matrix {
public:
  Matrix(size_t _width = 0, size_t _height = 0) {
    Resize(_width, _height);
  }

  float* GetDataPtr() noexcept {
    return m_data.data();
  }

  void Resize(size_t _width, size_t _height) {
    std::vector<float> temp_data(_width * _height, 0.0f);

    const size_t minWidth = std::min(_width, width);
    const size_t maxWidth = std::max(_width, width);

    const size_t minHeight = std::min(_height, height);

    for(int y = 0; y < minHeight; y++) {
      for(int x = 0; x < minWidth; x++) {
        temp_data[GetIndex(x,y, _width)] = m_data[GetIndex(x,y, width)];
      }
    }

    width = _width;
    height = _height;
    m_data = temp_data;
  }
  void PushBack(const std::vector<float>& _floatVec) {
    Resize(width, height + 1);

    for(int x = 0; x < width; x++) {
      m_data[GetIndex(x, height - 1)] = _floatVec[x];
    }
  }

  inline float& operator()(size_t _x, size_t _y) noexcept {
    return m_data[GetIndex(_x, _y)];
  }
  inline const float& operator()(size_t _x, size_t _y) const noexcept {
    return m_data[GetIndex(_x, _y)];
  }

  std::ostream& Print(std::ostream& os, const std::string& _indent = "") const {
    if(width == 0 || height == 0) return (os << "{}");

    os << _indent << "{\n";

    for(int y = 0; y < height - 1; y++) {
      os << _indent << "\t{";
      for(int x = 0; x < width - 1; x++) {
        os << m_data[GetIndex(x,y)] << ", ";
      }
      os << m_data[GetIndex(width - 1,y)] << "},\n";
    }

    os << _indent << "\t{";
    for(int x = 0; x < width - 1; x++) {
      os << m_data[GetIndex(x, height - 1)] << ", ";
    }
    os << m_data.back() << "}\n" << _indent << "}";

    return os;
  }

  size_t width, height;


private:
  inline size_t GetIndex(size_t _x, size_t _y) const noexcept {
    return _x + width * _y;
  }
  static size_t GetIndex(size_t _x, size_t _y, size_t _width) noexcept {
    return _x + _width * _y;
  }

  std::vector<float> m_data;
};