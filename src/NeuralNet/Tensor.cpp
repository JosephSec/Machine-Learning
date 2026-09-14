#include <NeuralNet/Tensor.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>

#include <random>
#include <chrono>


namespace NeuralNetwork {
  template <typename T>
  Tensor<T> Tensor<T>::Copy(const Tensor<T> &_tensor) {
    Tensor<T> output(_tensor.m_height, _tensor.m_width);
    std::memcpy(output.m_data, _tensor.m_data, (_tensor.m_width * _tensor.m_height) * sizeof(T));
    return output;
  }
  template <typename T>
  Tensor<T> Tensor<T>::Random(uint32_t _h, uint32_t _w, float _min, float _max) {
    Tensor<T> output(_h, _w);
    
    std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
    std::uniform_real_distribution<float> rand(_min, _max);

    const uint32_t elementCount = _w * _h;
    for(int i = 0; i < elementCount; i++) {
      *(output.m_data + i) = rand(gen);
    }

    return output;
  }


  template <typename T>
  std::string Tensor<T>::ToString() const {
    std::stringstream ss;
    for(int y = 0; y < m_height - 1; y++) {
      ss << "{";
      for(int x = 0; x < m_width - 1; x++) {
        ss << m_data[x + y * m_width] << ", ";
      }
      ss << m_data[m_width - 1 + y * m_width] << "}\n";
    }
    
    ss << "{";
    for(int x = 0; x < m_width - 1; x++) {
      ss << m_data[x + (m_height - 1) * m_width] << ", ";
    }
    ss << m_data[m_width * m_height - 1] << "}";

    return ss.str();
  }


  template <typename T>
  Tensor<T>::~Tensor() {
    if(m_data == nullptr) return;

    delete[] m_data;
    m_data = nullptr;
  }

  template <typename T>
  Tensor<T>::Tensor(Tensor<T>&& _other) noexcept {
    m_width = _other.m_width;
    m_height = _other.m_height;
    m_data = _other.m_data;

    _other.m_data = nullptr;
    _other.m_width = 0;
    _other.m_height = 0;
  }

  template <typename T>
  Tensor<T> &Tensor<T>::operator=(Tensor<T>&& _other) noexcept {
    if (this != &_other) {
      delete[] m_data;

      m_width = _other.m_width;
      m_height = _other.m_height;
      m_data = _other.m_data;

      _other.m_data = nullptr;
      _other.m_width = 0;
      _other.m_height = 0;
    }
    return *this;
  }
  template <typename T>
  Tensor<T> &Tensor<T>::set(const std::vector<T> &_vec) noexcept {    
    std::memcpy(m_data, _vec.data(), _vec.size() * sizeof(T));
    return *this;
  }

  template <typename T>
  Tensor<T>::Tensor(uint32_t _h, uint32_t _w, T _v) {
    m_height = _h;
    m_width = _w;
    m_data = new T[_w * _h];
    std::fill_n(m_data, _w * _h, _v);
  }

  template <typename T>
  Tensor<T> Tensor<T>::dot(const Tensor<T> &_tensor) const {
    Tensor<T> output(m_height, _tensor.m_width, 0);

    for(int i = 0; i < m_height; i++) {
      const T* ptrA = get(i, 0); 

      for(int k = 0; k < m_width; k++) {
        T valA = *(ptrA + k); 

        const T* ptrB = _tensor.get(k, 0);
        T* ptrOut = output.get(i, 0); 

        for(int j = 0; j < _tensor.m_width; j++) {
          *(ptrOut + j) += valA * *(ptrB + j);
        }
      }
    }

    return output;
  }
  template <typename T>
  Tensor<T> Tensor<T>::add(const Tensor<T> &_tensor) const {
    Tensor<T> output(m_height, m_width);

    const uint32_t size = m_height * m_width;
    T* outPtr = output.get(0,0);
    for(int i = 0; i < size; i++) {
      *(outPtr + i) = *(m_data + i) + *(_tensor.m_data + i);
    }

    return output;
  }

  template <typename T>
  const T* Tensor<T>::get(uint32_t _row, uint32_t _col) const {
    return m_data + (_col + _row * m_width);
  }
  template <typename T>
  T* Tensor<T>::get(uint32_t _row, uint32_t _col) {
    return m_data + (_col + _row * m_width);
  }


  template struct Tensor<float>;
};