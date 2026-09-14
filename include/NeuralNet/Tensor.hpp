#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <NeuralNet/API.hpp>


namespace NEURALNET_API NeuralNetwork {
  template <typename T>
  struct Tensor {
  public:
    static Tensor<T> Copy(const Tensor<T> &_tensor);
    static Tensor<T> Random(uint32_t _h, uint32_t _w, float _min, float _max);


    std::string ToString() const;


    Tensor() {}
    ~Tensor();

    Tensor(Tensor<T> &&_other) noexcept;
    Tensor(const Tensor<T> &_other) = delete;

    Tensor<T> &operator=(Tensor<T> &&_other) noexcept;
    Tensor<T> &operator=(const Tensor<T> &_other) = delete;

    Tensor<T> &set(const std::vector<T> &_vec) noexcept;


    //Construct Tensor with set size and value
    //@param _h Row Count
    //@param _w Column Count
    //@param _v Data fill value
    Tensor(uint32_t _h, uint32_t _w, T _v = 0);

    Tensor<T> dot(const Tensor<T> &_tensor) const;
    Tensor<T> add(const Tensor<T> &_tensor) const;

    const T* get(uint32_t _row, uint32_t _col) const;
    T* get(uint32_t _row, uint32_t _col);


    uint32_t m_width = 0;
    uint32_t m_height = 0;
    T *m_data = nullptr;
  };
};