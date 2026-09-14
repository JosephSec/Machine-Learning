#pragma once

#include <vector>
#include <NeuralNet/Tensor.hpp>


namespace NeuralNetwork {

  struct DataPoint {
  public:
    DataPoint(
      const std::vector<float> &_input,
      const std::vector<float> &_output
    ) : input(1, _input.size()), output(1, _output.size()) {
      input.set(_input);
      output.set(_output);
    }

    DataPoint(DataPoint&&) noexcept = default;
    DataPoint(const DataPoint&) = delete;
    
    DataPoint& operator=(DataPoint&&) noexcept = default;
    DataPoint& operator=(const DataPoint&) = delete;

    Tensor<float> input;
    Tensor<float> output;
  };
};