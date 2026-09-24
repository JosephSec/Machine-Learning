#pragma once

#include <matrix2d.h>


typedef struct {
  unsigned int inputCount;
  unsigned int outputCount;

  Matrix weights;
  Matrix biases;
} Dense;


inline Dense create_dense(unsigned int _inputs, unsigned int _outputs) {
  Dense output;

  output.inputCount = _inputs;
  output.outputCount = _outputs;

  output.weights = create_uniform_matrix(_inputs,_outputs, -.5,.5);
  output.biases = create_matrix(1, _outputs, 0);

  return output;
}

inline void free_dense(Dense *_dst) {
  free_matrix(&_dst->weights);
  free_matrix(&_dst->biases);
}


inline void forward_inplace(Matrix *_dst, const Dense *_dense, const Matrix *_inputs) {
  multiply_matrices_inplace(_dst, _inputs, &_dense->weights);
  add_matrices_inplace(_dst, _dst, &_dense->biases);
}
inline Matrix forward(const Dense *_dense, const Matrix *_inputs) {
  Matrix output = create_matrix(1,_dense->outputCount, 0);
  forward_inplace(&output, _dense, _inputs);
  return output;
}