#pragma once

#include <stdlib.h>

#include <dense.h>


typedef struct {
  Dense *layers;
  unsigned int layerCount;
} FNN;


inline FNN create_fnn(unsigned int *_layerConfig, unsigned int _layerCount) {
  FNN output;
  output.layerCount = _layerCount;

  output.layers = malloc(output.layerCount * sizeof(Dense));
  for(int i = 0; i < output.layerCount; i++) {
    output.layers[i] = create_dense(_layerConfig[i], _layerConfig[i + 1]);
  }

  return output;
}


inline void reshape_matrix_to_input_layer(Matrix *_dst, const FNN *_fnn) {
  const Dense *inputLayer = &_fnn->layers[0];

  _dst->rows = 1;
  _dst->cols = inputLayer->inputCount;
  _dst->data = realloc(_dst->data, _dst->rows * _dst->cols * sizeof(float));
}


inline void forward_fnn_inplace(const FNN *_fnn, Matrix *_dst, const Matrix *_input) {
  Matrix temp_inputs;
  copy_matrix_inplace(&temp_inputs, _input);

  for(int i = 0; i < _fnn->layerCount; i++) {
    Matrix next_output;
    forward_dense_inplace(&_fnn->layers[i], &next_output, &temp_inputs);

    free_matrix(&temp_inputs);

    temp_inputs = next_output;
  }

  copy_matrix_inplace(_dst, &temp_inputs);
  free_matrix(&temp_inputs);
}
