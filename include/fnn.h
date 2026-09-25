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
  return;
}
inline Matrix forward_fnn(const FNN *_fnn, const Matrix *_input) {
  Matrix output;
  forward_fnn_inplace(_fnn, &output, _input);
  return output;
}


inline char* get_string_fnn_structure(const FNN *_fnn) {
  const size_t byteSize = _fnn->layerCount * (10 + 2);
  char *buffer = malloc(byteSize * sizeof(char));

  size_t byteIndex = 0;
  for(int i = 0; i < _fnn->layerCount; i++) {
    byteIndex += snprintf(buffer + byteIndex, byteSize - byteIndex, "%i->", _fnn->layers[i].inputCount);
  }
  byteIndex += snprintf(buffer + byteIndex, byteSize - byteIndex, "%i", _fnn->layers[_fnn->layerCount - 1].outputCount);

  buffer[byteIndex] = '\0';
  byteIndex += 1;

  buffer = realloc(buffer, byteIndex);

  return buffer;
}

inline void print_fnn_structure(const char *_str, const FNN *_fnn) {
  char *fnnStr = get_string_fnn_structure(_fnn);
  printf(_str, fnnStr);
  free(fnnStr);
}