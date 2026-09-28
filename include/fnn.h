#pragma once

#include <stdlib.h>

#include <dense.h>
#include <loss.h>


typedef struct {
  Dense *layers;
  unsigned int layerCount;

  LossType loss;
  LossFunction lossFunc;
} FNN;

inline void set_loss_function_fnn(FNN *_fnn, LossType _loss) {
  _fnn->loss = _loss;

  switch(_loss) {
    default:
    case LOSS_MSE: {
      _fnn->lossFunc = calculate_mse_loss;
      break;
    }
    case LOSS_BCE: {
      _fnn->lossFunc = calculate_bce_loss;
      break;
    }
  }
}
inline FNN create_fnn(unsigned int *_layerSizes, ActivationType *_layerActivations, Initializer *_layerInitializers, unsigned int _layerCount, LossType _loss) {
  FNN output;
  output.layerCount = _layerCount;
  set_loss_function_fnn(&output, _loss);

  output.layers = malloc(output.layerCount * sizeof(Dense));
  for(int i = 0; i < output.layerCount; i++) {
    output.layers[i] = create_dense(_layerSizes[i], _layerSizes[i + 1], _layerActivations[i], _layerInitializers + (i * 2));
  }

  for(int i = 0; i < output.layerCount * 2; i++) {
    free_initializer(&_layerInitializers[i]);
  }

  return output;
}


inline void forward_fnn_inplace(const FNN *_fnn, Matrix *_dst, const Matrix *_input) {
  Matrix current_in = copy_matrix(_input);

  free_matrix(_dst);
  // *_dst = EMPTY_MATRIX;

  for(int j = 0; j < _fnn->layerCount; j++) {
    reshape_matrix_to_output_dense(&_fnn->layers[j], _dst); //MEMORY LEAK IN ONE OF THESE
    fill_value_matrix(_dst, 0); //MEMORY LEAK IN ONE OF THESE
    
    continue;
    forward_dense_inplace(&_fnn->layers[j], _dst, &current_in);

    free_matrix(&current_in);

    if(j < _fnn->layerCount - 1) {
      current_in = *_dst;
      *_dst = EMPTY_MATRIX;
    }
  }
  
  free_matrix(&current_in);
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