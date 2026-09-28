#pragma once

#include <stdlib.h>
#include <math.h>

#include <debug.h>

#include <matrix2d.h>


typedef enum {
  INIT_CONSTANT,
  INIT_RANDOM
} InitializerType;
typedef struct {
  InitializerType type;
  float *params;
  size_t paramCount;
} Initializer;
#define INITIALIZER_ZERO create_initializer(INIT_CONSTANT, (float[]){0}, 1)
#define INITIALIZER_ONE  create_initializer(INIT_CONSTANT, (float[]){1}, 1)
#define INITIALIZER_UNIFORM_ONE  create_initializer(INIT_RANDOM, (float[]){ -1,  1}, 2)
#define INITIALIZER_UNIFORM_HALF create_initializer(INIT_RANDOM, (float[]){-.5, .5}, 2)

inline Initializer create_initializer(InitializerType _type, const float *_params, size_t _paramCount) {
  Initializer output;
  output.type = _type;

  output.paramCount = _paramCount;
  output.params = malloc(output.paramCount * sizeof(float));
  memcpy(output.params, _params, output.paramCount * sizeof(float));

  return output;
}

inline void free_initializer(Initializer *_initializer) {
  free(_initializer->params);
}


typedef enum {
  ACTIVATION_LINEAR,
  ACTIVATION_SIGMOID,
  ACTIVATION_RELU,
  ACTIVATION_LEAKYRELU,
} ActivationType;

typedef struct {
  unsigned int inputCount;
  unsigned int outputCount;

  ActivationType activation;

  Matrix weights;
  Matrix biases;
} Dense;

inline Dense create_dense(unsigned int _inputs, unsigned int _outputs, ActivationType _activation,const Initializer *_initializers) {
  Dense output;

  output.inputCount = _inputs;
  output.outputCount = _outputs;

  output.activation = _activation;

  output.weights = create_uniform_matrix(_inputs,_outputs, -.5,.5);
  output.biases = create_matrix(1, _outputs, 0);

  Matrix *matrices[2] = {&output.weights, &output.biases};
  for(int i = 0; i < 2; i++) {
    const Initializer *initializer = &_initializers[i];
    switch(initializer->type) {
      default:
      case INIT_CONSTANT:
        fill_value_matrix(matrices[i], initializer->params[0]);
        break;

      case INIT_RANDOM:
        fill_uniform_matrix(matrices[i], initializer->params[0], initializer->params[1]);
        break;
    }
  }

  return output;
}

inline void free_dense(Dense *_dst) {
  free_matrix(&_dst->weights);
  free_matrix(&_dst->biases);
}


inline void reshape_matrix_to_input_dense(const Dense *_dense, Matrix *_dst) {
  reshape_matrix(_dst, 1, _dense->inputCount);
}
inline void reshape_matrix_to_output_dense(const Dense *_dense, Matrix *_dst) {
  reshape_matrix(_dst, 1, _dense->outputCount);
}


inline void add_bias_with_activation(Matrix *_dst, const Matrix *_inputs, const Matrix *_biases, ActivationType _activation) {
  ASSERT_MSG(_inputs->cols == _biases->cols && _inputs->rows == _biases->rows, "MATRIX",
    "Can't add matrices (%ix%i) and (%ix%i)",
    _inputs->rows, _inputs->cols, _biases->rows, _biases->cols);

  ASSERT_MSG(_dst->data != NULL, "MATRIX",
    "Can't add matrices, _dst must be pre-allocated");
    
  ASSERT_MSG(_dst->rows == _inputs->rows && _dst->cols == _inputs->cols, "MATRIX",
    "Can't add matrices, _dst(%ix%i) matrix shape should match (%ix%i)",
    _dst->rows, _dst->cols, _inputs->rows, _inputs->cols);
  
  const unsigned int elementCount = _inputs->rows * _inputs->cols;

  switch(_activation) {
    default:
    case ACTIVATION_LINEAR: {
      for(int i = 0; i < elementCount; i++) {
        _dst->data[i] = _inputs->data[i] + _biases->data[i];
      }
      break;
    }
    
    case ACTIVATION_RELU: {
      for(int i = 0; i < elementCount; i++) {
        _dst->data[i] = fmaxf(0, _inputs->data[i] + _biases->data[i]);
      }
      break;
    }
    
    case ACTIVATION_LEAKYRELU: {
      for(int i = 0; i < elementCount; i++) {
        _dst->data[i] = fmaxf(0.01f, _inputs->data[i] + _biases->data[i]);
      }
      break;
    }
    
    case ACTIVATION_SIGMOID: {
      for(int i = 0; i < elementCount; i++) {
        _dst->data[i] = 1.0f / (1.0f + expf(-(_inputs->data[i] + _biases->data[i])));
      }
      break;
    }
  }
}
inline void forward_dense_inplace(const Dense *_dense, Matrix *_dst, const Matrix *_inputs) {
  multiply_matrices_inplace(_dst, _inputs, &_dense->weights);
  add_bias_with_activation(_dst, _dst, &_dense->biases, _dense->activation);
}
inline Matrix forward_dense(const Dense *_dense, const Matrix *_inputs) {
  Matrix output = create_matrix(1,_dense->outputCount, 0);
  forward_dense_inplace(_dense, &output, _inputs);
  return output;
}


inline char* get_string_dense(const Dense *_dense) {  
  char *weightsStr = get_string_matrix(&_dense->weights, 4);
  char *biasesStr = get_string_matrix(&_dense->biases, 4);

  const size_t weightsStrLen = 10 + strlen(weightsStr) + 1;
  const size_t biasesStrLen = 9 + strlen(biasesStr);

  char *buffer = malloc((weightsStrLen + biasesStrLen + 1) * sizeof(char));
  
  size_t byteIndex = 0;
  { //Weights
    strcpy(buffer + byteIndex, "weights = ");
    byteIndex += 10;

    strcpy(buffer + byteIndex, weightsStr);
    byteIndex += weightsStrLen - 11;

    buffer[byteIndex] = '\n';
    byteIndex += 1;
  } //Weights
  { //Biases
    strcpy(buffer + byteIndex, "biases = ");
    byteIndex += 9;

    strcpy(buffer + byteIndex, weightsStr);
    byteIndex += weightsStrLen - 10;
  } //Biases
  buffer[byteIndex] = '\0';

  free(weightsStr);
  free(biasesStr);

  return buffer;
}