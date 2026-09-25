#pragma once

#include <stdlib.h>

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


inline void reshape_matrix_to_input_dense(const Dense *_dense, Matrix *_dst) {
  reshape_matrix(_dst, 1, _dense->inputCount);
}
inline void reshape_matrix_to_output_dense(const Dense *_dense, Matrix *_dst) {
  reshape_matrix(_dst, 1, _dense->outputCount);
}


inline void forward_dense_inplace(const Dense *_dense, Matrix *_dst, const Matrix *_inputs) {
  multiply_matrices_inplace(_dst, _inputs, &_dense->weights);
  add_matrices_inplace(_dst, _dst, &_dense->biases);
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