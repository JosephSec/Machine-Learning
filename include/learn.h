#pragma once

#include <stdlib.h>

#include <datapoint.h>
#include <dense.h>
#include <fnn.h>


inline float calculate_error(const Matrix *_output, const Matrix *_expectedOutput) {
  float errorSum = 0;
  for(int i = 0; i < _output->cols; i++) {
    float error = get_element_matrix(_output, 0,i) - get_element_matrix(_expectedOutput, 0,i);
    errorSum += error * error;
  }

  return errorSum / (float)(_output->cols);
}

inline void log_data_set_loss(const FNN *_fnn, const DataPoint *_dataSet, unsigned int _dataSetSize, unsigned int _precision) {
  Matrix output;
  
  for(int i = 0; i < _dataSetSize; i++) {
    reshape_matrix_to_input_layer(&output, _fnn);
    
    forward_fnn_inplace(_fnn, &output, &_dataSet[i].inputs);
    float error = calculate_error(&output, &_dataSet[i].expectedOutput);

    char *inputStr = get_string_matrix_row(_dataSet[i].inputs.data, _dataSet[i].inputs.cols, _precision);
    char *outputStr = get_string_matrix_row(output.data, output.cols, _precision);

    printf("input: %s\n", inputStr);
    printf("output: %s\n", outputStr);
    printf("error: %.*f\n\n", _precision, error);

    free(inputStr);
    free(outputStr);
  }

  free_matrix(&output);
}