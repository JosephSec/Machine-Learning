#pragma once

#include <math.h>
#include <matrix2d.h>


typedef enum {
  LOSS_MSE,
  LOSS_BCE,
} LossType;
typedef float (*LossFunction)(const Matrix*, const Matrix*);

float calculate_mse_loss(const Matrix *_output, const Matrix *_expectedOutput) {
  float totalLoss = 0;
  for(int i = 0; i < _output->cols; i++) {
    float loss = get_element_matrix(_output, 0,i) - get_element_matrix(_expectedOutput, 0,i);
    totalLoss += loss * loss;
  }

  return totalLoss / (float)(_output->cols);
}
float calculate_bce_loss(const Matrix *_output, const Matrix *_expectedOutput) {
  float totalLoss = 0;
  
  for(int i = 0; i < _output->cols; i++) {
    float y_hat = get_element_matrix(_output, 0,i);
    float y = get_element_matrix(_expectedOutput, 0,i);

    if(y_hat < 1e-7f) y_hat = 1e-7f;
    if(y_hat > 1.0f - 1e-7f) y_hat = 1.0f - 1e-7f;

    float loss = -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
    totalLoss += loss;
  }

  return totalLoss / (float)_output->cols;
}
