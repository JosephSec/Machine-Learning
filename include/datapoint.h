#pragma once

#include <matrix2d.h>


typedef struct {
  Matrix inputs;
  Matrix expectedOutput;
} DataPoint;


void free_datapoint(DataPoint *_dataPoint) {
  free_matrix(&_dataPoint->inputs);
  free_matrix(&_dataPoint->expectedOutput);
}