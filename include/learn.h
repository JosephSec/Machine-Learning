#pragma once

#include <stdlib.h>

#include <datapoint.h>
#include <dense.h>
#include <fnn.h>


typedef struct {
  DataPoint *data;
  unsigned int size;
} DataSet;

inline DataSet create_dataset(unsigned int _size) {
  return (DataSet){
    .data = malloc(_size * sizeof(DataPoint)),
    .size = _size,
  };
}


typedef struct {
  float duration;
  float averageLoss;
  size_t index;
} EpochMetrics;
typedef struct {
  float learnRate;
  float nudge;
  size_t epochs;

  DataSet dataSet;

  EpochMetrics *samples;
  size_t sampleCount;
} Optimizer;

inline Optimizer create_optimizer(float _learnRate, float _nudge, size_t _epochs, size_t _maxSamples) {
  Optimizer output;
  output.learnRate = _learnRate;
  output.nudge = _nudge;
  output.epochs = _epochs;

  if(_epochs <= _maxSamples) output.sampleCount = _epochs;
  else output.sampleCount = _maxSamples;

  output.samples = malloc(output.sampleCount * sizeof(EpochMetrics));

  return output;
}

inline void free_optimizer(Optimizer *_optimizer) {
  free(_optimizer->samples);
}


inline void get_match_model(FNN *_fnn, Optimizer *_optimizer) {
  *_fnn = create_fnn(
    (unsigned int[]){2,2},
    (ActivationType[]){ACTIVATION_LINEAR},
    (Initializer[]){INITIALIZER_UNIFORM_HALF, INITIALIZER_ZERO},
    1,
    LOSS_MSE
  );


  _optimizer->learnRate = 1e-2;
  _optimizer->nudge = 1e-7;

  const float input[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
  const float output[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};

  _optimizer->dataSet = create_dataset(4);

  for(int i = 0; i < 4; i++) {
    _optimizer->dataSet.data[i].inputs = create_list_matrix(1,2, input[i]);
    _optimizer->dataSet.data[i].expectedOutput = create_list_matrix(1,2, output[i]);
  }
}
inline void get_xor_model(FNN *_fnn, Optimizer *_optimizer) {
  *_fnn = create_fnn(
    (unsigned int[]){2,2,1},
    (ActivationType[]){ACTIVATION_SIGMOID, ACTIVATION_SIGMOID},
    (Initializer[]){
      create_initializer(INIT_RANDOM, (float[]){-1.22,1.22}, 2), INITIALIZER_ZERO,
      create_initializer(INIT_RANDOM, (float[]){-1.44,1.44}, 2), INITIALIZER_ZERO
    },
    2,
    LOSS_BCE
  );


  _optimizer->learnRate = 1;
  _optimizer->nudge = 1e-2;

  const float input[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
  const float output[4][1] = {{0}, {1}, {1}, {0}};

  _optimizer->dataSet = create_dataset(4);

  for(int i = 0; i < 4; i++) {
    _optimizer->dataSet.data[i].inputs = create_list_matrix(1,2, input[i]);
    _optimizer->dataSet.data[i].expectedOutput = create_list_matrix(1,1, output[i]);
  }
}


inline float calculate_data_set_loss(const FNN *_fnn, const DataSet *_dataSet) {
  float totalError = 0;

  Matrix output;
  
  for(int i = 0; i < _dataSet->size; i++) {
    forward_fnn_inplace(_fnn, &output, &_dataSet->data[i].inputs);
    totalError += _fnn->lossFunc(&output, &_dataSet->data[i].expectedOutput);
  }

  free_matrix(&output);

  return totalError / (float)(_dataSet->size);
}
inline void log_data_set_loss(const FNN *_fnn, const DataSet *_dataSet, unsigned int _precision) {
  Matrix output;
  
  for(int i = 0; i < _dataSet->size; i++) {
    forward_fnn_inplace(_fnn, &output, &_dataSet->data[i].inputs);
    float error = _fnn->lossFunc(&output, &_dataSet->data[i].expectedOutput);

    print_matrix_row("Input: %s\n", &_dataSet->data[i].inputs, 0, _precision);
    print_matrix_row("Output: %s\n", &output, 0, _precision);
    printf("error: %.*f\n\n", _precision, error);
  }

  free_matrix(&output);
}

inline float finite_difference_train_fnn(FNN *_fnn, const Optimizer *_optimizer, unsigned int _epochs) {
  Matrix output = EMPTY_MATRIX;

  Clock clock = create_clock();
  for(int i = 0; i < _epochs; i++) {
    for(int j = 0; j < _optimizer->dataSet.size; j++) {
      const DataPoint *dataPoint = &_optimizer->dataSet.data[j];
 
      for(int layerIndex = 0; layerIndex < _fnn->layerCount; layerIndex++) {
        Dense *layer = &_fnn->layers[layerIndex];

        for(int out = 0; out < layer->outputCount; out++) {
          forward_fnn_inplace(_fnn, &output, &dataPoint->inputs);
          float errorA = _fnn->lossFunc(&output, &dataPoint->expectedOutput);
          
          float *biasPtr = get_element_ptr_matrix(&layer->biases, 0,out);

          *biasPtr += _optimizer->nudge;
          
          forward_fnn_inplace(_fnn, &output, &dataPoint->inputs);
          float errorB = _fnn->lossFunc(&output, &dataPoint->expectedOutput);

          *biasPtr -= _optimizer->nudge;

          *biasPtr -= _optimizer->learnRate * ((errorB - errorA) / _optimizer->nudge);

          for(int in = 0; in < layer->inputCount; in++) {
            forward_fnn_inplace(_fnn, &output, &dataPoint->inputs);
            float errorA = _fnn->lossFunc(&output, &dataPoint->expectedOutput);

            float *weightPtr = get_element_ptr_matrix(&layer->weights, in,out);

            *weightPtr += _optimizer->nudge;

            forward_fnn_inplace(_fnn, &output, &dataPoint->inputs);
            float errorB = _fnn->lossFunc(&output, &dataPoint->expectedOutput);

            *weightPtr -= _optimizer->nudge;

            *weightPtr -= _optimizer->learnRate * ((errorB - errorA) / _optimizer->nudge);
          }
        }
      }
    }
  }

  free_matrix(&output);

  return restart_clock(&clock);
}
inline void log_finite_difference_train_fnn(FNN *_fnn, Optimizer *_optimizer, unsigned int _precision) {
  const size_t epochRate = _optimizer->epochs / _optimizer->sampleCount;

  for(int i = 0; i < _optimizer->sampleCount; i++) {
    float trainTime = finite_difference_train_fnn(_fnn, _optimizer, epochRate);
    float loss = calculate_data_set_loss(_fnn, &_optimizer->dataSet);
    size_t epochIndex = epochRate * (i + 1);

    _optimizer->samples[i] = (EpochMetrics){
      .duration = trainTime,
      .averageLoss = loss,
      .index = epochIndex
    };

    printf("%ims | loss: %.*f | epochs: %i/%i\n", (int)(trainTime * 1000), _precision, loss, epochIndex, _optimizer->epochs);
  }
}