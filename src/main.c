#include <stdio.h>
#include <stdlib.h>

#include <clock.h>
#include <dense.h>
#include <datapoint.h>


void test_matrix(Matrix *_inputs, Matrix *_weights, Matrix *_biases, Matrix *_output, bool log_values) {
  multiply_matrices_inplace(_output, _inputs, _weights);
  add_matrices_inplace(_output, _output, _biases);

  if(log_values == true) {
    char *inputsStr = get_string_matrix(_inputs, 4);
    char *weightsStr = get_string_matrix(_weights, 4);
    char *biasesStr = get_string_matrix(_biases, 4);
    char *outputStr = get_string_matrix(_output, 4);

    printf("inputs = %s\n", inputsStr);
    printf("weights = %s\n",weightsStr);
    printf("biases = %s\n", biasesStr);
    printf("output = %s\n", outputStr);

    free(inputsStr);
    free(weightsStr);
    free(biasesStr);
    free(outputStr);
  }
}
float test_matrix_functions(uint64_t iterations) {
  const unsigned int layerInput = 2;
  const unsigned int layerOutput = 2;

  Matrix inputs = create_uniform_matrix(1,layerInput, -1,1);
  Matrix weights = create_uniform_matrix(layerInput,layerOutput, -.5,.5);
  Matrix biases = create_matrix(1,layerOutput, 1);
  Matrix output = create_matrix(1,layerOutput, 0);

  Clock clock = create_clock();
  for(uint64_t i = 0; i < iterations; i++) {
    inputs.data[0] += 1e-12;
    test_matrix(&inputs, &weights, &biases, &output, false);
  }
  test_matrix(&inputs, &weights, &biases, &output, true);

  free_matrix(&inputs);
  free_matrix(&weights);
  free_matrix(&biases);
  free_matrix(&output);

  return restart_clock(&clock);
}

void test_dense(const Dense *_dense, const Matrix *_inputs, Matrix *_outputs, bool log_values) {
  forward_dense_inplace(_outputs, _dense, _inputs);

  if(log_values == true) {
    char *denseStr = get_string_dense(_dense);

    char *inputsStr = get_string_matrix(_inputs, 4);
    char *outputsStr = get_string_matrix(_outputs, 4);

    printf("dense\n%s\n\n", denseStr);
    printf("inputs = %s\n", inputsStr);
    printf("output = %s\n", outputsStr);

    free(denseStr);
    free(inputsStr);
    free(outputsStr);
  }
}
float test_dense_functions(uint64_t iterations) {
  
  Dense dense = create_dense(2,2);
  
  Matrix inputs = create_uniform_matrix(1,dense.inputCount, -1,1);
  Matrix outputs = create_matrix(1,dense.outputCount, 0);
  
  Clock clock = create_clock();
  for(uint64_t i = 0; i < iterations; i++) {
    inputs.data[0] += 1e-12;
    test_dense(&dense, &inputs, &outputs, false);
  }
  test_dense(&dense, &inputs, &outputs, true);
  
  free_matrix(&inputs);
  free_matrix(&outputs);
  free_dense(&dense);
  
  return restart_clock(&clock);
}

void test_speed() {
  // float deltaTime = test_matrix_functions(1e+9);
  float deltaTime = test_dense_functions(1e+8);
  printf("\n\n%ims", (int)(deltaTime * 1000));
}


void create_dataset(DataPoint **_dst) {
  Matrix input = create_matrix(1,2, 0);
  Matrix output = create_matrix(1,2, 0);
  
  (*_dst) = malloc(4 * sizeof(DataPoint));

  input.data[0] = 0;
  input.data[1] = 0;
  output.data[0] = 0;
  output.data[1] = 0;
  (*_dst)[0] = (DataPoint){
    .inputs = copy_matrix(&input),
    .expectedOutput = copy_matrix(&output)
  };

  input.data[0] = 0;
  input.data[1] = 1;
  output.data[0] = 0;
  output.data[1] = 1;
  (*_dst)[1] = (DataPoint){
    .inputs = copy_matrix(&input),
    .expectedOutput = copy_matrix(&output)
  };

  input.data[0] = 1;
  input.data[1] = 0;
  output.data[0] = 1;
  output.data[1] = 0;
  (*_dst)[2] = (DataPoint){
    .inputs = copy_matrix(&input),
    .expectedOutput = copy_matrix(&output)
  };

  input.data[0] = 1;
  input.data[1] = 1;
  output.data[0] = 1;
  output.data[1] = 1;
  (*_dst)[3] = (DataPoint){
    .inputs = copy_matrix(&input),
    .expectedOutput = copy_matrix(&output)
  };

  free_matrix(&input);
  free_matrix(&output);
}
float calculate_dense_error(const Matrix *_output, const Matrix *_expectedOutput) {
  float errorSum = 0;
  for(int i = 0; i < _output->cols; i++) {
    float error = get_element_matrix(_output, 0,i) - get_element_matrix(_expectedOutput, 0,i);
    errorSum += error * error;
  }

  return errorSum / (float)(_output->cols);
}

void log_data_set_loss(const Dense *_dense, const DataPoint *_dataSet, unsigned int _dataSetSize, unsigned int _precision) {
  Matrix output = create_matrix(1,_dense->outputCount, 0);

  for(int i = 0; i < _dataSetSize; i++) {
    forward_dense_inplace(&output, _dense, &_dataSet[i].inputs);
    float error = calculate_dense_error(&output, &_dataSet[i].expectedOutput);

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
int main(int argc, char* argv[]) {
  // test_speed();


  DataPoint *dataSet;
  create_dataset(&dataSet);

  Dense layer = create_dense(2,2);

  log_data_set_loss(&layer, dataSet, 4, 24);

  getchar();
  system("cls");

  const float NUDGE = 1e-7;
  const size_t epochs = 1e+4;
  const float learnRate = 1e-2;

  Matrix output = create_matrix(1,layer.outputCount, 0);

  Clock clock = create_clock();
  for(int i = 0; i < epochs; i++) {
    for(int j = 0; j < 4; j++) {
      const DataPoint *dataPoint = &dataSet[j];

      for(int out = 0; out < layer.outputCount; out++) {
        forward_dense_inplace(&output, &layer, &dataPoint->inputs);
        float errorA = calculate_dense_error(&output, &dataPoint->expectedOutput);

        float *biasPtr = get_element_ptr_matrix(&layer.biases, 0,out);

        *biasPtr += NUDGE;
        
        forward_dense_inplace(&output, &layer, &dataPoint->inputs);
        float errorB = calculate_dense_error(&output, &dataPoint->expectedOutput);

        *biasPtr -= NUDGE;

        *biasPtr -= learnRate * ((errorB - errorA) / NUDGE);

        for(int in = 0; in < layer.inputCount; in++) {
          forward_dense_inplace(&output, &layer, &dataPoint->inputs);
          float errorA = calculate_dense_error(&output, &dataPoint->expectedOutput);

          float *weightPtr = get_element_ptr_matrix(&layer.weights, in,out);

          *weightPtr += NUDGE;
          
          forward_dense_inplace(&output, &layer, &dataPoint->inputs);
          float errorB = calculate_dense_error(&output, &dataPoint->expectedOutput);

          *weightPtr -= NUDGE;

          *weightPtr -= learnRate * ((errorB - errorA) / NUDGE);
        }
      }
    }
  }
  float trainingTime = restart_clock(&clock);
  
  free_matrix(&output);

  log_data_set_loss(&layer, dataSet, 4, 24);
  printf("training time: %ims\n", (int)(trainingTime * 1000));


  printf("Press Enter to Continue...");
  getchar();
  return 0;
}