#include <stdio.h>
#include <stdlib.h>

#include <dense.h>

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
  struct timespec previous_time, current_time;
  
  const unsigned int layerInput = 2;
  const unsigned int layerOutput = 2;

  Matrix inputs = create_uniform_matrix(1,layerInput, -1,1);
  Matrix weights = create_uniform_matrix(layerInput,layerOutput, -.5,.5);
  Matrix biases = create_matrix(1,layerOutput, 1);
  Matrix output = create_matrix(1,layerOutput, 0);

  timespec_get(&previous_time, 1);
  for(uint64_t i = 0; i < iterations; i++) {
    inputs.data[0] += 1e-12;
    test_matrix(&inputs, &weights, &biases, &output, false);
  }
  test_matrix(&inputs, &weights, &biases, &output, true);
  timespec_get(&current_time, 1);

  free_matrix(&inputs);
  free_matrix(&weights);
  free_matrix(&biases);
  free_matrix(&output);

  return (current_time.tv_sec - previous_time.tv_sec) + (current_time.tv_nsec - previous_time.tv_nsec) / 1000000000.0;
}

void test_dense(const Dense *_dense, const Matrix *_inputs, Matrix *_outputs, bool log_values) {
  forward_inplace(_outputs, _dense, _inputs);

  if(log_values == true) {
    char *inputsStr = get_string_matrix(_inputs, 4);
    char *weightsStr = get_string_matrix(&_dense->weights, 4);
    char *biasesStr = get_string_matrix(&_dense->biases, 4);
    char *outputsStr = get_string_matrix(_outputs, 4);

    printf("inputs = %s\n", inputsStr);
    printf("weights = %s\n",weightsStr);
    printf("biases = %s\n", biasesStr);
    printf("output = %s\n", outputsStr);

    free(inputsStr);
    free(weightsStr);
    free(biasesStr);
    free(outputsStr);
  }
}
float test_dense_functions(uint64_t iterations) {
  struct timespec previous_time, current_time;

  Dense dense = create_dense(2,2);

  Matrix inputs = create_uniform_matrix(1,dense.inputCount, -1,1);
  Matrix outputs = create_matrix(1,dense.outputCount, 0);

  timespec_get(&previous_time, 1);
  for(uint64_t i = 0; i < iterations; i++) {
    inputs.data[0] += 1e-12;
    test_dense(&dense, &inputs, &outputs, false);
  }
  test_dense(&dense, &inputs, &outputs, true);
  timespec_get(&current_time, 1);

  free_matrix(&inputs);
  free_matrix(&outputs);
  free_dense(&dense);

  return (current_time.tv_sec - previous_time.tv_sec) + (current_time.tv_nsec - previous_time.tv_nsec) / 1000000000.0;
}

int main(int argc, char* argv[]) {
  // float deltaTime = test_matrix_functions(1e+9);
  float deltaTime = test_dense_functions(1e+9);
  printf("\n\n%ims", (int)(deltaTime * 1000));


  int ch = getchar();
  return 0;
}