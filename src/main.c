#include <stdio.h>

#include <clock.h>

#include <fnn.h>
#include <datapoint.h>


#define DO_MATRIX_TEST false

void test_matrix(Matrix *_inputs, Matrix *_weights, Matrix *_biases, Matrix *_output, bool log_values) {
  multiply_matrices_inplace(_output, _inputs, _weights);
  add_matrices_inplace(_output, _output, _biases);


  if(log_values == true) {
    char *inputsStr = get_string_matrix(_inputs, 4);
    char *weightsStr = get_string_matrix(_weights, 4);
    char *biasesStr = get_string_matrix(_biases, 4);
    char *outputStr = get_string_matrix(_output, 4);

    printf("inputs = %s\n", inputsStr);
    printf("weights = %s\n", weightsStr);
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


void print_matrix(const char *_str, const Matrix *_matrix, unsigned int _precision) {
  char *matrixStr = get_string_matrix(_matrix, 4);
  printf(_str, matrixStr);
  free(matrixStr);
}
void print_matrix_row(const char *_str, const Matrix *_matrix, unsigned int _row, unsigned int _precision) {
  char *rowStr = get_string_matrix_row(_matrix->data + _row * _matrix->cols, _matrix->cols, _precision);
  printf(_str, rowStr);
  free(rowStr);
}

size_t get_console_ull(const char *_msg, size_t bufferSize, bool clear_after) {
  char inputBuffer[bufferSize];
  
  printf("%s: ", _msg);
  
  fgets(inputBuffer, sizeof(inputBuffer), stdin);
  inputBuffer[strcspn(inputBuffer, "\n")] = '\0';

  if(clear_after == true) system("cls");

  return strtoull(inputBuffer, nullptr, 10);
}
int main(int argc, char* argv[]) {
  #if DO_MATRIX_TEST
    float deltaTime = test_matrix_functions(get_console_ull("Enter Test Iterations", 64, false));
    printf("test time: %ims\n", (int)(deltaTime * 1000));
    getchar();
    system("cls");
  #endif


  FNN fnn = create_fnn((unsigned int[]){2,2,1}, 2);

  DataPoint dataPoint = (DataPoint) {
    .inputs = create_list_matrix(1,2, (float[]){0,1}),
    .expectedOutput = create_list_matrix(1,2, (float[]){1}),
  };

  Matrix outputA = forward_dense(&fnn.layers[0], &dataPoint.inputs);
  Matrix outputB = forward_dense(&fnn.layers[1], &dataPoint.inputs);

  print_matrix_row("Input: %s\n", &dataPoint.inputs, 0, 4);
  print_matrix_row("Output A: %s\n", &outputA, 0, 4);
  print_matrix_row("Output B: %s\n", &outputB, 0, 4);

  free_datapoint(&dataPoint);
  free_matrix(&outputA);
  free_matrix(&outputB);

  getchar();
  return 0;
}