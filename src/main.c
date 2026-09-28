#include <stdio.h>

#include <debug.h>
#include <clock.h>

#include <fnn.h>
#include <datapoint.h>
#include <learn.h>


#define DO_MATRIX_TEST false
#define LOG_PRECISION 6

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


//Optimizer learnRate, nudge, and dataSet are overwritten by the loaded model
void load_model_data(FNN *_fnn, Optimizer *_optimizer) {
  const size_t option = get_console_ull("Enter Model Option\n[1]: Match\n[2]: Xor\nEnter Choice: ", 2, true);

  switch(option) {
    default:
    case 1: //Match
      get_match_model(_fnn, _optimizer);
      break;
    case 2: //Xor
      get_xor_model(_fnn, _optimizer);
      break;
  }
}
int main(int argc, char* argv[]) {
  #if DO_MATRIX_TEST
    float deltaTime = test_matrix_functions(get_console_ull("Enter Test Iterations: ", 64, false));
    printf("test time: %ims\n", (int)(deltaTime * 1000));
    getchar();
    system("cls");
  #endif

  const size_t epochs = get_console_ull("Enter Epoch Count: ", 64, false);
  Optimizer optimizer = create_optimizer(1e-2, 1e-7, epochs, 10);

  FNN fnn;
  load_model_data(&fnn, &optimizer);
  flush_stdin();

  print_fnn_structure("FNN Structure: %s\n\n", &fnn);
  printf("Pre Training Results\n\n");
  log_data_set_loss(&fnn, &optimizer.dataSet, LOG_PRECISION);
  getchar();
  system("cls");

  printf("training for %i epochs...\n", epochs);
  log_finite_difference_train_fnn(&fnn, &optimizer, LOG_PRECISION);
  printf("training complete.");
  getchar();
  system("cls");

  print_fnn_structure("FNN Structure: %s\n\n", &fnn);
  printf("Post Training Results (%i epochs)\n\n", epochs);
  log_data_set_loss(&fnn, &optimizer.dataSet, LOG_PRECISION);

  getchar();
  return 0;
}