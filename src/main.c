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


#define MATCH_DATASET false
#define XOR_DATASET true

void create_dataset(DataPoint **_dst) {
  
  #if MATCH_DATASET
    (*_dst) = malloc(4 * sizeof(DataPoint));

    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float outputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};

    for(int i = 0; i < 4; i++) {
      (*_dst)[i] = (DataPoint){
        .inputs = create_list_matrix(1,2, inputs[i]),
        .expectedOutput = create_list_matrix(1,2, outputs[i]),
      };
    }
  #elif XOR_DATASET
    (*_dst) = malloc(4 * sizeof(DataPoint));

    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float outputs[4][1] = {{0}, {1}, {1}, {0}};

    for(int i = 0; i < 4; i++) {
      (*_dst)[i] = (DataPoint){
        .inputs = create_list_matrix(1,2, inputs[i]),
        .expectedOutput = create_list_matrix(1,1, outputs[i]),
      };
    }
  #endif
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


  #if MATCH_DATASET
    FNN fnn = create_fnn((unsigned int[]){2,2}, 1);
  #elif XOR_DATASET
    FNN fnn = create_fnn((unsigned int[]){2,2,1}, 2);
  #endif

  print_fnn_structure("FNN Structure: %s\n\n", &fnn);

  DataPoint *dataSet;
  create_dataset(&dataSet);

  Matrix current_in;
  Matrix current_out;
  for(int i = 0; i < 4; i++) {
    print_matrix_row("Input: %s\n", &dataSet[i].inputs, 0, 4);
    
    current_in = copy_matrix(&dataSet[i].inputs);

    for(int j = 0; j < fnn.layerCount; j++) {
      reshape_matrix_to_output_dense(&fnn.layers[j], &current_out);    
      fill_value_matrix(&current_out, 0);
      
      forward_dense_inplace(&fnn.layers[j], &current_out, &current_in);

      free_matrix(&current_in);

      if(j < fnn.layerCount - 1) current_in = current_out;
    }

    print_matrix_row("Output: %s\n\n", &current_out, 0, 4);

  }
  free_matrix(&current_in);
  free_matrix(&current_out);

  getchar();
  return 0;
}