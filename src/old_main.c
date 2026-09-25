#include <stdio.h>
#include <stdlib.h>

#include <clock.h>
#include <datapoint.h>
#include <dense.h>
#include <fnn.h>
#include <learn.h>


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
  forward_dense_inplace(_dense, _outputs, _inputs);

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


// #define MATCH_DATASET
#define XOR_DATASET
void create_dataset(DataPoint **_dst) {
  (*_dst) = malloc(4 * sizeof(DataPoint));

  #if defined(MATCH_DATASET)
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float outputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};

    for(int i = 0; i < 4; i++) {
      (*_dst)[i] = (DataPoint){
        .inputs = create_list_matrix(1,2, inputs[i]),
        .expectedOutput = create_list_matrix(1,2, outputs[i]),
      };
    }

  #elif defined(XOR_DATASET)
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
int old_main(int argc, char* argv[]) {
  // test_speed();

  const size_t epochs = get_console_ull("Enter Epoch Count", 64, false);
  const uint8_t logPrecision = get_console_ull("Enter Log Precision", 64, true);


  DataPoint *dataSet;
  create_dataset(&dataSet);

  #if defined(MATCH_DATASET)
    FNN fnn = create_fnn((unsigned int[]){2,2}, 1);
  #elif defined(XOR_DATASET)
    FNN fnn = create_fnn((unsigned int[]){2,2,1}, 2);
  #endif

  log_data_set_loss(&fnn, dataSet, 4, logPrecision);

  getchar();
  system("cls");

  const float NUDGE = 1e-7;
  const float learnRate = 1e-2;

  Matrix output;
  reshape_matrix_to_input_layer(&output, &fnn);

  Clock clock = create_clock();
  for(int i = 0; i < epochs; i++) {
    for(int j = 0; j < 4; j++) {
      const DataPoint *dataPoint = &dataSet[j];
 
      for(int layerIndex = 0; layerIndex < fnn.layerCount; layerIndex++) {
        Dense *layer = &fnn.layers[layerIndex];

        for(int out = 0; out < layer->outputCount; out++) {
          forward_fnn_inplace(&fnn, &output, &dataPoint->inputs);
          float errorA = calculate_error(&output, &dataPoint->expectedOutput);

          float *biasPtr = get_element_ptr_matrix(&layer->biases, 0,out);

          *biasPtr += NUDGE;
          
          forward_fnn_inplace(&fnn, &output, &dataPoint->inputs);
          float errorB = calculate_error(&output, &dataPoint->expectedOutput);

          *biasPtr -= NUDGE;

          *biasPtr -= learnRate * ((errorB - errorA) / NUDGE);

          for(int in = 0; in < layer->inputCount; in++) {
            forward_fnn_inplace(&fnn, &output, &dataPoint->inputs);
            float errorA = calculate_error(&output, &dataPoint->expectedOutput);

            float *weightPtr = get_element_ptr_matrix(&layer->weights, in,out);

            *weightPtr += NUDGE;

            forward_fnn_inplace(&fnn, &output, &dataPoint->inputs);
            float errorB = calculate_error(&output, &dataPoint->expectedOutput);

            *weightPtr -= NUDGE;

            *weightPtr -= learnRate * ((errorB - errorA) / NUDGE);
          }
        }
      }
    }
  }
  float trainingTime = restart_clock(&clock);
  
  free_matrix(&output);

  log_data_set_loss(&fnn, dataSet, 4, logPrecision);
  printf("training time: %ims\n", (int)(trainingTime * 1000));


  printf("Press Enter to Continue...");
  getchar();
  return 0;
}