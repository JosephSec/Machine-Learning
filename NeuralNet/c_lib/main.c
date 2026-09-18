#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>


#if defined(_WIN32) || defined(_WIN64)
  #if defined(PYLIB_EXPORT)
    #define PYLIB __declspec(dllexport)
  #elif defined(PYLIB_IMPORT)
    #define PYLIB __declspec(dllimport)
  #else
    #define PYLIB
  #endif
#else
  #define PYLIB
#endif


uint64_t xg_state = 0;
void seed_fast_rng() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  xg_state = ((uint64_t)ts.tv_sec << 32) | ts.tv_nsec;
  if (xg_state == 0) xg_state = 1;
}
uint64_t next_random() {
  uint64_t x = xg_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  xg_state = x;
  return x * 0x2545F4914F6CDD1DULL;
}
float get_uniform_real(float low, float high) {
  float scaled = (float)next_random() / (float)UINT64_MAX;
  return low + scaled * (high - low);
}


#ifdef __cplusplus
extern "C" {
#endif

PYLIB void InitRandomFloat(float *_data, unsigned int _count, float _min, float _max) {
  seed_fast_rng();
  for(int i = 0; i < _count; i++) _data[i] = get_uniform_real(_min, _max);
}
PYLIB void ForwardTensor(float *_output, float *_input, float *_weights, float *_biases, unsigned int _in, unsigned int _out) {
  for(int o = 0; o < _out; o++) {
    float weightedInput = _biases[o];
    for(int i = 0; i < _in; i++) {
      weightedInput += _weights[i + o * _in] * _input[i];
    }

    _output[o] = weightedInput;
  }
}

#ifdef __cplusplus
}
#endif