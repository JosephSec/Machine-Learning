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


#ifdef __cplusplus
extern "C" {
#endif

PYLIB void HelloWorld() {
  printf("Hello, World!");
}

PYLIB void Loop(int _iterations, const char* _msg) {
  if(_msg == NULL) return;

  for(int i = 0; i < _iterations; i++) {
    printf("%i: %s\n", i, _msg);
  }
}


uint64_t xg_state;

void seed_fast_rng() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  xg_state = ((uint64_t)ts.tv_sec << 32) | ts.tv_nsec;
  if (xg_state == 0) xg_state = 1; // Xorshift state must never be 0
}
uint64_t next_random() {
  uint64_t x = xg_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  xg_state = x;
  return x * 0x2545F4914F6CDD1DULL;
}
double get_uniform_real(double low, double high) {
  // Map 64-bit integer uniformly into a [0, 1) double space
  double scaled = (double)next_random() / (double)UINT64_MAX;
  return low + scaled * (high - low);
}

PYLIB void InitParams(float *_a, float *_b, int _count) {
  seed_fast_rng();

  for(int i = 0; i < _count; i++) {
    unsigned int start = i * 2;

    _a[start+0] = get_uniform_real(-1,1);
    _a[start+1] = get_uniform_real(-1,1);
    
    _b[start+0] = get_uniform_real(-1,1);
    _b[start+1] = get_uniform_real(-1,1);
  }
}
PYLIB void ApplyForces(float *_pos, float *_vel, int _count) {
  for(int i = 0; i < _count; i++) {
    const unsigned int start = i * 2;

    _pos[start + 0] += _vel[start + 0];
    _pos[start + 1] += _vel[start + 1];
  }
}

#ifdef __cplusplus
}
#endif