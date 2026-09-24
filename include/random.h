#include <stdint.h>
#include <time.h>


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


void fill_random_uniform(float *_data, unsigned int _count, float _min, float _max) {
  seed_fast_rng();
  for(int i = 0; i < _count; i++) _data[i] = get_uniform_real(_min, _max);
}
