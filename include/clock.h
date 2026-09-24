#pragma once

#include <time.h>


typedef struct {
  struct timespec startTime;
} Clock;


Clock create_clock() {
  Clock output;
  timespec_get(&output.startTime, 1);
  return output;
}

float restart_clock(Clock *_clock) {
  struct timespec current;
  timespec_get(&current, 1);

  float output = (current.tv_sec - _clock->startTime.tv_sec) + (current.tv_nsec - _clock->startTime.tv_nsec) / 1000000000.0;
  _clock->startTime = current;

  return output;
}