#pragma once

#include <stdio.h>
#include <stdlib.h>


#if defined(NDEBUG)
  #define ASSERT_MSG(expression) ((void)0)
#else
  #define ASSERT_MSG(expression, err_type, format, ...) \
    do { \
      if(!(expression)) { \
        fprintf(stderr, "[%s ERROR]: Assertion '%s' failed at %s:%d\n", \
                err_type, #expression, __FILE__, __LINE__); \
        fprintf(stderr, format, ##__VA_ARGS__); \
        fprintf(stderr, "\n"); \
        abort(); \
      } \
    } while(0)
#endif