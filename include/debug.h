#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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


inline void flush_stdin() {
  int c;
  while((c = getchar()) != '\n' && c != EOF);
}
inline size_t get_console_ull(const char *_msg, size_t bufferSize, bool clear_after) {
  char inputBuffer[bufferSize];
  
  printf("%s", _msg);
  
  fgets(inputBuffer, sizeof(inputBuffer), stdin);
  inputBuffer[strcspn(inputBuffer, "\n")] = '\0';

  if(clear_after == true) system("cls");

  return strtoull(inputBuffer, nullptr, 10);
}