#include <iostream>

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


extern "C" {
  PYLIB void HelloWorld() {
    std::cout << "Hello World";
  }
}