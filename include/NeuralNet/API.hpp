#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #if defined(NEURALNET_DLL_BUILD)
    #define NEURALNET_API __declspec(dllexport)
  #elif defined(NEURALNET_DLL)
    #define NEURALNET_API __declspec(dllimport)
  #else
    #define NEURALNET_API
  #endif
#else
  #define NEURALNET_API
#endif