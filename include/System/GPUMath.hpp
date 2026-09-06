#pragma once

#include <System/Matrix.hpp>


namespace GPUMath {
  void Init();
  void ClearGPUMemory();

  Matrix ASubtractBMulScalar(Matrix& a, Matrix& b, float scalar);
};