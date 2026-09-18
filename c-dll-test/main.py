import os
import time
import ctypes
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
bin_folder = os.path.join(dir_path, "bin")
os.add_dll_directory(bin_folder)

if os.name == 'nt':  # Windows
  lib = ctypes.CDLL(os.path.join(bin_folder, "clib.dll"))

lib.HelloWorld.argtypes = []
lib.HelloWorld.restype = None

lib.Loop.argtypes = [ctypes.c_int, ctypes.c_char_p]
lib.Loop.restype = None

lib.InitParams.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.InitParams.restype = None

lib.ApplyForces.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.ApplyForces.restype = None


positions: None
velocitys: None

def init_params(count: int) -> None:
  global positions, velocitys

  float_count = count * 2
  InputArrayType = ctypes.c_float * float_count

  positions = [0] * float_count
  velocitys = [0] * float_count

  for i in range(count):
    start = i * 2

    positions[start + 0] = random.uniform(-1,1)
    positions[start + 1] = random.uniform(-1,1)

    velocitys[start + 0] = random.uniform(-1,1)
    velocitys[start + 1] = random.uniform(-1,1)

  positions = InputArrayType(*positions)
  velocitys = InputArrayType(*velocitys)
def apply_forces(count: int) -> None:
  global positions, velocitys

  for i in range(count):
    start = i * 2

    positions[start + 0] += velocitys[start + 0]
    positions[start + 1] += velocitys[start + 1]

def test_py(count: int) -> int: # returns ms test took
  prev_time = time.perf_counter()
  init_params(count)
  apply_forces(count)
  return int((time.perf_counter() - prev_time) * 1000)
def test_c(count: int) -> int: # returns ms test took
  global positions, velocitys

  prev_time = time.perf_counter()

  float_count = count * 2
  InputArrayType = ctypes.c_float * float_count

  positions = InputArrayType()
  velocitys = InputArrayType()

  lib.InitParams(positions, velocitys, count)
  lib.ApplyForces(positions, velocitys, count)
  return int((time.perf_counter() - prev_time) * 1000)

test_count = int(input("Enter Test Count: "))
# print(test_py(test_count))
print(test_c(test_count))