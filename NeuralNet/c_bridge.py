import os
import ctypes


dir_path = os.path.dirname(os.path.realpath(__file__))
bin_folder = os.path.join(dir_path, "c_lib")
lib = None

if os.name == 'nt':  # Windows
  os.add_dll_directory(bin_folder)
  lib = ctypes.CDLL(os.path.join(bin_folder, "c_lib.dll"))

lib.InitRandomFloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_float, ctypes.c_float]
lib.InitRandomFloat.restype = None

lib.ForwardTensor.argtypes = [
  ctypes.POINTER(ctypes.c_float), # output
  ctypes.POINTER(ctypes.c_float), # input
  ctypes.POINTER(ctypes.c_float), # weights
  ctypes.POINTER(ctypes.c_float), # biases
  ctypes.c_uint, #input count
  ctypes.c_uint, #output count
]
lib.ForwardTensor.restype = None