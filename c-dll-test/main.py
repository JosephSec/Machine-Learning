import ctypes
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
bin_folder = os.path.join(dir_path, "bin")
os.add_dll_directory(bin_folder)

if os.name == 'nt':  # Windows
  lib = ctypes.CDLL(os.path.join(bin_folder, "clib.dll"))

lib.HelloWorld.argtypes = []
lib.HelloWorld.restype = None

lib.HelloWorld()