import ctypes
import os
import sys

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "ygopro-core", "bin", "release",
                   "libocgcore.dylib" if sys.platform == "darwin" else "libocgcore.so")

lib = ctypes.CDLL(LIB)
lib.OCG_GetVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib.OCG_GetVersion.restype = None

major = ctypes.c_int(0)
minor = ctypes.c_int(0)
lib.OCG_GetVersion(ctypes.byref(major), ctypes.byref(minor))
print(f"OCG_GetVersion -> {major.value}.{minor.value}")
