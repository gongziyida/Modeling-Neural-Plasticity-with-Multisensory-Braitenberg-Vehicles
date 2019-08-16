from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
    	name="*",
        sources=["*.pyx"],
        libraries=["core"],
        library_dirs=["lib"],
        include_dirs=[numpy.get_include(), "lib"]
    ),
]

setup(name="BV Sim",
      ext_modules=cythonize(ext_modules))