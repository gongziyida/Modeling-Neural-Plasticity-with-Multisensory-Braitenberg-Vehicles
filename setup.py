from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
    	name="Space",
        sources=["Space.pyx"],
        libraries=["core"],
        library_dirs=["lib"],
        include_dirs=[numpy.get_include(), "lib"]
    ),
    Extension(
    	name="Layers",
        sources=["Layers.pyx"],
        libraries=["core"],
        library_dirs=["lib"],
        include_dirs=[numpy.get_include(), "lib"]
    ),
    Extension(
        name="Movement",
        sources=["Movement.pyx"],
        libraries=["core"],
        library_dirs=["lib"],
        include_dirs=[numpy.get_include(), "lib"]
    ),
    # Extension(
    #     name="BraitenbergVehicles",
    #     sources=["BraitenbergVehicles.pyx"],
    #     libraries=["core"],
    #     library_dirs=["lib"],
    #     include_dirs=[numpy.get_include(), "lib"]
    # ),
    # Extension(
    #     name="Simulation",
    #     sources=["Simulation.pyx"],
    #     libraries=["core"],
    #     library_dirs=["lib"],
    #     include_dirs=[numpy.get_include(), "lib"]
    # )
]

setup(name="Space",
      ext_modules=cythonize(ext_modules))