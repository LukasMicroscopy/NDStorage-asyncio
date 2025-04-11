from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("ndstorage/simple_file.pyx", language_level="3"),
)
