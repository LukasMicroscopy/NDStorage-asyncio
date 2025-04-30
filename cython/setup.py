from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ndcfile/ndtiff_cfile.pyx", language_level="3", annotate=True),
    name="ndcfile",
    version="0.1",
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
