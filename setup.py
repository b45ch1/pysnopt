#!/usr/bin/env python
# author: Sebastian F. Walter, Manuel Kudruss

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os
import numpy as np

BASEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(BASEDIR)

extra_params = {}
extra_params['include_dirs'] = [
    '/usr/include',
    os.path.join(BASEDIR, 'include'),
    np.get_include()]
extra_params['extra_compile_args'] = ["-O2"]
extra_params['extra_link_args'] = ["-Wl,-O1", "-Wl,--as-needed"]

extra_params = extra_params.copy()
extra_params['libraries'] = ['snopt7']

extra_params['library_dirs'] = ['/usr/lib', os.path.join(BASEDIR, 'lib')]
extra_params['language'] = 'fortran'

if os.name == 'posix':
    extra_params['runtime_library_dirs'] = extra_params['library_dirs']

ext_modules = [
    Extension("snopt",  ["snopt.pyx", "snopt.pxd"],   **extra_params),
]

setup(
    name='snopt interface',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
)
