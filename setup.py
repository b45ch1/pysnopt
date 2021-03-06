#!/usr/bin/env python
# This file is part of pysnopt, a Python interface to SNOPT.
# Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter
# License: GPL v3, see LICENSE.txt for details.

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
    # os.path.join(BASEDIR, 'f2c/src'),
    os.path.join(BASEDIR, 'cppsrc'),
    np.get_include()]
extra_params['extra_compile_args'] = ["-O2", "-g"]
extra_params['extra_link_args'] = ["-Wl,-O1", "-Wl,--as-needed"]

extra_params = extra_params.copy()
extra_params['libraries'] = ['pysnopt7']

extra_params['library_dirs'] = [os.path.join(BASEDIR, 'python')]
extra_params['language'] = 'c++'

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
