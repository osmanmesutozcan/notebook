#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from distutils.core import setup, Extension


setup(name='hough_module', version='0.0',
      ext_modules=[Extension('hough_module', ['hough_module.c'])],
      include_dirs=[np.get_include()])
