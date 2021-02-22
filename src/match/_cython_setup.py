"""python _cython_setup.py build_ext --inplace
"""

import os
import shutil
from distutils.core import setup
from Cython.Build import cythonize

setup(name='keyword processor',
      ext_modules=cythonize("keywordprocessor.pyx"))

sofile = [f for f in os.listdir('./match') if f.endswith('.so') and not f.startswith('.')][0]
assert sofile.startswith('keywordprocessor')

os.system(f'mv ./match/{sofile} ./keywordprocessor.so')
shutil.rmtree('./build')
shutil.rmtree('./match')
os.remove('./keywordprocessor.c')
