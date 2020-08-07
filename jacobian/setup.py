#
#  setup.py
#  Jacobian
#

from distutils.core import setup

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

setup(
  name        = 'cmake_cpp_pybind11',
  version     = '${PACKAGE_VERSION}',
  packages    = [ 'jacobian' ],
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  package_data = {
    '': ['jacobian.cpython-38-darwin.so']
  }
)
