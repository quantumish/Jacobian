#
#  setup.py
#  Jacobian
#
#  Created by David Freifeld
#  Copyright © 2020 David Freifeld. All rights reserved.
#

from setuptools import setup, Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

setup(
    name='Jacobian',
    version='1.0',
    description='A library for neural networks.',
    packages=['jacobian'],
    package_data={
        'jacobian': ['jacobian.cpython-37m-darwin.so'],
    },
    distclass=BinaryDistribution
)
