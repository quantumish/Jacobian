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
    description='Python Library for Neural Nets.',
    packages=['mrbpnn'],
    package_data={
        'mrbpnn': ['mrbpnn.cpython-37m-darwin.so'],
    },
    distclass=BinaryDistribution
)
