import os
import re
import sys
import sysconfig
import platform
import subprocess

from subprocess import CalledProcessError
from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            if ext is None:
                continue
            self.build_extension(ext)
        
kwargs = dict(
    name='Jacobian',
    version='0.8',
    author='David Freifeld',
    author_email='freifeld.david@gmail.com',
    description='A Python library for neural networks.',
    long_description='',
    ext_modules=[CMakeExtension('_jacobian')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=True,
    packages=['jacobian']
)

# likely there are more exceptions, take a look at yarl example
try:
    setup(**kwargs
        # name='Jacobian',
        # version='0.8',
        # author='David Freifeld',
        # author_email='freifeld.david@gmail.com',
        # description='A Python library for neural networks.',
        # long_description='',
        # ext_modules=[CMakeExtension('_jacobian')],
        # cmdclass=dict(build_ext=CMakeBuild),
        # zip_safe=True,
        # package=['jacobian']
    )        
except CalledProcessError:
    print('Failed to build extension!')
    del kwargs['ext_modules']
    setup(**kwargs)
