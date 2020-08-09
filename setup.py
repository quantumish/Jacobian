from subprocess import CalledProcessError

kwargs = dict(
    name='Jacobian',
    version='0.8',
    author='David Freifeld',
    author_email='freifeld.david@gmail.com',
    description='A Python library for neural networks.',
    long_description='',
    ext_modules=[CMakeExtension('jacobian._jacobian')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=['cmake_example']
)

# likely there are more exceptions, take a look at yarl example
try:
    setup(**kwargs)        
except CalledProcessError:
    print('Failed to build extension!')
    del kwargs['ext_modules']
    setup(**kwargs)
