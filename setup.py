from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='arpack',
    version="0.1",
    ext_modules=[
        CppExtension('arpack', ['eigsh.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch>=1.8']
)
