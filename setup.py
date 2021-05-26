from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='arpack',
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CppExtension('arpack._C', ['arpack/cpp/bind.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch>=1.8']
)
