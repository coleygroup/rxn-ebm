import os
import platform
import subprocess
from distutils.command.build import build

# build cuda lib
import torch
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

BASEPATH = os.path.dirname(os.path.abspath(__file__))

compile_args = []
link_args = []

if platform.system() != 'Darwin':  # add openmp
    compile_args.append('-fopenmp')
    link_args.append('-lgomp')
    
ext_modules=[CppExtension('extlib', 
                          ['gln/mods/torchext/src/extlib.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args)]

if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('extlib_cuda',
                                    ['gln/mods/torchext/src/extlib_cuda.cpp', 'gln/mods/torchext/src/extlib_cuda_kernels.cu']))

class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        folders = [
            os.path.join(BASEPATH, 'gln/mods/mol_gnn/mg_clib'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()

setup(name='gln',
      py_modules=['gln'],
      ext_modules=ext_modules,
      install_requires=[
          'torch',
      ],
      cmdclass={
          'develop': custom_develop,
          'build_ext': BuildExtension,
        }
)
