import os
from setuptools import setup
import versioneer
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fenicsprecice',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='FEniCS-preCICE adapter is a preCICE adapter for the open source computing platform FEniCS.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/precice/fenics-adapter',
      author='the preCICE developers',
      author_email='info@precice.org',
      license='LGPL-3.0',
      packages=['fenicsprecice'],
      install_requires=['pyprecice>=3.0.0.0', 'scipy', 'numpy>=1.13.3, <2', 'mpi4py<4'],
      test_suite='tests',
      zip_safe=False)
