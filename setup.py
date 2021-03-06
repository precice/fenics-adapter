import os
from setuptools import setup
import versioneer
import warnings

# from https://stackoverflow.com/a/9079062
import sys
if sys.version_info[0] < 3:
    raise Exception("fenicsprecice only supports Python3. Did you run $python setup.py <option>.? "
                    "Try running $python3 setup.py <option>.")

try:
    from fenics import *
except ModuleNotFoundError:
    warnings.warn("No FEniCS installation found on system. Please install FEniCS and check the installation.\n\n"
                  "You can check this by running the command\n\n"
                  "python3 -c 'from fenics import *'\n\n"
                  "Please check https://fenicsproject.org/download/ for installation guidance.\n"
                  "The installation will continue, but please be aware that your installed version of the "
                  "fenics-adapter might not work as expected.")

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
      install_requires=['pyprecice>=2.0.0', 'scipy', 'numpy>=1.13.3', 'mpi4py'],
      test_suite='tests',
      zip_safe=False)
