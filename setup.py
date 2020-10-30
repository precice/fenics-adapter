import os
from setuptools import setup

# from https://stackoverflow.com/a/9079062
import sys  
if sys.version_info[0] < 3:
    raise Exception("fenicsprecice only supports Python3. Did you run $python setup.py <option>.? Try running $python3 setup.py <option>.")

# using point 3. in https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
version = {}
with open("fenicsprecice/version.py") as fp:
    exec(fp.read(), version)
    
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fenicsprecice',
      version=version['__version__'],
      description='FEniCS-preCICE adapter is a preCICE adapter for the open source computing platform FEniCS.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/precice/fenics-adapter',
      author="Benjamin Rueth",
      author_email='benjamin.rueth@tum.de',
      license='LGPL-3.0',
      packages=['fenicsprecice'],
      install_requires=['pyprecice>=2.0.0', 'fenics', 'scipy', 'numpy>=1.13.3'],
      test_suite='tests',
      zip_safe=False)
