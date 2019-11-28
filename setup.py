from setuptools import setup

# from https://stackoverflow.com/a/9079062
import sys  
if sys.version_info[0] < 3:
    raise Exception("The fenicsadapter only supports Python3. Did you run $python setup.py <option>.? Try running $python3 setup.py <option>.")

setup(name='fenicsadapter',
      version='0.2',
      description='preCICE-adapter for the open source computing platform FEniCS ',
      url='https://github.com/precice/fenics-adapter',
      author="Benjamin Rueth",
      author_email='benjamin.rueth@tum.de',
      license='LGPL-3.0',
      packages=['fenicsadapter'],
      install_requires=['precice>=2.0.0', 'fenics', 'scipy', 'numpy>=1.13.3'],
      test_suite='tests',
      zip_safe=False)
