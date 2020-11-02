FEniCS-preCICE adapter
----------------------

<a style="text-decoration: none" href="https://travis-ci.org/precice/fenics-adapter" target="_blank">
    <img src="https://travis-ci.org/precice/fenics-adapter.svg?branch=master" alt="Build status">
</a>
<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/precice/fenics-adapter.svg" alt="GNU LGPL license">
</a>

**experimental** preCICE-adapter for the open source computing platform FEniCS

_**Note:** This adapter is currently purely expermental and limited in functionality. If you are interested in using it or you want to contribute, feel free to contact us via the [preCICE mailing list](https://mailman.informatik.uni-stuttgart.de/mailman/listinfo/precice)._

**currently only supports 2D simulations in FEniCS**

This adapter was developed by [Benjamin Rüth](https://www5.in.tum.de/wiki/index.php/Benjamin_R%C3%BCth,_M.Sc._(hons)) during his research stay at Lund University in the group for [Numerical Analysis](http://www.maths.lu.se/english/research/research-divisions/numerical-analysis/) in close collaboration with [Peter Meisrimel](https://www.lunduniversity.lu.se/lucat/user/09d80f0367a060bcf2a22d7c22e5e504).

# Installing the package

## Using pip3 to install from PyPI

It is recommended to install [fenicsprecice from PyPI](https://pypi.org/project/fenicsprecice/) via
```
$ pip3 install --user fenicsprecice
```
This should work out of the box, if all dependencies are installed correctly. If you face problems during installation or you want to run the tests, see below for a list of dependencies and alternative installation procedures

## Clone this repository and use pip3

### Required dependencies

Make sure to install the following dependencies:

* [preCICE](https://github.com/precice/precice/wiki)
* python3 (this adapter **only supports python3**)
* [the python language bindings for preCICE](https://github.com/precice/python-bindings)
* [FEniCS](https://fenicsproject.org/) (with python interface, installed by default)
* and scipy (`pip3 install scipy`)

### Build and install the adapter

After cloning this repository and switching to the root directory (`fenics-adapter`), run ``pip3 install --user .`` from your shell.

### Test the adapter

As a first test, try to import the adapter via `python3 -c "import fenicsprecice"`.

You can run the other tests via `python3 setup.py test`.

Single tests can be also be run. For example the test `test_vector_write` in the file `test_write_read.py` can be run as follows:
```
python3 -m unittest tests.test_write_read.TestWriteandReadData.test_vector_write
```

# Use the adapter

Add ``from fenicsprecice import Adapter`` in your FEniCS code. Please refer to the examples in the [tutorials repository](https://github.com/precice/tutorials) for usage examples:

The adapter is configured via a `json` configuration file. For example configuration files and usage refer to the tutorials ([fenics-fenics](https://github.com/precice/tutorials/tree/master/HT/partitioned-heat/fenics-fenics)).

# Packaging

To create and install the `fenicsprecice` python package the following instructions were used: https://python-packaging.readthedocs.io/en/latest/index.html.

# Citing

preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite us](https://www.precice.org/publications/):

*H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*

If you are using FEniCS, please also consider the information on https://fenicsproject.org/citing/.

# Disclaimer

This offering is not approved or endorsed by the FEniCS Project, producer and distributor of the FEniCS software via https://fenicsproject.org/.
