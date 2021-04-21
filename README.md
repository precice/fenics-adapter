# FEniCS-preCICE adapter

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/precice/fenics-adapter.svg" alt="GNU LGPL license">
</a>

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/actions/workflows/build-and-test.yml" target="_blank">
    <img src="https://github.com/precice/fenics-adapter/actions/workflows/build-and-test.yml/badge.svg" alt="Build and Test">
</a>
<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/actions/workflows/run-tutorials.yml" target="_blank">
    <img src="https://github.com/precice/fenics-adapter/actions/workflows/run-tutorials.yml/badge.svg" alt="Run preCICE Tutorials">
</a>
<a style="text-decoration: none" href="https://pypi.org/project/fenicsprecice/" target="_blank">
    <img src="https://github.com/precice/fenics-adapter/actions/workflows/pythonpublish.yml/badge.svg" alt="Upload Python Package">
</a>

preCICE-adapter for the open source computing platform FEniCS. Note: The adapter **currently only supports 2D simulations in FEniCS.**

## Installing the package

### Using pip3 to install from PyPI

It is recommended to install [fenicsprecice from PyPI](https://pypi.org/project/fenicsprecice/) via

```bash
pip3 install --user fenicsprecice
```

This should work out of the box, if all dependencies are installed correctly. If you face problems during installation or you want to run the tests, see below for a list of dependencies and alternative installation procedures

### Clone this repository and use pip3

#### Required dependencies

Make sure to install the following dependencies:

* [preCICE](https://github.com/precice/precice/wiki)
* python3 (this adapter **only supports python3**)
* [the python language bindings for preCICE](https://github.com/precice/python-bindings)
* [FEniCS](https://fenicsproject.org/) (with python interface, installed by default)
* and scipy (`pip3 install scipy`)

#### Build and install the adapter

After cloning this repository and switching to the root directory (`fenics-adapter`), run ``pip3 install --user .`` from your shell.

#### Test the adapter

As a first test, try to import the adapter via `python3 -c "import fenicsprecice"`.

You can run the other tests via `python3 setup.py test`.

Single tests can be also be run. For example the test `test_vector_write` in the file `test_write_read.py` can be run as follows:

```bash
python3 -m unittest tests.test_write_read.TestWriteandReadData.test_vector_write
```

#### Troubleshooting

**FEniCS is suddenly broken:** There are two known issues with preCICE, fenicsprecice and FEniCS:

* If you see `ImportError: cannot import name 'sub_forms_by_domain'` run `pip3 uninstall -y fenics-ufl`. For details, refer to [issue #103](https://github.com/precice/fenics-adapter/issues/103).
* If you see `ModuleNotFoundError: No module named 'dolfin'` and have installed PETSc from source, refer to [this forum post](https://fenicsproject.discourse.group/t/modulenotfounderror-no-module-named-dolfin-if-petsc-dir-is-set/4407). Short version: Try to use the PETSc that comes with your system, if possible. Note that you can also [compile preCICE without PETSc](https://www.precice.org/installation-source-configuration.html), if necessary.

If this does not help, you can contact us on [gitter](https://gitter.im/precice/lobby) or [open an issue](https://github.com/precice/fenics-adapter/issues/new).

## Use the adapter

Please refer to [our website](https://www.precice.org/adapter-fenics.html#how-can-i-use-my-own-solver-with-the-adapter-).

## Packaging

To create and install the `fenicsprecice` python package the following instructions were used: [How To Package Your Python Code from python-packaging.readthedocs.io](https://python-packaging.readthedocs.io/en/latest/index.html).

## Citing

If you are using this adapter, please refer to the [citing information on the FEniCS adapter](https://www.precice.org/adapter-fenics.html#how-to-cite).

preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite us](https://www.precice.org/publications/):

*H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*

If you are using FEniCS, please also consider the information on [the official FEniCS website on citing](https://fenicsproject.org/citing/).

## Development history

The initial version of this adapter was developed by [Benjamin Rodenberg](https://www.in.tum.de/i05/personen/personen/benjamin-rodenberg/) during his research stay at Lund University in the group for [Numerical Analysis](http://www.maths.lu.se/english/research/research-divisions/numerical-analysis/) in close collaboration with [Peter Meisrimel](https://www.lunduniversity.lu.se/lucat/user/09d80f0367a060bcf2a22d7c22e5e504).

[Richard Hertrich](https://github.com/richahert) contributed the possibility to perform FSI simulations using the adapter in his [Bachelor thesis](https://mediatum.ub.tum.de/node?id=1520579).

[Ishaan Desai](https://www.ipvs.uni-stuttgart.de/institute/team/Desai/) improved the user interface and extended the adapter to also allow for parallel FEniCS computations.
