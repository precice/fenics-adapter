# FEniCS-preCICE adapter

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/precice/fenics-adapter.svg" alt="GNU LGPL license">
</a>

<a style="text-decoration: none" href="https://doi.org/10.1016/j.softx.2021.100807" target="_blank">
    <img src="https://zenodo.org/badge/DOI/10.1016/j.softx.2021.100807.svg" alt="DOI">
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

preCICE-adapter for the open source computing platform FEniCS.

## Installing the package

Generally, we recommend to create a virtual environment and install the package there. The following commands will create the virtual environment `.venv`:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
```

You can replace `.venv` with `path/of/your/venv/venv_name` if you want to create the virtual environment somewhere else. The command `--system-site-packages` will allow the virtual environment to access the FEniCS installation on your system. Refer to the [Python documentation](https://docs.python.org/3/library/venv.html) for further details on virtual environments.

If you do not want to use a virtual environment, you might encounter the `error: externally-managed-environment` during installation of the adapter; this depends on your pip version and operating system. You can read why this error shows up [here](https://packaging.python.org/en/latest/specifications/externally-managed-environments/).

### Using pip to install from PyPI

It is recommended to install [fenicsprecice from PyPI](https://pypi.org/project/fenicsprecice/) via

```bash
pip install fenicsprecice
```

This should work out of the box, if all dependencies are installed correctly. If you face problems during installation or you want to run the tests, see below for a list of dependencies and alternative installation procedures

### Clone this repository and use pip

#### Required dependencies

Make sure to install the following dependencies:

* [preCICE](https://github.com/precice/precice/wiki)
* python3 (this adapter **only supports python3**)
* [the python language bindings for preCICE](https://github.com/precice/python-bindings)
* [FEniCS](https://fenicsproject.org/) (with Python interface, installed by default)
* and scipy (`pip install scipy`)

#### Build and install the adapter

After cloning this repository and switching to the root directory (`fenics-adapter`), run ``pip install .`` from your shell.

#### Test the adapter

As a first test, try to import the adapter via `python3 -c "import fenicsprecice"`.

You can run the other tests via `tox`.
If you want to run them separately, use `tox -e unit` and `tox -e integration` respectively.

Single tests can be also be run. For example the test `test_vector_write` in the file `test_write_read.py` can be run as follows:

```bash
tox -e all -- tests/integration/test_write_read.py::TestWriteandReadData::test_vector_write
```

#### Troubleshooting

**FEniCS is suddenly broken:** There are two known issues with preCICE, fenicsprecice and FEniCS:

* If you see `ImportError: cannot import name 'sub_forms_by_domain'` run `pip3 uninstall -y fenics-ufl`. For details, refer to [issue #103](https://github.com/precice/fenics-adapter/issues/103).
* If you see `ImportError: cannot import name 'cellname2facetname' from 'ufl.cell'`, refer to [issue #154](https://github.com/precice/fenics-adapter/issues/154).
* If you see `ModuleNotFoundError: No module named 'dolfin'` and have installed PETSc from source, refer to [this forum post](https://fenicsproject.discourse.group/t/modulenotfounderror-no-module-named-dolfin-if-petsc-dir-is-set/4407). Short version: Try to use the PETSc that comes with your system, if possible. Note that you can also [compile preCICE without PETSc](https://www.precice.org/installation-source-configuration.html), if necessary.

If this does not help, you can contact us on [gitter](https://gitter.im/precice/lobby) or [open an issue](https://github.com/precice/fenics-adapter/issues/new).

## Use the adapter

Please refer to [our website](https://www.precice.org/adapter-fenics.html#how-can-i-use-my-own-solver-with-the-adapter-).

## Packaging

To create and install the `fenicsprecice` Python package the following instructions were used: ["How To Package Your Python Code" from python-packaging.readthedocs.io](https://python-packaging.readthedocs.io/en/latest/index.html).

## Citing

* FEniCS-preCICE: If you are using this adapter (`fenics-adapter`), please refer to the [citing information on the FEniCS adapter](https://www.precice.org/adapter-fenics.html#how-to-cite).
* preCICE: preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite preCICE](https://precice.org/publications.html#how-to-cite-precice).
* FEniCS: If you are using FEniCS, please also consider the information on [the official FEniCS website on citing](https://fenicsproject.org/citing/).

## Development history

The initial version of this adapter was developed by [Benjamin Rodenberg](https://www.cs.cit.tum.de/sccs/personen/benjamin-rodenberg/) during his research stay at Lund University in the group for [Numerical Analysis (Philipp Birken)](https://www.maths.lu.se/forskning/forskningsavdelningar/numerisk-analys/forskning/) in close collaboration with Peter Meisrimel.

[Richard Hertrich](https://github.com/richahert) contributed the possibility to perform FSI simulations using the adapter in his [Bachelor thesis](https://mediatum.ub.tum.de/node?id=1520579).

[Ishaan Desai](https://www.ipvs.uni-stuttgart.de/institute/team/Desai/) improved the user interface and extended the adapter to allow for parallel FEniCS computations and 3D cases in certain scenarios.
