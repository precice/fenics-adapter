**This branch of the fenics-adapter repository represents the version of the code used for producing the results presented at the GAMM CSE 2018. See the corresponding presentation [here](https://mediatum.ub.tum.de/1467488). [preCICE v 1.3.0](https://github.com/precice/precice/releases/tag/v1.3.0) was used for the experiments.**

# fenics-adapter
**experimental** preCICE-adapter for the open source computing platform FEniCS

_**Note:** This adapter is currently purely expermental and limited in functionality. If you are interested in using it or you want to contribute, feel free to contact us via the [preCICE mailing list](https://mailman.informatik.uni-stuttgart.de/mailman/listinfo/precice)._

**currently only supports 2D simulations in FEniCS**

This adapter was developed by [Benjamin Rüth](https://www5.in.tum.de/wiki/index.php/Benjamin_R%C3%BCth,_M.Sc._(hons)) during his research stay at Lund University in the group for [Numerical Analysis](http://www.maths.lu.se/english/research/research-divisions/numerical-analysis/) in close collaboration with [Peter Meisrimel](https://www.lunduniversity.lu.se/lucat/user/09d80f0367a060bcf2a22d7c22e5e504).

## Installation

### Dependencies

Make sure to install 

* preCICE (https://github.com/precice/precice/wiki)
* python3 (this adapter supports **only python3**)
* the python language bindings for preCICE (https://github.com/precice/precice/blob/develop/src/precice/bindings/python/README.md)
* fenics (https://fenicsproject.org/)
* and scipy (`pip3 install scipy`)

### Build and install the adapter

Run ``python3 setup.py install`` from your shell.

### Test the adapter

As a first test, try to import the adapter via `python3 -c "import fenicsadapter"`.

You can also run the tests in `fenics-adapter/tests` from this folder. Example: `$python3 tests/testCustomExpression.py`.

## Use the adapter

Add ``from fenicsadapter import Coupling`` in your FEniCS code. Please refer to the examples in the [tutorials repository](https://github.com/precice/tutorials) for usage examples.

## Packaging

To create and install the `fenicsadapter` python package the following instructions were used: https://python-packaging.readthedocs.io/en/latest/index.html.

## Citing

preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite us](https://www.precice.org/publications/):

*H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*

If you are using FEniCS, please also consider the information on https://fenicsproject.org/citing/.

## Disclaimer

This offering is not approved or endorsed by the FEniCS Project, producer and distributor of the FEniCS software via https://fenicsproject.org/.
