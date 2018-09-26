# fenics-adapter
**experimental** preCICE-adapter for the open source computing platform FEniCS

**currently only supports 2D simulations in FEniCS**

## Installation

Run ``python setup.py install`` from your shell.

## Use the adapter

Add ``from fenicsadapter import Coupling`` in your FEniCS code. Please refer to the examples in the `tutorials` folder for usage examples.

## Packaging

To create and install the `fenicsadapter` python package the following instructions were used: https://python-packaging.readthedocs.io/en/latest/index.html.

## Citing

preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite us](https://www.precice.org/publications/):

*H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*

If you are using FEniCS, please also consider the information on https://fenicsproject.org/citing/.

## Disclaimer

This offering is not approved or endorsed by the FEniCS Project, producer and distributor of the FEniCS software via https://fenicsproject.org/.
