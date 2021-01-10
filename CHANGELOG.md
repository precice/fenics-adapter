# FEniCS-preCICE adapter changelog

## 1.0.1

* Bugfix for PointSources https://github.com/precice/fenics-adapter/issues/109
* Bugfix in parallelization https://github.com/precice/fenics-adapter/pull/110

## 1.0.0

* The paper *FEniCS-preCICE: Coupling FEniCS to other Simulation Software* (in preparation) describes features, usage and API of the adapter.
* The software is called FEniCS-preCICE adapter, the corresponding python package `fenicsprecice`. Our software uses the naming proposed in https://github.com/precice/fenics-adapter/issues/85.
* `fenicsprecice` is published via PyPI https://github.com/precice/fenics-adapter/pull/94.
* FEniCS PointSource and Expressions are used to create boundary conditions for coupling.
* The adapter uses a `SegregatedRBFinterpolationExpression` for interpolation, if an Expression is used https://github.com/precice/fenics-adapter/pull/83.
* The adapter supports one-way coupling and two-way coupling.
* The adapter supports explicit and implicit coupling schemes.
* The adapter supports checkpointing and subcycling.
* The adapter supports up to one read and one write data set.
* The current state of the adapter API was mainly designed in https://github.com/precice/fenics-adapter/pull/59.
* Supports parallel solvers for Expressions, but not for PointSources as coupling boundary conditions. See https://github.com/precice/fenics-adapter/pull/71.
