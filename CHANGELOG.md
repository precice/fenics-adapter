# FEniCS-preCICE adapter changelog

## latest

* Update adapter to use preCICE v3 API [#153](https://github.com/precice/fenics-adapter/pull/153).
* Remove functionality to define mesh connectivity in 2D cases in the form of triangles due to lack of testing and compatibility problems (might be added again). See [#162](https://github.com/precice/fenics-adapter/issues/162).

## 1.4.0

* Adding CITATION.cff to link the adapter repository to the relevant publication in the journal SoftwareX.
* Add functionality to define mesh connectivity in 2D cases in the form of triangles.

## 1.3.0

* Adding functionality for 3D cases with PointSource objects at coupling boundaries. See PRs [#133](https://github.com/precice/fenics-adapter/pull/133), [#146](https://github.com/precice/fenics-adapter/pull/146) and [#147](https://github.com/precice/fenics-adapter/pull/147).
* Fixing an issue of the `config` object not being able to find the adapter configuration file in a Jupyter notebook. [See PR #144](https://github.com/precice/fenics-adapter/pull/144)

## 1.2.0

* Reduce complexity of initialization to reduce runtime for large cases. [See PR #135](https://github.com/precice/fenics-adapter/pull/135)
* Raise warning, if FEniCS is not found on system. [See PR #130](https://github.com/precice/fenics-adapter/pull/130)
* Add test for python3 setup.py sdist. [See PR #127](https://github.com/precice/fenics-adapter/pull/127)

## 1.1.0

* Only warn during initialization, if duplicate boundary point is found for point sources.
* Remove deprecated package `fenicsadapter`. Don't use `import fenicsadapter`. Please use `import fenicsprecice`. [See PR #121](https://github.com/precice/fenics-adapter/pull/121)

## 1.0.1

* Bugfix for PointSources in [PR #109](https://github.com/precice/fenics-adapter/issues/109)
* Bugfix in parallelization in [PR #110](https://github.com/precice/fenics-adapter/pull/110)

## 1.0.0

* The paper *FEniCS-preCICE: Coupling FEniCS to other Simulation Software* (in preparation) describes features, usage and API of the adapter.
* The software is called FEniCS-preCICE adapter, the corresponding python package `fenicsprecice`. Our software uses the naming proposed in [PR #85](https://github.com/precice/fenics-adapter/issues/85).
* `fenicsprecice` is published via PyPI. [See PR #94](https://github.com/precice/fenics-adapter/pull/94).
* FEniCS PointSource and Expressions are used to create boundary conditions for coupling.
* The adapter uses a `SegregatedRBFinterpolationExpression` for interpolation, if an Expression is used. [See PR #83](https://github.com/precice/fenics-adapter/pull/83).
* The adapter supports one-way coupling and two-way coupling.
* The adapter supports explicit and implicit coupling schemes.
* The adapter supports checkpointing and subcycling.
* The adapter supports up to one read and one write data set.
* The current state of the adapter API was mainly designed in [PR #59](https://github.com/precice/fenics-adapter/pull/59).
* Supports parallel solvers for Expressions, but not for PointSources as coupling boundary conditions. [See PR #71](https://github.com/precice/fenics-adapter/pull/71).
