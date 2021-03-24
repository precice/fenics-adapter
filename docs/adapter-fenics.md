---
title: The FEniCS adapter
permalink: adapter-fenics.html
keywords: adapter, fenics
summary: "A general adapter for the open source computing platform FEniCS"
---

## What is FEniCS ?
From the FEniCS website: _FEniCS is a popular open-source (LGPLv3) computing platform for solving partial differential equations (PDEs). FEniCS enables users to quickly translate scientific models into efficient finite element code. With the high-level Python and C++ interfaces to FEniCS, it is easy to get started, but FEniCS offers also powerful capabilities for more experienced programmers. FEniCS runs on a multitude of platforms ranging from laptops to high-performance clusters._ More details can be found at [fenicsproject.org](https://fenicsproject.org/).


## Aim of this adapter
This adapter supports the Python interface of FEniCS and offers an API that allows the user to use FEniCS-style data structures for solving coupled problems. We provide usage example for the adapter for heat transport, conjugate heat transfer and fluid-structure interaction. However, the adapter is designed in a general fashion and can be used to couple any code using the FEniCS library.


## How to install the adapter?
The adapter requires FEniCS and preCICE version 2.0 or greater and the preCICE language bindings for Python. The adapter is [published on PyPI](https://pypi.org/project/fenicsprecice/). After installing preCICE and the python language bindings one can simply run `pip3 install --user fenicsprecice` to install the adapter via your Python package manager.

Please refer to the installation instructions provided [here](https://github.com/precice/fenics-adapter#installing-the-package) for alternative installation procedures.

## Examples for coupled codes.

The following tutorials can be used as a usage example for the FEniCS adapter:

* Solving the heat equation in a partitioned fashion (heat equation solved via FEniCS for both participants), see [1]
* Flow over plate (heat equation solved via FEniCS for solid participant), see [1]
* Perpendicular flap (structure problem solved via FEniCS), see [1, 2]
* Cylinder with flap (structure problem solved via FEniCS), see [2]

## How can I use my own solver with the adapter ?
The FEniCS adapter does not couple your code out-of-the-box, but you have to call the adapter API from within your code. You can use the tutorials from above as an example. The API of the adapter and the design is explained and usage examples are given in [1].

## You need more information?
Please don't hesitate to ask questions about the FEniCS adapter on [discourse](https://precice.discourse.group/) or in [gitter](https://gitter.im/precice/Lobby).

## How to cite

If you are using our adapter, please consider citing our paper "FEniCS-preCICE: Coupling FEniCS to other Simulation Software" [1].

## Related literature

[1] Benjamin Rodenberg, Ishaan Desai, Richard Hertrich, Alexander Jaust, Benjamin Uekermann. FEniCS-preCICE: Coupling FEniCS to other Simulation Software. [preprint on arXiv, arxiv.org/abs/2103.11191](https://arxiv.org/abs/2103.11191), 2021  
[2] Richard Hertrich. Partitioned fluid structure interaction: Coupling FEniCS and OpenFOAM via preCICE. Bachelor's thesis, Munich School of Engineering, Technical University of Munich, 2019.

