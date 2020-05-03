# Tutorial for a partitioned heat equation using FEniCS

This tutorial is described in the [preCICE wiki](https://github.com/precice/precice/wiki/Tutorial-for-solving-the-heat-equation-in-a-partitioned-fashion-using-FEniCS).

## Running in parallel 

```
mpirun -np 3 python3 heat.py -d
```

and 

```
mpirun -np 1 python3 heat.py -n
```
