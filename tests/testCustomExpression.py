import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root + "/../fenicsadapter")

print(root + "../fenicsadapter")

import numpy as np
import fenicsadapter

x_vert = np.array([1,2,3])        
y_vert = np.array([1,2,3])
vals = np.array([1,2,3])
        
coupling_bc_expression = fenicsadapter.CustomExpression()
coupling_bc_expression.set_boundary_data(vals=vals, coords_x=x_vert, coords_y=y_vert)

from dolfin import DirichletBC, UnitSquareMesh, FunctionSpace
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,"P", 1)

def boundary_D(x, on_boundary):
    if on_boundary:        
        return True        

DirichletBC(V, coupling_bc_expression, boundary_D)