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
        
coupling_bc_expression = fenicsadapter.CustomExpression(vals=vals, coords_x=x_vert, coords_y=y_vert)