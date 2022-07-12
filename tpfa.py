from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
import numpy as np
from scipy.sparse import csc_matrix as csc
from scipy.sparse.linalg import spsolve

mesh = FineScaleMesh("quad_mesh.msh", dim=2)

M = np.zeros((9,9), dtype=float)

for i in range(3):
    for j in range(3):
        if i == j and j == 9:
            M[0][0] = 1
        elif i == 2 and j == 2:
            M[8][8] = 1
        else:
            for k in mesh.faces.bridge_adjacencies(3*i + j, 1, 2):
                M[3*i+j][k] = -1
                M[3*i+j][3*i+j] = 4

M = csc(M)
b = np.zeros((9,))
b[0] = 1
sol = spsolve(M,b)

for i in range(9):
    mesh.pressure[i] = sol[i]