from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
import numpy as np
from scipy.sparse import csc_matrix as csc
from scipy.sparse.linalg import spsolve

norm = lambda x: np.dot(x,x)**0.5

def length(mesh: FineScaleMesh, i: int) -> float:
    a,b = mesh.edges.adjacencies[i]
    return norm(a-b)

def normal(mesh: FineScaleMesh, x: int, i: int) -> np.ndarray:
    return (mesh.faces.center[x] - mesh.faces.center[i])/norm(mesh.faces.center[x] - mesh.faces.center[i])

# np.ndarray is numpy array
def permProjec(mesh: FineScaleMesh, x: int, i: int) -> float: # K (permeability tensor) and  N (normal vector)
    K = mesh.permeability[x]
    N = normal(mesh, x, i) 
    return (N.T @ K @ N).item

def area(mesh: FineScaleMesh, x: int) -> float:
    edges = mesh.faces.adjacencies[x]
    k = 0
    a,b = mesh.edges.adjacencies[edges[k]] 
    c, d = a,b 
    while (c == a and d == b) or (c == b and d == a):
        k += 1
        c, d = mesh.edges.adjacencies[edges[k]]
    return abs(np.cross(a-b, c-d).item())    

def delta(mesh: FineScaleMesh, x: int, i: int) -> float:
    xadj = mesh.faces.bridge_adjacencies(x, 1, 2)
    iadj = mesh.faces.bridge_adjacencies(i, 1, 2)
    x = [e for e in xadj if e in iadj][0]
    index = -1
    for j in mesh.faces.bridge_adjacencies(i,1,2):
        if mesh.edges[j] == x:
            index = j 
            break
    return length(mesh, index)

def tpfa2d(mesh: FineScaleMesh, boundConds: list((int,float))) -> np.ndarray: # output is a numpy array
    numCells = len(mesh.faces())
    M = np.zeros((len,len))
    b = np.zeros((len,1))

    for (index, pressure) in boundConds:
        M[index][index] = 1
        b[index] = pressure
        
    boundCondsIndexs = [x for (x,_) in boundConds]
    
    for i in range(len):
        if i in boundCondsIndexs:
            continue
        coeff = lambda x : permProjec(mesh, x, i) * area(mesh, x)/delta(mesh, x, i)
        coeffs = np.vectorize(coeff)
        coeffs(mesh.faces.bridge_adjacencies(i,1,2))
        M[i][i] = sum(coeffs(mesh.faces.bridge_adjacencies(i,1,2)))
        for j in mesh.faces.bridge_adjacencies(i, 1, 2):
            M[i][j] = - coeff(j)


mesh = FineScaleMesh("quad_mesh.msh", dim=2)

    
M = csc(M)
b = np.zeros((9,))
b[0] = 1
sol = spsolve(M,b)

for i in range(9):
    mesh.pressure[i] = sol[i]


# M = np.zeros((9,9), dtype=float)

# for i in range(3):
#     for j in range(3):
#         if i == j and j == 9:
#             M[0][0] = 1
#         elif i == 2 and j == 2:
#             M[8][8] = 1
#         else:
#             M[3*i+j][3*i+j] = 0
#             for k in mesh.faces.bridge_adjacencies(3*i + j, 1, 2):
#                 M[3*i+j][k] = -1
#                 M[3*i+j][3*i+j] += 1
# M = csc(M)
# b = np.zeros((9,))
# b[0] = 1
# sol = spsolve(M,b)

# for i in range(9):
#     mesh.pressure[i] = sol[i]