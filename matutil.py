# matutil.py
# Copyright: Yong Y. Auh
# 
"""
Matrices and vectors are represented as numpy.ndarray.

import numpy as np
from matutil import *
x = [1,2,3]         # M(x) is a row vector
v = t(M(x))         # column vector
w = M('1; 2; 3')    # column vector
A = M('1 2 3')
B = M('1 2; 3 4')
B.shape #=> (2, 2)

type(M(x)) => numpy.ndarray

A = M('1 2 3; 4 5 6; 7 8 12')
b = M('1;0;3')
linalg.solve(A,b)
sympy.Matrix(hcat(A,b)).rref()

m = hcat(A,b)
arr = np.squeeze(np.asarray(Matrix(m).rref()[0]))
float_arr = np.vstack(arr[:, :]).astype(float)
def rref(m):
    arr = np.squeeze(np.asarray(Matrix(m).rref()[0]))
    float_arr = np.vstack(arr[:, :]).astype(float)
    return float_arr
"""

import numpy as np
from scipy import linalg
import sympy
#from sympy import Matrix

def M(s):
    """ returns an array object
    """
    return np.matrix(s).A

def hcat(m1, m2):
    """ horizontal concat
    """
    return np.append(m1, m2, axis=1)
    
def vcat(m1, m2):
    """ vertical concat
    """
    return np.append(m1, m2, axis=0)

""" close enough
"""
equal = np.allclose
#def equal(m1, m2):

#    return np.allclose(m1, m2)

""" matrix rank
    Matrix rank should be checked before calling inv
""" 
rank = np.linalg.matrix_rank
#def rank(m):
#    return np.linalg.matrix_rank(m)

""" matrix transpose
"""
t = np.transpose
#def t(m):

#    return np.transpose(m)

""" matrix inverse
    Matrix rank should be checked before calling inv
""" 
inv = linalg.inv
#def inv(m):
#    return linalg.inv(m)

""" solve Ax = v
"""
solve = linalg.solve
#def solve(m, v):
#    return linalg.solve(m, v)

def rref(m):
    """ reduced row echelon form
    """
    arr = np.asarray(sympy.Matrix(m).rref()[0])
    float_arr = np.vstack(arr[:, :]).astype(float)
    return float_arr

""" vector norm
"""
norm = linalg.norm
#def norm(v):
#    return linalg.norm(v)

def row(m, n):
    """ n'th row of the matrix m
    """
    return m[n]
    
def col(m, n):
    """ n'th column of the matrix m
    """
    return m[:,n]
    
def dot(m1, m2):
    """ dot product
    """
    try:
        return np.dot(m1, m2)
    except Exception as inst:
        if isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
            if m1.shape[1] == 1: # column vector
                a1 = m1[:,0]
            elif m1.shape[0] == 1: # column vector
                a1 = m1[0]
            else:
                return np.dot(m1, m2)
            if m2.shape[1] == 1: # column vector
                a2 = m2[:,0]
            elif m2.shape[0] == 1: # column vector
                a2 = m2[0]
            else:
                return np.dot(m1, m2)            
            return np.dot(a1, a2)
        else:
            return np.dot(m1, m2)
        
    #ans = np.dot(m1, m2)
    #if isinstance(ans, np.ndarray) and len(ans) == 1:
    #    return ans[0,0]
    #return ans
        
""" inner product
"""
inner = np.inner
#def inner(m1, m2):
#    return np.inner(m1, m2)

""" cross product of 3 dim matrices
"""
cross = np.cross
#def cross(m1, m2):
#    return np.cross(m1, m2)

""" matrix trace
"""
tr = np.trace
#def tr(m):
#    return np.trace(m)

def zeros(r, c=1):
    """ r x c zero matrix
    """
    return np.zeros((r, c))

""" n x n identity matrix
"""
identity = np.eye
#def identity(n): # in julia, it's just I 
#    return np.eye(n)

""" n x n identity matrix
"""
eye = np.eye
#def eye(n):
#    return np.eye(n)

""" If v is a 2-D array, return a copy of its k-th diagonal. 
    If v is a 1-D array, return a 2-D array with v on the k-th diagonal.
    The default of k is 0. 
    Use k>0 for diagonals above the main diagonal, 
        and k<0 for diagonals below the main diagonal.
"""
diag = np.diag
#def diag(v, k=0): 
#    return np.diag(v)

""" matrix determinant
"""
det = np.linalg.det
#def det(m):
#    return np.linalg.det(m)

""" matrix eigenvals and eigen vectors
"""
eig = np.linalg.eig
#def eig(m):
#    return np.linalg.eig(m)

""" matrix eigenvals
"""
eigvals = np.linalg.eigvals
#def eigvals(m):
#    return np.linalg.eigvals(m)

def eigvecs(m):
    """ matrix eigen vectors
    """
    return np.linalg.eig(m)[1]

""" matrix PLU decomp (P, L, U)
"""
plu = linalg.lu
#def plu(m):
#    return linalg.lu(m)

#def lu(m):
#    """ matrix LU decomp (L, U)
#        L, U = lu(A) or lst = lu(A)
#    """
#    return linalg.lu(m)[1:3]

""" matrix QR decomp 
"""
qr = linalg.qr
#def qr(m):
#    return linalg.qr(m)
"""

shur


"""

""" Cholesky decomposition 
"""
cholesky = linalg.cholesky
#def cholesky(m):
#    return linalg.cholesky(m)

""" Singular Value Decomposition 
"""
svd = linalg.svd
#def svd(m):
#    return linalg.svd(m)

