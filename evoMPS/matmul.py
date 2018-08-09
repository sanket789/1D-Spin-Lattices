# -*- coding: utf-8 -*-
# cython: profile=False
"""
Created on Fri Nov  4 13:05:59 2011

@author: Ashley Milsted

TODO:
    Look into using e.g. sp.linalg.fblas.zgemm._cpointer from cython? Or
    link it to blas at compile time using distutils...
"""
#from __future__ import absolute_import, division, print_function

import scipy as sp
import scipy.linalg as la
#import scipy.sparse as spa

class eyemat(object):
    __array_priority__ = 10.1 #makes right-ops work, ala sparse
    
    def __init__(self, D, dtype=sp.float64):
        self.shape = (D, D)
        self.dtype = dtype
        self.data = None
        
    def __array__(self):
        return self.toarray()
    
    def toarray(self):
        return sp.eye(self.shape[0], dtype=self.dtype)
    
    def __mul__(self, other):
        if sp.isscalar(other):
            return simple_diag_matrix(sp.ones(self.shape[0], self.dtype) * other)
        
        try:
            if other.shape == self.shape:
                return simple_diag_matrix(other.diagonal())
        except:
            return NotImplemented
        
        return self.toarray() * other
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __add__(self, other):
        try:
            return self.toarray() + other
        except:
            return NotImplemented
            
    def __radd__(self, other):
        try:
            return other + self.toarray()
        except:
            return NotImplemented
            
    def __sub__(self, other):
        try:
            return self.toarray() - other
        except:
            return NotImplemented
            
    def __rsub__(self, other):
        try:
            return other - self.toarray()
        except:
            return NotImplemented
        
    def __coerce__(self, other):
        try:
            other = sp.asanyarray(other)
            if other.shape == self.shape or sp.isscalar(other):
                return (self.toarray(), other)
            else:
                return NotImplemented
        except:
            return NotImplemented
        
    def dot(self, other):
        if self.shape[1] == other.shape[0]:
            return other
        else:
            raise BaseException
            
    def dot_left(self, other):
        if self.shape[0] == other.shape[1]:
            return other
        else:
            raise BaseException
            
    def conj(self):
        return self
        
    def transpose(self):
        return self
        
    def trace(self, offset=0):
        if offset == 0:
            return self.shape[0]
        else:
            return 0
            
    def diagonal(self):
        return sp.ones((self.shape[0]), dtype=self.dtype)
            
    def sqrt(self):
        return self
        
    def inv(self):
        return self
        
    def ravel(self):
        return self.toarray().ravel()
        
    def copy(self, order='C'):
        return eyemat(self.shape[0], dtype=self.dtype)
        
    def __getattr__(self, attr):
        if attr == 'A':
            return self.toarray()
        elif attr == 'T':
            return self.transpose()
        else:
            raise AttributeError(attr + " not found")
    

class simple_diag_matrix:
    __array_priority__ = 10.1 #makes right-ops work, ala sparse
    
    diag = None
    shape = None
    dtype = None
    
    def __init__(self, diag, dtype=None):
        diag = sp.asanyarray(diag, dtype=dtype)
        self.dtype = diag.dtype
        assert diag.ndim == 1
        self.diag = diag
        self.shape = (diag.shape[0], diag.shape[0])
        
    def __array__(self):
        return self.toarray()
        
    def dot(self, b):
        if isinstance(b, simple_diag_matrix):
            return simple_diag_matrix(self.diag * b.diag)
        elif isinstance(b, eyemat):
            return self
            
        return mmul_diag(self.diag, b)
        
    def dot_left(self, a):
        if isinstance(a, simple_diag_matrix):
            return simple_diag_matrix(self.diag * a.diag)
        elif isinstance(a, eyemat):
            return self
            
        return mmul_diag(self.diag, a, act_right=False)
        
    def conj(self):
        return simple_diag_matrix(self.diag.conj())
        
    def transpose(self):
        return self
        
    def inv(self):
        return simple_diag_matrix(1. / self.diag)
        
    def sqrt(self):
        return simple_diag_matrix(sp.sqrt(self.diag))
        
    def ravel(self):
        return sp.diag(self.diag).ravel()
        
    def diagonal(self):
        return self.diag
        
    def trace(self, offset=0):
        if offset == 0:
            return self.diag.sum()
        else:
            return 0
        
    def toarray(self):
        return sp.diag(self.diag)
        
    def copy(self, order='C'):
        return simple_diag_matrix(self.diag.copy())
        
    def __mul__(self, other):
        if sp.isscalar(other):
            return simple_diag_matrix(self.diag * other)
        
        try:
            other = sp.asanyarray(other)
    
            if other.shape == self.shape:
                return simple_diag_matrix(self.diag * other.diagonal())
            
            return self.toarray() * other
        except:
            return NotImplemented
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __add__(self, other):
        try:
            return self.toarray() + other
        except:
            return NotImplemented
            
    def __radd__(self, other):
        try:
            return other + self.toarray()
        except:
            return NotImplemented
            
    def __sub__(self, other):
        try:
            return self.toarray() - other
        except:
            return NotImplemented
            
    def __rsub__(self, other):
        try:
            return other - self.toarray()
        except:
            return NotImplemented
    
    def __coerce__(self, other):
        try:
            other = sp.asanyarray(other)
            if other.shape == self.shape:
                return (self.toarray(), other)
            else:
                return None
        except:
            return None
            
    def __getattr__(self, attr):
        if attr == 'A':
            return self.toarray()
        elif attr == 'T':
            return self.transpose()
        else:
            raise AttributeError(attr + " not found")

def mmul(*args):
    """Multiplies a chain of matrices (2-d ndarrays)
        
    All matrices must have dimensions compatible with matrix multiplication.
    This function actually calls the dot() method of the objects passed in
    as arguments. It thus handles any object that provides a dot() method
    that accepts 2D ndarrays.
    
    We also try to call dot_left(), in case an optimized left-acting
    dot operation is available.
    
    This function is intended to work nicely with the above defined "sparse"
    matrix objects.
    
    Parameters
    ----------
    *args : ndarray
        The chain of matrices to multiply together.

    Returns
    -------
    out : ndarray
        The result.
    """
    #if not out is None and (args.count == 2 and out in args or args[-1] is out):
    #    raise

    res = args[0]
    
    for x in args[1:]:
        try:
            res = x.dot_left(res)
        except:
            res = res.dot(x)
    
    #Since, for some reason, the method version of dot() does not generally
    #take an "out" argument, I ignored this (for now, minor) optimization.
    return res

#        if out is None:
#            return res.dot(args[-1])
#        elif out.size == 1: #dot() seems to dislike this
#            out[...] = res.dot(args[-1])
#            return out
#        else:
#            return sp.dot(res, args[-1], out=out)

#Inplace dot assuming dense output
def dot_inplace(A, B, out):
    if isinstance(A, eyemat):
        out[:] = B
        return out
    elif isinstance(B, eyemat):
        out[:] = A
        return out
    elif isinstance(A, simple_diag_matrix):
        return mmul_diag(A.diag, B, out=out)
    elif isinstance(B, simple_diag_matrix):
        return mmul_diag(B.diag, A, out=out, act_right=False)
    else:
        return sp.dot(A, B, out=out)

def adot(a, b):
    """
    Calculates the scalar product for the ancilla, expecting
    the arguments in matrix form.
    Equivalent to trace(dot(H(a), b))
    """    
    return sp.inner(a.ravel().conj(), b.ravel())
    
def adot_noconj(a, b):
    """
    Calculates the scalar product for the ancilla, expecting
    the arguments in matrix form.
    Equivalent to trace(dot(a, b))
    """    
    return sp.inner(a.T.ravel(), b.ravel())

def H(m, out=None):
    """Matrix conjugate transpose (adjoint).
    
    This is just a shortcut for performing this operation on normal ndarrays.
    
    Parameters
    ----------
    m : ndarray
        The input matrix.
    out : ndarray
        A matrix to hold the final result (dimensions must be correct). May be None.
        May also be the same object as m.

    Returns
    -------
    out : ndarray
        The result.    
    """
    if out is None:
        return m.T.conj()
    else:
        out = sp.conjugate(m.T, out)
        return out
    
def randomize_cmplx(x, a=-0.5, b=0.5, aj=-0.5, bj=0.5):
    """Randomizes a complex matrix in place.
    """
    x[:] = (((b - a) * sp.random.ranf(x.shape) + a) 
            + 1.j * ((bj - aj) * sp.random.ranf(x.shape) + aj))
    return x

def sqrtmh(A, ret_evd=False, evd=None):
    """Return the matrix square root of a hermitian or symmetric matrix

    Uses scipy.linalg.eigh() to diagonalize the input efficiently.

    Parameters
    ----------
    A : ndarray
        A hermitian or symmetric two-dimensional square array (a matrix).
    evd : (ev, EV)
        A tuple containing the 1D array of eigenvalues ev and the matrix of eigenvectors EV.
    ret_evd : Boolean
        Return the eigenvalue decomposition of the result.

    Returns
    -------
    sqrt_A : ndarray
        An array of the same shape and type as A containing the matrix square root of A.
    (ev, EV) : (ndarray, ndarray)
        A 1D array of eigenvalues and the matrix of eigenvectors.
        
    Notes
    -----
    The result is also Hermitian.

    """
    if not evd is None:
        (ev, EV) = evd
    else:
        ev, EV = la.eigh(A) #uses LAPACK ***EVR
    
    ev = sp.sqrt(ev) #we don't require positive (semi) definiteness, so we need the scipy sqrt here
    
    #Carry out multiplication with the diagonal matrix of eigenvalue square roots with H(EV)
    B = mmul_diag(ev, H(EV))
        
    if ret_evd:
        return mmul(EV, B), (ev, EV)
    else:
        return mmul(EV, B)
        
def mmul_diag(Adiag, B, act_right=True, out=None):
    if act_right:
        assert B.shape[0] == Adiag.shape[0]
    else:
        assert B.shape[1] == Adiag.shape[0]
        
    assert Adiag.ndim == 1
    assert B.ndim == 2
    
    if act_right:
        if out is None:
            out = sp.empty((Adiag.shape[0], B.shape[1]), dtype=sp.promote_types(Adiag.dtype, B.dtype))
        out = out.T
        sp.multiply(Adiag, B.T, out)
        out = out.T
    else:
        if out is None:
            out = sp.empty((B.shape[0], Adiag.shape[0]), dtype=sp.promote_types(Adiag.dtype, B.dtype))
        sp.multiply(Adiag, B, out)
        
    return out
        
def invmh(A, ret_evd=False, evd=None):
    if not evd is None:
        (ev, EV) = evd
    else:
        ev, EV = la.eigh(A)
    
    ev = 1. / ev
        
    B = mmul_diag(ev, H(EV))
    
    if ret_evd:
        return mmul(EV, B), (ev, EV)
    else:
        return mmul(EV, B)   
    
    
def invtr(A, overwrite=False, lower=False):
    """Compute the inverse of a triangular matrix

    Uses the corresponding LAPACK routine.

    Parameters
    ----------
    A : ndarray
        An upper or lower triangular matrix.
        
    overwrite : bool
        Whether to overwrite the input array (may increase performance).
                
    lower : bool
        Whether the input array is lower-triangular, rather than upper-triangular.

    Returns
    -------
    inv_A : ndarray
        The inverse of A, which is also triangular.    

    """    
    trtri, = la.lapack.get_lapack_funcs(('trtri',), (A,))
    
    inv_A, info = trtri(A, lower=lower, overwrite_c=overwrite)
    
    if info > 0:
        raise sp.LinAlgError("%d-th diagonal element of the matrix is zero" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potri'
                                                                    % -info)       
                                                                    
    return inv_A

'''
                    MY FUNCTIONS
==============================================================================================================================
******************************************************************************************************************************

    Function to multiply three matrices
'''
def mult_3(A,B,C):
    return sp.dot(A,sp.dot(B,C))
'''
    Function to evaluate contractions. Ref: Eq 93 from U. Schollowk
    Input:
        A : bra vector MPS
        B : ket vector MPS
    Output:
        < psi[A] | phi[B] >
'''
def contract(A,B):
    M0 = sp.dot(H(A[1][0]),B[1][0]) + sp.dot(H(A[1][1]),B[1][1])
    N = sp.shape(A)[0]
    for n in range(2,N):
        Ml = A[n]
        Mr = B[n]
        out = mult_3(H(Ml[0]),M0,Mr[0]) + mult_3(H(Ml[1]),M0,Mr[1])
        M0 = out
    return sp.asscalar(M0)

def calc_r_seq(x,A):
    q = sp.shape(A)[0]
    out = sp.zeros((sp.shape(A)[1],sp.shape(A)[1]))
    for i in range(q):
        out = out + mult_3(A[i],x,H(A[i]))
    return out