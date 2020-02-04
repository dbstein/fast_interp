import numpy as np
import numba
from scipy.linalg import lu_factor, lu_solve
from collections import OrderedDict

################################################################################
# basic utilities

def affine_transformation(xin, min_in, max_in, min_out, max_out):
    ran_in = max_in - min_in
    ran_out = max_out - min_out
    rat = ran_out/ran_in
    return (xin - min_in)*rat + min_out

def get_chebyshev_nodes(lb, ub, order):
    xc = np.cos( (2*np.arange(1, order+1)-1)/(2*order)*np.pi )
    x = affine_transformation(xc[::-1], -1, 1, lb, ub)
    return xc[::-1], x

################################################################################
# numba functions

@numba.njit(fastmath=True)
def bisect_search(x, ordered_array):
    n1 = 0
    n2 = ordered_array.size
    d = n2 - n1
    while d > 1:
        m = n1 + d // 2
        if x < ordered_array[m]:
            n2 = m
        else:
            n1 = m
        d = n2 - n1
    return n1

@numba.njit(fastmath=True)
def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x

@numba.njit(parallel=True, fastmath=True)
def _numba_multieval_check(xs, lbs, ubs, iscale, cs, out):
    n = xs.size
    for i in numba.prange(n):
        out[i] = _numba_eval_check(xs[i], lbs, ubs, iscale, cs)

@numba.njit(parallel=True, fastmath=True)
def _numba_multieval(xs, lbs, ubs, iscale, cs, out):
    n = xs.size
    for i in numba.prange(n):
        x = xs[i]
        ind = bisect_search(x, lbs)
        a = lbs[ind]
        b = ubs[ind]
        iba = iscale[ind]
        _x = (x-a)*iba - 1.0
        c = cs[ind]
        out[i] = _numba_chbevl(_x, c)

@numba.njit(fastmath=True)
def _numba_eval_check(x, lbs, ubs, iscale, cs):
    if x >= lbs[0] and x <= ubs[-1]:
        return _numba_eval(x, lbs, ubs, iscale, cs)
    else:
        return np.nan

@numba.njit(fastmath=True)
def _numba_eval(x, lbs, ubs, iscale, cs):
    ind = bisect_search(x, lbs)
    a = lbs[ind]
    b = ubs[ind]
    iba = iscale[ind]
    _x = (x-a)*iba - 1.0
    return _numba_chbevl(_x, cs[ind])

class ChebyshevFunctionGenerator(object):
    """
    This class provides a simple way to construct a fast "function evaluator"
    For 1-D functions defined on an interval
    """
    def __init__(self, f, a, b, tol=1e-10, n=12, mw=1e-15, verbose=False):
        """
        f:       function to create evaluator for
        a:       lower bound of evaluation interval
        b:       upper bound of evaluation interval
        tol:     accuracy to recreate function to
        n:       degree of chebyshev polynomials to be used
        mw:      minimum width of interval (accuracy no longer guaranteed!)
        verbose: generate verbose output
        """
        self.f = f
        r = b-a
        a1 = a + r/3
        a2 = a + 2*r/3
        self.dtype = self.f(np.array([a1, a2])).dtype
        self.a = float(a)
        self.b = float(b)
        self.tol = tol
        self.n = n
        self.mw = mw
        self.verbose = verbose
        self.lbs = []
        self.ubs = []
        self.coefs = []
        _x, _ = get_chebyshev_nodes(-1, 1, self.n)
        self.V = np.polynomial.chebyshev.chebvander(_x, self.n-1)
        self.VLU = lu_factor(self.V)
        self._fit(self.a, self.b)
        self.lbs = np.array(self.lbs)
        self.ubs = np.array(self.ubs)
        self.iscale = 2.0/(self.ubs - self.lbs)
        self.coef_mat = np.row_stack(self.coefs)
    def __call__(self, x, check_bounds=True, out=None):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            if out is None: out = np.empty(x.shape, dtype=self.dtype)
            if check_bounds:
                _numba_multieval_check(x.ravel(), self.lbs, self.ubs, self.iscale, self.coef_mat, out.ravel())
            else:
                _numba_multieval(x.ravel(), self.lbs, self.ubs, self.iscale, self.coef_mat, out.ravel())
            return out
        else:
            if check_bounds:
                return _numba_eval_check(x, self.lbs, self.ubs, self.iscale, self.coef_mat)
            else:
                return _numba_eval(x, self.lbs, self.ubs, self.iscale, self.coef_mat)
    def _fit(self, a, b):
        m = (a+b)/2.0
        if self.verbose:
            print('[', a, ',', b, ']')
        _, x = get_chebyshev_nodes(a, b, self.n)
        coefs = lu_solve(self.VLU, self.f(x))
        tail_energy = np.abs(coefs[-2:]).max()/max(1, np.abs(coefs[0]))
        if tail_energy < self.tol or b-a < self.mw:
            self.lbs.append(a)
            self.ubs.append(b)
            self.coefs.append(coefs)
        else:
            self._fit(a, m)
            self._fit(m, b)

    def get_base_function(self):
        lbs = self.lbs
        ubs = self.ubs
        cs = self.coef_mat
        @numba.njit(fastmath=True)
        def func(x):
            if x > lbs[0] and x < ubs[-1]:
                ind = _get_ind(x, ubs)
                a = lbs[ind]
                b = ubs[ind]
                _x = 2*(x-a)/(b-a) - 1.0
                return _numba_chbevl(_x, cs[ind])
            else:
                return np.nan
        return func

################################################################################
# This is slow for now :(

leaf_type = numba.deferred_type()
leaf_spec = OrderedDict()
leaf_spec['a'] = numba.float64
leaf_spec['b'] = numba.float64
leaf_spec['ancestor'] = numba.optional(leaf_type)
leaf_spec['m'] = numba.float64
leaf_spec['ind'] = numba.int64
leaf_spec['parent'] = numba.boolean
leaf_spec['left'] = numba.optional(leaf_type)
leaf_spec['right'] = numba.optional(leaf_type)

@numba.jitclass(leaf_spec)
class Leaf(object):
    def __init__(self, a, b, ancestor, ind):
        self.a = a
        self.b = b
        self.ancestor = ancestor
        self.m = (self.a + self.b)/2.0
        self.ind = ind
        self.parent = False
        self.left = None
        self.right = None
    def create_left(self):
        self.left = Leaf(self.a, self.m, self, self.ind)
        self.parent = True
    def create_right(self):
        self.right = Leaf(self.m, self.b, self, self.ind+1)
        ancestry = self
        while ancestry is not None:
            ancestry.ind += 1
            ancestry = ancestry.ancestor
    def get_left(self):
        return self.left
    def get_right(self):
        return self.right
    def get_ind(self):
        return self.ind
    def get_parent(self):
        return self.parent
    def get_m(self):
        return self.m
    def get_bounds(self):
        return self.a, self.b

leaf_type.define(Leaf.class_type.instance_type)

@numba.njit
def get_ind(tree, x):
    leaf = tree
    while leaf.get_parent():
        leaf = leaf.get_left() if x < leaf.get_m() else leaf.get_right()
    return leaf.get_ind()



