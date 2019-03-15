import numpy as np
import numba
from .fast_interp import interp1d

class FunctionGenerator(object):
    """
    This class provides a simple way to construct a fast "function evaluator"
    For 1-D functions defined on an interval
    """
    def __init__(self, f, a, b, tol=1e-10, n=1000, k=5):
        """
        f:   function to create evaluator for
        a:   lower bound of evaluation interval
        b:   upper bound of evaluation interval
        tol: accuracy to recreate function to
        n:   number of points used in interpolations
        k:   degree of polynomial used (1, 3, 5, or 7)
        """
        self.f = f
        self.a = float(a)
        self.b = float(b)
        self.tol = tol
        self.n = n
        self.k = k
        self.lbs = []
        self.ubs = []
        self.hs = []
        self.fs = []
        self._fit(self.a, self.b)
        self.lbs = np.array(self.lbs)
        self.ubs = np.array(self.ubs)
        self.hs = np.array(self.hs)
        self.fs = np.row_stack(self.fs)
    def __call__(self, x, out=None):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            xr = x.ravel()
            outr = np.zeros_like(xr) if out is None else out.ravel()
            _evaluates[self.k](self.fs, xr, outr, self.lbs, self.ubs, self.hs, self.n)
            return outr.reshape(x.shape)
        else:
            return _evaluate1s[self.k](self.fs, x, self.lbs, self.ubs, self.hs, self.n)
    def _fit(self, a, b):
        x, h = np.linspace(a, b, self.n, retstep=True)
        interp = interp1d(a, b, h, self.f(x), self.k, c=True)
        check_x = x[:-1] + h/2.0
        check_f = self.f(check_x)
        estim_f = interp(check_x)
        reg = np.abs(check_f)
        reg[reg < 1] = 1.0
        err = np.abs((check_f-estim_f)/reg).max()
        if err < self.tol:
            self.lbs.append(a)
            self.ubs.append(b)
            self.fs.append(interp._f)
            self.hs.append(h)
        else:
            m = a + (b-a)/2
            self._fit(a, m)
            self._fit(m, b)

@numba.njit
def _single_interp_1d_k1(f, x, a, h, n):
    xx = x - a
    ix = min(int(xx//h), n-2)
    ratx = xx/h - (ix+0.5)
    asx = np.empty(2)
    asx[0] = 0.5 - ratx
    asx[1] = 0.5 + ratx
    fout = 0.0
    for i in range(2):
        fout += f[ix+i]*asx[i]
    return fout
@numba.njit
def _single_interp_1d_k3(f, x, a, h, n):
    xx = x - a
    ix = min(int(xx//h), n-2)
    ratx = xx/h - (ix+0.5)
    asx = np.empty(4)
    asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
    asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
    asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
    asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
    fout = 0.0
    for i in range(4):
        fout += f[ix+i]*asx[i]
    return fout
@numba.njit
def _single_interp_1d_k5(f, x, a, h, n):
    xx = x - a
    ix = min(int(xx//h), n-2)
    ratx = xx/h - (ix+0.5)
    asx = np.empty(6)
    asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
    asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
    asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
    asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
    asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
    asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
    fout = 0.0
    for i in range(6):
        fout += f[ix+i]*asx[i]
    return fout
@numba.njit
def _single_interp_1d_k7(f, x, a, h, n):
    xx = x - a
    ix = min(int(xx//h), n-2)
    ratx = xx/h - (ix+0.5)
    asx = np.empty(8)
    asx[0] =   -5/2048 + ratx*(     75/107520 + ratx*(  259/11520/2 + ratx*(  -37/1920/6 + ratx*(  -7/48/24 + ratx*(   5/24/120 + ratx*( 1/2/720 -  1/5040*ratx))))))
    asx[1] =   49/2048 + ratx*(  -1029/107520 + ratx*(-2495/11520/2 + ratx*(  499/1920/6 + ratx*(  59/48/24 + ratx*( -59/24/120 + ratx*(-5/2/720 +  7/5040*ratx))))))
    asx[2] = -245/2048 + ratx*(   8575/107520 + ratx*(11691/11520/2 + ratx*(-3897/1920/6 + ratx*(-135/48/24 + ratx*( 225/24/120 + ratx*( 9/2/720 - 21/5040*ratx))))))
    asx[3] = 1225/2048 + ratx*(-128625/107520 + ratx*(-9455/11520/2 + ratx*( 9455/1920/6 + ratx*(  83/48/24 + ratx*(-415/24/120 + ratx*(-5/2/720 + 35/5040*ratx))))))
    asx[4] = 1225/2048 + ratx*( 128625/107520 + ratx*(-9455/11520/2 + ratx*(-9455/1920/6 + ratx*(  83/48/24 + ratx*( 415/24/120 + ratx*(-5/2/720 - 35/5040*ratx))))))
    asx[5] = -245/2048 + ratx*(  -8575/107520 + ratx*(11691/11520/2 + ratx*( 3897/1920/6 + ratx*(-135/48/24 + ratx*(-225/24/120 + ratx*( 9/2/720 + 21/5040*ratx))))))
    asx[6] =   49/2048 + ratx*(   1029/107520 + ratx*(-2495/11520/2 + ratx*( -499/1920/6 + ratx*(  59/48/24 + ratx*(  59/24/120 + ratx*(-5/2/720 -  7/5040*ratx))))))
    asx[7] =   -5/2048 + ratx*(    -75/107520 + ratx*(  259/11520/2 + ratx*(   37/1920/6 + ratx*(  -7/48/24 + ratx*(  -5/24/120 + ratx*( 1/2/720 +  1/5040*ratx))))))
    fout = 0.0
    for i in range(8):
        fout += f[ix+i]*asx[i]
    return fout

@numba.njit
def _get_ind(x, ubs):
    ind = 0
    while(x > ubs[ind]):
        ind += 1
    return ind

@numba.njit(fastmath=True)
def _evaluate1_1(fs, x, lbs, ubs, hs, n):
    ind = _get_ind(x, ubs)
    return _single_interp_1d_k1(fs[ind], x, lbs[ind], hs[ind], n)
@numba.njit(fastmath=True)
def _evaluate1_3(fs, x, lbs, ubs, hs, n):
    ind = _get_ind(x, ubs)
    return _single_interp_1d_k3(fs[ind], x, lbs[ind], hs[ind], n)
@numba.njit(fastmath=True)
def _evaluate1_5(fs, x, lbs, ubs, hs, n):
    ind = _get_ind(x, ubs)
    return _single_interp_1d_k5(fs[ind], x, lbs[ind], hs[ind], n)
@numba.njit(fastmath=True)
def _evaluate1_7(fs, x, lbs, ubs, hs, n):
    ind = _get_ind(x, ubs)
    return _single_interp_1d_k7(fs[ind], x, lbs[ind], hs[ind], n)

@numba.njit(parallel=True, fastmath=True)
def _evaluate_1(fs, xs, out, lbs, ubs, hs, n):
    m = xs.shape[0]
    for i in numba.prange(m):
        out[i] = _evaluate1_1(fs, xs[i], lbs, ubs, hs, n)
@numba.njit(parallel=True, fastmath=True)
def _evaluate_3(fs, xs, out, lbs, ubs, hs, n):
    m = xs.shape[0]
    for i in numba.prange(m):
        out[i] = _evaluate1_3(fs, xs[i], lbs, ubs, hs, n)
@numba.njit(parallel=True, fastmath=True)
def _evaluate_5(fs, xs, out, lbs, ubs, hs, n):
    m = xs.shape[0]
    fplop = np.empty((6, m), dtype=np.float64)
    asxs = np.empty((6, m), dtype=np.float64)
    ixs = np.empty(m, dtype=np.int32)
    ratxs = np.empty(m, dtype=np.float64)
    inds = np.empty(m, dtype=np.int32)
    for k in numba.prange(m):
        x = xs[k]
        ind = 0
        while(x > ubs[ind]):
            ind += 1
        a = lbs[ind]
        h = hs[ind]
        xx = x - a
        ix = min(int(xx//h), n-2)
        ratxs[k] = xx/h - (ix+0.5)
        ixs[k] = ix
        inds[k] = ind
    for k in numba.prange(m):
        ratx = ratxs[k]
        asxs[0, k] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
        asxs[1, k] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
        asxs[2, k] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
        asxs[3, k] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
        asxs[4, k] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
        asxs[5, k] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
    for k in numba.prange(m):
        ix = ixs[k]
        ind = inds[k]
        for i in range(6):
            fplop[i, k] = fs[ind, ix+i]
    for k in numba.prange(m):
        out[k] = 0.0
        for i in range(6):
            out[k] += fplop[i,k]*asxs[i,k]
@numba.njit(parallel=True, fastmath=True)
def _evaluate_7(fs, xs, out, lbs, ubs, hs, n):
    m = xs.shape[0]
    for i in numba.prange(m):
        out[i] = _evaluate1_7(fs, xs[i], lbs, ubs, hs, n)

_evaluate1s = [None, _evaluate1_1, None, _evaluate1_3, None, _evaluate1_5, None, _evaluate1_7]
_evaluates  = [None, _evaluate_1,  None, _evaluate_3,  None, _evaluate_5,  None, _evaluate_7 ]

