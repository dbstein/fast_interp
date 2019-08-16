import numpy as np
import numba
# from scipy.special import k0, hankel1

def affine_transformation(xin, min_in, max_in, min_out, max_out):
    ran_in = max_in - min_in
    ran_out = max_out - min_out
    rat = ran_out/ran_in
    return (xin - min_in)*rat + min_out

@numba.njit
def _get_ind(x, ubs):
    ind = 0
    while(x > ubs[ind]):
        ind += 1
    return ind

@numba.njit
def _numba_chbevl(x, c):
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

@numba.njit(parallel=True)
def _numba_multieval(xs, lbs, ubs, cs, out):
    n = xs.size
    for i in numba.prange(n):
        out[i] = _numba_eval(xs[i], lbs, ubs, cs)

@numba.njit
def _numba_eval(x, lbs, ubs, cs):
    if x > lbs[0] and x < ubs[-1]:
        ind = _get_ind(x, ubs)
        a = lbs[ind]
        b = ubs[ind]
        _x = 2*(x-a)/(b-a) - 1.0
        return _numba_chbevl(_x, cs[ind])
    else:
        return np.nan

def get_chebyshev_nodes(lb, ub, order):
    xc = np.cos( (2*np.arange(1, order+1)-1)/(2*order)*np.pi )
    x = affine_transformation(xc[::-1], -1, 1, lb, ub)
    return xc[::-1], x

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
        self.VI = np.linalg.inv(self.V)
        self._fit(self.a, self.b)
        self.lbs = np.array(self.lbs)
        self.ubs = np.array(self.ubs)
        self.coef_mat = np.row_stack(self.coefs)
    def __call__(self, x):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            out = np.empty(x.shape, dtype=self.dtype)
            _numba_multieval(x.ravel(), self.lbs, self.ubs, self.coef_mat, out.ravel())
            return out
        else:
            return _numba_eval(x, self.lbs, self.ubs, self.coef_mat)
    def _fit(self, a, b):
        if self.verbose:
            print('[', a, ',', b, ']')
        _, x = get_chebyshev_nodes(a, b, self.n)
        coefs = self.VI.dot(self.f(x))
        tail_energy = np.abs(coefs[-2:]).max()/coefs[0]
        if tail_energy < self.tol or b-a < self.mw:
            self.lbs.append(a)
            self.ubs.append(b)
            self.coefs.append(coefs)
        else:
            m = a + (b-a)/2
            self._fit(a, m)
            self._fit(m, b)

    def get_base_function(self):
        lbs = self.lbs
        ubs = self.ubs
        cs = self.coef_mat
        @numba.njit
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

# def test_func(x):
#     # return np.exp(np.sin(17*x))
#     # return k0(x)
#     return hankel1(0, x)
# test_vals = np.random.rand(1000*1000)

# cutoff = 1e-10

# test_vals = test_vals[test_vals > cutoff]

# interp = ChebyshevFunctionGenerator(test_func, cutoff, 10, tol=1e-14, n=32, verbose=True)

# np.abs(test_func(test_vals) - interp(test_vals)).max()

# %timeit test_func(test_vals)
# %timeit interp(test_vals)

