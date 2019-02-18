import numpy as np
import numba

################################################################################
# 1D Extrapolation Routines

def _extrapolate1d_x(fb, k, periodic):
    if periodic:
        __extrapolate1d_x_periodic(fb, k)
    else:
        __extrapolate1d_x(fb, k)
def _extrapolate1d_y(fb, k, periodic):
    if periodic:
        __extrapolate1d_y_periodic(fb, k)
    else:
        __extrapolate1d_y(fb, k)
def __extrapolate1d_x(fb, k):
    if k == 3:
        fb[ 0] = 4*fb[ 1] - 6*fb[ 2] + 4*fb[ 3] - fb[ 4]
        fb[-1] = 4*fb[-2] - 6*fb[-3] + 4*fb[-4] - fb[-5]
    if k == 5:
        fb[ 0] = 21*fb[ 2] - 70*fb[ 3] + 105*fb[ 4] - 84*fb[ 5] + 35*fb[ 6] - 6*fb[ 7]
        fb[ 1] =  6*fb[ 2] - 15*fb[ 3] +  20*fb[ 4] - 15*fb[ 5] +  6*fb[ 6] -   fb[ 7]
        fb[-2] =  6*fb[-3] - 15*fb[-4] +  20*fb[-5] - 15*fb[-6] +  6*fb[-7] -   fb[-8]
        fb[-1] = 21*fb[-3] - 70*fb[-4] + 105*fb[-5] - 84*fb[-6] + 35*fb[-7] - 6*fb[-8]
def __extrapolate1d_y(fb, k):
    if k == 3:
        fb[:, 0] = 4*fb[:, 1] - 6*fb[:, 2] + 4*fb[:, 3] - fb[:, 4]
        fb[:,-1] = 4*fb[:,-2] - 6*fb[:,-3] + 4*fb[:,-4] - fb[:,-5]
    if k == 5:
        fb[:, 0] = 21*fb[:, 2] - 70*fb[:, 3] + 105*fb[:, 4] - 84*fb[:, 5] + 35*fb[:, 6] - 6*fb[:, 7]
        fb[:, 1] =  6*fb[:, 2] - 15*fb[:, 3] +  20*fb[:, 4] - 15*fb[:, 5] +  6*fb[:, 6] -   fb[:, 7]
        fb[:,-2] =  6*fb[:,-3] - 15*fb[:,-4] +  20*fb[:,-5] - 15*fb[:,-6] +  6*fb[:,-7] -   fb[:,-8]
        fb[:,-1] = 21*fb[:,-3] - 70*fb[:,-4] + 105*fb[:,-5] - 84*fb[:,-6] + 35*fb[:,-7] - 6*fb[:,-8]
def __extrapolate1d_x_periodic(fb, k):
    if k == 1:
        fb[-1] = fb[ 0]
    if k == 3:
        fb[ 0] = fb[-3]
        fb[-2] = fb[ 1]
        fb[-1] = fb[ 2]
    if k == 5:
        fb[ 0] = fb[-5]
        fb[ 1] = fb[-4]
        fb[-3] = fb[ 2]
        fb[-2] = fb[ 3]
        fb[-1] = fb[ 4]
def __extrapolate1d_y_periodic(fb, k):
    if k == 1:
        fb[:,-1] = fb[:, 0]
    if k == 3:
        fb[:, 0] = fb[:,-3]
        fb[:,-2] = fb[:, 1]
        fb[:,-1] = fb[:, 2]
    if k == 5:
        fb[:, 0] = fb[:,-5]
        fb[:, 1] = fb[:,-4]
        fb[:,-3] = fb[:, 2]
        fb[:,-2] = fb[:, 3]
        fb[:,-1] = fb[:, 4]
def _extrapolate1d_z(fb, k, periodic):
    if periodic:
        __extrapolate1d_z_periodic(fb, k)
    else:
        __extrapolate1d_z(fb, k)
def __extrapolate1d_z(fb, k):
    if k == 3:
        fb[:,:, 0] = 4*fb[:,:, 1] - 6*fb[:,:, 2] + 4*fb[:,:, 3] - fb[:,:, 4]
        fb[:,:,-1] = 4*fb[:,:,-2] - 6*fb[:,:,-3] + 4*fb[:,:,-4] - fb[:,:,-5]
    if k == 5:
        fb[:,:, 0] = 21*fb[:,:, 2] - 70*fb[:,:, 3] + 105*fb[:,:, 4] - 84*fb[:,:, 5] + 35*fb[:,:, 6] - 6*fb[:,:, 7]
        fb[:,:, 1] =  6*fb[:,:, 2] - 15*fb[:,:, 3] +  20*fb[:,:, 4] - 15*fb[:,:, 5] +  6*fb[:,:, 6] -   fb[:,:, 7]
        fb[:,:,-2] =  6*fb[:,:,-3] - 15*fb[:,:,-4] +  20*fb[:,:,-5] - 15*fb[:,:,-6] +  6*fb[:,:,-7] -   fb[:,:,-8]
        fb[:,:,-1] = 21*fb[:,:,-3] - 70*fb[:,:,-4] + 105*fb[:,:,-5] - 84*fb[:,:,-6] + 35*fb[:,:,-7] - 6*fb[:,:,-8]
def __extrapolate1d_z_periodic(fb, k):
    if k == 1:
        fb[:,:,-1] = fb[:,:, 0]
    if k == 3:
        fb[:,:, 0] = fb[:,:,-3]
        fb[:,:,-2] = fb[:,:, 1]
        fb[:,:,-1] = fb[:,:, 2]
    if k == 5:
        fb[:,:, 0] = fb[:,:,-5]
        fb[:,:, 1] = fb[:,:,-4]
        fb[:,:,-3] = fb[:,:, 2]
        fb[:,:,-2] = fb[:,:, 3]
        fb[:,:,-1] = fb[:,:, 4]

################################################################################
# One dimensional routines

class interp1d(object):
    def __init__(self, xv, f, k=3, periodic=False):
        """
        xv are the equispaced data nodes
        f is the data, sampled at xv
        k is the order of the local taylor expansions (int)
            k gives interp accuracy of order k+1
            only 1, 3, 5, supported
        periodic gives whether the dimension is taken to be periodic
            if it is taken to be periodic, it is assumed that the last index
            is skipped, and is equal to the first index; that is, the interp
            range is taken to be:
                xv[0] --> xv[-1] + xh
            no checking as to whether values live in this region is done
            at least for now...
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')
        self.f = f
        self.k = k
        self.periodic = periodic
        self._dtype = f.dtype
        self._hx, self._fb = _prepare1d(xv, k, periodic, f)
    def __call__(self, xout, fout=None):
        """
        Interpolate to xout
        xout must be a float or a ndarray of floats
        """
        if isinstance(xout, np.ndarray):
            m = int(np.prod(xout.shape))
            copy_made = False
            if fout is None:
                _out = np.empty(m, dtype=self._dtype)
            else:
                _out = fout.ravel()
                if _out.base is None:
                    copy_made = True
            _xout = xout.ravel()
            if self.k == 1:
                _interp1d_k1(self._fb, _xout, _out, self._hx)
            elif self.k == 3:
                _interp1d_k3(self._fb, _xout, _out, self._hx)
            else:
                _interp1d_k5(self._fb, _xout, _out, self._hx)
        if copy_made:
            fout[:] = _out
        return _out.reshape(xout.shape)

# the actual interpolation routines
@numba.njit(parallel=True)
def _interp1d_k1(f, xout, fout, hx):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        ix = int(xx//hx)
        dx = xx - (ix+0.5)*hx
        ratx = dx/hx
        asx = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        sp = 0.0
        for i in range(2):
            sp += f[ix+i]*asx[i]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp1d_k3(f, xout, fout, hx):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        ix = int(xx//hx)
        dx = xx - (ix+0.5)*hx
        ratx = dx/hx
        asx = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        sp = 0.0
        for i in range(4):
            sp += f[ix+i]*asx[i]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp1d_k5(f, xout, fout, hx):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        ix = int(xx//hx)
        dx = xx - (ix+0.5)*hx
        ratx = dx/hx
        asx = np.empty(6)
        asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
        asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
        asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
        asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
        asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
        asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
        sp = 0.0
        for i in range(6):
            sp += f[ix+i]*asx[i]
        fout[mi] = sp

# preparation and extrapolation routines
def _prepare1d(xv, k, periodic, f):
    hx = xv[1] - xv[0]
    fb = _extrapolate1d(f, k, periodic)
    return hx, fb
def _extrapolate1d(f, k, periodic):
    if k == 1 and not periodic:
        return f
    else:
        return _extrapolate1d_periodic(f, k, periodic)
def _extrapolate1d_periodic(f, k, periodic):
    offset = k//2
    fb = np.empty(f.shape[0] + 2*offset + int(periodic), dtype=f.dtype)
    _fill1(f, fb, offset)
    _extrapolate1d_x(fb, k, periodic)
    return fb
def _fill1(f, fb, offset):
    nx = f.shape[0]
    fb[offset:offset+nx] = f

################################################################################
# Two dimensional routines

class interp2d(object):
    def __init__(self, xv, yv, f, k=3, periodic=[False, False]):
        """
        xv, yv are the equispaced data nodes
        f is the data, sampled at meshgrid(xv, yv)
        k is the order of the local taylor expansions (int)
            k gives interp accuracy of order k+1
            only 1, 3, 5, supported
        periodic gives whether the dimension is taken to be periodic
            if it is taken to be periodic, it is assumed that the last index
            is skipped, and is equal to the first index; that is, the interp
            range is taken to be:
                xv[0] --> xv[-1] + xh
            no checking as to whether values live in this region is done
            at least for now...
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')
        self.f = f
        self.k = k
        self.periodic = periodic
        self._dtype = f.dtype
        self._hx, self._hy, self._fb = _prepare2d(xv, yv, k, periodic, f)
    def __call__(self, xout, yout, fout=None):
        """
        Interpolate to xout
        For 1-D interpolation, xout must be a float
            or a ndarray of floats
        """
        if isinstance(xout, np.ndarray):
            m = int(np.prod(xout.shape))
            copy_made = False
            if fout is None:
                _out = np.empty(m, dtype=self._dtype)
            else:
                _out = fout.ravel()
                if _out.base is None:
                    copy_made = True
            _xout = xout.ravel()
            _yout = yout.ravel()
            if self.k == 1:
                _interp2d_k1(self._fb, _xout, _yout, _out, self._hx, self._hy)
            elif self.k == 3:
                _interp2d_k3(self._fb, _xout, _yout, _out, self._hx, self._hy)
            else:
                _interp2d_k5(self._fb, _xout, _yout, _out, self._hx, self._hy)
        if copy_made:
            fout[:] = _out
        return _out.reshape(xout.shape)

# the actual interpolation routines
@numba.njit(parallel=True)
def _interp2d_k1(f, xout, yout, fout, hx, hy):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        ratx = dx/hx
        raty = dy/hy
        asx = np.empty(2)
        asy = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        asy[0] = 0.5 - raty
        asy[1] = 0.5 + raty
        sp = 0.0
        for i in range(2):
            for j in range(2):
                sp += f[ix+i,iy+j]*asx[i]*asy[j]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp2d_k3(f, xout, yout, fout, hx, hy):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        ratx = dx/hx
        raty = dy/hy
        asx = np.empty(4)
        asy = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        asy[0] = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))
        asy[1] =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))
        asy[2] =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))
        asy[3] = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))
        sp = 0.0
        for i in range(4):
            for j in range(4):
                sp += f[ix+i,iy+j]*asx[i]*asy[j]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp2d_k5(f, xout, yout, fout, hx, hy):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        ratx = dx/hx
        raty = dy/hy
        asx = np.empty(6)
        asy = np.empty(6)
        asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
        asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
        asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
        asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
        asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
        asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
        asy[0] =   3/256 + raty*(   -9/1920 + raty*( -5/48/2 + raty*(  1/8/6 + raty*( 1/2/24 -  1/8/120*raty))))
        asy[1] = -25/256 + raty*(  125/1920 + raty*( 39/48/2 + raty*(-13/8/6 + raty*(-3/2/24 +  5/8/120*raty))))
        asy[2] = 150/256 + raty*(-2250/1920 + raty*(-34/48/2 + raty*( 34/8/6 + raty*( 2/2/24 - 10/8/120*raty))))
        asy[3] = 150/256 + raty*( 2250/1920 + raty*(-34/48/2 + raty*(-34/8/6 + raty*( 2/2/24 + 10/8/120*raty))))
        asy[4] = -25/256 + raty*( -125/1920 + raty*( 39/48/2 + raty*( 13/8/6 + raty*(-3/2/24 -  5/8/120*raty))))
        asy[5] =   3/256 + raty*(    9/1920 + raty*( -5/48/2 + raty*( -1/8/6 + raty*( 1/2/24 +  1/8/120*raty))))
        sp = 0.0
        for i in range(6):
            for j in range(6):
                sp += f[ix+i,iy+j]*asx[i]*asy[j]
        fout[mi] = sp

# preparation and extrapolation routines
def _prepare2d(xv, yv, k, periodic, f):
    hx = xv[1] - xv[0]
    hy = yv[1] - yv[0]
    fb = _extrapolate2d(f, k, periodic)
    return hx, hy, fb
def _extrapolate2d(f, k, periodic):
    any_periodic = periodic[0] or periodic[1]
    if k == 1 and not any_periodic:
        return f
    else:
        return _extrapolate2d_periodic(f, k, periodic)
def _extrapolate2d_periodic(f, k, periodic):
    offset = k//2
    newsh = [n + 2*offset + int(p) for n, p in zip(f.shape, periodic)]
    fb = np.empty(newsh, dtype=f.dtype)
    _fill2(f, fb, offset)
    _extrapolate1d_x(fb, k, periodic[0])
    _extrapolate1d_y(fb, k, periodic[1])
    return fb
def _fill2(f, fb, offset):
    nx = f.shape[0]
    ny = f.shape[1]
    if nx*ny < 100000:
        fb[offset:offset+nx, offset:offset+ny] = f
    else:
        __fill2(f, fb, offset)
@numba.njit(parallel=True)
def __fill2(f, fb, offset):
    nx = f.shape[0]
    ny = f.shape[1]
    for i in numba.prange(nx):
        for j in range(ny):
            fb[i+offset,j+offset] = f[i,j]

################################################################################
# Three dimensional routines

class interp3d(object):
    def __init__(self, xv, yv, zv, f, k=3, periodic=[False, False, False]):
        """
        xv, yv, zv are the equispaced data nodes
        f is the data, sampled at meshgrid(xv, yv, zv)
        k is the order of the local taylor expansions (int)
            k gives interp accuracy of order k+1
            only 1, 3, 5, supported
        periodic gives whether the dimension is taken to be periodic
            if it is taken to be periodic, it is assumed that the last index
            is skipped, and is equal to the first index; that is, the interp
            range is taken to be:
                xv[0] --> xv[-1] + xh
            no checking as to whether values live in this region is done
            at least for now...
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')
        self.f = f
        self.k = k
        self.periodic = periodic
        self._dtype = f.dtype
        self._hx, self._hy, self._hz, self._fb = _prepare3d(xv, yv, zv, k, periodic, f)
    def __call__(self, xout, yout, zout, fout=None):
        """
        Interpolate to xout
        For 1-D interpolation, xout must be a float
            or a ndarray of floats
        """
        if isinstance(xout, np.ndarray):
            m = int(np.prod(xout.shape))
            copy_made = False
            if fout is None:
                _out = np.empty(m, dtype=self._dtype)
            else:
                _out = fout.ravel()
                if _out.base is None:
                    copy_made = True
            _xout = xout.ravel()
            _yout = yout.ravel()
            _zout = zout.ravel()
            if self.k == 1:
                _interp3d_k1(self._fb, _xout, _yout, _zout, _out, self._hx, self._hy, self._hz)
            elif self.k == 3:
                _interp3d_k3(self._fb, _xout, _yout, _zout, _out, self._hx, self._hy, self._hz)
            else:
                _interp3d_k5(self._fb, _xout, _yout, _zout, _out, self._hx, self._hy, self._hz)
        if copy_made:
            fout[:] = _out
        return _out.reshape(xout.shape)

# the actual interpolation routines
@numba.njit(parallel=True)
def _interp3d_k1(f, xout, yout, zout, fout, hx, hy, hz):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        zz = zout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        iz = int(zz//hz)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        dz = zz - (iz+0.5)*hz
        ratx = dx/hx
        raty = dy/hy
        ratz = dz/hz
        asx = np.empty(2)
        asy = np.empty(2)
        asz = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        asy[0] = 0.5 - raty
        asy[1] = 0.5 + raty
        asz[0] = 0.5 - ratz
        asz[1] = 0.5 + ratz
        sp = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sp += f[ix+i,iy+j,iz+k]*asx[i]*asy[j]*asz[k]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp3d_k3(f, xout, yout, zout, fout, hx, hy, hz):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        zz = zout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        iz = int(zz//hz)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        dz = zz - (iz+0.5)*hz
        ratx = dx/hx
        raty = dy/hy
        ratz = dz/hz
        asx = np.empty(4)
        asy = np.empty(4)
        asz = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        asy[0] = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))
        asy[1] =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))
        asy[2] =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))
        asy[3] = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))
        asz[0] = -1/16 + ratz*( 1/24 + ratz*( 1/4 - ratz/6))
        asz[1] =  9/16 + ratz*( -9/8 + ratz*(-1/4 + ratz/2))
        asz[2] =  9/16 + ratz*(  9/8 + ratz*(-1/4 - ratz/2))
        asz[3] = -1/16 + ratz*(-1/24 + ratz*( 1/4 + ratz/6))
        sp = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    sp += f[ix+i,iy+j,iz+k]*asx[i]*asy[j]*asz[k]
        fout[mi] = sp
@numba.njit(parallel=True)
def _interp3d_k5(f, xout, yout, zout, fout, hx, hy, hz):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = xout[mi]
        yy = yout[mi]
        zz = zout[mi]
        ix = int(xx//hx)
        iy = int(yy//hy)
        iz = int(zz//hz)
        dx = xx - (ix+0.5)*hx
        dy = yy - (iy+0.5)*hy
        dz = zz - (iz+0.5)*hz
        ratx = dx/hx
        raty = dy/hy
        ratz = dz/hz
        asx = np.empty(6)
        asy = np.empty(6)
        asz = np.empty(6)
        asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))
        asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))
        asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))
        asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))
        asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))
        asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))
        asy[0] =   3/256 + raty*(   -9/1920 + raty*( -5/48/2 + raty*(  1/8/6 + raty*( 1/2/24 -  1/8/120*raty))))
        asy[1] = -25/256 + raty*(  125/1920 + raty*( 39/48/2 + raty*(-13/8/6 + raty*(-3/2/24 +  5/8/120*raty))))
        asy[2] = 150/256 + raty*(-2250/1920 + raty*(-34/48/2 + raty*( 34/8/6 + raty*( 2/2/24 - 10/8/120*raty))))
        asy[3] = 150/256 + raty*( 2250/1920 + raty*(-34/48/2 + raty*(-34/8/6 + raty*( 2/2/24 + 10/8/120*raty))))
        asy[4] = -25/256 + raty*( -125/1920 + raty*( 39/48/2 + raty*( 13/8/6 + raty*(-3/2/24 -  5/8/120*raty))))
        asy[5] =   3/256 + raty*(    9/1920 + raty*( -5/48/2 + raty*( -1/8/6 + raty*( 1/2/24 +  1/8/120*raty))))
        asz[0] =   3/256 + ratz*(   -9/1920 + ratz*( -5/48/2 + ratz*(  1/8/6 + ratz*( 1/2/24 -  1/8/120*ratz))))
        asz[1] = -25/256 + ratz*(  125/1920 + ratz*( 39/48/2 + ratz*(-13/8/6 + ratz*(-3/2/24 +  5/8/120*ratz))))
        asz[2] = 150/256 + ratz*(-2250/1920 + ratz*(-34/48/2 + ratz*( 34/8/6 + ratz*( 2/2/24 - 10/8/120*ratz))))
        asz[3] = 150/256 + ratz*( 2250/1920 + ratz*(-34/48/2 + ratz*(-34/8/6 + ratz*( 2/2/24 + 10/8/120*ratz))))
        asz[4] = -25/256 + ratz*( -125/1920 + ratz*( 39/48/2 + ratz*( 13/8/6 + ratz*(-3/2/24 -  5/8/120*ratz))))
        asz[5] =   3/256 + ratz*(    9/1920 + ratz*( -5/48/2 + ratz*( -1/8/6 + ratz*( 1/2/24 +  1/8/120*ratz))))
        sp = 0.0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    sp += f[ix+i,iy+j,iz+k]*asx[i]*asy[j]*asz[k]
        fout[mi] = sp

# preparation and extrapolation routines
def _prepare3d(xv, yv, zv, k, periodic, f):
    hx = xv[1] - xv[0]
    hy = yv[1] - yv[0]
    hz = zv[1] - zv[0]
    fb = _extrapolate3d(f, k, periodic)
    return hx, hy, hz, fb
def _extrapolate3d(f, k, periodic):
    any_periodic = periodic[0] or periodic[1] or periodic[2]
    if k == 1 and not any_periodic:
        return f
    else:
        return _extrapolate3d_periodic(f, k, periodic)
def _extrapolate3d_periodic(f, k, periodic):
    offset = k//2
    newsh = [n + 2*offset + int(p) for n, p in zip(f.shape, periodic)]
    fb = np.empty(newsh, dtype=f.dtype)
    _fill3(f, fb, offset)
    _extrapolate1d_x(fb, k, periodic[0])
    _extrapolate1d_y(fb, k, periodic[1])
    _extrapolate1d_z(fb, k, periodic[2])
    return fb
def _fill3(f, fb, offset):
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    if nx*ny*nz < 100000:
        fb[offset:offset+nx, offset:offset+ny, offset:offset+nz] = f
    else:
        __fill3(f, fb, offset)
@numba.njit(parallel=True)
def __fill3(f, fb, offset):
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    for i in numba.prange(nx):
        for j in range(ny):
            for k in range(nz):
                fb[i+offset,j+offset,k+offset] = f[i,j,k]

