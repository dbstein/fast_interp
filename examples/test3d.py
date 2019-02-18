import numpy as np
import scipy as sp
import scipy.interpolate
import time
from fast_interp import interp3d
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

################################################################################
# Basic error sweep

print('\n----- Testing Error -----')

ntest = 10*2**np.arange(6)
ktest = [1, 3, 5]
my_errors = np.zeros([ntest.shape[0], 3], dtype=float)

for ki, k in enumerate(ktest):
	print('--- Testing for k =', k, '---')
	for ni, n in enumerate(ntest):
		print('   ...n =', n)

		v, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
		x, y, z = np.meshgrid(v, v, v, indexing='ij')
		xo = x[:-1, :-1, :-1].flatten()
		yo = y[:-1, :-1, :-1].flatten()
		zo = y[:-1, :-1, :-1].flatten()
		xo += np.random.rand(*xo.shape)*h
		yo += np.random.rand(*yo.shape)*h
		zo += np.random.rand(*zo.shape)*h

		test_function = lambda x, y, z: np.exp(x)*np.cos(y)*z**2 + np.sin(z)*x*y**3/(1 + x + y)
		f = test_function(x, y, z)
		fa = test_function(xo, yo, zo)

		interpolater = interp3d(v, v, v, f, k=k)
		fe = interpolater(xo, yo, zo)
		my_errors[ni, ki] = np.abs(fe - fa).max()

nts = ntest[1:-1].astype(float)

fig, ax = plt.subplots(1,1)
ax.plot(ntest, my_errors[:,0], color='black',  label='Linear')
ax.plot(ntest, my_errors[:,1], color='blue',   label='Cubic')
ax.plot(ntest, my_errors[:,2], color='purple', label='Qunitic')
ax.plot(nts, 2*nts**-2, color='black', alpha=0.7, label=r'$\mathcal{O}(h^2)$')
ax.plot(nts, 0.5*nts**-4, color='black', alpha=0.5, label=r'$\mathcal{O}(h^4)$')
ax.plot(nts, 0.5*nts**-6, color='black', alpha=0.3, label=r'$\mathcal{O}(h^6)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Error')
ax.legend()

################################################################################
# Test with an shaped output array

print('\n----- Testing Interpolation to Shaped Output -----')

n = 200
v, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
x, y, z = np.meshgrid(v, v, v, indexing='ij')
xo = x[:-1, :-1, :-1].copy()
yo = y[:-1, :-1, :-1].copy()
zo = z[:-1, :-1, :-1].copy()
xo += np.random.rand(*xo.shape)*h
yo += np.random.rand(*yo.shape)*h
zo += np.random.rand(*zo.shape)*h

test_function = lambda x, y, z: np.exp(x)*np.cos(y)*z**2 + np.sin(z)*x*y**3/(1 + x + y)
f = test_function(x, y, z)
fa = test_function(xo, yo, zo)

interpolater = interp3d(v, v, v, f, k=5)
fe = interpolater(xo, yo, zo)
err = np.abs(fe - fa).max()
print('...Error in interpolating to shaped array: {:0.1e}'.format(err))

################################################################################
# Test with periodic boundaries

print('\n----- Testing Periodic Qunitic Interpolation, both directions -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	x, y, z = np.meshgrid(v, v, v, indexing='ij')
	xo = x[:-1, :-1, :-1].copy()
	yo = y[:-1, :-1, :-1].copy()
	zo = z[:-1, :-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*h
	yo += np.random.rand(*yo.shape)*h
	zo += np.random.rand(*zo.shape)*h

	test_function = lambda x, y, z: np.exp(np.sin(x))*np.cos(y) + np.sin(2*z)
	f = test_function(x, y, z)
	fa = test_function(xo, yo, zo)

	interpolater = interp3d(v, v, v, f, k=5, periodic=[True,True,True])
	fe = interpolater(xo, yo, zo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Linear Interpolation, x-direction -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	xv, xh = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	yv, yh = np.linspace(0, 1, n, endpoint=True, retstep=True)
	zv, zh = np.linspace(0, 1, n, endpoint=True, retstep=True)
	x, y, z = np.meshgrid(xv, yv, zv, indexing='ij')
	xo = x[:-1, :-1, :-1].copy()
	yo = y[:-1, :-1, :-1].copy()
	zo = z[:-1, :-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh
	zo += np.random.rand(*zo.shape)*zh

	test_function = lambda x, y, z: np.exp(np.sin(x))*np.exp(y) + z**2
	f = test_function(x, y, z)
	fa = test_function(xo, yo, zo)

	interpolater = interp3d(xv, yv, zv, f, k=1, periodic=[True,False,False])
	fe = interpolater(xo, yo, zo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Cubic Interpolation, x, z-directions -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	xv, xh = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	yv, yh = np.linspace(0, 1, n, endpoint=True, retstep=True)
	zv, zh = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	x, y, z = np.meshgrid(xv, yv, zv, indexing='ij')
	xo = x[:-1, :-1, :-1].copy()
	yo = y[:-1, :-1, :-1].copy()
	zo = y[:-1, :-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh
	zo += np.random.rand(*zo.shape)*zh

	test_function = lambda x, y, z: np.exp(np.sin(x))*y**2 + np.cos(2*z)
	f = test_function(x, y, z)
	fa = test_function(xo, yo, zo)

	interpolater = interp3d(xv, yv, zv, f, k=3, periodic=[True,False,True])
	fe = interpolater(xo, yo, zo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))



