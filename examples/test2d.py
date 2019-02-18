import numpy as np
import scipy as sp
import scipy.interpolate
import time
from fast_interp import interp2d
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

################################################################################
# Basic error sweep

print('\n----- Testing Error and Timings vs. Scipy -----')

ntest = 10*2**np.arange(8)
ktest = [1, 3, 5]
my_errors = np.zeros([ntest.shape[0], 3], dtype=float)
sp_errors = np.zeros([ntest.shape[0], 3], dtype=float)
my_setup_time = np.zeros([ntest.shape[0], 3], dtype=float)
sp_setup_time = np.zeros([ntest.shape[0], 3], dtype=float)
my_eval_time = np.zeros([ntest.shape[0], 3], dtype=float)
sp_eval_time = np.zeros([ntest.shape[0], 3], dtype=float)

for ki, k in enumerate(ktest):
	print('--- Testing for k =', k, '---')
	for ni, n in enumerate(ntest):
		print('   ...n =', n)

		v, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
		x, y = np.meshgrid(v, v, indexing='ij')
		xo = x[:-1, :-1].flatten()
		yo = y[:-1, :-1].flatten()
		xo += np.random.rand(*xo.shape)*h
		yo += np.random.rand(*yo.shape)*h

		test_function = lambda x, y: np.exp(x)*np.cos(y) + x*y**3/(1 + x + y)
		f = test_function(x, y)
		fa = test_function(xo, yo)

		# run once to compile numba functions
		interpolater = interp2d(v, v, f, k=k)
		fe = interpolater(xo, yo)
		# fast_interp
		st = time.time()
		interpolater = interp2d(v, v, f, k=k)
		my_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater(xo, yo)
		my_eval_time[ni, ki] = (time.time()-st)*1000
		my_errors[ni, ki] = np.abs(fe - fa).max()

		# scipy interp
		st = time.time()
		interpolater = sp.interpolate.RectBivariateSpline(v, v, f, kx=k, ky=k)
		sp_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater.ev(xo, yo)
		sp_eval_time[ni, ki] = (time.time()-st)*1000
		sp_errors[ni, ki] = np.abs(fe - fa).max()

fig, ax = plt.subplots(1,1)
ax.plot(ntest, my_errors[:,0], color='black',  label='This, Linear')
ax.plot(ntest, my_errors[:,1], color='blue',   label='This, Cubic')
ax.plot(ntest, my_errors[:,2], color='purple', label='This, Qunitic')
ax.plot(ntest, sp_errors[:,0], color='black',  linestyle='--', label='Scipy, Linear')
ax.plot(ntest, sp_errors[:,1], color='blue',   linestyle='--', label='Scipy, Cubic')
ax.plot(ntest, sp_errors[:,2], color='purple', linestyle='--', label='Scipy, Qunitic')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Error')
ax.legend()

fig, ax = plt.subplots(1,1)
ax.plot(ntest, my_setup_time[:,0], color='black',  label='This, Linear')
ax.plot(ntest, my_setup_time[:,1], color='blue',   label='This, Cubic')
ax.plot(ntest, my_setup_time[:,2], color='purple', label='This, Qunitic')
ax.plot(ntest, sp_setup_time[:,0], color='black',  linestyle='--', label='Scipy, Linear')
ax.plot(ntest, sp_setup_time[:,1], color='blue',   linestyle='--', label='Scipy, Cubic')
ax.plot(ntest, sp_setup_time[:,2], color='purple', linestyle='--', label='Scipy, Qunitic')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Setup Time')
ax.legend()

fig, ax = plt.subplots(1,1)
ax.plot(ntest, my_eval_time[:,0], color='black',  label='This, Linear')
ax.plot(ntest, my_eval_time[:,1], color='blue',   label='This, Cubic')
ax.plot(ntest, my_eval_time[:,2], color='purple', label='This, Qunitic')
ax.plot(ntest, sp_eval_time[:,0], color='black',  linestyle='--', label='Scipy, Linear')
ax.plot(ntest, sp_eval_time[:,1], color='blue',   linestyle='--', label='Scipy, Cubic')
ax.plot(ntest, sp_eval_time[:,2], color='purple', linestyle='--', label='Scipy, Qunitic')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Evaluation Time')
ax.legend()

################################################################################
# Test with an shaped output array

print('\n----- Testing Interpolation to Shaped Output -----')

n = 1000
v, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
xo = x[:-1, :-1].copy()
yo = y[:-1, :-1].copy()
xo += np.random.rand(*xo.shape)*h
yo += np.random.rand(*yo.shape)*h

test_function = lambda x, y: np.exp(x)*np.cos(y) + x*y**3/(1 + x + y)
f = test_function(x, y)
fa = test_function(xo, yo)

interpolater = interp2d(v, v, f, k=5)
fe = interpolater(xo, yo)
err = np.abs(fe - fa).max()
print('...Error in interpolating to shaped array: {:0.1e}'.format(err))

################################################################################
# Test with periodic boundaries

print('\n----- Testing Periodic Qunitic Interpolation, both directions -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	x, y = np.meshgrid(v, v, indexing='ij')
	xo = x[:-1, :-1].copy()
	yo = y[:-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*h
	yo += np.random.rand(*yo.shape)*h

	test_function = lambda x, y: np.exp(np.sin(x))*np.cos(y)
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d(v, v, f, k=5, periodic=[True,True])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Linear Interpolation, x-direction -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	xv, xh = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	yv, yh = np.linspace(0, 1, n, endpoint=True, retstep=True)
	x, y = np.meshgrid(xv, yv, indexing='ij')
	xo = x[:-1, :-1].copy()
	yo = y[:-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh

	test_function = lambda x, y: np.exp(np.sin(x))*np.exp(y)
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d(xv, yv, f, k=1, periodic=[True,False])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Cubic Interpolation, y-direction -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	xv, xh = np.linspace(0, 1, n, endpoint=True, retstep=True)
	yv, yh = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
	x, y = np.meshgrid(xv, yv, indexing='ij')
	xo = x[:-1, :-1].copy()
	yo = y[:-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh

	test_function = lambda x, y: np.exp(x)*np.exp(np.sin(y))
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d(xv, yv, f, k=3, periodic=[False,True])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))



