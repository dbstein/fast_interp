import numpy as np
import scipy as sp
import scipy.interpolate
import time
from fast_interp import interp2d
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import gc

################################################################################
# Basic error sweep

print('\n----- Testing Error and Timings vs. Scipy -----')

ntest = 10*2**np.arange(9)
ktest = [1, 3, 5]
my_errors = np.zeros([ntest.shape[0], 3], dtype=float)
sp_errors = np.zeros([ntest.shape[0], 3], dtype=float)
my_setup_time = np.zeros([ntest.shape[0], 3], dtype=float)
sp_setup_time = np.zeros([ntest.shape[0], 3], dtype=float)
my_eval_time = np.zeros([ntest.shape[0], 3], dtype=float)
sp_eval_time = np.zeros([ntest.shape[0], 3], dtype=float)

gc.disable()

for ki, k in enumerate(ktest):
	print('--- Testing for k =', k, '---')
	for ni, n in enumerate(ntest):
		print('   ...n =', n)

		a = 1/np.e
		v, h = np.linspace(a, 1, n, endpoint=True, retstep=True)
		x, y = np.meshgrid(v, v, indexing='ij')
		xo = x[:-1, :-1].flatten()
		yo = y[:-1, :-1].flatten()
		xo += np.random.rand(*xo.shape)*h
		yo += np.random.rand(*yo.shape)*h

		test_function = lambda x, y: np.exp(x)*np.cos(y) + x*y**3/(1 + x + y)
		f = test_function(x, y)
		fa = test_function(xo, yo)

		# run once to compile numba functions
		interpolater = interp2d([a,a], [1.0,1.0], [h,h], f, k=k)
		fe = interpolater(xo, yo)
		del interpolater
		gc.collect()

		# fast_interp
		st = time.time()
		interpolater = interp2d([a,a], [1.0,1.0], [h,h], f, k=k)
		my_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater(xo, yo)
		my_eval_time[ni, ki] = (time.time()-st)*1000
		my_errors[ni, ki] = np.abs(fe - fa).max()

		del interpolater
		gc.collect()

		# scipy interp
		st = time.time()
		interpolater = sp.interpolate.RectBivariateSpline(v, v, f, kx=k, ky=k)
		sp_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater.ev(xo, yo)
		sp_eval_time[ni, ki] = (time.time()-st)*1000
		sp_errors[ni, ki] = np.abs(fe - fa).max()

		del interpolater
		gc.collect()

gc.enable()

fig, ax = plt.subplots(1,1)
ax.plot(ntest, my_errors[:,0], color='black',  label='This, Linear')
ax.plot(ntest, my_errors[:,1], color='blue',   label='This, Cubic')
ax.plot(ntest, my_errors[:,2], color='purple', label='This, Qunitic')
ax.plot(ntest, sp_errors[:,0], color='black',  linestyle='--', label='Scipy, Linear')
ax.plot(ntest, sp_errors[:,1], color='blue',   linestyle='--', label='Scipy, Cubic')
ax.plot(ntest, sp_errors[:,2], color='purple', linestyle='--', label='Scipy, Qunitic')
ax.set_xlabel(r'$n$')
ax.set_ylabel('Maximum Error')
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
ax.set_xlabel(r'$n$')
ax.set_ylabel('Time (ms)')
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
ax.set_xlabel(r'$n$')
ax.set_ylabel('Time (ms)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Evaluation Time')
ax.legend()

fig, ax = plt.subplots(1,1)
ax.plot(ntest, sp_eval_time[:,0]/my_eval_time[:,0], color='black',  label='Linear')
ax.plot(ntest, sp_eval_time[:,1]/my_eval_time[:,1], color='blue',   label='Cubic')
ax.plot(ntest, sp_eval_time[:,2]/my_eval_time[:,2], color='purple', label='Qunitic')
ax.set_xlabel(r'$n$')
ax.set_ylabel('Ratio (scipy/this)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Evaluation Time Ratio')
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

interpolater = interp2d([0.0,0.0], [1.0,1.0], [h,h], f, k=5)
fe = interpolater(xo, yo)
err = np.abs(fe - fa).max()
print('...Error in interpolating to shaped array: {:0.1e}'.format(err))

################################################################################
# Test with periodic boundaries

print('\n----- Testing Periodic Qunitic Interpolation, both directions -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	a = 0.0
	b = 2*np.pi
	v, h = np.linspace(a, b, n, endpoint=False, retstep=True)
	x, y = np.meshgrid(v, v, indexing='ij')
	xo = np.random.rand(*x.shape)*10
	yo = np.random.rand(*x.shape)*10

	test_function = lambda x, y: np.exp(np.sin(x))*np.cos(y)
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d([a,a], [b,b], [h,h], f, k=5, p=[True,True])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Linear Interpolation, x-direction -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	ax, bx = 0.0, 2*np.pi
	ay, by = 0.0, 1.0
	xv, xh = np.linspace(ax, bx, n, endpoint=False, retstep=True)
	yv, yh = np.linspace(ay, by, n, endpoint=True, retstep=True)
	x, y = np.meshgrid(xv, yv, indexing='ij')
	xo = x[:-1, :-1].copy()
	yo = y[:-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh

	test_function = lambda x, y: np.exp(np.sin(x))*np.exp(y)
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d([ax,ay], [bx,by], [xh,yh], f, k=1, p=[True,False])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

print('\n----- Testing Periodic Cubic Interpolation, y-direction -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	ax, bx = 0.0, 1.0
	ay, by = 0.0, 2*np.pi
	xv, xh = np.linspace(ax, bx, n,   endpoint=True, retstep=True)
	yv, yh = np.linspace(ay, by, 2*n, endpoint=False, retstep=True)
	x, y = np.meshgrid(xv, yv, indexing='ij')
	xo = x[:-1, :-1].copy()
	yo = 2*y[:-1, :-1].copy()
	xo += np.random.rand(*xo.shape)*xh
	yo += np.random.rand(*yo.shape)*yh

	test_function = lambda x, y: np.exp(x)*np.exp(np.sin(y))
	f = test_function(x, y)
	fa = test_function(xo, yo)

	interpolater = interp2d([ax,ay], [bx,by], [xh,yh], f, k=3, p=[False,True])
	fe = interpolater(xo, yo)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

