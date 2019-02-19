import numpy as np
import scipy as sp
import scipy.interpolate
import time
from fast_interp import interp1d
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import gc
gc.disable()

################################################################################
# Basic error sweep

print('\n----- Testing Error and Timings vs. Scipy -----')

ntest = 10*2**np.arange(18)
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

		x, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
		test_x = (x + np.random.rand(*x.shape)*h)[:-1]

		def test_function(x):
		    return np.exp(x)*np.cos(x) + x**2/(1+x)

		f = test_function(x)
		fa = test_function(test_x)

		# run once to compile numba functions
		interpolater = interp1d(x, f, k=k)
		fe = interpolater(test_x)
		# fast_interp
		st = time.time()
		interpolater = interp1d(x, f, k=k)
		my_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater(test_x)
		my_eval_time[ni, ki] = (time.time()-st)*1000
		my_errors[ni, ki] = np.abs(fe - fa).max()

		# scipy interp
		st = time.time()
		interpolater = sp.interpolate.InterpolatedUnivariateSpline(x, f, k=k)
		sp_setup_time[ni, ki] = (time.time()-st)*1000
		st = time.time()
		fe = interpolater(test_x)
		sp_eval_time[ni, ki] = (time.time()-st)*1000
		sp_errors[ni, ki] = np.abs(fe - fa).max()

		gc.collect()

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
x, h = np.linspace(0, 1, n, endpoint=True, retstep=True)
test_x = np.random.rand(n).reshape(100, 10)

def test_function(x):
    return np.exp(x)*np.cos(x) + x**2/(1+x)

f = test_function(x)
fa = test_function(test_x)

interpolater = interp1d(x, f, k=5)
fe = interpolater(test_x)
err = np.abs(fe - fa).max()
print('...Error in interpolating to shaped array: {:0.1e}'.format(err))

gc.collect()

################################################################################
# Test with periodic boundaries

print('\n----- Testing Periodic Interpolation -----')

for n in [20, 40, 80, 160]:
	print('...for n =', n)
	x, h = np.linspace(0, 1, n, endpoint=False, retstep=True)
	test_x = (x + np.random.rand(*x.shape)*h)[:-1]

	def test_function(x):
	    return np.exp(np.sin(2*np.pi*x))

	f = test_function(x)
	fa = test_function(test_x)

	interpolater = interp1d(x, f, k=5, periodic=True)
	fe = interpolater(test_x)
	err = np.abs(fe - fa).max()
	print('...Error is: {:0.1e}'.format(err))

	gc.collect()


