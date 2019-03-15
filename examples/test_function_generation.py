import numpy as np
import fast_interp
import time
import numexpr as ne
from scipy.special import struve, y0

n = 1000000

# function to test evaluation of
# in this case, nasty and singular at the origin
def true_func(x):
	return struve(0, x) - y0(x)
approx_range = [1e-14, 10]

# generate approximation
approx_func5 = fast_interp.FunctionGenerator(true_func, approx_range[0], approx_range[1], k=5)
approx_func7 = fast_interp.FunctionGenerator(true_func, approx_range[0], approx_range[1], k=7)

def random_in(n, a, b):
	x = np.random.rand(n)
	x *= (b-a)
	x += a
	return x

xtest = random_in(n, approx_range[0], approx_range[1])

st = time.time()
ft = true_func(xtest)
true_func_time = time.time() - st

fa = approx_func5(xtest)
st = time.time()
fa5 = approx_func5(xtest)
approx_func5_time = time.time()-st

fa = approx_func7(xtest)
st = time.time()
fa7 = approx_func7(xtest)
approx_func7_time = time.time()-st

reg = np.abs(ft)

reg[reg < 1] = 1
err5 = np.abs(fa5-ft)/reg
err7 = np.abs(fa7-ft)/reg

print('')
print('Error (5):        {:0.1e}'.format(err5.max()))
print('Error (7):        {:0.1e}'.format(err7.max()))
print('True time:        {:0.1f}'.format(true_func_time*1000))
print('Approx time (5):  {:0.1f}'.format(approx_func5_time*1000))
print('Approx time (7):  {:0.1f}'.format(approx_func7_time*1000))

n = 10000
x = np.random.rand(n)
y = np.random.rand(n)
r = np.empty([n, n], dtype=float)
G = np.empty([n, n], dtype=float)
tau = np.ones(n)

def ker(x, y, tau, r, G):
	xT = x[:,None]
	yT = y[:,None]
	ne.evaluate('sqrt((x-xT)**2 + (y-yT)**2)', out=r)
	approx_func5(r, out=G)
	return G.dot(tau)

out = ker(x, y, tau, r, G)
st = time.time()
out = ker(x, y, tau, r, G)
kernel_time = time.time()-st

print('Kernel Eval Time: {:0.1f}'.format(kernel_time*1000))


