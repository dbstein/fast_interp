import numpy as np
import fast_interp
import time
from scipy.special import struve

n = 100000

# function to test evaluation of
# in this case, nasty and singular at the origin
def true_func(x):
	return struve(-3, x)*np.exp(x/10.0)*np.sin(10*x)/(1 + 0.99*np.cos(x))*np.log(np.sqrt(x+1.0)+1.0)
approx_range = [1e-14, 10]

# generate approximation
approx_func = fast_interp.FunctionGenerator(true_func, approx_range[0], approx_range[1], k=5)

def random_in(n, a, b):
	x = np.random.rand(n)
	x *= (b-a)
	x += a
	return x

xtest = random_in(n, approx_range[0], approx_range[1])

st = time.time()
ft = true_func(xtest)
true_func_time = time.time() - st

fa = approx_func(xtest)
st = time.time()
fa = approx_func(xtest)
approx_func_time = time.time()-st

reg = np.abs(ft)
reg[reg < 1] = 1
err = np.abs(fa-ft)/reg
print('')
print('Error:       {:0.1e}'.format(err.max()))
print('True time:   {:0.1f}'.format(true_func_time*1000))
print('Approx time: {:0.1f}'.format(approx_func_time*1000))
