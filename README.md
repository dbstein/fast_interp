# fast_interp: numba accelerated interpolation on regular grids in 1, 2, and 3 dimensions

This code provides functionality similar to the scipy.interpolation functions for *smooth functions* defined on *regular* arrays in 1, 2, and 3 dimensions. Like the scipy.interpolate functions (and unlike map_coordinates or some other fast interpolation packages), this function is *asmptotically accurate up to the boundary*, meaning that the interpolation accuracy is second-, fourth-, and sixth-order accurate for k=1, 3, and 5, respectively, even when interpolating to points that are close to the edges of the domains on which the data is defined. It is even asymptotically accurate when extrapolating, although this in general is not recommended as it is numerically unstable. This package also supports k=7 and 9, providing eighth and tenth order accuracy, respectively. These are *use at your own risk*, as high-order interpolation from equispaced points is generally inadvisable.

Unlike the scipy.interpolate functions, this is not based on spline interpolation, but rather the evaluation of local Taylor expansions to the required order, with derivatives estimated using finite differences. Some rearrangement of terms and the order in which things are evaluated makes the code surprisingly fast and stable. The provided data is padded (by local extrapolation, or periodic wrapping when the user specifies) in order to maintain accuracy at the boundary. If near boundary interpolation is not needed, the user can specify this, and the padding step is skipped.

For dimensions that the user specifies are periodic, the interpolater does the correct thing for any input value. For non-periodic dimensions, constant extrapolation is done outside of the specified interpolation region. The user can request that extrapolation is done along a dimension to some distance (specified in units of gridspacing). Although I have attempted to make the computation of this reasonably stable, extrapolation is dangerous, use at your own risk.

In the most recent update, this code fixes a few issues and makes a few improvements:
1. Any of the list-of-float / list-of-int / list-of-bool parameters, such as 'a' for the lower bound of the interpolation regions, can be specified with type-heterogeneity. For example, you should be able to specify a=[0, 1.0, np.pi], or p=[0, True]. These will now all be dumbly typecast to the appropriate type, so unless you do something rather odd, they should do the right thing. Let me know if not.
2. All of these lists are now packaged into numba.typed.List objects, so that the deprecation warnings that numba used to spit out should all be gone. If you have a very old version of numba (pre-typed-Lists), this may not work. Upgrade your numba installation. I have not udpated the below performance diagnostics, but thanks to performance improvements in numba's TypedList implementation these shouldn't have changed much, if at all.
3. Under the hood, the code now compiles both serial and parallel versions, and calls the different versions depending on the size of the vector being interpolated to. The dimension-dependent default switchover is at n=[2000, 400, 100], which seemed reasonable when doing some quick benchmarking; you can adjust this (for each dimension independently), by calling "set_serial_cutoffs(dimension, cutoff)". I.e. if you want 3D interpolation to switch to parallel when the number of points being interpolated to is bigger than 1000, call "fast_interp.set_serial_cutoffs(3, 1000)". If you always want to use a serial version, set cutoff=np.Inf). This change improves the performance when interpolating to a small number of points, although scipy typically still wins for very small numbers of points. I haven't yet updated the timing tests below.
4. The checking on k has been updated to allow k=9 (which was implemented before, but rejected by the checks).
5. A bug associated with a missed index when a value was exactly at or above the edge of the extrapolation region has been fixed.

Usage is as follows:

```python
from fast_interp import interp2d
import numpy as np

nx = 50
ny = 37
xv, xh = np.linspace(0, 1,       nx, endpoint=True,  retstep=True)
yv, yh = np.linspace(0, 2*np.pi, ny, endpoint=False, retstep=True)
x, y = np.meshgrid(xv, yv, indexing='ij')

test_function = lambda x, y: np.exp(x)*np.exp(np.sin(y))
f = test_function(x, y)
test_x = -xh/2.0
test_y = 271.43
fa = test_function(test_x, test_y)

interpolater = interp2d([0,0], [1,2*np.pi], [xh,yh], f, k=5, p=[False,True], e=[1,0])
fe = interpolater(test_x, test_y)

print('Error is: {:0.2e}'.format(np.abs(fe-fa)))
```

In the case given above, the y-dimension is specified to be periodic, and the user has specified that extrapolation should be done to a distance xh from the boundary in the x-dimension. Thus this function will provide asymptotically accurate interpolation for x in [-xh, 1+xh] and y in [-Inf, Inf]. For values of xh outside of this region, extrapolation will be constant. The code given above produces an error of 4.53e-06. If test_x and test_y were numpy arrays, this will return a numpy array of the same shape with the interpolated values. It does not do any kind of broadcasting, or check if you provided different shaped arrays, or any such nicety. There are quite a few examples, in all dimensions, included in the files in the examples folder.

## Performance (on my 12 core machine...)

For fitting, this greatly outperforms the scipy options, since... it doesn't have to fit anything. In the general case, it does allocate and copy a padded array the size of the data, so that's slightly inefficient if you'll only be interpolating to a few points, but its still much cheaper (often orders of magnitude) than the fitting stage of the scipy functions. If the function can avoid making a copy, it will, this happens if all dimensions are periodic, linear with no extrapolation, or the user has requested to ignore close evaluation by setting the variable c. Here is the setup cost in 2D, where copies are required, compared to scipy.interpolate.RectBivariateSpline:

![Solution](fast_interp_setup.png?raw=true "Title")

For small interpolation problems, the provided scipy.interpolate functions are a bit faster. In 2D, this code breaks even on a grid of ~30 by 30, and by ~100 by 100 is about 10 times faster. For a 2000 by 2000 grid this advantage is at least a factor of 100, and can be as much as 1000+. Besides getting the parallel and SIMD boost from numba, the algorithm actually scales better, since on a regular grid locating the points on the grid is an order one operation. You can get a sense of break-even points on your system for 1D and 2D by running the tests in the examples folder. Shown below are timings in 2D, on an n by n grid, interpolating to n^2 points, comparing scipy and fast_interp:

![Solution](fast_interp_eval.png?raw=true "Title")

Performance on this system approximately 20,000,000 points per second per core. The ratio between scipy.interpolate.RectBivariateSpline evaluation time and fast_interp evaluation time:

![Solution](fast_interp_ratio.png?raw=true "Title")

In terms of error, the algorithm scales in the same way as the scipy.interpolate functions, although the scipy functions provide *slightly* better constants. The error on this code could probably be improved a bit by making slightly different choices about the points at which finite-differences are computed and how wide the stencils are, but this would require wider padding of the input data. When the grid spacing becomes fine, the algorithm appears to be slightly more stable than the scipy.interpolate functions, with a bit less digit loss on very fine grids. Here is an error comparison in 2D:

![Solution](fast_interp_error.png?raw=true "Title")

A final consideration is numerical stability. In the following plot, I show a test of interpolation accuracy when some random noise is added to the function that is being interpolated. This test is done in 1D, so I can go to enormously large n to really push the bounds of stability. The gray line shows the level of noise that was added; even for k=5 the algorithm is stable for all n (and for all k, more stable than the scipy.interpolate) functions:

![Solution](fast_interp_stability.png?raw=true "Title")
