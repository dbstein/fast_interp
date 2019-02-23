# fast_interp: numba accelerated interpolation on regular grids in 1, 2, and 3 dimensions

This code provides functionality similar to the scipy.interpolation functions for *smooth functions* defined on *regular* arrays in 1, 2, and 3 dimensions. Like the scipy.interpolate functions (and unlike map_coordinates or some other fast interpolation packages), this function is *asmptotically accurate up to the boundary*, meaning that the interpolation accuracy is second-, fourth-, and sixth-order accurate for k=1, 3, and 5, respectively, even when interpolating to points that are close to the edges of the domains on which the data is defined. It is even asymptotically accurate when extrapolating, although this in general is not recommended as it is numerically unstable.

Unlike the scipy.interpolate functions, this is not based on spline interpolation, but rather the evaluation of local Taylor expansions to the required order, with derivatives estimated using finite differences. Some rearrangement of terms and the order in which things are evaluated makes the code surprisingly stable. The provided data is padded (by local extrapolation, or periodic wrapping when the user specifies) in order to maintain accuracy at the boundary. If near boundary interpolation is not needed, the user can specify this, and the padding step is skipped.

For dimensions that the user specifies are periodic, the interpolater does the correct thing for any input value. For non-periodic dimensions, constant extrapolation is done outside of the specified interpolation region. The user can request that extrapolation is done along a dimension to some distance (specified in units of gridspacing). Although I have attempted to make the computation of this reasonably stable, extrapolation is dangerous, use at your own risk.

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

In the case given above, the y-dimension is specified to be periodic, and the user has specified that extrapolation should be done to a distance xh from the boundary in the x-dimension. Thus this function will provide asymptotically accurate interpolation for x in [-xh, 1+xh] and y in [-Inf, Inf]. For values of xh outside of this region, extrapolation will be constant. The code given above produces an error of 4.53e-06. Of course, if test_x and test_y were numpy arrays, this will return a numpy array of the same shape with the interpolated values. It does not do any kind of broadcasting, or check if you provided different shaped arrays, or any such nicety. There are quite a few examples, in all dimensions, included in the files in the examples folder.

## Performance (on my 12 core machine...)

For fitting, this greatly outperforms the scipy options, since... it doesn't have to fit anything. In the general case, it does allocate and copy a padded array the size of the data, so that's slightly inefficient if you'll only be interpolating to a few points, but its still much cheaper (often orders of magnitude) than the fitting stage of the scipy functions. If the function can avoid making a copy, it will, this happens if all dimensions are periodic, linear with no extrapolation, or the user has requested to ignore close evaluation by setting the variable c. Here is the setup cost in 2D, where copies are required, compared to scipy.RectBivariateSpline:



For extremely small interpolation problems, the provided scipy.interpolate functions are a bit faster. In 2D, this code breaks even on a grid of ~30 by 30, and by ~100 by 100 is about 10 times faster. For a 2000 by 2000 grid this advantage is at least a factor of 100, and can be as much as 1000+. Besides getting the parallel and SIMD boost from numba, the algorithm actually scales better, since on a regular grid locating the points on the grid is an order one operation. You can get a sense of break-even points on your system for 1D and 2D by running the tests in the examples folder.

In terms of error, the algorithm scales in the same way as the scipy.interpolate functions, although the scipy functions provide *slightly* better constants. You can see this by running the provide 1D and 2D tests. The error on this code could probably be improved a bit by making slightly different choices about the points at which finite-differences are computed and how wide the stencils are, but this would require wider padding of the input data and a bit of effort. When the grid spacing becomes fine, the algorithm appears to be slightly more stable than the scipy.interpolate functions, with a bit less digit loss on very fine grids.

## To do, perhaps:

1. For interpolation in 2D and 3D, k is forced to be the same in the different coordinate directions. It could easily be allowed to be different, although I'm not convinced there's a great use case for this, so I haven't implemented it.

