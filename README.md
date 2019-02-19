# fast_interp: numba accelerated interpolation on regular grids in 1, 2, and 3 dimensions

This code provides a nearly drop-in replacement for the scipy interpolation functions for *smooth functions* defined on *regular* arrays in 1, 2, and 3 dimensions. Like the scipy.interpolate functions (and unlike map_coordinates or some other fast interpolation packages), this function is *asmptotically accurate up to the boundary*, meaning that the interpolation accuracy is second-, fourth-, and sixth-order accurate for k=1, 3, and 5, respectively, even when interpolating to points that are close to the edges of the domains on which the data is defined.

Unlike the scipy.interpolate functions, this is not based on spline interpolation, but rather the evaluation of local Taylor expansions to the required order, with derivatives estimated using finite differences. Some rearrangement of terms and the order in which things are evaluated makes the code surprisingly stable. The provided data is padded (by local extrapolation, or periodic wrapping when the user specifies) in order to maintain accuracy at the boundary. If near boundary interpolation is not needed, the user can specify this, and the padding step is skipped.

```python
from fast_interp import interp1d

interpolater = interp1d(xv, f, k=3)
out = interper(xo)
```

The user can specify that any number of axes are periodic; if they are it assumes the last point is not provided and is the same as the first.

## Performance (on my 12 core machine...)

For fitting, this greatly outperforms the scipy options, since... it doesn't have to fit anything. It does allocate and copy a padded array the size of the data, so that's slightly inefficient if you'll only be interpolating to a few points, but its still much cheaper (often orders of magnitude) than the fitting stage of the scipy functions. Linear interpolation, when no periodic arrays are present, requires no padding, and doesn't incur any startup cost.

For extremely small interpolation problems, the provided scipy.interpolate functions are a bit faster. In 2D, this code breaks even on a grid of ~30 by 30, and by ~100 by 100 is about 10 times faster. For a 2000 by 2000 grid this advantage is at least a factor of 100, and can be as much as 1000+. Besides getting the parallel and SIMD boost from numba, the algorithm actually scales better, since on a regular grid locating the points on the grid is an order one operation.

In terms of error, the algorithm scales in the same way as the scipy.interpolate functions, although the scipy functions provide *slightly* better constants (by about a factor of 2, usually). You can see this by running the provide 1D and 2D tests. The error on this code could probably be improved a bit by making slightly different choices about the points at which finite-differences are computed and how wide the stencils are, but this would require wider padding of the input data and a bit of effort. I might look into it at some point, but I doubt it. When the grid spacing becomes fine, the algorithm appears to be slightly more stable than the scipy.interpolate functions, with a bit less digit loss on very fine grids.

## To do, perhaps:

1. For cubic and qunitic interpolation, the code, by default, allocates a slightly larger array than the specified values and pads it with an extrapolation (or wrapping, if the array is specified to be periodic). The user can override this behavior for speeds sake, if they *know that there are no points close to the domain edges* (within h for cubic, 2h for quintic, and an extra h for linear, cubic, and quintic on the right edge of periodic dimensions). This is not an ideal solution, but fixing it requires more complex numba code (which may be hard to ensure gets compiled to SIMD instructions).
2. For interpolation in 2D and 3D, k is forced to be the same in the different coordinate directions. It could easily be allowed to be different, although I'm not convinced there's a great use case for this, so I haven't implemented it.
3. No extrapolation is supported. It might be convenient to allow extrapolation by a certain safe-ish distance (say, < h), this could be supported by extrapolating the function by one more grid point than it is currently extrapolated.

