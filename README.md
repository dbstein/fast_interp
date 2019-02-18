# fast_interp: numba accelerated interpolation on regular grids in 1, 2, and 3 dimensions

This code provides a nearly drop-in replacement for the scipy interpolation functions for *regular* arrays that is significantly faster. Like the scipy.interpolate functions, this function is **asmptotically accurate up to the boundary**. Interpolation in 1D is done by evaluating local Taylor expansions to the required order; derivatives are estimated using finite differences. Some rearrangment of terms and order of evaluation makes the code surprisingly stable. Usage is as follows:

```python
from fast_interp import interp1d

interpolater = interp1d(xv, f, k=3)
out = interper(xo)
```

It also allows the user to specify that any number of axes are periodic; if they are it assumes the last point is not provided and is the same as the first.

## Performance

On my machine with 12 cores, this function varies between being about the same speed as the scipy function (for small evaluations), to being 1000+ times faster for large evaluations (besides getting the parallel and SIMD boost from numba, it actually scales better, since it doesn't have to hunt for the cell that the interpolation point is located in).

While there are other fast interpolation routines (like map_coordinates), and other numba based interpolation routines out there (e.g. [this one](https://github.com/EconForge/interpolation.py)), they are not accurate to the domain boundaries for k > 1, and require a fitting stage. This function requires no fitting; the setup cost, is extremely minimal, only requiring the allocation of a padded array that is slightly larger than the given data. Non-periodic linear interpolation does not require any padding and has effectively no setup cost.

## To do, perhaps:

1. For interpolation in 2D and 3D, k is forced to be the same in the different coordinate directions. It could easily be allowed to be different, although I'm not convinced there's a great use case for this, so I haven't implemented that yet.
2. For cubic and qunitic interpolation, the code allocates a slightly larger array than the specified values and pads it with an extrapolation (or wrapping, if the array is specified to be periodic). This could potentially be eliminated, but at the cost of more complex code. The cost of doing this isn't particularly large if the user is interpolating to *many* points, but will be nontrivial if the user is only interpolating to a few points. Even more simply, one could allow the user to turn this off if they are sure there are no interpolation points within a certain distance of the boundary (h for cubic, 2h for quintic).
3. No extrapolation is supported. It might be convenient to allow extrapolation by a certain safe-ish distance (say, < h), this could be supported by extrapolating the function by one more grid point than it is currently extrapolated.
