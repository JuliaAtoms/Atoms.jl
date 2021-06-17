# Atoms.jl

Documentation for Atoms.jl

## A note on usage

Atoms.jl (and the whole JuliaAtoms cluster of libraries) is _research
grade software_. This means, that although we try our best to ensure
that the individual components behave as they should, and we do have a
fair amount of complete calculations that we compare against values
from the literature. However, Atoms.jl is not an _expert system_, in
that if the calculation does not converge, it will not try another
route (such as e.g. [ATSP](https://github.com/compas/atsp-book) or
[Grasp](https://github.com/compas/grasp) would); it also does not
adapt the grid to the problem, this has to be set by the user before
the calculation starts.

Some things to check:

- Is the grid large enough? Too small grids will force all eigenvalues
  to increase.
- Is the grid spacing small enough? Too coarse grids will not be able
  to resolve the finer details of the orbitals, which may or may not
  be a problem, depending on your use-case (i.e. the energies may be
  good enough, but matrix elements between orbitals not, etc).
- If it seems that the optimizer cannot converge to the minimum, but
  instead just "cycles" a few energies, it could be worthwhile to try
  another line-search algorithm,
  e.g. [More–Thuente](https://julianlsolvers.github.io/LineSearches.jl/stable/reference/linesearch.html#LineSearches.MoreThuente)
  instead of the default
  [Hager–Zhang](https://julianlsolvers.github.io/LineSearches.jl/stable/reference/linesearch.html#LineSearches.HagerZhang).
- Try with another basis set, increase or decrease polynomial order if
  using B-splines or FE-DVR, try with different knot-sets or placement
  of finite-elements.
- For point-charge nuclei, there is a singularity at the origin (for
  ``\ell=0``) which needs to be accounted for (in terms of fulfilling
  the boundary conditions). For the finite-differences,
  [CompactBases.jl](https://github.com/JuliaApproximation/CompactBases.jl/)
  provides one-point fixes that approximately satisfy the boundary
  conditions, for B-splines one can place the intervals densely close
  to the origin, for FE-DVR one can increase the polynomial order of
  the first element.
