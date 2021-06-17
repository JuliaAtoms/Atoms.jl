# Examples

To find the ground state of various atoms, we need a radial grid, a
set of electronic configurations, and an atomic nucleus. Atoms.jl does
not yet support multi-configurational Hartree–Fock (it is implemented
but not properly working, yet), and furthermore
[AngularMomentumAlgebra.jl](https://github.com/JuliaAtoms/AngularMomentumAlgebra.jl)
can only derive energy expressions for [Slater
determinants](https://en.wikipedia.org/wiki/Slater_determinant) at the
moment, i.e. [configuration state
functions](https://en.wikipedia.org/wiki/Configuration_state_function)
are not yet supported.

First we load some required packages:
```julia
using Atoms
using AtomicLevels
using SCF

using LinearAlgebra
using CompactBases
using IntervalSets
```

We can use the following function to quickly set up the radial grid:
```julia
function get_atom_grid(grid_type, rₘₐₓ, ρ, nucleus; fedvr_order=10)
    Z = charge(nucleus)
    amend_order=nucleus isa PointCharge # For correct boundary
                                        # conditions at r=0.

    if grid_type == :fedvr
        # FEDVR is more accurate, but can be expensive to use
        N = max(ceil(Int, rₘₐₓ/(ρ*fedvr_order)),2)
        t = range(0.0, stop=rₘₐₓ, length=N)
        amended_order = vcat(fedvr_order+5, fill(fedvr_order,length(t)-2))
        FEDVR(t, amend_order ? amended_order : fedvr_order)[:,2:end-1]
    else
        # Finite-differences are much lighter, but may require very
        # fine grids to converge.
        N = ceil(Int, rₘₐₓ/ρ + 1/2)
        StaggeredFiniteDifferences(N, ρ)
    end
end
```

## Hydrogen

Since hydrogen can be solved exactly, no Hartree–Fock iterations are
needed, and the orbital is in fact initialized to its hydrogenic shape
upon construction:

```julia-repl
julia> nucleus = pc"H"
Z = 1 [hydrogen]

julia> R = get_atom_grid(:fedvr, 10.0, 0.1, nucleus)
FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87

julia> gst = ground_state(nucleus)
1s

julia> atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
Atom{Float64}(R=FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87; Z = 1 [hydrogen]; 1 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α
```

![Hydrogen orbital](figures/hydrogen.svg)

## Helium

The setup is very similar to that of [Hydrogen](@ref), but since
helium has two electrons, it is a three-body problem which cannot be
solved exactly. Instead, we make the mean-field approximation, where
every electron is assumed to move independently in the potential
formed from the nucleus and _all other electrons_:

1. We first make an initial guess, simply solving the hydrogen
   problem, but for the helium atomic nucleus with ``Z=2``,
2. We then form an electron–electron repulsion potential for each
   electron, constructed from the other electron,
3. We use these potentials to solve for new electron orbitals.
4. Repeat 2–3 until convergence.
5. Additionally, when the solution is converged enough, we can switch
   to non-linear optimization via
   [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl/), which is
   usually much faster. This is not strictly necessary for helium, but
   very helpful for heavier elements.

The initial setup is very similar to hydrogen:
```julia-repl
julia> nucleus = pc"He"
Z = 2 [helium]

julia> R = get_atom_grid(:fedvr, 10.0, 0.1, nucleus)
FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87

julia> gst = ground_state(nucleus)
1s²

julia> atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
Atom{Float64}(R=FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87; Z = 2 [helium]; 2 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β
```
Then we create a Fock operator, which will automatically form the
energy expression and from that derive the orbital Hartree–Fock
equations:
```julia-repl
julia> fock = Fock(atom)
Fock operator with
- quantum system: Atom{Float64}(R=FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87; Z = 2 [helium]; 2 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -0.7500000039950884 Ha = -20.408250108710348 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -0.7500000039950884 Ha = -20.408250108710348 eV
```
Note that the orbital energies pre-optimization corresponds to that of
a ``1s`` electron in a ``Z=2`` potential. We are now ready to perform
the optimization:
```julia-repl
julia> optimize!(fock)
[ Info: Performing initial SCF iterations
Self-Consistent-Field calculation of
- Atom{Float64}(R=FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87; Z = 2 [helium]; 2 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β
- Maximum amount of iterations: 200
- Stopping tolerance: 1.00×10⁻³

Iteration Tolerance  Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[  1/200] 3.84×10⁻²    -2.75000 Ha =  -74.83025 eV   +2.56003 Ha   -5.31003 Ha -2.07421 (7.42×10⁻² )
[  2/200] 2.08×10⁻¹²   -2.85247 Ha =  -77.61859 eV   +2.56003 Ha   -5.41251 Ha -2.11423 (1.14×10⁻¹ )

Finished in 0.06665205955505371 seconds
┌───────┬────────────────────┬───────────────────┬───────────────────┐
│ i – j │              ⟨i|j⟩ │           ⟨i|𝔣|j⟩ │           ⟨j|𝔣|i⟩ │
├───────┼────────────────────┼───────────────────┼───────────────────┤
│ 2 – 2 │ 0.9999999999999999 │ -0.94488715344265 │ -0.94488715344265 │
│ 1 – 1 │ 0.9999999999999999 │ -0.94488715344265 │ -0.94488715344265 │
└───────┴────────────────────┴───────────────────┴───────────────────┘
Iteration   |g|        Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[   1/1000] 3.15×10⁻¹    -2.85247 Ha =  -77.61859 eV   +2.56003 Ha   -5.41251 Ha -2.11423 (1.14×10⁻¹ )
[   2/1000] 5.62×10⁻²    -2.85775 Ha =  -77.76214 eV   +2.77906 Ha   -5.63680 Ha -2.02832 (2.83×10⁻² )
[   3/1000] 1.78×10⁻¹    -2.85867 Ha =  -77.78732 eV   +2.79272 Ha   -5.65139 Ha -2.02362 (2.36×10⁻² )
[   4/1000] 2.87×10⁻¹    -2.85877 Ha =  -77.79001 eV   +2.79551 Ha   -5.65428 Ha -2.02263 (2.26×10⁻² )
[   5/1000] 2.43×10⁻¹    -2.85889 Ha =  -77.79333 eV   +2.79827 Ha   -5.65716 Ha -2.02166 (2.17×10⁻² )
[   6/1000] 1.40×10⁻¹    -2.85922 Ha =  -77.80224 eV   +2.80846 Ha   -5.66768 Ha -2.01807 (1.81×10⁻² )
[   7/1000] 3.56×10⁻¹    -2.85932 Ha =  -77.80504 eV   +2.81610 Ha   -5.67542 Ha -2.01535 (1.53×10⁻² )
[   8/1000] 1.06×10⁻¹    -2.85960 Ha =  -77.81245 eV   +2.83038 Ha   -5.68998 Ha -2.01032 (1.03×10⁻² )
[   9/1000] 1.61×10⁻¹    -2.85981 Ha =  -77.81828 eV   +2.82407 Ha   -5.68388 Ha -2.01265 (1.27×10⁻² )
[  10/1000] 1.70×10⁻¹    -2.85991 Ha =  -77.82088 eV   +2.82151 Ha   -5.68142 Ha -2.01361 (1.36×10⁻² )
⁞
[ 120/1000] 3.39×10⁻⁸    -2.86168 Ha =  -77.86917 eV   +2.86168 Ha   -5.72336 Ha -2.00000 (1.00×10⁻⁸ )
[ 121/1000] 4.03×10⁻⁹    -2.86168 Ha =  -77.86917 eV   +2.86168 Ha   -5.72336 Ha -2.00000 (1.00×10⁻⁸ )
  2.081946 seconds (3.29 M allocations: 256.888 MiB, 2.89% gc time)
 * Status: success

 * Candidate solution
    Minimizer: [1.59e-02, 6.35e-02, 1.35e-01,  ...]
    Minimum:   -2.861680e+00

 * Found with
    Algorithm:     BFGS
    Initial Point: [1.49e-02, 5.98e-02, 1.27e-01,  ...]

 * Convergence measures
    |x - x'|               = 5.13e-10 ≰ 0.0e+00
    |x - x'|/|x'|          = 1.59e-09 ≰ 0.0e+00
    |f(x) - f(x')|         = 1.02e-14 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 3.57e-15 ≰ 0.0e+00
    |g(x)|                 = 4.03e-09 ≤ 1.0e-08

 * Work counters
    Seconds run:   2  (vs limit Inf)
    Iterations:    120
    f(x) calls:    243
    ∇f(x) calls:   244

Finished in 2.150846004486084 seconds
┌───────┬────────────────────┬─────────────────────┬─────────────────────┐
│ i – j │              ⟨i|j⟩ │             ⟨i|𝔣|j⟩ │             ⟨j|𝔣|i⟩ │
├───────┼────────────────────┼─────────────────────┼─────────────────────┤
│ 2 – 2 │ 0.9999999999999997 │ -0.9179555568639189 │ -0.9179555568639189 │
│ 1 – 1 │ 1.0000000000000002 │ -0.9179555568624107 │ -0.9179555568624107 │
└───────┴────────────────────┴─────────────────────┴─────────────────────┘
Fock operator with
- quantum system: Atom{Float64}(R=FEDVR{Float64} basis with 9 elements on 0.0..10.0, restricted to basis functions 2..86 ⊂ 1..87; Z = 2 [helium]; 2 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -0.9179555568624103 Ha = -24.978488657783043 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -0.9179555568639188 Ha = -24.978488657824094 eV
```
The first section is the self-consistent iteration procedure described
above, then the non-linear optimization refines the solution. The
Hartree–Fock limit for helium is ``-2.8616800\;\textrm{Ha}``, which we achieved!

![Helium orbitals](figures/helium.svg)

Comparing with hydrogen, we see that the 1s orbitals are more
contracted in helium, due to the larger nuclear charge. However, they
are not as contracted as a single 1s electron in a ``Z=2`` potential
would be, due to the screening from the other electron.

## Beryllium

In order to make the self-consistent iterations more stable, we
introduce the successive-relaxation parameter ``\omega``; if ``w_i``
is the previous solution, and ``\tilde{w}`` the candidate solution,
the new values are calculated as
```math
w_{i+1} = (1-ω)\tilde{w} + ωw_i
```
For ``\omega=0``, the candidate solution is chosen, for all other
values in the range ``(0,1)``, a mixture is chosen. This helps avoid
oscillatory behaviour around local minima.

Additionally, since the ``1s`` and ``2s`` orbitals are non-orthogonal
in the angular coordinates, orthogonality has to be enforced along the
radial coordinate. This is accomplished via projectors in the
self-consistent iterations together with explicit orbital rotations,
to keep the Fock operator diagonal. In the non-linear optimization,
optimization on Riemannian manifolds is used.

### FEDVR

```julia-repl
julia> nucleus = pc"Be"
Z = 4 [beryllium]

julia> R = get_atom_grid(:fedvr, 15.0, 0.1, nucleus)
FEDVR{Float64} basis with 14 elements on 0.0..15.0, restricted to basis functions 2..131 ⊂ 1..132

julia> gst = ground_state(nucleus)
1s² 2s²

julia> atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
Atom{Float64}(R=FEDVR{Float64} basis with 14 elements on 0.0..15.0, restricted to basis functions 2..131 ⊂ 1..132; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β

julia> fock = Fock(atom)
Fock operator with
- quantum system: Atom{Float64}(R=FEDVR{Float64} basis with 14 elements on 0.0..15.0, restricted to basis functions 2..131 ⊂ 1..132; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩ … - 1r⁻¹×Y⁰(2s₀α,1s₀α)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -3.9087791916120778 Ha = -106.36179058295625 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩ … + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀β⟩ - 1r⁻¹×Y⁰(2s₀β,1s₀β)|2s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -3.908779191612078 Ha = -106.36179058295626 eV

  - Hartree–Fock equation: E|2s₀α⟩ = OrbitalEquation(2s₀α):
  [1, 1]  =  + 1ĥ₀|2s₀α⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀α⟩ … + 1r⁻¹×Y⁰(1s₀β,1s₀β)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|2s₀α⟩

⟨2s₀α| 𝓗 |2s₀α⟩ = 0.19278329234852803 Ha = 5.245826168095796 eV

  - Hartree–Fock equation: E|2s₀β⟩ = OrbitalEquation(2s₀β):
  [1, 1]  =  + 1ĥ₀|2s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀β⟩ … - 1r⁻¹×Y⁰(1s₀β,2s₀β)|1s₀β⟩ + 1r⁻¹×Y⁰(2s₀α,2s₀α)|2s₀β⟩

⟨2s₀β| 𝓗 |2s₀β⟩ = 0.19278329234852803 Ha = 5.245826168095796 eV

julia> optimize!(fock,ω=0.999,ωmax=1-1e-3)
[ Info: Performing initial SCF iterations
Self-Consistent-Field calculation of
- Atom{Float64}(R=FEDVR{Float64} basis with 14 elements on 0.0..15.0, restricted to basis functions 2..131 ⊂ 1..132; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- Maximum amount of iterations: 200
- Stopping tolerance: 1.00×10⁻³
- Successive relaxation: ω = 0.999

Iteration Tolerance  1-ω         Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[  1/200] 4.30×10⁻¹  1.00×10⁻³   -13.71600 Ha = -373.22596 eV  +19.99226 Ha  -33.70825 Ha -1.68607 (3.14×10⁻¹ )
[  2/200] 4.29×10⁻¹  1.00×10⁻³   -13.71758 Ha = -373.26911 eV  +19.98485 Ha  -33.70243 Ha -1.68640 (3.14×10⁻¹ ) R
[  3/200] 4.28×10⁻¹  1.00×10⁻³   -13.71913 Ha = -373.31132 eV  +19.97745 Ha  -33.69658 Ha -1.68673 (3.13×10⁻¹ ) R
[  4/200] 4.27×10⁻¹  1.00×10⁻³   -13.72068 Ha = -373.35344 eV  +19.97006 Ha  -33.69074 Ha -1.68706 (3.13×10⁻¹ ) R
[  5/200] 4.25×10⁻¹  1.00×10⁻³   -13.72222 Ha = -373.39546 eV  +19.96269 Ha  -33.68491 Ha -1.68739 (3.13×10⁻¹ ) R
[  6/200] 4.24×10⁻¹  1.00×10⁻³   -13.72377 Ha = -373.43738 eV  +19.95533 Ha  -33.67909 Ha -1.68772 (3.12×10⁻¹ ) R
[  7/200] 4.23×10⁻¹  1.00×10⁻³   -13.72530 Ha = -373.47920 eV  +19.94798 Ha  -33.67328 Ha -1.68805 (3.12×10⁻¹ ) R
[  8/200] 4.21×10⁻¹  1.00×10⁻³   -13.72684 Ha = -373.52092 eV  +19.94064 Ha  -33.66748 Ha -1.68838 (3.12×10⁻¹ ) R
[  9/200] 4.20×10⁻¹  1.00×10⁻³   -13.72837 Ha = -373.56255 eV  +19.93332 Ha  -33.66168 Ha -1.68871 (3.11×10⁻¹ ) R
[ 10/200] 4.18×10⁻¹  1.01×10⁻¹   -13.72989 Ha = -373.60408 eV  +19.92600 Ha  -33.65590 Ha -1.68904 (3.11×10⁻¹ ) R
[ 11/200] 4.17×10⁻¹  1.91×10⁻¹   -13.73141 Ha = -373.64552 eV  +19.20938 Ha  -32.94079 Ha -1.71483 (2.85×10⁻¹ ) R
[ 12/200] 3.16×10⁻¹  2.72×10⁻¹   -13.87507 Ha = -377.55466 eV  +18.05594 Ha  -31.93101 Ha -1.76845 (2.32×10⁻¹ ) R
[ 13/200] 1.30×10⁻¹  3.45×10⁻¹   -14.08097 Ha = -383.15728 eV  +16.85133 Ha  -30.93230 Ha -1.83560 (1.64×10⁻¹ ) R
[ 14/200] 1.91×10⁻²  4.10×10⁻¹   -14.26919 Ha = -388.27893 eV  +15.86314 Ha  -30.13233 Ha -1.89952 (1.00×10⁻¹ ) R
[ 15/200] 6.65×10⁻²  3.45×10⁻¹   -14.40575 Ha = -391.99490 eV  +15.18855 Ha  -29.59430 Ha -1.94846 (5.15×10⁻² ) R
[ 16/200] 5.48×10⁻²  2.72×10⁻¹   -14.48906 Ha = -394.26184 eV  +14.90772 Ha  -29.39678 Ha -1.97192 (2.81×10⁻² ) R
[ 17/200] 3.91×10⁻²  1.91×10⁻¹   -14.52224 Ha = -395.16458 eV  +14.79295 Ha  -29.31518 Ha -1.98170 (1.83×10⁻² ) R
[ 18/200] 2.92×10⁻²  1.01×10⁻¹   -14.53737 Ha = -395.57624 eV  +14.74264 Ha  -29.28001 Ha -1.98608 (1.39×10⁻² ) R
[ 19/200] 2.38×10⁻²  1.00×10⁻³   -14.54466 Ha = -395.77469 eV  +14.72277 Ha  -29.26743 Ha -1.98790 (1.21×10⁻² ) R
[ 20/200] 2.15×10⁻²  1.00×10⁻³   -14.54770 Ha = -395.85739 eV  +14.72260 Ha  -29.27030 Ha -1.98812 (1.19×10⁻² ) R
[ 21/200] 2.14×10⁻²  1.00×10⁻³   -14.54772 Ha = -395.85812 eV  +14.72244 Ha  -29.27016 Ha -1.98813 (1.19×10⁻² ) R
[ 22/200] 2.14×10⁻²  1.00×10⁻³   -14.54775 Ha = -395.85885 eV  +14.72227 Ha  -29.27002 Ha -1.98815 (1.19×10⁻² ) R
[ 23/200] 2.14×10⁻²  1.00×10⁻³   -14.54778 Ha = -395.85958 eV  +14.72210 Ha  -29.26988 Ha -1.98816 (1.18×10⁻² ) R
[ 24/200] 2.14×10⁻²  1.01×10⁻¹   -14.54780 Ha = -395.86031 eV  +14.72194 Ha  -29.26974 Ha -1.98817 (1.18×10⁻² ) R
[ 25/200] 2.14×10⁻²  1.91×10⁻¹   -14.54783 Ha = -395.86104 eV  +14.70519 Ha  -29.25302 Ha -1.98930 (1.07×10⁻² ) R
[ 26/200] 1.92×10⁻²  2.72×10⁻¹   -14.55051 Ha = -395.93392 eV  +14.67672 Ha  -29.22723 Ha -1.99140 (8.60×10⁻³ ) R
[ 27/200] 1.56×10⁻²  3.45×10⁻¹   -14.55499 Ha = -396.05594 eV  +14.64443 Ha  -29.19943 Ha -1.99389 (6.11×10⁻³ ) R
[ 28/200] 1.14×10⁻²  4.10×10⁻¹   -14.56005 Ha = -396.19352 eV  +14.61572 Ha  -29.17577 Ha -1.99619 (3.81×10⁻³ ) R
[ 29/200] 7.52×10⁻³  4.69×10⁻¹   -14.56461 Ha = -396.31766 eV  +14.59479 Ha  -29.15940 Ha -1.99793 (2.07×10⁻³ ) R
[ 30/200] 4.45×10⁻³  5.22×10⁻¹   -14.56810 Ha = -396.41260 eV  +14.58217 Ha  -29.15027 Ha -1.99904 (9.65×10⁻⁴ )
[ 31/200] 2.37×10⁻³  5.70×10⁻¹   -14.57042 Ha = -396.47578 eV  +14.57594 Ha  -29.14636 Ha -1.99962 (3.78×10⁻⁴ )
[ 32/200] 1.13×10⁻³  6.13×10⁻¹   -14.57178 Ha = -396.51281 eV  +14.57357 Ha  -29.14535 Ha -1.99988 (1.22×10⁻⁴ )
[ 33/200] 4.87×10⁻⁴  6.52×10⁻¹   -14.57249 Ha = -396.53204 eV  +14.57297 Ha  -29.14546 Ha -1.99997 (3.26×10⁻⁵ )

Finished in 61.561668157577515 seconds
┌───────┬───────────────────────┬───────────────────────┬────────────────────────┐
│ i – j │                 ⟨i|j⟩ │               ⟨i|𝔣|j⟩ │                ⟨j|𝔣|i⟩ │
├───────┼───────────────────────┼───────────────────────┼────────────────────────┤
│ 2 – 2 │    0.9999956866353903 │    -4.732776785129098 │     -4.732776785129083 │
│ 2 – 4 │   9.8612728676473e-10 │ 0.0002050777971931087 │ 0.00020507485106526969 │
│ 4 – 4 │    0.9997331924711604 │   -0.3093444536176635 │   -0.30934445361767793 │
│ 1 – 1 │    0.9999956866354134 │    -4.732776784904216 │    -4.7327767849042015 │
│ 1 – 3 │ 9.861345986044206e-10 │ 0.0002050780474937991 │ 0.00020507510136494084 │
│ 3 – 3 │    0.9997331924496939 │  -0.30934445298883984 │   -0.30934445298885427 │
└───────┴───────────────────────┴───────────────────────┴────────────────────────┘
Iteration   |g|        Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[   1/1000] 2.12×10⁰    -14.57302 Ha = -396.54653 eV  +14.57297 Ha  -29.14599 Ha -2.00000 (3.88×10⁻⁶ )
[   2/1000] 3.51×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57379 Ha  -29.14682 Ha -1.99995 (5.29×10⁻⁵ )
[   3/1000] 3.55×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57380 Ha  -29.14682 Ha -1.99995 (5.33×10⁻⁵ )
[   4/1000] 9.83×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57380 Ha  -29.14682 Ha -1.99995 (5.30×10⁻⁵ )
[   5/1000] 7.69×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57375 Ha  -29.14677 Ha -1.99995 (4.96×10⁻⁵ )
[   6/1000] 4.17×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57373 Ha  -29.14676 Ha -1.99995 (4.88×10⁻⁵ )
[   7/1000] 3.84×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57372 Ha  -29.14674 Ha -1.99995 (4.78×10⁻⁵ )
[   8/1000] 3.62×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57371 Ha  -29.14674 Ha -1.99995 (4.75×10⁻⁵ )
[   9/1000] 3.08×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57367 Ha  -29.14670 Ha -1.99996 (4.47×10⁻⁵ )
[  10/1000] 8.47×10⁻⁴   -14.57302 Ha = -396.54653 eV  +14.57366 Ha  -29.14669 Ha -1.99996 (4.40×10⁻⁵ )
⁞
[ 215/1000] 1.05×10⁻⁸   -14.57302 Ha = -396.54653 eV  +14.57303 Ha  -29.14605 Ha -2.00000 (2.46×10⁻⁷ )
[ 216/1000] 8.73×10⁻⁹   -14.57302 Ha = -396.54653 eV  +14.57303 Ha  -29.14605 Ha -2.00000 (2.46×10⁻⁷ )
 55.969735 seconds (45.75 M allocations: 4.256 GiB, 2.00% gc time)
 * Status: success

 * Candidate solution
    Minimizer: [4.47e-02, 1.65e-01, 3.10e-01,  ...]
    Minimum:   -1.457302e+01

 * Found with
    Algorithm:     BFGS
    Initial Point: [4.47e-02, 1.65e-01, 3.10e-01,  ...]

 * Convergence measures
    |x - x'|               = 5.34e-10 ≰ 0.0e+00
    |x - x'|/|x'|          = 1.19e-09 ≰ 0.0e+00
    |f(x) - f(x')|         = 4.62e-14 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 3.17e-15 ≰ 0.0e+00
    |g(x)|                 = 8.73e-09 ≤ 1.0e-08

 * Work counters
    Seconds run:   56  (vs limit Inf)
    Iterations:    215
    f(x) calls:    434
    ∇f(x) calls:   435

Finished in 117.59166884422302 seconds
┌───────┬────────────────────────┬────────────────────────┬────────────────────────┐
│ i – j │                  ⟨i|j⟩ │                ⟨i|𝔣|j⟩ │                ⟨j|𝔣|i⟩ │
├───────┼────────────────────────┼────────────────────────┼────────────────────────┤
│ 2 – 2 │     0.9999999999999999 │    -4.7326681293957815 │    -4.7326681293957815 │
│ 2 – 4 │  -6.94182183900668e-18 │  0.0001925532657846884 │  0.0001925504372876625 │
│ 4 – 4 │     1.0000000000000002 │   -0.30926888559986515 │   -0.30926888559986515 │
│ 1 – 1 │     0.9999999999999999 │     -4.732668129395627 │     -4.732668129395627 │
│ 1 – 3 │ -1.226564706038428e-17 │ 0.00019255346268659193 │ 0.00019255063418685092 │
│ 3 – 3 │     0.9999999999999998 │   -0.30926888559996896 │   -0.30926888559996896 │
└───────┴────────────────────────┴────────────────────────┴────────────────────────┘
Fock operator with
- quantum system: Atom{Float64}(R=FEDVR{Float64} basis with 14 elements on 0.0..15.0, restricted to basis functions 2..131 ⊂ 1..132; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩ … - 1r⁻¹×Y⁰(2s₀α,1s₀α)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -4.732668129395628 Ha = -128.78063246898444 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩ … + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀β⟩ - 1r⁻¹×Y⁰(2s₀β,1s₀β)|2s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -4.7326681293957815 Ha = -128.7806324689886 eV

  - Hartree–Fock equation: E|2s₀α⟩ = OrbitalEquation(2s₀α):
  [1, 1]  =  + 1ĥ₀|2s₀α⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀α⟩ … + 1r⁻¹×Y⁰(1s₀β,1s₀β)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|2s₀α⟩

⟨2s₀α| 𝓗 |2s₀α⟩ = -0.30926888559996896 Ha = -8.415515646060754 eV

  - Hartree–Fock equation: E|2s₀β⟩ = OrbitalEquation(2s₀β):
  [1, 1]  =  + 1ĥ₀|2s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀β⟩ … - 1r⁻¹×Y⁰(1s₀β,2s₀β)|1s₀β⟩ + 1r⁻¹×Y⁰(2s₀α,2s₀α)|2s₀β⟩

⟨2s₀β| 𝓗 |2s₀β⟩ = -0.30926888559986515 Ha = -8.41551564605793 eV
```

![Beryllium FEDVR orbitals](figures/beryllium-fedvr.svg)

### Finite-differences

We can repeat the same calculation using finite-differences instead of
FEDVR. In this case, we have to switch from LOBPCG to Arnoldi
iterations when solving the self-consistent problem, since the former
is prone to errors in the Cholesky factorization if the grid is too
coarse, whereas the latter struggle with too fine grids:

```julia-repl
julia> nucleus = pc"Be"
Z = 4 [beryllium]

julia> R = get_atom_grid(:fd, 15.0, 0.1, nucleus)
Radial finite differences basis {Float64} on 0.0..15.15 with 151 points spaced by ρ = 0.1

julia> gst = ground_state(nucleus)
1s² 2s²

julia> atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
Atom{Float64}(R=Radial finite differences basis {Float64} on 0.0..15.15 with 151 points spaced by ρ = 0.1; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β

julia> fock = Fock(atom)
Fock operator with
- quantum system: Atom{Float64}(R=Radial finite differences basis {Float64} on 0.0..15.15 with 151 points spaced by ρ = 0.1; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩ … - 1r⁻¹×Y⁰(2s₀α,1s₀α)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -3.9701516322712 Ha = -108.03179606573161 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩ … + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀β⟩ - 1r⁻¹×Y⁰(2s₀β,1s₀β)|2s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -3.9701516322712 Ha = -108.03179606573161 eV

  - Hartree–Fock equation: E|2s₀α⟩ = OrbitalEquation(2s₀α):
  [1, 1]  =  + 1ĥ₀|2s₀α⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀α⟩ … + 1r⁻¹×Y⁰(1s₀β,1s₀β)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|2s₀α⟩

⟨2s₀α| 𝓗 |2s₀α⟩ = 0.17124757357291542 Ha = 4.659817724492601 eV

  - Hartree–Fock equation: E|2s₀β⟩ = OrbitalEquation(2s₀β):
  [1, 1]  =  + 1ĥ₀|2s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀β⟩ … - 1r⁻¹×Y⁰(1s₀β,2s₀β)|1s₀β⟩ + 1r⁻¹×Y⁰(2s₀α,2s₀α)|2s₀β⟩

⟨2s₀β| 𝓗 |2s₀β⟩ = 0.17124757357291542 Ha = 4.659817724492601 eV


julia> optimize!(fock,ω=0.999,ωmax=1-1e-3,scf_method=:arnoldi)
[ Info: Performing initial SCF iterations
Self-Consistent-Field calculation of
- Atom{Float64}(R=Radial finite differences basis {Float64} on 0.0..15.15 with 151 points spaced by ρ = 0.1; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- Maximum amount of iterations: 200
- Stopping tolerance: 1.00×10⁻³
- Successive relaxation: ω = 0.999

Iteration Tolerance  1-ω         Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[  1/200] 4.16×10⁻¹  1.00×10⁻³   -13.79823 Ha = -375.46366 eV  +20.83039 Ha  -34.62862 Ha -1.66241 (3.38×10⁻¹ )
[  2/200] 4.15×10⁻¹  1.00×10⁻³   -13.79976 Ha = -375.50517 eV  +20.82292 Ha  -34.62268 Ha -1.66272 (3.37×10⁻¹ ) R
[  3/200] 4.13×10⁻¹  1.00×10⁻³   -13.80125 Ha = -375.54581 eV  +20.81546 Ha  -34.61671 Ha -1.66303 (3.37×10⁻¹ ) R
[  4/200] 4.12×10⁻¹  1.00×10⁻³   -13.80274 Ha = -375.58636 eV  +20.80802 Ha  -34.61076 Ha -1.66334 (3.37×10⁻¹ ) R
[  5/200] 4.11×10⁻¹  1.00×10⁻³   -13.80423 Ha = -375.62682 eV  +20.80059 Ha  -34.60481 Ha -1.66365 (3.36×10⁻¹ ) R
[  6/200] 4.09×10⁻¹  1.00×10⁻³   -13.80571 Ha = -375.66718 eV  +20.79316 Ha  -34.59887 Ha -1.66395 (3.36×10⁻¹ ) R
[  7/200] 4.08×10⁻¹  1.00×10⁻³   -13.80719 Ha = -375.70745 eV  +20.78575 Ha  -34.59295 Ha -1.66426 (3.36×10⁻¹ ) R
[  8/200] 4.07×10⁻¹  1.00×10⁻³   -13.80867 Ha = -375.74763 eV  +20.77836 Ha  -34.58702 Ha -1.66457 (3.35×10⁻¹ ) R
[  9/200] 4.05×10⁻¹  1.00×10⁻³   -13.81014 Ha = -375.78772 eV  +20.77097 Ha  -34.58111 Ha -1.66488 (3.35×10⁻¹ ) R
[ 10/200] 4.04×10⁻¹  1.01×10⁻¹   -13.81161 Ha = -375.82772 eV  +20.76359 Ha  -34.57520 Ha -1.66518 (3.35×10⁻¹ ) R
[ 11/200] 4.03×10⁻¹  1.91×10⁻¹   -13.81308 Ha = -375.86763 eV  +20.04038 Ha  -33.85346 Ha -1.68926 (3.11×10⁻¹ ) R
[ 12/200] 3.05×10⁻¹  2.72×10⁻¹   -13.95161 Ha = -379.63715 eV  +18.86506 Ha  -32.81667 Ha -1.73955 (2.60×10⁻¹ ) R
[ 13/200] 1.27×10⁻¹  3.45×10⁻¹   -14.15161 Ha = -385.07934 eV  +17.62484 Ha  -31.77645 Ha -1.80294 (1.97×10⁻¹ ) R
[ 14/200] 1.70×10⁻²  4.10×10⁻¹   -14.33537 Ha = -390.07973 eV  +16.61120 Ha  -30.94657 Ha -1.86299 (1.37×10⁻¹ ) R
[ 15/200] 6.41×10⁻²  3.45×10⁻¹   -14.46760 Ha = -393.67792 eV  +15.93288 Ha  -30.40048 Ha -1.90803 (9.20×10⁻² ) R
[ 16/200] 5.34×10⁻²  2.72×10⁻¹   -14.54737 Ha = -395.84854 eV  +15.65666 Ha  -30.20403 Ha -1.92915 (7.09×10⁻² ) R
[ 17/200] 3.81×10⁻²  1.91×10⁻¹   -14.57934 Ha = -396.71832 eV  +15.54476 Ha  -30.12410 Ha -1.93789 (6.21×10⁻² ) R
[ 18/200] 2.84×10⁻²  1.01×10⁻¹   -14.59405 Ha = -397.11857 eV  +15.49565 Ha  -30.08970 Ha -1.94182 (5.82×10⁻² ) R
[ 19/200] 2.32×10⁻²  1.00×10⁻³   -14.60117 Ha = -397.31232 eV  +15.47621 Ha  -30.07738 Ha -1.94346 (5.65×10⁻² ) R
[ 20/200] 2.09×10⁻²  1.00×10⁻³   -14.60414 Ha = -397.39320 eV  +15.47604 Ha  -30.08018 Ha -1.94366 (5.63×10⁻² ) R
[ 21/200] 2.09×10⁻²  1.00×10⁻³   -14.60416 Ha = -397.39391 eV  +15.47588 Ha  -30.08004 Ha -1.94367 (5.63×10⁻² ) R
[ 22/200] 2.08×10⁻²  1.00×10⁻³   -14.60419 Ha = -397.39462 eV  +15.47572 Ha  -30.07991 Ha -1.94368 (5.63×10⁻² ) R
[ 23/200] 2.08×10⁻²  1.00×10⁻³   -14.60422 Ha = -397.39534 eV  +15.47555 Ha  -30.07977 Ha -1.94370 (5.63×10⁻² ) R
[ 24/200] 2.08×10⁻²  1.01×10⁻¹   -14.60424 Ha = -397.39605 eV  +15.47539 Ha  -30.07963 Ha -1.94371 (5.63×10⁻² ) R
[ 25/200] 2.08×10⁻²  1.91×10⁻¹   -14.60427 Ha = -397.39676 eV  +15.45898 Ha  -30.06325 Ha -1.94471 (5.53×10⁻² ) R
[ 26/200] 1.87×10⁻²  2.72×10⁻¹   -14.60689 Ha = -397.46809 eV  +15.43104 Ha  -30.03793 Ha -1.94659 (5.34×10⁻² ) R
[ 27/200] 1.52×10⁻²  3.45×10⁻¹   -14.61128 Ha = -397.58758 eV  +15.39927 Ha  -30.01055 Ha -1.94883 (5.12×10⁻² ) R
[ 28/200] 1.11×10⁻²  4.10×10⁻¹   -14.61624 Ha = -397.72243 eV  +15.37090 Ha  -29.98714 Ha -1.95090 (4.91×10⁻² ) R
[ 29/200] 7.32×10⁻³  4.69×10⁻¹   -14.62071 Ha = -397.84423 eV  +15.35013 Ha  -29.97084 Ha -1.95248 (4.75×10⁻² ) R
[ 30/200] 4.33×10⁻³  5.22×10⁻¹   -14.62414 Ha = -397.93747 eV  +15.33755 Ha  -29.96169 Ha -1.95349 (4.65×10⁻² )
[ 31/200] 2.30×10⁻³  5.70×10⁻¹   -14.62642 Ha = -397.99953 eV  +15.33134 Ha  -29.95776 Ha -1.95402 (4.60×10⁻² )
[ 32/200] 1.10×10⁻³  6.13×10⁻¹   -14.62776 Ha = -398.03592 eV  +15.32898 Ha  -29.95674 Ha -1.95426 (4.57×10⁻² )
[ 33/200] 4.74×10⁻⁴  6.52×10⁻¹   -14.62845 Ha = -398.05482 eV  +15.32839 Ha  -29.95684 Ha -1.95434 (4.57×10⁻² )

Finished in 0.9416050910949707 seconds
┌───────┬────────────────────────┬────────────────────────┬───────────────────────┐
│ i – j │                  ⟨i|j⟩ │                ⟨i|𝔣|j⟩ │               ⟨j|𝔣|i⟩ │
├───────┼────────────────────────┼────────────────────────┼───────────────────────┤
│ 2 – 2 │     0.9999961059152673 │     -4.766687758807457 │    -4.766687758807457 │
│ 2 – 4 │ -1.4285951450204454e-8 │  0.0002211661797713788 │ 0.0002211661797724507 │
│ 4 – 4 │     0.9997403317297171 │    -0.3181852936952018 │   -0.3181852936952018 │
│ 1 – 1 │     0.9999961059152672 │     -4.766687758805878 │    -4.766687758805878 │
│ 1 – 3 │ -1.4285197072494597e-8 │ 0.00022116616548698294 │ 0.0002211661654858277 │
│ 3 – 3 │     0.9997403317297165 │    -0.3181852936952392 │   -0.3181852936952392 │
└───────┴────────────────────────┴────────────────────────┴───────────────────────┘
Iteration   |g|        Energy                         ⟨T̂⟩           ⟨V̂⟩           ⟨V̂⟩/⟨T̂⟩ ⟨V̂⟩/⟨T̂⟩ + 2    Flags
[   1/1000] 6.65×10⁰    -14.62898 Ha = -398.06906 eV  +15.32839 Ha  -29.95737 Ha -1.95437 (4.56×10⁻² )
[   2/1000] 1.12×10⁻³   -14.62898 Ha = -398.06906 eV  +15.32927 Ha  -29.95825 Ha -1.95432 (4.57×10⁻² )
[   3/1000] 1.14×10⁻³   -14.62898 Ha = -398.06906 eV  +15.32928 Ha  -29.95826 Ha -1.95432 (4.57×10⁻² )
[   4/1000] 1.00×10⁻³   -14.62898 Ha = -398.06906 eV  +15.32917 Ha  -29.95814 Ha -1.95432 (4.57×10⁻² )
[   5/1000] 8.28×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32914 Ha  -29.95811 Ha -1.95432 (4.57×10⁻² )
[   6/1000] 7.03×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32906 Ha  -29.95803 Ha -1.95433 (4.57×10⁻² )
[   7/1000] 6.47×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32898 Ha  -29.95795 Ha -1.95433 (4.57×10⁻² )
[   8/1000] 6.23×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32890 Ha  -29.95788 Ha -1.95434 (4.57×10⁻² )
[   9/1000] 6.02×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32884 Ha  -29.95781 Ha -1.95434 (4.57×10⁻² )
[  10/1000] 5.81×10⁻⁴   -14.62898 Ha = -398.06906 eV  +15.32878 Ha  -29.95776 Ha -1.95435 (4.57×10⁻² )
⁞
[ 104/1000] 1.14×10⁻⁶   -14.62898 Ha = -398.06906 eV  +15.32847 Ha  -29.95745 Ha -1.95437 (4.56×10⁻² )
[ 105/1000] 1.05×10⁻⁶   -14.62898 Ha = -398.06906 eV  +15.32847 Ha  -29.95745 Ha -1.95437 (4.56×10⁻² )
  1.692566 seconds (1.99 M allocations: 140.474 MiB, 16.79% gc time)
 * Status: success

 * Candidate solution
    Minimizer: [6.00e-01, 1.23e+00, 1.39e+00,  ...]
    Minimum:   -1.462898e+01

 * Found with
    Algorithm:     BFGS
    Initial Point: [6.00e-01, 1.23e+00, 1.39e+00,  ...]

 * Convergence measures
    |x - x'|               = 1.85e-08 ≰ 0.0e+00
    |x - x'|/|x'|          = 1.33e-08 ≰ 0.0e+00
    |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
    |g(x)|                 = 1.05e-06 ≰ 1.0e-08

 * Work counters
    Seconds run:   1  (vs limit Inf)
    Iterations:    104
    f(x) calls:    213
    ∇f(x) calls:   214

Finished in 2.8848788738250732 seconds
┌───────┬─────────────────────────┬────────────────────────┬────────────────────────┐
│ i – j │                   ⟨i|j⟩ │                ⟨i|𝔣|j⟩ │                ⟨j|𝔣|i⟩ │
├───────┼─────────────────────────┼────────────────────────┼────────────────────────┤
│ 2 – 2 │      0.9999999999999997 │     -4.766576166630178 │     -4.766576166630178 │
│ 2 – 4 │ -1.9455136301643975e-17 │  0.0002081264173954475 │ 0.00020812641739532037 │
│ 4 – 4 │      0.9999999999999997 │    -0.3181123088680883 │    -0.3181123088680883 │
│ 1 – 1 │      0.9999999999999994 │     -4.766576166630179 │     -4.766576166630179 │
│ 1 – 3 │   2.076818550021598e-18 │ 0.00020812640451178777 │  0.0002081264045114301 │
│ 3 – 3 │                     1.0 │   -0.31811230886808745 │   -0.31811230886808745 │
└───────┴─────────────────────────┴────────────────────────┴────────────────────────┘
Fock operator with
- quantum system: Atom{Float64}(R=Radial finite differences basis {Float64} on 0.0..15.15 with 151 points spaced by ρ = 0.1; Z = 4 [beryllium]; 4 e⁻ ⇒ Q = 0) with 1 Configuration{SpinOrbital{Orbital{Int64},Tuple{Int64,HalfIntegers.Half{Int64}}}}: 1s₀α 1s₀β 2s₀α 2s₀β
- SCF equations:
  - Hartree–Fock equation: E|1s₀α⟩ = OrbitalEquation(1s₀α):
  [1, 1]  =  + 1ĥ₀|1s₀α⟩ + 1r⁻¹×Y⁰(1s₀β,1s₀β)|1s₀α⟩ … - 1r⁻¹×Y⁰(2s₀α,1s₀α)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀α⟩

⟨1s₀α| 𝓗 |1s₀α⟩ = -4.766576166630178 Ha = -129.70330407017377 eV

  - Hartree–Fock equation: E|1s₀β⟩ = OrbitalEquation(1s₀β):
  [1, 1]  =  + 1ĥ₀|1s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|1s₀β⟩ … + 1r⁻¹×Y⁰(2s₀β,2s₀β)|1s₀β⟩ - 1r⁻¹×Y⁰(2s₀β,1s₀β)|2s₀β⟩

⟨1s₀β| 𝓗 |1s₀β⟩ = -4.766576166630178 Ha = -129.70330407017377 eV

  - Hartree–Fock equation: E|2s₀α⟩ = OrbitalEquation(2s₀α):
  [1, 1]  =  + 1ĥ₀|2s₀α⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀α⟩ … + 1r⁻¹×Y⁰(1s₀β,1s₀β)|2s₀α⟩ + 1r⁻¹×Y⁰(2s₀β,2s₀β)|2s₀α⟩

⟨2s₀α| 𝓗 |2s₀α⟩ = -0.31811230886808733 Ha = -8.656154036609523 eV

  - Hartree–Fock equation: E|2s₀β⟩ = OrbitalEquation(2s₀β):
  [1, 1]  =  + 1ĥ₀|2s₀β⟩ + 1r⁻¹×Y⁰(1s₀α,1s₀α)|2s₀β⟩ … - 1r⁻¹×Y⁰(1s₀β,2s₀β)|1s₀β⟩ + 1r⁻¹×Y⁰(2s₀α,2s₀α)|2s₀β⟩

⟨2s₀β| 𝓗 |2s₀β⟩ = -0.3181123088680882 Ha = -8.656154036609548 eV
```

![Beryllium FEDVR orbitals](figures/beryllium-fd.svg)

Comparing with the FEDVR calculation, the total and orbital energies
are slightly worse, but the calculation is much faster!

| Energy | Hartree–Fock limit | FEDVR         | FD            |
| -----  | ------------------ | ------------- | ------------- |
| Total  | -14.573023 Ha      | -14.57302 Ha  | -14.62898 Ha  |
| 1s     | -4.7326698 Ha      | -4.7326681 Ha | -4.7665762 Ha |
| 2s     | -0.3092695 Ha      | -0.3092689 Ha | -0.3181123 Ha |
    
