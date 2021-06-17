# One-body Hamiltonians

```@meta
CurrentModule = Atoms
DocTestSetup = quote
    using Atoms
end
```

The one-body Hamiltonian for electron $i$ in an atom is given by

$$\begin{equation}
\hamiltonian_i \defd
-\frac{\nabla_i^2}{2} +
V(\vec{r}_i).
\end{equation}$$

In spherical coordinates (and using reduced wavefunctions), the
Laplacian transforms to

$$\begin{equation}
\operator{T}_{r_i} = -\frac{\partial_{r_i}^2}{2} + \frac{\ell(\ell+1)}{2r_i^2},
\end{equation}$$

where the second term, called the centrifugal potential, although
originating from the Laplacian, is usually treated together with the
nuclear potential $V(r_i)$.

```@docs
one_body_hamiltonian
KineticEnergyHamiltonian
PotentialEnergyHamiltonian
AtomicOneBodyHamiltonian
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:AtomicOneBodyHamiltonian, Source, Dest}) where {T,Source,Dest}
```

## Diagonalization of one-body Hamiltonians

```@docs
diagonalize_one_body
```

```@meta
CurrentModule = nothing
DocTestSetup = nothing
```
