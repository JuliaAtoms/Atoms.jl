# Orbital equations

```@meta
CurrentModule = Atoms
DocTestSetup = quote
    using Atoms
end
```

## Hamiltonian

```@docs
OrbitalHamiltonianTerm
coefficient
OrbitalHamiltonian
Projector
projectout!
SCF.energy_matrix!(H::HM, hamiltonian::OrbitalHamiltonian{aO,bO,O,T}, ϕ::RadialOrbital{T}) where {HM<:AbstractMatrix,aO,bO,O,T}
Base.filter(fun::Function, H::OrbitalHamiltonian)
Base.copyto!(dest::M, hamiltonian::OrbitalHamiltonian) where {T,M<:AbstractMatrix{T}}
Base.:(+)(h::OrbitalHamiltonian{O,T,B,OV,Proj}, λ::UniformScaling) where {O,T,B,OV,Proj}
Base.:(-)(h::OrbitalHamiltonian, λ::UniformScaling)
SCF.KrylovWrapper
LinearAlgebra.mul!(y::V₁, A::KrylovWrapper{T,Hamiltonian}, x::V₂) where {V₁,V₂,T,B,Hamiltonian<:OrbitalHamiltonian}
```

## Orbital integrals and terms

```@docs
OrbitalIntegral
OrbitalOverlapIntegral
SCF.update!(oo::OrbitalOverlapIntegral; kwargs...)
HFPotential
DirectPotential
SCF.update!(p::DirectPotential{O,T,B,OV,RO,P}; kwargs...) where {O,T,B,OV,RO,P}
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:DirectPotential, Source, Dest}) where {T,Source,Dest}
ExchangePotential
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:ExchangePotential, Source, Dest}) where {T,Source,Dest}
SourceTerm
ShiftTerm
```

## Orbital equations

```@docs
AtomicOrbitalEquation
SCF.energy(hfeq::AtomicOrbitalEquation, which::Symbol=:total)
```

```@meta
CurrentModule = nothing
DocTestSetup = nothing
```
