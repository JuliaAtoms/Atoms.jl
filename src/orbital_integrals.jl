# * Orbital integrals

"""
    OrbitalIntegral{N}

Abstract type for integrals of rank `N` of orbitals, whose values need
to be recomputed every time the orbitals are updated. Rank 0
corresponds to a scalar value, rank 1 to a diagonal matrix, etc.
"""
abstract type OrbitalIntegral{N,aO<:AbstractOrbital,bO<:AbstractOrbital,T} end

Base.iszero(::OrbitalIntegral) = false

# ** Zero integral
struct ZeroIntegral{aO,bO,T} <: OrbitalIntegral{0,aO,bO,T} end

integral_value(::ZeroIntegral{aO,bO,T}) where {aO,bO,T} = zero(T)

SCF.update!(::ZeroIntegral, ::Atom; kwargs...) = nothing

# ** Orbital overlap integral

"""
    OrbitalOverlapIntegral(a, b, av, bv, value)

Represents the orbital overlap integral `⟨a|b⟩`, for orbitals `a` and
`b`, along with `view`s of their radial orbitals `av` and `bv` and the
current `value` of the integral.
"""
mutable struct OrbitalOverlapIntegral{aO,bO,T,OV,Metric} <: OrbitalIntegral{0,aO,bO,T}
    a::aO
    b::bO
    av::OV
    bv::OV
    value::T
    S̃::Metric
end

function OrbitalOverlapIntegral(a::aO, b::bO, atom::Atom{T}) where {aO,bO,T}
    oo = OrbitalOverlapIntegral(a, b,
                                view(atom, a).args[2],
                                view(atom, b).args[2],
                                zero(T), atom.S̃)
    SCF.update!(oo)
    oo
end

Base.show(io::IO, oo::OrbitalOverlapIntegral) =
    write(io, "⟨$(oo.a)|$(oo.b)⟩")

"""
    SCF.update!(oo::OrbitalOverlapIntegral)

Update the value of the integral `oo`.
"""
function SCF.update!(oo::OrbitalOverlapIntegral; kwargs...)
    oo.value = dot(oo.av, oo.S̃, oo.bv)
end

"""
    SCF.update!(oo::OrbitalOverlapIntegral)

Update the value of the integral `oo` with respect to `atom`.
"""
function SCF.update!(oo::OrbitalOverlapIntegral, atom::Atom; kwargs...)
    oo.av = view(atom, oo.a).args[2]
    oo.bv = view(atom, oo.b).args[2]
    SCF.update!(oo; kwargs...)
end

integral_value(oo::OrbitalOverlapIntegral) = oo.value

# ** Operator matrix element

"""
    OperatorMatrixElement(a, b, Â, coeff, value)

Represents the matrix element `coeff*⟨a|Â|b⟩`, for the operator `Â`
and the orbitals `a` and `b`, along with the current `value` of the
integral. Typically, `Â` is a radial part of an operator and `coeff`
is the associated angular coefficient; `coeff` can be of any type
convertible to a scalar.
"""
mutable struct OperatorMatrixElement{aO,bO,OV,RO,T,Coeff,Metric,Tmp} <: OrbitalIntegral{0,aO,bO,T}
    a::aO
    b::bO
    av::OV
    bv::OV
    Â::RO
    coeff::Coeff
    value::T
    S̃::Metric
    tmp::Tmp
end

function OperatorMatrixElement(a::aO, b::bO, Â, atom::Atom{T}, coeff) where {aO,bO,T}
    av = view(atom, a).args[2]
    bv = view(atom, b).args[2]
    tmp = similar(bv)

    ome = OperatorMatrixElement(a, b, av, bv,
                                Â, coeff, zero(T), atom.S̃, tmp)
    SCF.update!(ome)
    ome
end

OperatorMatrixElement(a::aO, b::bO, Â::RadialOperator, atom::Atom{T}, coeff) where {aO,bO,T} =
    OperatorMatrixElement(a, b, Â.args[2], atom, coeff)

function Base.show(io::IO, ome::OperatorMatrixElement)
    write(io, "⟨$(ome.a)|")
    show(io, ome.Â)
    write(io, "|$(ome.b)⟩")
end

function SCF.update!(ome::OperatorMatrixElement{aO,bO,OV,RO,T}; kwargs...) where {aO,bO,OV,RO,T}
    # NB: It is assumed that ome.Â, if necessary, is updated /before/
    # SCF.update!(ome) is called.
    mul!(ome.tmp, ome.Â, ome.bv)
    ome.value = convert(T,ome.coeff)*dot(ome.av, ome.S̃, ome.tmp)
end

function SCF.update!(ome::OperatorMatrixElement, atom::Atom; kwargs...)
    ome.av = view(atom, ome.a)
    ome.bv = view(atom, ome.b)
    SCF.update!(ome; kwargs...)
end

integral_value(ome::OperatorMatrixElement) = ome.value

# ** Hartree–Fock potentials

const HFPotentialOperator{T,B<:Basis} = RadialOperator{T,B,Diagonal{T,<:AbstractVector{T}}}

"""
    HFPotential(k, a, b, av, bv, V̂, poisson)

Represents the `k`:th multipole exansion of the Hartree–Fock potential
formed by orbitals `a` and `b` (`av` and `bv` being `view`s of their
corresponding radial orbitals). `V̂` is the resultant one-body
potential formed, which can act on a third orbital and `poisson`
computes the potential by solving Poisson's problem.
"""
mutable struct HFPotential{kind,aO,bO,T,
                           OV<:RadialOrbital{T},
                           Density,
                           Potential} <: OrbitalIntegral{1,aO,bO,T}
    k::Int
    a::aO
    b::bO
    av::OV
    bv::OV
    ρ::Density
    V̂::Potential
end

HFPotential(kind::Symbol, k::Int, a::aO, b::bO, av::OV, bv::OV, ρ::Density, V̂::Potential) where {aO,bO,T,OV<:RadialOrbital{T},
                                                                                                 Density,Potential} =
    HFPotential{kind,aO,bO,T,OV,Density,Potential}(k, a, b, av, bv, ρ, V̂)

get_coulomb_repulsion_potential(::CoulombInteraction{Nothing}, args...; kwargs...) =
    CoulombRepulsionPotential(args...; kwargs...)

get_coulomb_repulsion_potential(g::CoulombInteraction{<:AbstractQuasiMatrix}, R, k, T; apply_metric_inverse=true, kwargs...) =
    CoulombRepulsionPotential(R, AsymptoticPoissonProblem(R, k, g.o, T; kwargs...);
                              apply_metric_inverse=apply_metric_inverse)

function HFPotential(kind::Symbol, k::Int, a::aO, b::bO, atom::Atom{T}, g::CoulombInteraction; kwargs...) where {aO,bO,T}
    av, bv = view(atom, a), view(atom, b)
    R = av.args[1]
    ρ = Density(av, bv)
    V̂ = get_coulomb_repulsion_potential(g, R, k, T; apply_metric_inverse=false, kwargs...)
    update!(HFPotential(kind, k, a, b, av, bv, ρ, V̂), atom)
end

Base.convert(::Type{HFPotential{kind,aO₁,bO₁,T,OV,Density,Potential}},
             hfpotential::HFPotential{kind,aO₂,bO₂,T,OV,Density,Potential}) where {kind,aO₁,bO₁,aO₂,bO₂,T,OV,Density,Potential} =
                 HFPotential{kind,aO₁,bO₁,T,OV,Density,Potential}(hfpotential.k,
                                                                  hfpotential.a, hfpotential.b,
                                                                  hfpotential.av, hfpotential.bv,
                                                                  hfpotential.ρ, hfpotential.V̂)

Base.eltype(::HFPotential{<:Any,<:Any,<:Any,T}) where T = T

# *** Direct potential

"""
    DirectPotential

Special case of [`HFPotential`](@ref) for the direct interaction, in
which case the potential formed from two orbitals can be precomputed
before acting on a third orbital.
"""
const DirectPotential{aO,bO,T,OV,Density,Potential} = HFPotential{:direct,aO,bO,T,OV,Density,Potential}

Base.show(io::IO, Y::DirectPotential) =
    write(io, "r⁻¹×Y", to_superscript(Y.k), "($(Y.a), $(Y.b))")

"""
    SCF.update!(p::DirectPotential)

Update the direct potential `p` by solving the Poisson problem with
the current values of the orbitals forming the mutual density.
"""
function SCF.update!(p::DirectPotential{aO,bO,T,OV,Density,Potential}; kwargs...) where {aO,bO,T,OV,Density,Potential}
    copyto!(p.ρ, p.av, p.bv)
    copyto!(p.V̂, p.ρ)
    p
end

"""
    SCF.update!(p::DirectPotential, atom::Atom)

Update the direct potential `p` by solving the Poisson problem with
the current values of the orbitals of `atom` forming the mutual
density.
"""
function SCF.update!(p::DirectPotential{aO,bO,T,OV,Density,Potential}, atom::Atom; kwargs...) where {aO,bO,T,OV,Density,Potential}
    p.av = view(atom, p.a)
    p.bv = view(atom, p.b)
    SCF.update!(p; kwargs...)
end

LinearAlgebra.mul!(y::AbstractVecOrMat, p::DirectPotential, x::AbstractVecOrMat,
                   α::Number=true, β::Number=false) =
                       mul!(y, p.V̂, x, α, β)

"""
    materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:DirectPotential, Source, Dest})

Materialize the lazy multiplication–addition of the type `y ←
α*V̂*x + β*y` where `V̂` is a [`DirectPotential`](@ref) (with a
precomputed direct potential computed via `SCF.update!`) and `x` and
`y` are [`RadialOrbital`](@ref)s.
"""
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, <:Any, <:DirectPotential, <:Any, <:Any}) =
    mul!(ma.C.args[2], ma.A, ma.B.args[2], ma.α, ma.β)

# *** Exchange potential

"""
    ExchangePotential

Special case of [`HFPotential`](@ref) for the exchange interaction, in
which case the potential is formed from the orbital acted upon, along
with another orbital, and then applied to a third orbital. Thus this
potential *cannot* be precomputed, but must be recomputed every time
the operator is applied. This makes this potential expensive to handle
and the number of times it is applied should be minimized, if possible.
"""
const ExchangePotential{aO,bO,T,OV,Density,Potential} = HFPotential{:exchange,aO,bO,T,OV,Density,Potential}

Base.show(io::IO, Y::ExchangePotential) =
    write(io, "|$(Y.b)⟩r⁻¹×Y", to_superscript(Y.k), "($(Y.a), ●)")

SCF.update!(p::ExchangePotential; kwargs...) = p

function SCF.update!(p::ExchangePotential, atom::Atom; kwargs...)
    p.av = view(atom, p.a)
    p.bv = view(atom, p.b)
    p
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, p::ExchangePotential, x::AbstractVecOrMat,
                            α::Number=true, β::Number=false)
    # Form exchange potential from the mutual density conj(p.a)*x
    copyto!(p.ρ, p.av.args[2], x)
    copyto!(p.V̂, p.ρ)
    # Act with the exchange potential on p.bv
    mul!(y, p.V̂, p.bv.args[2], α, β)
end

"""
    materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:ExchangePotential, Source, Dest})

Materialize the lazy multiplication–addition of the type `y ← α*V̂*x +
β*y` where `V̂` is a [`ExchangePotential`](@ref) (by solving the
Poisson problem with `x` as one of the constituent source orbitals in
the mutual density) and `x` and `y` are [`RadialOrbital`](@ref)s.
"""
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, <:Any, <:ExchangePotential, <:Any, <:Any}) =
    mul!(ma.C.args[2], ma.A, ma.B.args[2], ma.α, ma.β)

# * Source terms

"""
    SourceTerm(operator, source_orbital, ov)

The point of `SourceTerm` is to implement inhomogeneous terms that
contribute to the equation for an orbital, and whose input is some
other `source_orbital`. This kind of term appears in
multi-configurational problems.
"""
mutable struct SourceTerm{QO,O,OV}
    operator::QO
    source_orbital::O
    ov::OV
end

Base.show(io::IO, st::SourceTerm) = write(io, "SourceTerm($(st.operator)|$(st.source_orbital)⟩)")

Base.iszero(::SourceTerm) = false
Base.similar(st::SourceTerm) = similar(st.ov)

function Base.copyto!(dest::Applied{<:Any,typeof(*),<:Tuple{<:AbstractQuasiMatrix,<:AbstractArray{<:Any,N}}},
                      src::Applied{<:Any,typeof(*),<:Tuple{<:AbstractQuasiMatrix,<:AbstractArray{<:Any,N}}}) where N
    d = last(dest.args)
    s = last(src.args)
    copyto!(IndexStyle(d), d, IndexStyle(s), s)
    dest
end

update!(st::SourceTerm; kwargs...) = nothing

function update!(st::SourceTerm, atom::Atom; kwargs...)
    st.ov = view(atom, st.source_orbital)
end

# These are source terms that do not depend on the atom (e.g. external
# source term, or a constant orbital).
update!(::SourceTerm{<:IdentityOperator,<:AbstractString}, ::Atom) = nothing

LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:SourceTerm, Source, Dest}) where {T,Source,Dest} =
    materialize!(MulAdd(ma.α, ma.A.operator, ma.A.ov, ma.β, ma.C))

function LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:IdentityOperator{1}, Source, Dest}) where {T,Source,Dest}
    isone(ma.β) || lmul!(ma.β, ma.C.args[2])
    BLAS.axpy!(ma.α, ma.B.args[2], ma.C.args[2])
end

# * Shift terms

"""
    ShiftTerm(λ)

The point of `ShiftTerm` is to implement an overall energy shift of
the Hamiltonian.
"""
struct ShiftTerm{T}
    shift::UniformScaling{T}
end

Base.iszero(::ShiftTerm) = false

LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:ShiftTerm, Source, Dest}) where {T,Source,Dest} =
    BLAS.axpy!(ma.α*ma.A.shift.λ, ma.B.args[2], ma.C.args[2])
