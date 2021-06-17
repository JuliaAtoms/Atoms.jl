# * Generation of one-body Hamiltonian

function potential_matrix(V::Function, R)
    r = axes(R,1)
    apply(*, R', QuasiDiagonal(V.(r)), R)
end

operator(M, R) = applied(*, R, M, R')

"""
    one_body_hamiltonian(::Type{Tuple}, atom, orb)

Return the kinetic and one-body potential energy operators (as a
tuple) for the orbital `orb` of `atom`.
"""
function one_body_hamiltonian(::Type{Tuple}, R::AbstractQuasiMatrix,
                              potential, orb)
    ℓ = getℓ(orb)
    Z = float(effective_charge(potential))

    # This way we can apply the correct boundary condition at r = 0.
    D = Derivative(axes(R,1))
    # CD = (potential isa PointCharge && R isa AbstractFiniteDifferences
    #       ? CoulombDerivative(D, Z, ℓ) : D)
    CD = (R isa AbstractFiniteDifferences ? CoulombDerivative(D, Z, ℓ) : D)

    T = apply(*, R', CD', CD, R)
    T /= -2

    # Add in centrifugal part
    T += potential_matrix(r -> ℓ*(ℓ+1)/(2r^2), R)

    V = potential_matrix(r -> potential(orb, r), R)

    operator(T, R), operator(V, R)
end

one_body_hamiltonian(::Type{Tuple}, atom::Atom, orb) =
    one_body_hamiltonian(Tuple, radial_basis(atom),
                         atom.potential, orb)

"""
    one_body_hamiltonian(::Type{Tuple}, atom, orb)

Return the one-body energy operator for the orbital `orb` of `atom`.
"""
one_body_hamiltonian(args...) = +(one_body_hamiltonian(Tuple, args...)...)

# * One-body Hamiltonian operators

# ** T̂
"""
    KineticEnergyHamiltonian

The kinetic energy part of the one-body Hamiltonian, ubcluding the
centrifugal potential. It is diagonal in spin, i.e. it does not couple
orbitals of opposite spin.
"""
struct KineticEnergyHamiltonian <: OneBodyOperator end
Base.iszero(me::OrbitalMatrixElement{1,A,KineticEnergyHamiltonian,B}) where {A<:SpinOrbital{<:Orbital},B<:SpinOrbital{<:Orbital}} =
    me.a[1].m[2] != me.b[1].m[2]

Base.show(io::IO, ::KineticEnergyHamiltonian) = write(io, "T̂")

# ** V̂
"""
    PotentialEnergyHamiltonian

The potential energy part of the one-body Hamiltonian. It is diagonal
in spin, i.e. it does not couple orbitals of opposite spin.
"""
struct PotentialEnergyHamiltonian <: OneBodyOperator end
Base.iszero(me::OrbitalMatrixElement{1,A,PotentialEnergyHamiltonian,B}) where {A<:SpinOrbital{<:Orbital},B<:SpinOrbital{<:Orbital}} =
    me.a[1].m[2] != me.b[1].m[2]

Base.show(io::IO, ::PotentialEnergyHamiltonian) = write(io, "V̂")

# * AtomicOneBodyHamiltonian types

abstract type AbstractAtomicOneBodyHamiltonian end
Base.iszero(::AbstractAtomicOneBodyHamiltonian) = false

# ** ZeroAtomicOneBodyHamiltonian

struct ZeroAtomicOneBodyHamiltonian{M,N} <: AbstractAtomicOneBodyHamiltonian
    a::Tuple{M,N}
end

Base.axes(ĥ::ZeroAtomicOneBodyHamiltonian) = ĥ.a
Base.axes(ĥ::ZeroAtomicOneBodyHamiltonian,i) = ĥ.a[i]

Base.iszero(::ZeroAtomicOneBodyHamiltonian) = true

Base.show(io::IO, ĥ::ZeroAtomicOneBodyHamiltonian) =
    write(io, "0")

# ** AtomicOneBodyHamiltonian
"""
    AtomicOneBodyHamiltonian(op, orbital)

Structure holding a one-body energy operator `op` acting on its
associated `orbital`.
"""
struct AtomicOneBodyHamiltonian{LT,O} <: AbstractAtomicOneBodyHamiltonian
    op::LT
    orbital::O
end

"""
    AtomicOneBodyHamiltonian(atom, orb)

Create the one-body Hamiltonian corresponding to the orbital `orb` of
`atom`.
"""
AtomicOneBodyHamiltonian(atom::Atom, orb::AbstractOrbital) =
    AtomicOneBodyHamiltonian(one_body_hamiltonian(atom, orb), orb)

Base.axes(ĥ::AtomicOneBodyHamiltonian, args...) =
    axes(ĥ.op, args...)

Base.zero(ĥ::AtomicOneBodyHamiltonian) =
    ZeroAtomicOneBodyHamiltonian(axes(ĥ))

SCF.update!(::AtomicOneBodyHamiltonian) = nothing
SCF.update!(::AtomicOneBodyHamiltonian, ::Atom) = nothing

matrix(ĥ::AtomicOneBodyHamiltonian) = matrix(ĥ.op)

"""
    materialize!(::MulAdd{<:Any, <:Any, <:Any, T, <:AtomicOneBodyHamiltonian, Source, Dest})

Materialize the lazy multiplication–addition of the type `y ← α*H*x +
β*y` where `H` is a [`AtomicOneBodyHamiltonian`](@ref) and `x` and `y`
are [`RadialOrbital`](@ref)s.
"""
LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T, <:AtomicOneBodyHamiltonian, Source, Dest}) where {T,Source,Dest} =
    mul!(ma.C.args[2], ma.A.op.args[2], ma.B.args[2], ma.α, ma.β)

LinearAlgebra.mul!(y, h::AtomicOneBodyHamiltonian, x,
                   α::Number=true, β::Number=false) =
                       mul!(y, h.op.args[2], x, α, β)

Base.show(io::IO, ĥ::AtomicOneBodyHamiltonian) =
    write(io, "ĥ($(ĥ.orbital))")

# * Diagonalization of one-body Hamiltonian

symmetric_transform(H::AbstractMatrix, U::AbstractMatrix) = U*H*U
symmetric_transform(H::SymTridiagonal, U::Diagonal) = SymTridiagonal(U*H*U)

function symmetric_orthogonalization(H, ::Type{<:Diagonal}, R)
    S = Diagonal(metric(R))
    S⁻¹ᐟ² = inv(√(S))
    symmetric_transform(H, S⁻¹ᐟ²), S⁻¹ᐟ²
end

function symmetric_orthogonalization(H, ::Any, R)
    S = metric(R)
    S⁻¹ᐟ² = inv(√(Symmetric(S)))
    symmetric_transform(H, S⁻¹ᐟ²), S⁻¹ᐟ²
end

symmetric_orthogonalization(H::AbstractMatrix, R::AbstractQuasiMatrix) =
    symmetric_orthogonalization(H, metric_shape(R), R)

"""
    diagonalize_one_body(H, nev; method=:arnoldi_shift_invert, tol=1e-10, σ=-1)

Diagonalize the one-body Hamiltonian `H` and find the `nev` lowest
eigenpairs, using the specified diagonalization `method`; valid
choices are

- `:arnoldi` which performs the standard Krylov iteration looking for
  the eigenvalues with smallest real values,

- `:arnoldi_shift_invert` which performs the Krylov iteration but with
  the shifted and inverted matrix `(H - I*σ)⁻¹` looking for the
  eigenvalues with _largest_ real values,

- `:eigen` which uses Julia's built-in eigensolver.

`tol` sets the Krylov tolerance.
"""
function diagonalize_one_body(H::RadialOperator, R, nev::Int;
                              method::Symbol=:arnoldi_shift_invert, tol=1e-10, σ=-1,
                              verbosity=0, io=stdout)
    Hm = matrix(H)
    verbosity > 2 && println(io, "Diagonalizing via $(method)")
    if method == :arnoldi || method == :arnoldi_shift_invert
        A,target = method == :arnoldi ? (LinearOperator(Hm, R),SR()) : (ShiftAndInvert(Hm, R, σ),LR())

        schur,history = partialschur(A, nev=nev, tol=tol, which=target)
        verbosity > 3 && println(io, history)

        θ = diag(schur.R)
        λ = method == :arnoldi ? θ : σ .+ inv.(θ)
        length(λ) < nev &&
            error("Could not converge the requested orbitals: $(history)")
        λ,schur.Q
    elseif method == :eigen
        H′, U = symmetric_orthogonalization(Hm, R)

        !(metric_shape(R) <: Diagonal) && size(Hm,1) > 1000 &&
            @warn "Attempting to diagonalize a matrix of size $(size(Hm)), prepare to wait"

        ee = eigen(H′)
        ee.values[1:nev], U*ee.vectors[:,1:nev]
    else
        throw(ArgumentError("Unknown diagonalization method $(method)"))
    end
end
