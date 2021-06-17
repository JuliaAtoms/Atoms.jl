# * Split Hamiltonian

"""
    OrbitalHamiltonianTerm(i, j, coeff, A, integrals)

Represents a term in the orbital Hamiltonian arising from a variation
of the energy expressions between configurations `i` and `j` in the
multi-configurational expansion. `coeff` is the numeric coefficient,
`A` is the operator acting on the orbital, and `integrals` is a vector
of [`OrbitalIntegral`](@ref)s arising from the presence of
non-orthogonal orbitals and whose values should be multiplied to form
the overall coefficient.
"""
struct OrbitalHamiltonianTerm{aO,bO,T,QO,VOint}
    i::Int
    j::Int
    coeff::T
    A::QO
    integrals::VOint
end

function Base.show(io::IO, t::OrbitalHamiltonianTerm)
    write(io, "OrbitalHamiltonianTerm at [$(t.i),$(t.j)]: ")
    show(io, t.A)
    write(io, " × ($(t.coeff)) (#∫: $(length(t.integrals)))")
end

Base.show(io::IO, ::MIME"text/plain", t::OrbitalHamiltonianTerm) = show(io, t)

"""
    coefficient(term::OrbitalHamiltonianTerm)

Return the multiplicative coefficient pertaining to `term`,
_excluding_ the `conj(c_i)*c_j` mixing coefficients, due to the
configuration-interaction.
"""
function coefficient(term::OrbitalHamiltonianTerm)
    isempty(term.integrals) && return term.coeff
    term.coeff*prod(integral_value, term.integrals)
end

"""
    coefficient(term::OrbitalHamiltonianTerm, c::Vector)

Return the multiplicative coefficient pertaining to `term`,
_including_ the `conj(c_i)*c_j` mixing coefficients, due to the
configuration-interaction.
"""
function coefficient(term::OrbitalHamiltonianTerm, c::AbstractVector)
    coeff = coefficient(term)
    # We use zero-valued i,j coordinates to designate terms which do
    # not originate from the configuration expansion, but rather added
    # in manually, such as energy shifts.
    (iszero(term.i) || iszero(term.j)) && return coeff

    conj(c[term.i])*c[term.j]*coeff
end

update!(::AbstractMatrix) = nothing
update!(::AbstractMatrix, ::Atom) = nothing
update!(t::OrbitalHamiltonianTerm) = update!(t.A)
update!(t::OrbitalHamiltonianTerm, atom::Atom) = update!(t.A, atom)

"""
    OrbitalHamiltonian(R, terms, mix_coeffs, projector, orbital)

The Hamiltonian for `orbital` is constructed from a radial basis `R`,
a set of [`OrbitalHamiltonianTerm`](@ref) `terms` that describe the
various interactions between orbitals, `mix_coeffs` which are the
mixing coefficents for the multi-configurational expansion. The
`projector` ensures orthogonality between orbital pairs which have
Lagrange multipliers associated with them, by projecting out
components of other orbitals every time the `OrbitalHamiltonian`
action on `orbital` is computed.
"""
mutable struct OrbitalHamiltonian{aO,bO,O,T,Proj,RT}
    R::RT
    terms::Vector{OrbitalHamiltonianTerm{aO,bO,T}}
    mix_coeffs::Vector{T}
    projector::Proj
    orbital::O
end

Base.axes(hamiltonian::OrbitalHamiltonian) =
    (axes(hamiltonian.R,1),axes(hamiltonian.R,1))

Base.axes(hamiltonian::OrbitalHamiltonian, i) =
    axes(hamiltonian)[i]

Base.eltype(hamiltonian::OrbitalHamiltonian{aO,bO,O,T,Proj}) where {aO,bO,O,T,Proj} = T

# This is strictly speaking, not correct, since the Hamiltonian
# formally acts on the quasiaxes, which are uncountably infinite, but
# it is practical.
Base.size(hamiltonian::OrbitalHamiltonian) =
    (size(hamiltonian.R,2),size(hamiltonian.R,2))
Base.size(hamiltonian::OrbitalHamiltonian, i) = size(hamiltonian)[i]


update!(h::OrbitalHamiltonian) = foreach(t -> update!(t), h.terms)
function update!(h::OrbitalHamiltonian, atom::Atom)
    foreach(t -> update!(t, atom), h.terms)
    h.mix_coeffs = atom.mix_coeffs
end

"""
    energy_matrix!(H, hamiltonian, ϕ)

Compute the contribution of `hamiltonian` to the Hamiltonian matrix
`H` by repeatedly acting on the associated radial orbital `ϕ` with the
different multi-configurational [`OrbitalHamiltonianTerm`](@ref)s of
`hamiltonian`.
"""
function SCF.energy_matrix!(H::HM, hamiltonian::OrbitalHamiltonian{aO,bO,O,T},
                            ϕ::RadialOrbital{T}) where {HM<:AbstractMatrix,aO,bO,O,T}
    tmp = similar(ϕ)
    for term in hamiltonian.terms
        materialize!(MulAdd(coefficient(term), term.A, ϕ, zero(T), tmp))
        H[term.i,term.j] += dot(ϕ.args[2],tmp.args[2])
    end
    H
end

"""
    filter(fun::Function, H::OrbitalHamiltonian)

Filter the [`OrbitalHamiltonianTerm`](@ref)s of `H` according to the
predicate `fun`.
"""
Base.filter(fun::Function, H::OrbitalHamiltonian) =
    OrbitalHamiltonian(H.R, filter(fun, H.terms),
                       H.mix_coeffs, H.projector, H.orbital)

function Base.getindex(H::OrbitalHamiltonian, term::Symbol)
    term == :total && return H
    operator_type = if term == :onebody
        AtomicOneBodyHamiltonian
    elseif term == :direct
        DirectPotential
    elseif term == :exchange
        ExchangePotential
    elseif term == :twobody
        HFPotential
    elseif term == :source
        SourceTerm
    elseif term == :shift
        ShiftTerm
    else
        throw(ArgumentError("Unknown Hamiltonian term $term"))
    end

    filter(t -> t.A isa operator_type, H)
end

# ** Materialization

const OrbitalHamiltonianMatrixElement{aO,bO,O,T} =
    Applied{<:Any,typeof(*),<:Tuple{<:AdjointRadialOrbital{T},
                                    <:OrbitalHamiltonian{aO,bO,O,T},
                                    <:RadialOrbital{T}}}

const OrbitalHamiltonianMatrixVectorProduct{aO,bO,O,T} =
    Applied{<:Any,typeof(*),<:Tuple{<:OrbitalHamiltonian{aO,bO,O,T},<:RadialOrbital{T}}}

const OrbitalHamiltonianMatrixMatrixProduct{aO,bO,O,T} =
    Applied{<:Any,typeof(*),<:Tuple{<:OrbitalHamiltonian{aO,bO,O,T},<:RadialOrbitals{T}}}

Base.eltype(::OrbitalHamiltonianMatrixVectorProduct{aO,bO,O,T}) where {aO,bO,O,T} = T

function Base.copyto!(dest::RadialOrbital{T},
                      matvec::OrbitalHamiltonianMatrixVectorProduct{aO,bO,O,T}) where {aO,bO,O,T}
    axes(dest) == axes(matvec) || throw(DimensionMismatch("axes must be the same"))
    R′,v = dest.args
    hamiltonian,b = matvec.args

    c = hamiltonian.mix_coeffs

    v .= zero(T)

    # Compute the additive action of each of the terms in the
    # Hamiltonian, weighting them by their coefficients which are
    # constructed from
    # 1) the configuration-interaction mixing coefficients,
    # 2) coefficients due to the energy expression, and
    # 3) values of various orbitals integrals resulting from
    #    non-orthogonal orbitals.
    for term in hamiltonian.terms
        coeff = coefficient(term, c)
        materialize!(MulAdd(coeff, term.A, b, one(T), dest))
    end

    dest
end

function Base.copyto!(dest::RadialOrbitals,
                      matvec::OrbitalHamiltonianMatrixMatrixProduct)
    axes(dest) == axes(matvec) || throw(DimensionMismatch("axes must be the same"))
    R′,dv = dest.args
    hamiltonian,b = matvec.args
    bv = b.args[2]

    n = size(dv,2)
    for j = 1:n
        copyto!(applied(*, hamiltonian.R, view(dv, :, j)),
                applied(*, hamiltonian, applied(*, hamiltonian.R, view(bv, :, j))))
    end
    dest
end

"""
    mul!(y, h::OrbitalHamiltonian, x)

Materialize the action of the [`OrbitalHamiltonian`](@ref) on the
linear algebra vector `x` and store the result in `y`, by wrapping
them both with the `QuasiMatrix` necessary to transform `x` and `y` to
the function space of the Hamiltonian.
"""
function LinearAlgebra.mul!(y, h::OrbitalHamiltonian, x, α::Number=true, β::Number=false)
    @assert !β
    copyto!(applied(*, h.R, y),
            applied(*, h, (applied(*, h.R, x))))
    !isone(α) && lmul!(α, y)
    y
end

Base.similar(matvec::OrbitalHamiltonianMatrixVectorProduct) =
    similar(matvec.args[2])

LazyArrays.materialize(matvec::OrbitalHamiltonianMatrixVectorProduct) =
    copyto!(similar(matvec), matvec)

function LazyArrays.materialize(matel::OrbitalHamiltonianMatrixElement)
    a,op,b = matel.args
    op.R isa CompactBases.BSplineOrRestricted &&
        @warn "Implementation not correct for non-orthogonal bases"
    apply(*, a, apply(*, op, b))
end

# *** Materialization into a matrix

get_operator_matrix(A::AtomicOneBodyHamiltonian) = A.op.args[2]
get_operator_matrix(A::DirectPotential) = A.V̂.args[2]
get_operator_matrix(A::ShiftTerm) = A.shift
get_operator_matrix(A::RadialOperator) = A.args[2]
get_operator_matrix(A::Diagonal) = A
get_operator_matrix(A::IdentityOperator) = I
get_operator_matrix(A::SourceTerm) = get_operator_matrix(A.operator)

"""
    copyto!(dest::AbstractMatix, hamiltonian::OrbitalHamiltonian)

Materialize the orbital `hamiltonian` into matrix form and store it in
`dest`, using the current values of all other orbitals. This is only
possible if the orbital `hamiltonian` does *not* contain any
[`ExchangePotential`](@ref)s or [`SourceTerm`](@ref)s (which are not
diagonal in orbital space), since the former is non-local (and thus
not representable as a matrix) and the latter is not a linear operator
(but an affine one).

Typical usage is to compute an easily factorizable matrix that can be
used for preconditioning the solution of the full equation.
"""
function Base.copyto!(dest::M, hamiltonian::OrbitalHamiltonian) where {T,M<:AbstractMatrix{T}}
    m = size(hamiltonian.R,2)
    size(dest) == (m,m) || throw(DimensionMismatch("axes must be the same"))

    c = hamiltonian.mix_coeffs

    dest .= zero(T)

    for term in hamiltonian.terms
        term.A isa SourceTerm && term.A.source_orbital ≠ hamiltonian.orbital &&
            throw(ArgumentError("It is not possible to materialize an orbitally off-diagonal $(typeof(term.A)) as a matrix"))
        coeff = coefficient(term, c)

        dest += coeff*get_operator_matrix(term.A)
    end

    dest
end

function Base.similar(h::OrbitalHamiltonian{aO,bO,O,T,Proj,RT}, ::Type{T}) where {aO,bO,O,T,Proj,RT<:BasisOrRestricted{<:AbstractFiniteDifferences}}
    R = h.R
    m = size(R,2)
    # TODO: This is only valid for {,Staggered}FiniteDifferences of
    # CompactBases.jl
    o = ones(T, m)
    SymTridiagonal(o,0*o[2:end])
end

Base.similar(h::OrbitalHamiltonian{aO,bO,O,T,Proj,RT}, ::Type{T}) where {aO,bO,O,T,Proj,
                                                                         RT<:BasisOrRestricted{<:Union{<:FEDVR,<:BSpline}}} =
    Matrix(undef, h.R)

LazyArrays.materialize(h::OrbitalHamiltonian) =
    copyto!(similar(h, eltype(h)), h)

# ** Arithmetic

"""
    h::OrbitalHamiltonian + λ::UniformScaling

Shift the [`OrbitalHamiltonian`](@ref) `h` by `λ`.
"""
function Base.:(+)(h::OrbitalHamiltonian{aO,bO,O,T,Proj,RT}, λ::UniformScaling) where {aO,bO,O,T,Proj,RT}
    # The zeros designate that the shift is not to be weighted by the
    # mixing coefficients
    shift_term = OrbitalHamiltonianTerm(0, 0, one(T), ShiftTerm(λ),
                                        Vector{OrbitalIntegral{<:Any,aO,bO,T}}())
    OrbitalHamiltonian{aO,bO,O,T,Proj,RT}(h.R, vcat(h.terms, shift_term),
                                          h.mix_coeffs, h.projector, h.orbital)
end

"""
    h::OrbitalHamiltonian - λ::UniformScaling

Shift the [`OrbitalHamiltonian`](@ref) `h` by `-λ`.
"""
Base.:(-)(h::OrbitalHamiltonian, λ::UniformScaling) = h + (-λ)

# ** Krylov wrapper

"""
    SCF.KrylovWrapper(hamiltonian::OrbitalHamiltonian)

Construct a `KrylovWrapper` such that `hamiltonian`, that acts on
function spaces, can be used in a Krylov solver, which works with
linear algebra vector spaces.
"""
function SCF.KrylovWrapper(hamiltonian::OrbitalHamiltonian)
    T = eltype(hamiltonian)
    L = LinearOperator(hamiltonian, hamiltonian.R)
    KrylovWrapper{T,typeof(L)}(L)
end

function SCF.OrthogonalKrylovWrapper(hamiltonian::OrbitalHamiltonian)
    R = hamiltonian.R
    S = R'R
    SCF.OrthogonalKrylovWrapper(hamiltonian, LinearOperator(hamiltonian, hamiltonian.R),
                                length(hamiltonian.projector.orbitals), S)
end

function update!(okw::SCF.OrthogonalKrylovWrapper{<:OrbitalHamiltonian})
    for (j,ϕ) in enumerate(okw.h.projector.ϕs)
        copyto!(view(okw.C, :, j), ϕ.args[2])
    end
    okw
end
