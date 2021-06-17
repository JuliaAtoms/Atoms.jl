# TODO: Replace by FuncArray et al.
const RadialOrbitalArray{T,N,B<:Basis} = Applied{<:Any, typeof(*), <:Tuple{<:BasisOrRestricted{B},<:AbstractArray{T,N}}}
const AdjointRadialOrbitalArray{T,N,B<:Basis} = Applied{<:Any, typeof(*), <:Tuple{<:Adjoint{<:Any,<:AbstractArray{T,N}},<:AdjointBasisOrRestricted{B}}}

function Base.similar(A::RadialOrbitalArray)
    R,v = A.args
    applied(*, R, similar(v))
end

"""
    RadialOrbital

A radial orbital is represented using basis coupled with a vector of
expansion coefficients with respect to that basis; the basis
implemented as an `AbstractQuasimatrix`.
"""
const RadialOrbital{T,B<:Basis} = RadialOrbitalArray{T,1,B}
const AdjointRadialOrbital{T,B<:Basis} = AdjointRadialOrbitalArray{T,1,B}

Base.axes(v::RadialOrbital, p::Int) = p == 1 ? axes(v)[1] : 1

"""
    RadialOrbitals

A collection of radial orbitals is instead represented using a
matrix of such expansion coefficients, where each matrix column
corresponds to a single radial orbital.
"""
const RadialOrbitals{T,B<:Basis} = RadialOrbitalArray{T,2,B}
const AdjointRadialOrbitals{T,B<:Basis} = AdjointRadialOrbitalArray{T,2,B}

const RadialOperator{T,B<:Basis,M<:AbstractMatrix{T}} =
    Applied{<:Any,typeof(*),<:Tuple{<:BasisOrRestricted{B},M,<:AdjointBasisOrRestricted{B}}}

function LinearAlgebra.adjoint(op::RadialOperator)
    Ac,M,B = op.args
    applied(*, B', M', parent(Ac))
end

function Base.show(io::IO, ro::RadialOperator{T,B,M}) where {T,B,M}
    write(io, "RadialOperator(")
    show(io, M)
    write(io, ")")
end

Base.iszero(op::RadialOperator) = iszero(op.args[2])
Base.iszero(op::MulQuasiArray{<:Any,2,<:Tuple{
    <:BasisOrRestricted,<:AbstractMatrix,<:AdjointBasisOrRestricted}}) =
        iszero(op.applied)

matrix(o::RadialOperator) = o.args[2]

function Base.zero(o::RadialOperator)
    A,B,C = o.args
    applied(*, A, Diagonal(zeros(eltype(B),size(B,1))), C)
end

function Base.:(+)(a::RadialOperator{T,B}, b::RadialOperator{T,B}) where {T,B}
    R = first(a.args)
    applied(*, R, matrix(a) + matrix(b), R')
end
