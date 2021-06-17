"""
    Projector(ϕs, orbitals, S)

Represents the projector on the subspace spanned by the radial
orbitals `ϕs` (corresponding to `orbitals`).
"""
struct Projector{T,B<:AbstractQuasiMatrix,RO<:RadialOrbital{T,B},O,Metric,Selection} <: NBodyOperator{1}
    ϕs::Vector{RO}
    orbitals::Vector{O}
    S::Metric
    sel::Selection
    function Projector(ϕs::Vector{RO}, orbitals::Vector{O}, S::Metric, sel=nothing) where {T,B,RO<:RadialOrbital{T,B},O,Metric}
        m,n = size(S)
        all(ϕ -> length(ϕ.args[2]) == m, ϕs) ||
            throw(DimensionMismatch())
        if isnothing(sel)
            sel = m == n ? Colon() : 1:m
        end
        new{T,B,RO,O,Metric,typeof(sel)}(ϕs, orbitals, S, sel)
    end
end

Base.eltype(projector::Projector{T}) where T = T
function Base.size(projector::Projector)
    n = size(projector.S,2)
    n,n
end
Base.size(projector::Projector, i) = size(projector)[i]
Base.similar(projector::Projector{T}) where T = Matrix{T}(undef, size(projector))

Base.iszero(me::OrbitalMatrixElement{1,A,<:Projector,B}) where {A<:SpinOrbital,B<:SpinOrbital} =
    me.a[1] ∉ me.o.orbitals || me.b[1] ∉ me.o.orbitals

function Base.show(io::IO, projector::Projector)
    write(io, "P(")
    write(io, join(string.(projector.orbitals), " "))
    write(io, ")")
    if !(projector.sel isa Colon)
        write(io, " restricted to elements $(projector.sel)")
    end
end

# * Multiplication by projector

proj_view(y, ::Colon) = y
proj_view(y, sel) = view(y, sel)

function LinearAlgebra.mul!(y::AbstractVector, p::Projector, x::AbstractVector,
                            α::Number=true, β::Number=false)
    if iszero(β)
        y .= false
    elseif !isone(β)
        lmul!(β, y)
    end
    iszero(α) && return y

    yv = proj_view(y, p.sel)
    for ϕ in p.ϕs
        ϕv = ϕ.args[2]
        c = α*dot(ϕv, p.S, x)
        yv .+= c .* ϕv
    end

    y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, p::Projector, X::AbstractMatrix,
                            α::Number=true, β::Number=false)
    n = size(Y,2)
    for j = 1:n
        mul!(view(Y, :, j), p, view(X, :, j), α, β)
    end
    Y
end

LinearAlgebra.Matrix(p::Projector) =
    mul!(similar(p), p, I(size(p.S,2)))

# * Rejection

projectout!(y::RO, ::Nothing) where RO = y

function projectout!(y::AbstractVector{T}, projector::Projector) where T
    s = zero(T)
    yv = proj_view(y, projector.sel)
    for ϕ in projector.ϕs
        c = dot(ϕ.args[2], projector.S, y)
        yv .-= c .* ϕ.args[2]
        s += c
    end
    s
end

function projectout!(Y::AbstractMatrix{T}, projector::Projector) where T
    s = zero(T)
    n = size(Y, 2)
    for j = 1:n
        s += projectout!(view(Y, :, j), projector)
    end
    s
end

"""
    projectout!(y, projector)

Project out all components of `y` parallel to the radial orbitals
`projector.ϕs`.
"""
projectout!(y::RadialOrbital, projector::Projector) =
    projectout!(y.args[2], projector)
