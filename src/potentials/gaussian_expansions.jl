struct GaussianExpansion{T}
    n::Vector{Int}
    β::Vector{T}
    B::Vector{T}
end

Base.length(ge::GaussianExpansion) = length(ge.n)
Base.isempty(ge::GaussianExpansion) = length(ge) == 0

Base.iszero(ge::GaussianExpansion) = all(iszero, ge.B)

Base.:(==)(a::GaussianExpansion, b::GaussianExpansion) =
    a.n == b.n && a.β == b.β && a.B == b.B

Base.isapprox(a::GaussianExpansion, b::GaussianExpansion; kwargs...) =
    a.n == b.n && isapprox(a.β, b.β; kwargs...) && isapprox(a.B, b.B; kwargs...)

Base.hash(ge::GaussianExpansion, h::UInt) = hash((ge.n, ge.β, ge.B), h)

function (ge::GaussianExpansion{T})(r::AbstractVector) where T
    V = zeros(T, length(r))
    for k in 1:length(ge)
        V += ge.B[k] * r.^(ge.n[k]-2).*exp.(-ge.β[k]*r.^2)
    end
    V
end
