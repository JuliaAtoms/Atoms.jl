struct RelativisticEffectiveCorePotential{T} <: AbstractEffectiveCorePotential{T}
    name::String
    gst_config::Configuration{Orbital{Int}}
    Q::Int
    V₋::Vector{GaussianExpansion{T}} # For negative κ
    V₊::Vector{GaussianExpansion{T}} # For positive κ
    reference::String
end

function RelativisticEffectiveCorePotential(p::EffectiveCorePotential{T,true}; threshold=1e-4) where T
    isempty(p.Vℓ′) && throw(ArgumentError("No spin–orbit (difference) potential present"))

    V₋ = Vector{GaussianExpansion{T}}()
    V₊ = Vector{GaussianExpansion{T}}()

    push!(V₋, p.Vℓ[1])

    Vℓ = p.Vℓ[2:findlast(!iszero, p.Vℓ)]
    Vℓ′ = p.Vℓ′[1:end]

    length(Vℓ) == length(Vℓ′) && length.(Vℓ) == length.(Vℓ′) || throw(ArgumentError("Cannot join expansion of different lengths"))

    function push_ge!(V, n, β, d)
        sel = abs.(d) .> threshold
        push!(V, GaussianExpansion(n[sel], β[sel], d[sel]))
    end

    for (ℓ,(Vℓ,Vℓ′)) in enumerate(zip(Vℓ, Vℓ′))
        Vℓ.n == Vℓ′.n && Vℓ.β == Vℓ′.β || throw(ArgumentError("Gaussian expansion exponents and radial monomials must agree"))

        # Convert from average + spin–orbit (difference) potentials to
        # V₋ and V₊ potentials; the spin-orbit potentials are
        # pre-multiplied by 2/(2ℓ+1), so we have to undo this.
        d = ([ℓ/(2ℓ+1) (ℓ+1)/(2ℓ+1); -1 1] \ [Vℓ.B (2ℓ+1)/2*Vℓ′.B]')

        push_ge!(V₋, Vℓ.n, Vℓ.β, d[2,:])
        push_ge!(V₊, Vℓ.n, Vℓ.β, d[1,:])
    end

    RelativisticEffectiveCorePotential(p.name, p.gst_config, p.Q, filter(!isempty, V₋), filter(!isempty, V₊), p.reference)
end

Base.:(==)(a::RelativisticEffectiveCorePotential, b::RelativisticEffectiveCorePotential) =
    a.gst_config == b.gst_config && a.Q == b.Q &&
    a.V₋ == b.V₋ && a.V₊ == b.V₊

Base.isapprox(a::RelativisticEffectiveCorePotential, b::RelativisticEffectiveCorePotential; rtol=1e-6) =
    a.gst_config == b.gst_config && a.Q == b.Q  &&
    all(isapprox.(a.V₋, b.V₋; rtol=rtol)) &&
    all(isapprox.(a.V₊, b.V₊; rtol=rtol))

Base.hash(pp::RelativisticEffectiveCorePotential, h::UInt) =
    hash((pp.gst_config, pp.Q, pp.V₋, pp.V₊), h)

charge(pp::RelativisticEffectiveCorePotential) = num_electrons(pp.gst_config) + (pp.Q-num_electrons(peel(pp.gst_config)))
ground_state(pp::RelativisticEffectiveCorePotential) = pp.gst_config

Base.show(io::IO, pp::RelativisticEffectiveCorePotential{T}) where {T} =
    write(io, "Relativistic effective core potential for $(pp.name) ($(pp.gst_config)), Z = $(charge(pp))")

function κrange(pp::RelativisticEffectiveCorePotential)
    κ₋max = length(pp.V₋)
    κ₊max = length(pp.V₊)

    sort(vcat(-1:-1:-κ₋max,1:κ₊max), lt=(a,b)->abs(a)<abs(b) || abs(a)==abs(b) && a<0)
end

function Base.show(io::IO, ::MIME"text/plain", pp::RelativisticEffectiveCorePotential)
    show(io, pp)
    println(io)
    κ₋max = length(pp.V₋)
    κ₊max = length(pp.V₊)
    print(io, "Long-range Q = $(pp.Q), κ₋ ∈ -1:-1:$(-κ₋max)")
    κ₊max > 0 && print(io, ", κ₊ ∈ 1:$(κ₊max)")
    println(io, "\nData from \"$(pp.reference)\"")

    κs = κrange(pp)

    data = map(κs) do κ
        data = κ > 0 ? pp.V₊[κ] : pp.V₋[-κ]
        nk = length(data)
        ℓ = κ2ℓ(κ)
        j = κ2j(κ)
        hcat([spectroscopic_label(ℓ);repeat([""],nk-1)],
             [j;repeat([""],nk-1)],
             1:nk, data.n, data.B, data.β)
    end |> v -> vcat(v...)

    headers = ["ℓ", "j", "k", "n", "B", "β"]

    pretty_table(io, data, header=headers)
end

function (pp::RelativisticEffectiveCorePotential{T})(orb::RelativisticOrbital, r::AbstractVector{T}) where T
    V = -pp.Q./r

    κ = orb.κ

    data = if κ < 0
        -κ > length(pp.V₋) && return V
        pp.V₋[-κ]
    else
        κ > length(pp.V₊) && return V
        pp.V₊[κ]
    end
    V += data(r)

    V
end

function spin_orbit_potential(pp::RelativisticEffectiveCorePotential{T}, r::AbstractVector{T},
                              a::SpinOrbital{<:Orbital}, b::SpinOrbital{<:Orbital}) where T
    # This returns the (off-)diagonal, spin–orbit part of the
    # relativistic ECP.
    s = half(1)
    ℓ,mℓa = jmⱼ(a)
    mas  = spin(a)
    ℓb,mℓb = jmⱼ(b)
    mbs = spin(b)

    V = zeros(T, length(r))

    ℓ == ℓb || return V

    κs = κrange(pp)

    for j = [ℓ+s, ℓ-s]
        κ = AtomicLevels.ℓj2κ(ℓ,j)
        j ≥ abs(mℓa+mas) && κ ∈ κs || continue

        coeff = clebschgordan(
            ℓ,mℓa,s,mas,j
        )*clebschgordan(
            ℓ,mℓb,s,mbs,j
        )
        V += coeff*((κ > 0 ? pp.V₊[κ] : pp.V₋[-κ])(r))
    end

    V
end

function spin_orbit_potential(pp::RelativisticEffectiveCorePotential{T}, r::AbstractVector{T},
                              a::SpinOrbital{<:RelativisticOrbital}, b::SpinOrbital{<:RelativisticOrbital}) where T
    # This returns the (off-)diagonal, spin–orbit part of the
    # relativistic ECP.
    κ,κb = a.orb.κ,b.orb.κ

    V = zeros(T, length(r))

    a.orb.ℓ == b.orb.ℓ && a.orb.j == b.orb.j && a.m == b.m || return V

    κs = κrange(pp)

    if κ ∈ κs # Is this correct? Perhaps for larger ℓ/j, we should use
              # the last pseudopotential term, or is it overshadowed
              # by the centrifugal potential?
        V += (κ > 0 ? pp.V₊[κ] : pp.V₋[-κ])(r)
    end

    V
end

function (pp::RelativisticEffectiveCorePotential{T})(orb::SpinOrbital, r::AbstractVector{T}) where T
    V = -pp.Q./r

    V += spin_orbit_potential(pp, r, orb, orb)

    V
end


(pp::RelativisticEffectiveCorePotential{T})(orb::AbstractOrbital, r::T) where T = pp(orb, [r])[1]
