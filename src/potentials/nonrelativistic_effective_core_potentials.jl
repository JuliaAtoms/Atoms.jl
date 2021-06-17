struct EffectiveCorePotential{T,relativistic} <: AbstractEffectiveCorePotential{T}
    name::String
    gst_config::Configuration{Orbital{Int}}
    Q::Int
    Vℓ::Vector{GaussianExpansion{T}} # ℓ ∈ 0:ℓmax
    Vℓ′::Vector{GaussianExpansion{T}} # ℓ′ ∈ 1:ℓmax′
    reference::String
end

Base.:(==)(a::EffectiveCorePotential{<:Any,ra}, b::EffectiveCorePotential{<:Any,rb}) where {ra,rb} =
    ra == rb && a.gst_config == b.gst_config && a.Q == b.Q &&
    a.Vℓ == b.Vℓ && a.Vℓ′ == b.Vℓ′

Base.:(≈)(a::EffectiveCorePotential{<:Any,ra}, b::EffectiveCorePotential{<:Any,rb}) where {ra,rb} =
    ra == rb && a.gst_config == b.gst_config && a.Q == b.Q &&
    a.Vℓ ≈ b.Vℓ && a.Vℓ′ ≈ b.Vℓ′

Base.hash(pp::EffectiveCorePotential{<:Any,relativistic}, h::UInt) where relativistic =
    hash((relativistic, pp.gst_config, pp.Q, pp.Vℓ, pp.Vℓ′), h)

const ScalarSORelativisticEffectiveCorePotential{T} = EffectiveCorePotential{T,true}

charge(pp::EffectiveCorePotential) = num_electrons(pp.gst_config) + (pp.Q-num_electrons(peel(pp.gst_config)))
ground_state(pp::EffectiveCorePotential) = pp.gst_config

function Base.show(io::IO, pp::EffectiveCorePotential{T,relativistic}) where {T,relativistic}
    if relativistic
        write(io, "Scalar-relativistic + spin–orbit effective core potential")
    else
        write(io, "Effective core potential")
    end
    write(io, " for $(pp.name) ($(pp.gst_config)), Z = $(charge(pp))")
end

function Base.show(io::IO, ::MIME"text/plain", pp::EffectiveCorePotential)
    show(io, pp)
    println(io)
    ℓmax = length(pp.Vℓ)-1
    ℓmax′ = length(pp.Vℓ′)
    print(io, "Long-range Q = $(pp.Q), ℓ ∈ 0:$(ℓmax)")
    ℓmax′ > 0 && print(io, ", ℓ′ ∈ 1:$(length(pp.Vℓ′))")
    println(io, "\nData from \"$(pp.reference)\"")
    ℓk = map(enumerate(0:ℓmax)) do (i,ℓ)
        nk = length(pp.Vℓ[i])
        [[spectroscopic_label(ℓ);repeat([""],nk-1)] 1:nk]
    end |> v -> vcat(v...)
    ℓ′k = map(enumerate(1:ℓmax′)) do (i,ℓ′)
        nk = length(pp.Vℓ′[i])
        [[spectroscopic_label(ℓ′);repeat([""],nk-1)] 1:nk]
    end |> v -> vcat(v...)
    data = [ℓk vcat([Any[V.n V.β V.B]
                     for V in pp.Vℓ]...)]
    headers = ["ℓ", "k", "n", "β", "B"]
    if ℓmax′ > 0
        data = [data vcat(repeat([""], length(pp.Vℓ[1]), 5),
                          [ℓ′k vcat([Any[V.n V.β V.B]
                                     for V in pp.Vℓ′]...)],
                          repeat([""], 1, 5))]
        headers = vcat(headers, ["ℓ′", "k", "n", "β", "B"])
    end
    pretty_table(io, data, header=headers)
end

#=
\[V_{\mathrm{pp}}(\vec{r})=
-\frac{Q}{r}+
\sum_{\ell jk}
B_{\ell j k}
r^{n_k-2}
\exp(-\beta_{\ell j k} r^2)
\mathcal{P}_{\ell j}\]
=#

function (pp::ScalarSORelativisticEffectiveCorePotential{T})(orb::SpinOrbital, r::AbstractVector{T}) where T
    # This returns the diagonal, average part of the relativistic ECP.
    V = -pp.Q./r

    ℓ = orb.orb.ℓ
    ℓ >= length(pp.Vℓ) && return V

    V += pp.Vℓ[ℓ+1](r)

    V
end

function spin_orbit_potential(pp::ScalarSORelativisticEffectiveCorePotential{T}, r::AbstractVector{T},
                              a::SpinOrbital{<:Orbital}, b::SpinOrbital{<:Orbital}) where T
    # This returns the (off-)diagonal, spin–orbit part of the
    # relativistic ECP.
    s = half(1)
    ℓ,mℓa = jmⱼ(a)
    mas  = spin(a)
    ℓb,mℓb = jmⱼ(b)
    mbs = spin(b)

    @assert ℓ == ℓb

    V = zeros(T, length(r))
    (ℓ > length(pp.Vℓ′) || iszero(ℓ)) && return V

    # Eq. (30) of
    # - Dolg, M. (2016). Relativistic Effective Core Potentials. In (Eds.),
    #   Handbook of Relativistic Quantum Chemistry (pp. 449–478). : Springer
    #   Berlin Heidelberg.
    coeff = ℓ*clebschgordan(
        ℓ,mℓa,s,mas,ℓ+s
    )*clebschgordan(
        ℓ,mℓb,s,mbs,ℓ+s
    )
    if abs(mℓa+mas) < ℓ
        coeff -= (ℓ+1)*clebschgordan(
            ℓ,mℓa,s,mas,ℓ-s
        )*clebschgordan(
            ℓ,mℓb,s,mbs,ℓ-s
        )
    end

    # According to
    # http://www.tc.uni-koeln.de/cgi-bin/pp.pl?language=en,job=getreadme
    # the spin–orbit potentials are pre-multiplied by 2/(2ℓ+1), but we
    # need 1/(2ℓ+1).
    coeff /= 2

    # We can index by ℓ, since Vℓ′ does not contain any entry for ℓ = s
    V += coeff*pp.Vℓ′[ℓ](r)

    V
end

function (pp::EffectiveCorePotential{T,false})(orb::SpinOrbital, r::AbstractVector{T}) where T
    V = -pp.Q./r
    ℓ = orb.orb.ℓ
    ℓ >= length(pp.Vℓ) && return V
    V += pp.Vℓ[ℓ+1](r)

    V
end

(pp::EffectiveCorePotential{T})(orb::AbstractOrbital, r::T) where T = pp(orb, [r])[1]
