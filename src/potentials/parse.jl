function parse_effective_core_potential_line(::Type{T}, pp_line::AbstractString) where T
    data = split(pp_line, ";")
    num_terms = parse(Int, data[1])
    length(data) == num_terms+2 || throw(ArgumentError("Unrecognizable potential string $(pp_line)"))
    n = Vector{Int}(undef, num_terms)
    β = Vector{T}(undef, num_terms)
    B = Vector{T}(undef, num_terms)
    map(1:num_terms) do i
        term = split(strip(data[i+1]),",")
        n[i] = parse(Int, term[1])
        β[i] = parse(T, term[2])
        B[i] = parse(T, term[3])
    end
    GaussianExpansion(n, β, B)
end

function parse_header(pp_string::String)
    lines = split(pp_string, "\n")
    gst_config = parse(Configuration{Orbital},
                       lstrip(lines[1], ['!', ' ']))
    i = findfirst(l -> first(l) != '!', lines)
    i === nothing && error("Could not find start of data")
    header = split(split(lines[i], ";")[1], ",")
    kind = header[1]

    element = String(header[2])
    Z = element_number(Symbol(element))
    nelec = num_electrons(gst_config)
    Z == nelec ||
        throw(ArgumentError("Atom number Z = $Z does not match number of electrons of ground state configuration $(gst_config) => $(nelec) electrons"))

    Q̃ = parse(Int, header[3])
    ncore = num_electrons(core(gst_config))
    Q̃ == ncore ||
        throw(ArgumentError("Charge modelled by effective core potential, Q̃ = $(Q̃), does not match number of core electrons in ground state configuration: $(core(gst_config)) => $(ncore) electrons"))

    Q = Z-Q̃ # Long-range charge of potential

    element, gst_config, Q, header, lines[i+1:end]
end

parse_reference(lines) =
    map(lines) do line
        lstrip(line, ['!', '[', ']', ' ', '0':'9'...])
    end |> l -> join(l, "\n")

function parse_effective_core_potential(::Type{T}, pp_string::String, relativistic::Bool) where T
    element, gst_config, Q, header, lines = parse_header(pp_string)

    ℓmax = parse(Int, header[4])
    ℓmax′ = parse(Int, header[5])
    ii = 0
    Vℓ = map(vcat(ℓmax,0:ℓmax-1)) do ℓ
        ℓ => parse_effective_core_potential_line(T, lines[ii+=1])
    end |> v -> last.(sort(v, by=first))
    Vℓ′ = map(1:ℓmax′) do ℓ′
        parse_effective_core_potential_line(T, lines[ii+=1])
    end |> Vector{GaussianExpansion{T}}
    reference = parse_reference(lines[ii+2:end])
    EffectiveCorePotential{T,relativistic}(element, gst_config, Q, Vℓ, Vℓ′, reference)
end

function parse_relativistic_effective_core_potential(::Type{T}, pp_string::String) where T
    element, gst_config, Q, header, lines = parse_header(pp_string)

    nκ₋ = parse(Int, header[4])
    nκ₊ = parse(Int, header[5])
    ii = 0
    V₋ = map(1:nκ₋) do ℓ
        parse_effective_core_potential_line(T, lines[ii+=1])
    end
    V₊ = map(1:nκ₊) do ℓ′
        parse_effective_core_potential_line(T, lines[ii+=1])
    end # |> Vector{GaussianExpansion{T}}
    reference = parse_reference(lines[ii+2:end])
    RelativisticEffectiveCorePotential{T}(element, gst_config, Q, V₋, V₊, reference)
end

macro ECP_str(pp_string, suffix="")
    if suffix == "" || suffix == "sso"
        parse_effective_core_potential(Float64, pp_string, suffix=="sso")
    elseif suffix == "r"
        parse_relativistic_effective_core_potential(Float64, pp_string)
    else
        throw(ArgumentError("Unkown effective core potential suffix $(suffix)"))
    end
end
