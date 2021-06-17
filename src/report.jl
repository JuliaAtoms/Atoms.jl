function shift_unit(u::U, d) where {U<:Unitful.Unit}
    d = Int(3floor(Int, d/3))
    iszero(d) && return u
    for tens = Unitful.tens(u) .+ (d:(-3*sign(d)):0)
        haskey(Unitful.prefixdict, tens) && return U(tens, u.power)
    end
    u
end

function shift_unit(u::Unitful.FreeUnits, d)
    tu = typeof(u)
    us,ds,a = tu.parameters

    uu = shift_unit(us[1], d)

    Unitful.FreeUnits{(uu,us[2:end]...), ds, a}()
end

format_number(fspec, v::Real) = format("{1:+$(fspec)}", v)
function format_number(fspec, v::Complex)
    iv = imag(v)
    format("{1:+$(fspec)} {2:s} {3:$(fspec)}im", real(v), iv ≥ 0 ? "+" : "-", abs(iv))
end

function si_round(q::Quantity; fspec="9.4f")
    v,u = ustrip(q), unit(q)
    if !iszero(v)
        u = shift_unit(u, log10(abs(v)))
        q = u(q)
    end
    format_number(fspec, ustrip(q))*" $(string(unit(q)))"
end

function radial_moments(atom::Atom{T}, powers) where T
    R = radial_basis(atom)
    r = axes(R,1)

    rᵏ = collect(R'*QuasiDiagonal(r.^k)*R for k in powers)
    rms = zeros(T, length(atom.orbitals), length(powers))
    for i = eachindex(atom.orbitals)
        ϕ = view(atom.radial_orbitals.args[2], :, i)
        for (j,rᵏ) in enumerate(rᵏ)
            rms[i,j] = dot(ϕ, rᵏ, ϕ)
        end
    end
    rms
end

function report(io::IO, fock::Fock{<:Atom{T}};
                orbital_energies=[:total],
                powers = (2,1,-1,-2,-3)) where T
    atom = fock.quantum_system
    Z = charge(atom.potential)
    N = num_electrons(atom)
    println(io, "Atom{$(T)}")
    println(io, "R = ", radial_basis(atom))
    show(io, atom.potential)
    println(io, ", $(N) e⁻ ⇒ Q = $(Z-N)")

    Eₜₒₜ = SCF.energy(fock, :total_energy)*u"hartree"
    Eₖᵢₙ = SCF.energy(fock, :kinetic_energy)*u"hartree"
    Eₚₒₜ = Eₜₒₜ - Eₖᵢₙ
    VT = Eₚₒₜ/Eₖᵢₙ

    # At some point we will support comparing to reference data, taken from e.g.
    # - Saito, S. L. (2009). Hartree-Fock-Roothaan energies and expectation
    #   values for the neutral atoms he to uuo: the B-spline expansion
    #   method. Atomic Data and Nuclear Data Tables, 95(6),
    #   836–870. http://dx.doi.org/10.1016/j.adt.2009.06.001
    total_energy_errors = Any["", "", "", VT+2]

    pretty_table(io, hcat(["Total energy", "Kinetic energy", "Potential energy", "Virial ratio"],
                          vcat(si_round.((Eₜₒₜ, Eₖᵢₙ, Eₚₒₜ), fspec="15.10f")..., format_number("15.10f", VT)),
                          total_energy_errors),
                 noheader=true,
                 alignment=:l,tf=tf_borderless)

    g = degeneracy.(atom.orbitals)
    ϵ = zeros(T, length(g), length(orbital_energies))
    for (j,which) in enumerate(orbital_energies)
        ϵ[:,j] = SCF.orbital_energies(fock, which)
    end
    ϵ = si_round.((ϵ ./ g) * u"hartree")

    rms = radial_moments(atom, powers)

    # Should also compute electron density ρ(0) and Kato cusp

    println(io)

    orbital_energy_headers = getindex.(Ref(Dict(:total => "ϵ",
                                                :onebody => "ϵ(1)",
                                                :twobody => "ϵ(2)",
                                                :direct => "ϵ(J)",
                                                :exchange => "ϵ(K)")),
                                       orbital_energies)

    pretty_table(io, [atom.orbitals ϵ rms],
                 vcat("Orbital", orbital_energy_headers, ["⟨r"*(k≠1 ? to_superscript(k) : "")*"⟩" for k in powers]),
                 tf=tf_borderless)
end

report(fock::Fock; kwargs...) = report(stdout, fock; kwargs...)
