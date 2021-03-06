exact_energies = Dict(pc"He" => [-2.8616800, [o"1s" => -0.91795555]],
                      pc"Be" => [-14.573023, [o"1s" => -4.7326698, o"2s" => -0.3092695]],

                      pc"Ne" => [-128.54710, [o"1s" => -32.7724455, o"2s" => -1.93039095, o"2p" => -0.85040965]],
                      PseudoPotentials.NeonHF => [-34.709465, [o"2s" => -1.93039095, o"2p" => -0.85040965]],
                      PseudoPotentials.NeonWB => [-34.708785, [o"2s" => -1.93039095, o"2p" => -0.85040965]],

                      pc"Mg" => [-199.61463, [o"1s" => -49.0317255, o"2s" => -3.767718, o"2p" => -2.2822236, o"3s" => -0.25305275]],

                      pc"Ar" => [-526.81751, [o"1s" => -237.22070/2, o"2s" => -24.644306/2, o"2p" => -19.142932/2,
                                             o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],
                      PseudoPotentials.ArgonHF => [-20.849867, [o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],
                      PseudoPotentials.ArgonWB => [-20.884584, [o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],

                      pc"Ca" => [-676.75818, [o"1s" => -298.72744/2, o"2s" => -33.645481/2, o"2p" => -27.258531/2,
                                          o"3s" => -4.4907488/2, o"3p" => -2.6814114/2, o"4s" => -0.3910594/2]],
                      pc"Zn" => [-1777.8481, [o"1s" => -706.60909/2, o"2s" => -88.723452/2, o"2p" => -77.849691/2,
                                             o"3s" => -11.275642/2, o"3p" => -7.6787581/2, o"3d" => -1.5650843/2,
                                             o"4s" => -0.5850141/2]],
                      pc"Kr" => [-2752.0550, [o"1s" => -1040.3309/2, o"2s" => -139.80617/2, o"2p" => -126.01957/2,
                                             o"3s" => -21.698934/2, o"3p" => -16.663004/2, o"3d" => -7.6504697/2,
                                             o"4s" => -2.3058703/2, o"4p" => -1.0483734/2]],
                      pc"Xe" => [-7232.1384, [o"1s" => -2448.7956/2, o"2s" => -378.68024/2, o"2p" => -355.56490/2,
                                             o"3s" => -80.351324/2, o"3p" => -70.443321/2, o"3d" => -52.237736/2,
                                             o"4s" => -15.712602/2, o"4p" => -12.016674, o"4d" => -5.5557597,
                                             o"5s" => -1.8888279/2, o"5p" => -0.9145793/2]],

                      PseudoPotentials.XenonHF => [-14.989100, [o"5s" => -1.8888279/2, o"5p" => -0.9145793/2]],
                      PseudoPotentials.XenonWB => [-15.277055, [o"5s" => -1.8888279/2, o"5p" => -0.9145793/2]],
                      PseudoPotentials.XenonDF2c => [-328.74543, [o"5s" => -1.0097, ro"5p-" => -0.4915, ro"5p" => -0.4398]])

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

function si_round(q::Quantity; fspec="{1:+9.4f} {2:s}")
    v,u = ustrip(q), unit(q)
    if !iszero(v)
        u = shift_unit(u, log10(abs(v)))
        q = u(q)
    end
    format(fspec, ustrip(q), unit(q))
end

function energy_errors(fock, exact_energies, Δ, δ)
    atom = fock.quantum_system

    Eexact,orbExact = exact_energies
    exact_energies = if first(atom.orbitals) isa SpinOrbital
        vcat(Eexact, [repeat([E], degeneracy(o)) for (o,E) in orbExact]...)
    else
        vcat(Eexact, [E for (o,E) in orbExact]...)
    end

    orbital_refs = unique(nonrelorbital.(first.(orbExact)))

    H = zeros(1,1)
    Etot = SCF.energy_matrix!(H, fock)[1,1]
    energies = vcat(Etot, collect(SCF.energy(fock.equations.equations[i])/degeneracy(o)
                                  for (i,o) in enumerate(atom.orbitals)
                                  if nonrelorbital(Atoms.getspatialorb(o)) ∈ orbital_refs))
    errors = energies - exact_energies

    labels = vcat("Total", string.(collect(o for o in atom.orbitals
                                           if nonrelorbital(Atoms.getspatialorb(o)) ∈ orbital_refs)))

    Ha_formatter = (v,i) -> si_round(v*u"hartree")
    eV_formatter = (v,i) -> si_round(v*u"eV")

    pretty_table([labels exact_energies energies errors 27.211energies 27.211errors errors./abs.(exact_energies)],
                 ["", "HF limit", "Energy", "Δ", "Energy", "Δ", "Δrel"],
                 formatter=Dict(
                     2 => Ha_formatter,
                     3 => Ha_formatter,
                     4 => Ha_formatter,
                     5 => eV_formatter,
                     6 => eV_formatter,
                     7 => (v,i) -> si_round(100v*u"percent")
                 ),
                 highlighters=(Highlighter((v,i,j) -> abs(v[i,7])>0.2, foreground=:red, bold=true),
                               Highlighter((v,i,j) -> abs(v[i,7])>0.01, foreground=:yellow, bold=true),))

    @test abs(errors[1]) < Δ
    @test all(abs.(errors[2:end]) .< δ)
end

function atom_calc(nucleus::AbstractPotential, grid_type::Symbol, rₘₐₓ, ρ,
                   Δ, δ;
                   config_transform=identity, kwargs...)
    # If we're using a pseudopotential, we don't really need higher
    # order of the FEDVR basis functions in the first finite element,
    # since the orbitals almost vanish there.
    R,r = get_atom_grid(grid_type, rₘₐₓ, ρ, nucleus,
                        amend_order=nucleus isa PointCharge)

    atom = Atom(R, [spin_configurations(config_transform(ground_state(nucleus)))[1]],
                nucleus, eltype(R))

    fock = Fock(atom)

    optimize!(fock; kwargs...)
    energy_errors(fock, exact_energies[nucleus], Δ, δ)
end

@testset "Calculation accuracy" begin
    @testset "Helium" begin
        @testset "$(orb_type) orbitals" for (orb_type,config_transform) in [
            ("Non-relativistic", identity),
            ("Relativistic", relconfigurations)
        ]
            @testset "$(grid_type)" for (grid_type,Δ,δ) in [(:fedvr,6e-9,7e-9),
                                                            (:fd,4e-3,4e-3)]
                atom_calc(pc"He", grid_type, 10.0, 0.1, Δ, δ, ω=0.9,
                          config_transform=config_transform)
            end
        end
    end

    @testset "Beryllium" begin
        @testset "$(grid_type)" for (grid_type,ρ,Δ,δ) in [(:fedvr,0.2,6e-7,2e-6),
                                                          (:fd,0.05,0.02,0.009)]
            atom_calc(pc"Be", grid_type, 15.0, ρ, Δ, δ, ω=0.9, scf_method=:arnoldi)
        end
    end

    @testset "Neon" begin
        @testset "$nucleus" for (nucleus,Δ,δ) in [(pc"Ne",0.005,0.005),
                                                  (PseudoPotentials.NeonHF,0.2,0.05)]
            atom_calc(nucleus, :fedvr, 10, 0.2, Δ, δ,
                      ω=0.999, ωmax=1.0-1e-3, scf_method=:arnoldi)
        end
    end

    @testset "Xenon" begin
        @testset "$nucleus" for (nucleus,grid_type,ρ,Δ,δ,config_transform) in [
            (PseudoPotentials.XenonHF,:fd,0.1,4e-3,4e-3,identity),
            (PseudoPotentials.XenonDF2c,:fd,0.1,0.2,1e-2,relconfigurations)
        ]
            atom_calc(nucleus, grid_type, 7.0, ρ, Δ, δ,
                      ω=0.999, ωmax=1.0-1e-3,
                      config_transform=config_transform, scf_method=:arnoldi)
        end
    end
end
