exact_energies = Dict(
    pc"He" => [-2.8616800, [o"1s" => -0.91795555]],
    pc"Li" => [-7.4327269, [o"1s" => -4.9554827/2, o"2s" => -0.3926457/4]],
    pc"Be" => [-14.573023, [o"1s" => -4.7326698, o"2s" => -0.3092695]],
    pc"B" => [-24.529061, [o"1s" => -15.390670/2, o"2s" => -0.9894116/2, o"2p" => -0.6197128/12]],
    pc"C" => [-37.688619, [o"1s" => -22.651038/2, o"2s" => -1.4112549/2, o"2p" => -0.8666811/12]],

    pc"Ne" => [-128.54710, [o"1s" => -32.7724455, o"2s" => -1.93039095, o"2p" => -0.85040965]],
    ECPs.NeonHF => [-34.685316, [o"2s" => -1.93039095, o"2p" => -0.85040965]],
    ECPs.NeonWB => [-34.708785, [o"2s" => -1.93039095, o"2p" => -0.85040965]],

    pc"Mg" => [-199.61463, [o"1s" => -49.0317255, o"2s" => -3.767718, o"2p" => -2.2822236, o"3s" => -0.25305275]],

    pc"Ar" => [-526.81751, [o"1s" => -237.22070/2, o"2s" => -24.644306/2, o"2p" => -19.142932/2,
                            o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],
    ECPs.ArgonHF => [-20.849867, [o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],
    ECPs.ArgonWB => [-20.884584, [o"3s" => -2.5547063/2, o"3p" => -1.1820348/2]],

    pc"Ca" => [-676.75818, [o"1s" => -298.72744/2, o"2s" => -33.645481/2, o"2p" => -27.258531/2,
                            o"3s" => -4.4907488/2, o"3p" => -2.6814114/2, o"4s" => -0.3910594/2]],
    ECPs.CalciumDF => [-36.44769593,
                       [o"3s" => -4.4907488/2, # HF limit, above
                        o"3p" => -(2*1.346583638 + 4*1.332801306)/6, o"4s" => -0.196693786]],

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
    ECPs.XenonHF => [-14.989100, [o"5s" => -1.8888279/2, o"5p" => -0.9145793/2]],
    ECPs.XenonWB => [-15.277055, [o"5s" => -1.8888279/2, o"5p" => -0.9145793/2]],
    ECPs.XenonDF => [-328.74543, [o"4s" => -15.712602/2, o"4p" => -12.016674, o"4d" => -5.5557597,
                                  o"5s" => -1.0097, ro"5p-" => -0.4915, ro"5p" => -0.4398]],
    ECPs.XenonDF2c => [-328.36818, [o"5s" => -1.0097, ro"5p-" => -0.4915, ro"5p" => -0.4398]],
)

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

function total_energy(fock)
    atom = fock.quantum_system
    c = atom.mix_coeffs
    nc = length(c)

    H = zeros(nc,nc)
    SCF.energy_matrix!(H, fock)
    Etot = dot(c, H, c)
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

    energies = vcat(total_energy(fock),
                    collect(SCF.energy(fock.equations.equations[i])/degeneracy(o)
                            for (i,o) in enumerate(atom.orbitals)
                            if nonrelorbital(Atoms.getspatialorb(o)) ∈ orbital_refs))
    errors = energies - exact_energies

    labels = vcat("Total", string.(collect(o for o in atom.orbitals
                                           if nonrelorbital(Atoms.getspatialorb(o)) ∈ orbital_refs)))


    pretty_table([labels exact_energies energies errors 27.211energies 27.211errors errors./abs.(exact_energies)],
                 header=["", "HF limit", "Energy", "Δ", "Energy", "Δ", "Δrel"],
                 formatters=((v,i,j) -> j ∈ 2:4 ? si_round(v*u"hartree") : v,
                             (v,i,j) -> j ∈ 5:6 ? si_round(v*u"eV") : v,
                             (v,i,j) -> j == 7 ? si_round(100v*u"percent") : v),
                 highlighters=(Highlighter((v,i,j) -> abs(v[i,7])>0.2, foreground=:red, bold=true),
                               Highlighter((v,i,j) -> abs(v[i,7])>0.01, foreground=:yellow, bold=true),))

    @test abs(errors[1]) < Δ
    @test all(abs.(errors[2:end]) .< δ)
end

function atom_calc(nucleus::AbstractPotential, grid_type::Symbol, rₘₐₓ, grid_kwargs,
                   Δ, δ;
                   config_transform=identity, kwargs...)
    R = get_atom_grid(grid_type, rₘₐₓ, nucleus; grid_kwargs...)

    atom = Atom(R, spin_configurations(config_transform(ground_state(nucleus))),
                nucleus, eltype(R), verbosity=0)
    # For open-shell atoms, we simply assume an equal distribution
    # among the available spin-configurations. This is correct for
    # e.g. lithium.
    atom.mix_coeffs .= 1 ./ √(length(atom.configurations))

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
            @testset "$(grid_type)" for (grid_type,grid_kwargs,optim_kwargs,Δ,δ) in [
                (:fd_uniform,   (ρ=0.1,),             (),             1e-2, 1e-2),
                (:fd_staggered, (ρ=0.1,),             (),             4e-3, 4e-3),
                (:fd_loglin,    (ρ=0.1,),             (),             1e-3, 1e-3),
                (:implicit_fd,  (ρ=0.1,),             (),             3e-3, 3e-3),
                (:fedvr,        (intervals=10,k=10,), (),             6e-9, 7e-9),
                (:bsplines,     (k=4,m=2,),           (scf_iters=0,), 8e-4, 8e-4)
            ]
                atom_calc(pc"He", grid_type, 10.0, grid_kwargs, Δ, δ;
                          ω=0.9, config_transform=config_transform,
                          optim_kwargs...)
            end
        end
    end

    @testset "Lithium" begin
        @testset "$(grid_type)" for (grid_type,grid_kwargs,Δ,δ) in [
            (:fd_uniform,   (ρ=0.1,),                3e-2, 1e-2),
            (:fd_staggered, (ρ=0.1,),                3e-3, 2e-3),
            (:fd_loglin,    (ρ=0.05,),               5e-4, 3e-4),
            # (:implicit_fd,  (ρ=0.1,),              0.0, 0.0), # This does not converge
            (:fedvr,        (intervals=10,k=10,),    1e-6, 8e-5),
            (:bsplines,     (intervals=30,k=4,m=2,), 2e-4, 3e-5)
        ]
            atom_calc(pc"Li", grid_type, 20.0, grid_kwargs, Δ, δ;
                      ω=0.9,
                      # Lithium is an open-shell atom, so we keep the
                      # mixing coefficients fixed, since we run an
                      # unrestricted Hartree–Fock calculation with
                      # Slater determinants, which would otherwise
                      # collapse into one of the spin projections.
                      update_mixing_coefficients=false,
                      scf_iters=0)
        end
    end

    @testset "Beryllium" begin
        @testset "$(grid_type)" for (grid_type,grid_kwargs,optim_kwargs,Δ,δ) in [
            (:fedvr,        (k=10,intervals=8,), (scf_iters=0,),        6e-7, 2e-6),
            (:fd_staggered, (ρ=0.2,),            (scf_method=:arnoldi,), 0.02, 0.009) # This works, but takes forever
        ]
            atom_calc(pc"Be", grid_type, 15.0, grid_kwargs, Δ, δ;
                      ω=0.9,
                      linesearch=LineSearches.MoreThuente(), optim_kwargs...)
        end
    end

    @testset "Neon" begin
        @testset "$nucleus" for (nucleus,Δ,δ) in [(pc"Ne",0.005,0.005),
                                                  (ECPs.NeonHF,0.2,0.05)]
            atom_calc(nucleus, :fedvr, 10.0, (k=10,intervals=6,), Δ, δ,
                      ω=0.999, ωmax=1.0-1e-3, scf_method=:arnoldi)
        end
    end

    @testset "Argon" begin
        atom_calc(pc"Ar", :bsplines, 8.0, (intervals=30,k=4,m=2), 8e-3, 2e-3;
                  g_tol=1e-8, scf_iters=0, opt_iters=1000,
                  linesearch=LineSearches.MoreThuente())
    end

    @testset "Xenon" begin
        @testset "$nucleus — $(grid_type)" for (nucleus,grid_type,grid_kwargs,Δ,δ,config_transform) in [
            (ECPs.XenonHF,   :fedvr,    (intervals=10, k=5,),     3e-2, 5e-3, identity)
            (ECPs.XenonDF2c, :fedvr,    (intervals=10, k=5,),     3e-2, 5e-3, relconfigurations)
            # (ECPs.XenonDF2c, :bsplines, (intervals=10, k=4, m=2), 1e-3, 3e-3, relconfigurations)
        ]
            atom_calc(nucleus, grid_type, 7.0, grid_kwargs, Δ, δ,
                      ω=0.999, ωmax=1.0-1e-3,
                      config_transform=config_transform, scf_iters=0)
        end
    end
end
