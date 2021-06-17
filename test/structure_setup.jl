nℓ(o::Orbital) = o.n,o.ℓ
nℓ(o::SpinOrbital) = nℓ(o.orb)

function test_hydrogenic_orbitals(atom::Atom, tol=1e-8)
    R = radial_basis(atom)
    r = locs(R)
    χ = R[r,:]
    Z = charge(atom.potential)
    for o in atom.orbitals
        n,ℓ = nℓ(o)
        n > 3 && continue
        ρ = Z*r/n
        C = (Z/n)^(3/2)
        # Table 2.2 of
        #
        #   Foot, C. J. (2005). Atomic physics. Oxford New York: Oxford
        #   University Press.
        exact = C * r .* exp.(-ρ) .* if n == 1
            2
        elseif n == 2
            if ℓ == 0
                2*(1 .- ρ)
            else
                2/√3 * ρ
            end
        elseif n == 3
            if ℓ == 0
                2*(1 .- 2ρ .+ 2/3*ρ.^2)
            elseif ℓ == 1
                4*√2/3 * ρ .* (1 .- ρ/2)
            else
                2*√2/(3*√5) * ρ.^2
            end
        end
        @test χ*view(atom, o).args[2] ≈ exact rtol=tol
    end
end

@testset "Structure setup" begin
    @testset "Hydrogenic orbitals" begin
        @testset "Hydrogen" begin
            nucleus = pc"H"
            rₘₐₓ = 50

            @testset "$(grid_type)" for (grid_type,grid_kwargs,tol) ∈ [(:fd_staggered, (ρ=0.15,),            2e-3),
                                                                       (:fedvr,        (k=10,intervals=20,), 1e-3)]
                R = get_atom_grid(grid_type, rₘₐₓ, nucleus; grid_kwargs...)
                @testset "$(method)" for method in vcat([:arnoldi_shift_invert, :arnoldi],
                                                        grid_type == :fd ? :eigen : [])
                    atom = Atom(R, csfs([c"1s", c"2s", c"2p", c"3s", c"3p", c"3d"]),
                                nucleus, method=method, verbosity=4)
                    test_hydrogenic_orbitals(atom, tol)
                end
            end
        end
        @testset "Helium" begin
            nucleus = pc"He"
            rₘₐₓ = 300
            ρ = 0.25
            R = get_atom_grid(:fd_staggered, rₘₐₓ, nucleus, ρ=ρ)
            @testset "CSFs" begin
                atom = Atom(R, csfs([c"1s2", c"1s 2s", c"1s 2p", c"3s 3p", c"4s 3d", c"5s 5d"]),
                            nucleus, verbosity=4)
            end
            @testset "Spin-configurations" begin
                atom = Atom(R, spin_configurations([c"1s2", c"1s 2s", c"1s 2p", c"3s 3p", c"4s 3d", c"5s 5d"]),
                            nucleus, verbosity=4)
            end
        end
    end

    @testset "Hartree–Fock equations" begin
        rₘₐₓ = 300
        ρ = 0.25
        N = ceil(Int, rₘₐₓ/ρ + 1/2)

        @testset "Hydrogen" begin
            nucleus = pc"H"
            R = StaggeredFiniteDifferences(N, ρ)
            # atom = Atom(R, csfs(c"1s"), nucleus)
            atom = Atom(R, [spin_configurations(c"1s")[1]], nucleus)

            # # Set up energy expression manually.
            # eng = DiagonalIntegral(o"1s")
            # equations = map(Atoms.hf_equations(first(atom.configurations), eng)) do (orb,equation)
            #     Atoms.HFEquation(atom, equation, orb)
            # end

            # fock = Fock(atom, equations)

            fock = Fock(atom)

            @test length(fock.equations) == 1
            @test isapprox(energy(first(fock.equations)), -0.5, atol=2e-5)
        end

        @testset "Helium" begin
            nucleus = pc"He"
            R = StaggeredFiniteDifferences(N, ρ)
            atom = Atom(R, spin_configurations(c"1s2"), nucleus, verbosity=Inf)

            @testset "Arnoldi" begin
                fock = Fock(atom; verbosity=Inf)
                scf!(fock, verbosity=Inf, num_printouts=typemax(Int))
                display(fock)
                println()

                @test all(isapprox.(energy.(fock.equations), -24.587387/27.211, rtol=0.1))

                @testset "Initialize multiconfigurational atom" begin
                    new_atom = Atom(atom, spin_configurations([c"1s2", c"1s 2s"]))
                    for o in atom.orbitals
                        @test view(new_atom, o).args[2] == view(atom, o).args[2]
                    end
                    @test new_atom.orbitals == vcat(atom.orbitals, SpinOrbital(o"2s", 0, half(1)), SpinOrbital(o"2s", 0, half(-1)))
                end
            end

            @testset "LOBPCG" begin
                fock = Fock(atom; verbosity=Inf)
                scf!(fock, verbosity=Inf, num_printouts=typemax(Int),
                     method=:lobpcg)
                display(fock)
                println()

                @test all(isapprox.(energy.(fock.equations), -24.587387/27.211, rtol=0.1))
            end
        end

        @testset "Beryllium" begin
            nucleus = pc"Be"
            Z = charge(nucleus)
            rₘₐₓ = 15.0
            R = StaggeredFiniteDifferences(0.05, 0.2, 0.01, rₘₐₓ)
            atom = Atom(R, spin_configurations(c"1s2 2s2"),
                        nucleus, verbosity=Inf)

            fock = Fock(atom)

            test_hydrogenic_slater_integrals(fock) do i
                # The energy expressions for the 1s₀α/β and 2s₂α/β
                # spin-orbitals, respectively, should be identical.
                if i ∈ 1:2
                    [(:onebody, -8, 0.07),
                     # A 1s orbital is screened by another 1s orbital and 2 2s orbitals
                     (:direct, Z*(exact_F⁰s[(o"1s",o"1s",0)] + 2exact_F⁰s[(o"2s",o"1s",0)]), 0.2),
                     (:exchange, -Z*(exact_Gᵏs[(o"2s",o"1s",0)]), 0.2)]
                elseif i ∈ 3:4
                    [(:onebody, -2, 0.01),
                     # A 2s orbital is screened by 2 1s orbitals and another 2s orbital
                     (:direct, Z*(2exact_F⁰s[(o"2s",o"1s",0)] + exact_F⁰s[(o"2s",o"2s",0)]), 0.2),
                     (:exchange, -Z*(exact_Gᵏs[(o"2s",o"1s",0)]), 0.2)]
                end
            end

            optimize!(fock, ω=0.9, scf_method=:arnoldi, verbosity=2, num_printouts=typemax(Int))
            display(fock)
            println()

            @test isapprox(energy(fock.equations.equations[1]), -4.7326698, rtol=1e-4)
            @test isapprox(energy(fock.equations.equations[2]), -4.7326698, rtol=1e-4)
            @test isapprox(energy(fock.equations.equations[3]), -0.3092695, rtol=1e-4)
            @test isapprox(energy(fock.equations.equations[4]), -0.3092695, rtol=1e-4)
        end
    end
end
