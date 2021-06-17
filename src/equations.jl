# * Atomic System of Equations

"""
    AtomicEquations(atom, equations, integrals)

Structure representing the (e.g. Hartree–Fock) `equations` for `atom`,
along with all `integrals` that are shared between the `equations`.
"""
mutable struct AtomicEquations{T, B<:AbstractQuasiMatrix,
                               O<:AbstractOrbital,
                               A<:Atom{T,B,O},
                               Equations<:AbstractVector{<:AtomicOrbitalEquation}}
    atom::A
    equations::Equations
    integrals::Vector{OrbitalIntegral}
    observables::Dict{Symbol,<:Observable}
end

# Iteration interface
Base.length(hfeqs::AtomicEquations) = length(hfeqs.equations)
Base.iterate(iter::AtomicEquations, args...) = iterate(iter.equations, args...)

Base.show(io::IO, eqs::AtomicEquations{T}) where T =
    write(io, "$(length(eqs)) AtomicEquations{$T}")

"""
    update!(equations::AtomicEquations[, atom::Atom])

Recompute all integrals using the current values for the radial
orbitals (optionally specifying which `atom` from which the orbitals
are taken).
"""
function SCF.update!(equations::AtomicEquations, args...; kwargs...)
    Threads.@threads for integral in equations.integrals
        SCF.update!(integral, args...; kwargs...)
    end
    # Updating the operators is apparently not thread-safe at the
    # moment. Why?
    for eq in equations.equations
        update!(SCF.hamiltonian(eq), args...; kwargs...)
    end
end

"""
    energy_matrix!(H, hfeqs::AtomicEquations[, which=:energy])

Compute the energy matrix by computing the energy observable and
storing it in `H`. Requires that `hfeqs` has the `:energy` and
`:kinetic_energy` [`Observable`](@ref)s registered (this is the
default).
"""
function SCF.energy_matrix!(H::HM, hfeqs::AtomicEquations,
                            which::Symbol=:total_energy) where {HM<:AbstractMatrix}
    observable = hfeqs.observables[which]
    observe!(H, observable)
    H
end

# * Orbital symmetries
"""
    find_symmetries(orbitals)

Group all orbitals according to their symmetries, e.g. ℓ for
`Orbital`s. This is used to determine which off-diagonal Lagrange
multipliers are necessary to maintain orthogonality.
"""
find_symmetries(orbitals::Vector{O}) where {O<:AbstractOrbital} =
    merge!(vcat, [Dict(symmetry(orb) => O[orb])
                  for orb in orbitals]...)

SCF.symmetries(atom::Atom) =
    map(values(find_symmetries(atom.orbitals))) do orbitals
        orbital_index.(Ref(atom), orbitals)
    end

# * Setup orbital equations

function get_operator(::FieldFreeOneBodyHamiltonian, atom::Atom,
                      orbital::aO, source_orbital::bO; kwargs...) where {aO,bO}
    symmetry(orbital) == symmetry(source_orbital) ||
        throw(ArgumentError("FieldFreeOneBodyHamiltonian between orbitals of different symmetries: ⟨$(orbital)|𝔥₀|$(source_orbital)⟩"))
    AtomicOneBodyHamiltonian(atom, source_orbital)
end

for (i,HT) in enumerate([:KineticEnergyHamiltonian, :PotentialEnergyHamiltonian])
    @eval begin
        get_operator(::$HT, atom::Atom, orbital::aO, ::bO; kwargs...) where {aO,bO} =
            AtomicOneBodyHamiltonian(one_body_hamiltonian(Tuple, atom, orbital)[$i],
                                     orbital)
    end
end

function get_operator(op::CoulombPotentialMultipole, atom::Atom,
                      orbital::aO, source_orbital::bO; kwargs...) where {aO,bO}
    a,b = op.a[1],op.b[1]
    if orbital == source_orbital
        HFPotential(:direct, op.o.k, a, b, atom, op.o.g; kwargs...)
    else
        HFPotential(:exchange, op.o.k, a, source_orbital, atom, op.o.g; kwargs...)
    end
end

get_operator(top::IdentityOperator, atom::Atom,
             ::aO, source_orbital::bO; kwargs...) where {aO,bO} =
                 SourceTerm(top, source_orbital, view(atom, source_orbital))
SCF.update!(::IdentityOperator; kwargs...) = nothing
SCF.update!(::IdentityOperator, ::Atom; kwargs...) = nothing

function get_operator(op::Op, atom::Atom, orbital::aO, source_orbital::bO;
                      kwargs...) where {Op<:Union{<:RadialOperator,<:CoulombRepulsionPotential}, aO, bO}
    if orbital == source_orbital
        # In this case, the operator is diagonal in orbital space,
        # i.e. it maps an orbital onto itself.
        op
    else
        # In this case, the operator maps from source_orbital to
        # orbital.
        SourceTerm(op, source_orbital, view(atom, source_orbital))
    end
end

function get_operator(M::AbstractMatrix, atom::Atom, a::aO, b::bO; kwargs...) where {aO, bO}
    R = radial_basis(atom)
    get_operator(applied(*, R, M, R'), atom, a, b; kwargs...)
end

# RadialOperators (which are built from matrices), are independent of
# the atom.
SCF.update!(::RadialOperator; kwargs...) = nothing
SCF.update!(::RadialOperator, ::Atom; kwargs...) = nothing

function get_operator(top::Projector, ::Atom, orbital::aO, source_orbital::bO) where {aO,bO}
    orbital == source_orbital || return 0
    SourceTerm(IdentityOperator{1}(), source_orbital,
               top.ϕs[findfirst(isequal(source_orbital), top.orbitals)])
end

"""
    generate_atomic_orbital_equations(atom::Atom, eqs::MCEquationSystem,
                                      integrals, integral_map)

For each variationally derived orbital equation in `eqs`, generate the
corresponding [`AtomicOrbitalEquation`](@ref).
"""
function generate_atomic_orbital_equations(atom::Atom{T,B,O}, eqs::MCEquationSystem,
                                           integrals::Vector,
                                           integral_map::Dict{Any,Int},
                                           symmetries::Dict;
                                           verbosity=0,
                                           kwargs...) where {T,B,O}
    p = if verbosity > 0
        @info "Generating atomic orbital equations"
        Progress(length(eqs.equations))
    end
    map(eqs.equations) do equation
        orbital = equation.orbital
        terms = Vector{OrbitalHamiltonianTerm{O,O,T}}()

        for (integral,equation_terms) in equation.terms
            if integral > 0
                operator = get_integral(integrals, integral_map, eqs.integrals[integral])
                # We first add all terms with operators diagonal in
                # orbital space.
                pushterms!(terms, operator, filter(t -> t.source_orbital == orbital, equation_terms),
                           integrals, integral_map, eqs.integrals)
                # We then add those terms are that are off-diagonal in
                # orbital space.
                for t in filter(t -> t.source_orbital ≠ orbital, equation_terms)
                    pushterms!(terms, SourceTerm(operator, t.source_orbital, view(atom, t.source_orbital)), [t],
                               integrals, integral_map, eqs.integrals)
                end
            else
                for t in equation_terms
                    operator = get_operator(t.operator, atom, orbital, t.source_orbital; kwargs...)
                    iszero(operator) && continue

                    pushterms!(terms, operator, [t],
                               integrals, integral_map, eqs.integrals)
                end
            end
        end

        # Find all other orbitals of the same symmetry as the current
        # one. These will be used to create a projector, that projects
        # out their components.
        #
        # TODO: Think of what the non-orthogonalities due to
        # `overlaps` imply for the Lagrange multipliers/projectors.
        symmetry_orbitals = filter(!isequal(orbital), symmetries[symmetry(orbital)])
        verbosity > 1 && println("Symmetry: ", symmetry_orbitals)

        !isnothing(p) && ProgressMeter.next!(p)

        AtomicOrbitalEquation(atom, equation, orbital, terms, symmetry_orbitals)
    end
end

atomic_hamiltonian(::Atom{T,B,O,TC,CV,P}) where {T,B,O,TC,CV,P<:AbstractPotential} =
    FieldFreeOneBodyHamiltonian() + CoulombInteraction()

"""
    diff(atom; H=atomic_hamiltonian(atom), overlaps=[], selector=outsidecoremodel, verbosity=0)

Differentiate the energy expression of the Hamiltonian `H` associated
with the `atom`'s configurations(s) with respect to the atomic
orbitals to derive the Hartree–Fock equations for the orbitals.

By default, the Hamiltonian
`H=FieldFreeOneBodyHamiltonian()+CoulombInteraction()`.

Non-orthogonality between orbitals can be specified by providing
`OrbitalOverlap`s between these pairs. Only those electrons not
modelled by `atom.potential` of each configuration are considered for
generating the energy expression, this can be changed by choosing
another value for `selector`.
"""
function Base.diff(atom::Atom,
                   energy_expression::EnergyExpression;
                   H::QuantumOperator=atomic_hamiltonian(atom),
                   overlaps::Vector{<:OrbitalOverlap}=OrbitalOverlap[],
                   selector::Function=default_selector(atom),
                   configurations = selector.(atom.configurations),
                   orbitals = unique_orbitals(configurations),
                   symmetries = find_symmetries(orbitals),
                   observables::Union{Nothing,Dict{Symbol,Tuple{<:QuantumOperator,Bool}}} =
                   Dict{Symbol,Tuple{<:QuantumOperator,Bool}}(
                       :total_energy => (H,false),
                       :double_counted_energy => (H,true),
                       :kinetic_energy => (KineticEnergyHamiltonian(),false),
                   ),
                   modify_eoms!::Function = eqs -> nothing,
                   modify_integrals!::Function = (eqs,integrals,integral_map) -> nothing,
                   modify_equations!::Function = hfeqs -> nothing,
                   verbosity=0, kwargs...)
    eqs = diff(energy_expression, conj.(orbitals); verbosity=verbosity)
    modify_eoms!(eqs)

    if verbosity > 1
        println("Energy expression:")
        display(energy_expression)
        println()

        println("Hartree–Fock equations:")
        display(eqs)
        println()

        println("Symmetries:")
        display(symmetries)
        println()

        if verbosity > 2
            println("Common integrals:")
            display(eqs.integrals)
            println()
        end
    end

    integrals, integral_map, poisson_cache = common_integrals(atom, eqs.integrals; verbosity=verbosity, kwargs...)
    modify_integrals!(eqs, integrals, integral_map)

    hfeqs = generate_atomic_orbital_equations(atom, eqs,
                                              integrals, integral_map,
                                              symmetries; verbosity=verbosity,
                                              poisson_cache=poisson_cache)
    modify_equations!(hfeqs)

    observables = if !isnothing(observables)
        map(collect(pairs(observables))) do (k,(operator,double_counted))
            verbosity > 3 && println("Observable: $k ($operator)")
            k => Observable(operator, atom, overlaps,
                            integrals, integral_map,
                            symmetries;
                            selector=selector,
                            double_counted=double_counted,
                            poisson_cache=poisson_cache,
                            verbosity=verbosity, kwargs...)
        end |> Dict{Symbol,Observable}
    else
        Dict{Symbol,Observable}()
    end
    AtomicEquations(atom, hfeqs, integrals, observables)
end

Base.Matrix(H::QuantumOperator, atom::Atom;
            overlaps::Vector{<:OrbitalOverlap}=OrbitalOverlap[],
            selector::Function=default_selector(atom),
            configurations = selector.(atom.configurations),
            kwargs...) =
    Matrix(H, configurations, overlaps; kwargs...)

function Base.diff(atom::Atom; H::QuantumOperator=atomic_hamiltonian(atom),
                   kwargs...)
    energy_expression = Matrix(H, atom; kwargs...)
    diff(atom, energy_expression; H=H, kwargs...)
end

# * Overlap matrix
function SCF.overlap_matrix(atom::Atom)
    R = radial_basis(atom)
    R'R
end
