get_integral(integrals::Vector, integral_map::Dict, symbolic_integral) =
    integrals[integral_map[symbolic_integral]]

function get_or_create_integral!(integrals, integral_map, symbolic_integral, atom; kwargs...)
    if symbolic_integral ∈ keys(integral_map)
        get_integral(integrals, integral_map, symbolic_integral)
    else
        i = create_integral(symbolic_integral, atom, integrals, integral_map; kwargs...)
        isnothing(i) && return
        push!(integrals, i)
        integral_map[symbolic_integral] = length(integrals)
        i
    end
end

function create_integral(symbolic_integral::OrbitalOverlap, atom::Atom, integrals, integral_map; kwargs...)
    a,b = symbolic_integral.a,symbolic_integral.b
    OrbitalOverlapIntegral(a, b, atom)
end

function create_integral(symbolic_integral::CoulombPotentialMultipole, atom::Atom, integrals, integral_map; kwargs...)
    a,b = symbolic_integral.a[1],symbolic_integral.b[1]
    k,g = symbolic_integral.o.k,symbolic_integral.o.g
    # All CoulombPotentialMultipoles that occur as common
    # integrals in an equation are direct potentials acting on
    # either the orbital, or on a source orbital coupled
    # through configuration interaction.
    HFPotential(:direct, k, a, b, atom, g; kwargs...)
end

function create_integral(ome::OrbitalMatrixElement{1,aO,<:OneBodyOperator,bO},
                         atom::Atom{T}, integrals, integral_map; kwargs...) where {aO<:SpinOrbital,bO<:SpinOrbital,T}
    a,o,b = ome.a,ome.o,ome.b
    op = Atoms.get_operator(o, atom, a[1], b[1]; kwargs...)
    iszero(op) && return ZeroIntegral{aO,bO,real(T)}()

    OperatorMatrixElement(a[1], b[1], op, atom, 1)
end

function create_integral(symbolic_integral::OrbitalMatrixElement{2,<:SpinOrbital,<:CoulombInteractionMultipole,<:SpinOrbital},
                         atom::Atom, integrals, integral_map; kwargs...)
    a,b = symbolic_integral.a[1],symbolic_integral.b[1]
    ∂I = diff(symbolic_integral, conj(a))
    # The operator in a two-electron integral is a HFPotential (that
    # may be shared with many other two-electron integrals). The
    # potential must be updated before the two-electron integral is
    # computed, and we achieve this by making sure it appears before
    # the two-electron integral in the list of integrals (which is
    # updated sequentially).
    hfpotential = get_or_create_integral!(integrals, integral_map, ∂I.operator, atom; kwargs...)
    # We know that the coefficient is purely numeric (and likely
    # unity), since we varied a single OrbitalMatrixElement with
    # respect to one of the orbitals.
    coeff = ∂I.factor.coeff
    OperatorMatrixElement(a, b, hfpotential.V̂, atom, coeff)
end

create_integral(symbolic_integral, atom::Atom, integrals, integral_map; kwargs...) =
    throw(ArgumentError("$(symbolic_integral) of type $(typeof(symbolic_integral)) not yet supported"))

function append_common_integrals!(integrals::Vector, integral_map::Dict, atom::Atom, equation_integrals;
                                  verbosity=0, kwargs...)
    nint = length(equation_integrals)
    p = if verbosity > 0 && nint > 0
        @info "Appending common integrals"
        Progress(nint)
    end
    for symbolic_integral in equation_integrals
        get_or_create_integral!(integrals, integral_map, symbolic_integral, atom; kwargs...)
        !isnothing(p) && ProgressMeter.next!(p)
    end
end

function common_integrals(atom::Atom, equation_integrals; kwargs...)
    integrals = Vector{OrbitalIntegral}()
    integral_map = Dict{Any,Int}()
    poisson_cache = Dict{Int,CoulombIntegrals.PoissonCache}()

    append_common_integrals!(integrals, integral_map,
                             atom, equation_integrals;
                             poisson_cache=poisson_cache, kwargs...)

    integrals, integral_map, poisson_cache
end

"""
    pushterms!(terms, operator, equation_terms,
               integrals, integral_map, symbolic_integrals)

For each term in `equation_terms`, push a term, located at CI
coordinates `i,j`, of the overall orbital Hamiltonian to `terms`,
constructed from `operator` and a product of orbital `integrals`,
multiplied by an overall factor given by expression and multipole
expansions. `integrals` contain common [`OrbitalIntegral`](@ref)s
and `integral_map` maps from `symbolic_integrals` to `integrals`.
"""
function pushterms!(terms::Vector{<:OrbitalHamiltonianTerm{aO,bO,T}},
                    operator::QO,
                    equation_terms::Vector,
                    integrals::Vector,
                    integral_map::Dict{Any,Int},
                    symbolic_integrals) where {aO,bO,T,QO}
    for eq_term in equation_terms
        integrals = OrbitalIntegral{0}[
            get_integral(integrals, integral_map, symbolic_integrals[i])
            for i in eq_term.integrals
        ]
        push!(terms,
              OrbitalHamiltonianTerm{aO,bO,T,QO,typeof(integrals)}(
                  eq_term.i, eq_term.j, T(eq_term.coeff),
                  operator, integrals))
    end
end
