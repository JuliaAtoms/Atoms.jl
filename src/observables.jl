"""
    Observable

Represents a physical quantity that can be observed, which is
calculated as the matrix element of an operator between two
configurations. All physical observables are real.
"""
mutable struct Observable{T,B<:Basis,Equations<:AbstractVector{<:AtomicOrbitalEquation},Integrals,Metric}
    equations::Equations
    integrals::Integrals
    tmp::RadialOrbital{T,B}
    S̃::Metric
end

function observable_eqs(operator::QuantumOperator, atom::Atom{T},
                        overlaps::Vector{<:OrbitalOverlap};
                        selector::Function=default_selector(atom),
                        double_counted::Bool=false) where T
    M = Matrix(operator, selector.(atom.configurations), overlaps)

    m,n = size(M)

    if double_counted
        numbodies(operator) > 2 &&
            throw(ArgumentError("Cannot construct double-counted observables for N-body operators with N > 2."))
        M *= one(T)/2
        # We need to filter out those terms that are due to zero- and
        # one-body interactions, since they would not be
        # double-counted.
        for i = 1:m
            for j = 1:n
                iszero(M[i,j]) && continue
                M[i,j] = NBodyMatrixElement(filter(t -> any(f -> numbodies(f) > 1,
                                                           t.factors),
                                                   M[i,j].terms))
            end
        end
    end

    diff(M, conj(atom.orbitals)) do M,corb
        # We need to filter out those terms that are dependent on
        # corb, to avoid double-counting when computing the
        # observables.
        for i = 1:m
            for j = 1:n
                (iszero(M[i,j]) || double_counted) && continue
                M[i,j] = NBodyMatrixElement(filter(t -> !isdependent(t, corb),
                                                   M[i,j].terms))
            end
        end
    end
end

function _Observable(atom, equation_system,
                     integrals, integral_map, symmetries;
                     selector=default_selector(atom), double_counted=false,
                     kwargs...)
    # TODO: Should observables really project out symmetry orbitals?
    eqs = generate_atomic_orbital_equations(
        atom, equation_system,
        integrals, integral_map,
        symmetries; kwargs...)

    tmp = similar(first(eqs).ϕ)
    tmp.args[2] .= false # To clear any NaNs

    Observable(eqs, integrals, tmp, atom.S̃)
end

"""
    Observable(operator, atom, overlaps, integrals, integral_map,
               [; selector=default_selector(atom), double_counted=false])

Construct an observable corresponding the `operator` acting on `atom`;
if `double_counted`, only return those terms that would be
double-counted, otherwise return the normal observable
equations. `overlaps` is a list of non-orthogonal, `integrals` a list
of common integrals, and `integral_map` is a mapping from symbolic
integrals to [`OrbitalIntegral`](@ref)s. `selector` selects those
orbitals not modelled by the potential, e.g. all orbitals in case of a
nuclear potential, fewer in case of a
pseudo-potential. `double_counted` can be used to only return those
terms of the sum that would be double-counted if simply summing over
orbital contributions (applicable to two-body operators only). Half
the result of the double-counted term is then subtracted the sum over
orbital contributions; this makes the expression slightly more
symmetric in orbital space than if the sum is derived avoiding
double-counting.
"""
function Observable(operator::QuantumOperator, atom::Atom{T},
                    overlaps::Vector{<:OrbitalOverlap},
                    integrals::Vector,
                    integral_map::Dict{Any,Int},
                    symmetries::Dict;
                    kwargs...) where T
    eqs = observable_eqs(operator, atom, overlaps)
    _Observable(atom, eqs, integrals, integral_map, symmetries; kwargs...)
end

function Observable(operator::QuantumOperator, atom::Atom,
                    overlaps::Vector{<:OrbitalOverlap};
                    kwargs...) where T
    eqs = observable_eqs(operator, atom, overlaps)
    integrals, integral_map, poisson_cache =
        common_integrals(atom, eqs.integrals)

    selector = get(kwargs, :selector, default_selector(atom))
    configurations = selector.(atom.configurations)
    orbitals = unique_orbitals(configurations)
    symmetries = find_symmetries(orbitals)

    _Observable(atom, eqs, integrals, integral_map, symmetries;
                poisson_cache=poisson_cache, kwargs...)
end

realifreal(::Type{R}, v) where {R<:Real} = real(v)
realifreal(::Type{R}, v) where {R} = v

function SCF.update!(o::Observable, args...; kwargs...)
    for integral in o.integrals
        SCF.update!(integral, args...; kwargs...)
    end
    # Updating the operators is apparently not thread-safe at the
    # moment. Why?
    for eq in o.equations
        update!(SCF.hamiltonian(eq), args...; kwargs...)
    end
end

"""
    observe!(A::M, o::Observable)

Compute the observable `o` between all configurations and store the
results as matrix elements of `A`.
"""
function observe!(A::M, o::Observable{T}) where {R,M<:AbstractMatrix{R},T}
    A .= zero(T)
    for eq in o.equations
        for term in eq.hamiltonian.terms
            materialize!(MulAdd(coefficient(term), term.A, eq.ϕ, zero(T), o.tmp))
            A[term.i,term.j] += realifreal(R, dot(eq.ϕ.args[2], o.S̃, o.tmp.args[2]))
        end
    end
    A
end

"""
    observe!(A::M, atom::Atom, o::Observable)

Compute the observable `o` between all configurations and store the
results as matrix elements of `A`. The right-hand side vectors are
taken from the equations, whereas the lefth-hand side vectors are taken from `atom`.

"""
function observe!(A::M, atom::Atom, o::Observable{T}) where {R,M<:AbstractMatrix{R},T}
    A .= zero(T)
    Φ = atom.radial_orbitals.args[2]
    for eq in o.equations
        for term in eq.hamiltonian.terms
            materialize!(MulAdd(coefficient(term), term.A, eq.ϕ, zero(T), o.tmp))
            ϕr = view(Φ, :, eq.index)
            A[term.i,term.j] += realifreal(R, dot(ϕr, o.S̃, o.tmp.args[2]))
        end
    end
    A
end

"""
    observe!(A::M, o::Observable, atom::Atom)

Compute the observable `o` between all configurations, store the
results as matrix elements of `A`, and finally contract with respect
to the mixing coefficients of `atom` and return the result as a
scalar. `o` is _not_ [`update!`](@ref)d with respect to `atom`.
"""
function observe!(A::M, o::Observable{T}, atom::Atom) where {R,M<:AbstractMatrix{R},T}
    observe!(A, o)
    c = atom.mix_coeffs
    dot(c, A, c)
end

"""
    observe!(A::M, o::Observable, a::Atom, b::Atom)

Compute the observable `o` between all configurations, store the
results as matrix elements of `A`, and finally contract with respect
to the mixing coefficients of atoms `a` & `b` and return the result as
a scalar, which corresponds to a transition between the states of the
respective atoms. `o` will be [`update!`](@ref)d with respect to `b`.
"""
function observe!(A::M, o::Observable{T}, a::Atom, b::Atom) where {R,M<:AbstractMatrix{R},T}
    update!(o, b)
    observe!(A, a, o)
    dot(a.mix_coeffs, A, b.mix_coeffs)
end
