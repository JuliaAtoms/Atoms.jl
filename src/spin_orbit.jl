struct SpinOrbitInteraction <: OneBodyOperator end

atomic_hamiltonian(::Atom{T,B,O,TC,CV,P}) where {T,B,O,TC,CV,P<:FullOrScalarRelativisticEffectiveCorePotential} =
    FieldFreeOneBodyHamiltonian() + CoulombInteraction() + SpinOrbitInteraction()

Base.show(io::IO, ::SpinOrbitInteraction) = write(io, "V̂ₛₒ")

"""
    mⱼ(o::SpinOrbital)

Return the total mⱼ that `o` couples to in LS coupling, i.e. `mℓ+mₛ`.
"""
mⱼ(o::SpinOrbital) = sum(o.m)

function Base.iszero(me::OrbitalMatrixElement{1,A,SpinOrbitInteraction,B}) where {A<:SpinOrbital,B<:SpinOrbital}
    a = me.a[1]
    b = me.b[1]

    ℓa = first(jmⱼ(a))
    ℓb = first(jmⱼ(b))
    # This is correct for both nonrelativistic as well as relativistic
    # spin-orbitals, since for the former, first(jmⱼ(o)),mⱼ(o) returns
    # ℓ_o,m_j_o, and for the latter j_o,m_j_o. The ECPs are formed
    # using 𝒫(ℓjmⱼ) projectors, which are diagonal in the basis of
    # relativistic spin-orbitals (except for the principal quantum
    # number n), but block-diagonal in the basis of non-relativistic
    # spin-orbitals, with the entries of the blocks given by the
    # Clebsch–Gordan coefficients ⟨ℓm_ℓsmₛ|ℓsjmⱼ⟩.  Additionally, for
    # relativistic spin-orbitals, we must check ℓ.
    a.orb.ℓ ≠ b.orb.ℓ || ℓa ≠ ℓb || iszero(ℓa) || mⱼ(a) != mⱼ(b)
end


function get_operator(::SpinOrbitInteraction, atom::Atom, a::SpinOrbital, b::SpinOrbital; kwargs...)
    if a == b && atom.potential isa RelativisticEffectiveCorePotential
        # In case of two-component relativistic ECPs, the
        # orbital-diagonal spin–orbit contribution is already
        # accounted for.
        return 0
    end
    R = radial_basis(atom)
    r = locs(R)
    Vₛₒ = spin_orbit_potential(atom.potential, r, a, b)
    operator(Diagonal(Vₛₒ), R)
end
