module Atoms

using Compat

using AtomicLevels
import AtomicLevels: AbstractOrbital, spectroscopic_label
using WignerSymbols
using HalfIntegers
using AngularMomentumAlgebra
import AngularMomentumAlgebra: jmⱼ, spin
using EnergyExpressions
import EnergyExpressions: NBodyMatrixElement, OrbitalMatrixElement,
    orbital_equation, MCEquationSystem, QuantumOperator

using ContinuumArrays
import ContinuumArrays: Basis
using QuasiArrays
import QuasiArrays: AbstractQuasiMatrix, MulQuasiArray, SubQuasiArray
using LazyArrays
import LazyArrays: materialize, materialize!, MulAdd
using ArrayLayouts
using FillArrays
using BandedMatrices
using BlockBandedMatrices

using CompactBases
import CompactBases: BasisOrRestricted, AdjointBasisOrRestricted

using LinearAlgebra
using SparseArrays

using ArnoldiMethod
using SCF
import SCF: norm_rot!, update!, KrylovWrapper, print_block
using CoulombIntegrals
import CoulombIntegrals: locs

using Unitful
using UnitfulAtomic

using PrettyTables
using Formatting
using UnicodeFun
import ProgressMeter
import ProgressMeter: Progress

function unique_orbitals(configurations::Vector{C}) where {O,C<:Configuration{O}}
    map(configurations) do config
        config.orbitals
    end |> o -> Vector{O}(vcat(o...)) |> unique |> sort
end

getspatialorb(orb::Union{Orbital,RelativisticOrbital}) = orb
getspatialorb(orb::SpinOrbital) = orb.orb

getn(orb::AbstractOrbital) = getspatialorb(orb).n
getℓ(orb::AbstractOrbital) = getspatialorb(orb).ℓ


include("potentials/potentials.jl")
include("radial_orbitals.jl")
include("atom_types.jl")
include("one_body.jl")
include("utils.jl")
include("hydrogenic.jl")
include("projectors.jl")
include("lazy_matmuls.jl")
include("orbital_integrals.jl")
include("orbital_hamiltonian.jl")
include("orbital_equation.jl")
include("common_integrals.jl")
include("observables.jl")
include("spin_orbit.jl")
include("equations.jl")
include("report.jl")

end # module
