using Atoms
import Atoms: ECPs
using CoulombIntegrals
import CoulombIntegrals: locs
using AtomicLevels
using HalfIntegers
using AngularMomentumAlgebra

using SCF
using LineSearches

using PrettyTables
using Unitful
using UnitfulAtomic
using Formatting

using CompactBases

using LinearAlgebra
using FillArrays
using LazyArrays

using Test

function bspline_atomic_grid(k, m, rmax, Z)
    h = 2.0^(-m)
    rlin = (1:2^m)*h
    η = 1+h
    a = rlin[end]
    n = floor(Int, log(Z*rmax/a)/log(η))
    rexp = a*η.^(1:n)
    r = vcat(0, rlin, rexp, Z*rmax)/Z

    ArbitraryKnotSet(k, r)
end

function get_atom_grid(grid_type, rmax, nucleus; ρ=0.1, ρmax=0.6, α=0.002, intervals=30, k=4, m=2)
    # Effective nuclear charge, subtracting those core electrons that
    # are modelled by e.g. a pseudopotential.
    Z = float(charge(nucleus)) - num_electrons(core(ground_state(nucleus)))
    ρ /= Z

    if grid_type == :fd_uniform
        N = ceil(Int, rmax/ρ)
        FiniteDifferences(N, ρ)
    elseif grid_type == :fd_staggered
        N = ceil(Int, rmax/ρ-1/2)
        StaggeredFiniteDifferences(N, ρ)
    elseif grid_type == :fd_loglin
        StaggeredFiniteDifferences(ρ, ρmax, α, rmax)
    elseif grid_type == :implicit_fd
        N = ceil(Int, rmax/ρ)
        ImplicitFiniteDifferences(N, ρ)
    elseif grid_type == :fedvr
        t = range(0, stop=rmax, length=intervals)
        FEDVR(t, Vcat(k+5,Fill(k,length(t)-2)))[:,2:end-1]
    elseif grid_type == :bsplines
        t = bspline_atomic_grid(k, m, rmax, Z)
        BSpline(t)[:,2:end-1]
    else
        error("Unknown grid type $(grid_type)")
    end
end

include("test_slater_integrals.jl")
include("structure_setup.jl")
include("calculation_accuracy.jl")
include("spin_orbit.jl")
include("projectors.jl")
