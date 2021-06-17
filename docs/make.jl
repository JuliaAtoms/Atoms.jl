using Documenter
using EnergyExpressions
using AtomicLevels
using Atoms

isdefined(Main, :NOPLOTS) && NOPLOTS || include("plots.jl")

makedocs(
    modules = [Atoms],
    sitename = "Atoms",
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Theory" => [
            "Density matrices" => "density_matrices.md"
        ],
        "Implementation" => [
            "Radial orbitals" => "radial_orbitals.md",
            "Atom types" => "atom_types.md",
            "One-body Hamiltonians" => "one_body.md",
            "Hydrogenic initialization" => "hydrogenic.md",
            "Orbital equations" => "orbital_equations.md",
            "Equation systems" => "equation_systems.md",
            "Observables" => "observables.md"
        ]
    ],
    format = Documenter.HTML(assets = ["assets/latex.js"],
                             mathengine = Documenter.MathJax()),
    doctest=false,
)

deploydocs(repo = "github.com/JuliaAtoms/Atoms.jl.git")
