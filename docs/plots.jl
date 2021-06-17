using PyPlot
using Jagot.plotting
plot_style("ggplot")

using PyCall
Cycler = pyimport("cycler")
plt.rc("axes", prop_cycle=(plt.rcParams["axes.prop_cycle"] +
                           Cycler.cycler("linestyle", ["-","--",":","-.",":","-","--"])))

using Statistics
using Random

using Atoms
using AtomicLevels
using SCF

using LinearAlgebra
using CompactBases
using IntervalSets

function mean_color(color::String)
    c = parse(Colorant, color)
    mean([c.r,c.g,c.b])
end

lerp(a,b,t) = (1-t)*a + t*b

mean_position(x, ϕ) = ϕ'*Diagonal(x)*ϕ/(ϕ'ϕ)

function cfigure(fun::Function, figname; clear=true, tight=true, kwargs...)
    figure(figname; kwargs...)
    clear && clf()
    fun()
    tight && tight_layout()
end

function csubplot(fun::Function, args...; nox=false, noy=false)
    ax = subplot(args...)
    fun()
    if nox
        no_tick_labels(:x)
        xlabel("")
    end
    if noy
        no_tick_labels(:y)
        ylabel("")
    end
    ax
end

function plot_orbitals(atom::Atom; nrplot=1000, plot_basis=false,
                       fig_name=string(atom.potential))
    R = radial_basis(atom)
    d = axes(R,1).domain
    r = range(leftendpoint(d), stop=rightendpoint(d), length=nrplot)
    χ = R[r, :]

    n = 0
    ℓ = 0
    i = 0
    orbitals = map(atom.orbitals) do o
        on,oℓ = Atoms.getn(o),Atoms.getℓ(o)
        if on == n && oℓ == ℓ
            i
        else
            n = on
            ℓ = oℓ
            i += 1
        end
    end
    m = orbitals[end] + plot_basis

    cfigure("$(fig_name)") do
        for i in 1:m
            csubplot(m,1,i, nox=i<m) do
                js = findall(isequal(i), orbitals)
                for j in js
                    plot(r, abs2.(χ*view(atom, j).args[2]), label="$(atom.orbitals[j])")
                end
                legend(framealpha=0.75,ncol=2)
                xscale("log")
                i == m && xlabel(L"$r$ [au]")
                spatial_orbs = unique([Atoms.getspatialorb(atom.orbitals[j])
                                       for j in js])
                ylabel(join(string.(spatial_orbs), ", "))
            end
        end
        if plot_basis
            csubplot(m,1,m) do
                plot(r, χ)
                xscale("log")
                xlabel(L"$r$ [au]")
            end
        end
    end
end

function get_atom_grid(grid_type, rₘₐₓ, ρ, nucleus; fedvr_order=10)
    Z = charge(nucleus)
    amend_order=nucleus isa PointCharge # For correct boundary
                                        # conditions at r=0.

    if grid_type == :fedvr
        # FEDVR is more accurate, but can be expensive to use
        N = max(ceil(Int, rₘₐₓ/(ρ*fedvr_order)),2)
        t = range(0.0, stop=rₘₐₓ, length=N)
        amended_order = vcat(fedvr_order+5, fill(fedvr_order,length(t)-2))
        FEDVR(t, amend_order ? amended_order : fedvr_order)[:,2:end-1]
    else
        # Finite-differences are much lighter, but may require very
        # fine grids to converge.
        N = ceil(Int, rₘₐₓ/ρ + 1/2)
        StaggeredFiniteDifferences(N, ρ)
    end
end

function hydrogen()
    nucleus = pc"H"
    R = get_atom_grid(:fedvr, 10.0, 0.1, nucleus)
    gst = ground_state(nucleus)
    atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
    plot_orbitals(atom)
    savefig("docs/src/figures/hydrogen.svg")
end

function helium()
    nucleus = pc"He"
    R = get_atom_grid(:fedvr, 10.0, 0.1, nucleus)
    gst = ground_state(nucleus)
    atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
    fock = Fock(atom)
    optimize!(fock)
    plot_orbitals(atom)
    begin
        hydrogenics = [Atom(R, [spin_configurations(ground_state(n))[1]], n)
                       for n in [pc"H", pc"He"]]
        d = axes(R,1).domain
        r = range(leftendpoint(d), stop=rightendpoint(d), length=1000)
        χ = R[r, :]
        for (hyd,label) in zip(hydrogenics, ["H 1s", "He+ 1s"])
            plot(r, abs2.(χ*view(hyd, 1).args[2]), label=label)
        end
        legend(framealpha=0.75,ncol=2)
    end
    savefig("docs/src/figures/helium.svg")
end

function beryllium(grid_type)
    nucleus = pc"Be"
    R = get_atom_grid(grid_type, 15.0, 0.1, nucleus)
    gst = ground_state(nucleus)
    atom = Atom(R, [spin_configurations(gst)[1]], nucleus)
    fock = Fock(atom)
    optimize!(fock,ω=0.999,ωmax=1-1e-3,scf_method=:arnoldi)
    plot_orbitals(atom)
    savefig("docs/src/figures/beryllium-$(grid_type).svg")
end

macro echo(expr)
    println(expr)
    :(@time $expr)
end

@info "Documentation plots"
mkpath("docs/src/figures")
@echo hydrogen()
@echo helium()
@echo beryllium(:fedvr)
@echo beryllium(:fd)
