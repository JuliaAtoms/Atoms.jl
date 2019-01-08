function one_body_hamiltonian(atom::Atom{T,T,B,O,TC,C,CM}, ℓ::Int) where {T,B,O,TC,C,CM}
    R = radial_basis(atom)
    
    D = Derivative(axes(R,1))
    Tm = R'D'D*R
    Tm /= -2
    
    V = Matrix(r -> ℓ*(ℓ+1)/(2r^2) + atom.potential(ℓ, r), R)

    Tm + V
end

struct ShiftInvert{M}
    A⁻¹::M
end

Base.size(S::ShiftInvert, args...) = size(S.A⁻¹, args...)
Base.eltype(S::ShiftInvert) = eltype(S.A⁻¹)

LinearAlgebra.mul!(y, S::ShiftInvert, x) =
    ldiv!(y, S.A⁻¹, x)

function diagonalize_one_body(H, nev::Int;
                              method::Symbol=:arnoldi_shift_invert, tol=1e-10, σ=-1,
                              verbosity=0, io=stdout)
    verbosity > 2 && println(io, "Diagonalizing via $(method)")
    if method == :arnoldi || method == :arnoldi_shift_invert
        A,target = if method == :arnoldi
            H,SR()
        else
            ShiftInvert(factorize(H - σ*I)),LR()
        end
        schur,history = partialschur(A, nev=nev, tol=tol, which=target)
        verbosity > 3 && println(io, history)
        λ = if method == :arnoldi
            diag(schur.R)
        else
            θ = diag(schur.R)
            verbosity > 2 && println(io, "Schur values: $θ")
            σ .+ inv.(θ)
        end
        length(λ) < nev &&
            error("Could not converge the requested orbitals: $(history)")
        λ,schur.Q
    elseif method == :eigen
        ee = eigen(H)
        ee.values[1:nev],ee.vectors[:,1:nev]
    else
        throw(ArgumentError("Unknown diagonalization method $(method)"))
    end    
end
