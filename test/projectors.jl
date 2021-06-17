@testset "Projectors" begin
    R = get_atom_grid(:fd_staggered, 10.0, pc"H")
    S = metric(R)
    v = rand(ComplexF64, size(R,2))
    v ./= √(dot(v, S, v))

    p = Atoms.Projector([applied(*, R, v)], [o"1s"], S)
    mp = Matrix(p)
    out = similar(mp)
    mul!(out, p, mp)
    @test out ≈ mp atol=1e-14

    Atoms.projectout!(out, p)
    @test norm(out) < 1e-14
end
