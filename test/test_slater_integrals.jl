include("exact_slater.jl")

function test_hydrogenic_slater_integrals!(data, tests,
                                           eq, term::Symbol, expected, rtol;
                                           print_res=false)
    eng = energy(eq, term)
    expectedf = convert(Float64, expected)
    δ = eng-expectedf
    test = :(isapprox($eng, $expected; rtol=$rtol))
    pass = @eval $test
    push!(data, [eq.orbital Dict(:onebody => "⟨h⟩", :direct => "⟨J⟩", :exchange => "⟨K⟩")[term] expected expectedf eng δ δ/abs(1e-10+abs(expectedf)) pass])
    push!(tests, test)
end

function test_hydrogenic_slater_integrals(fun::Function, fock::Fock, do_test::Bool=true)
    data = []
    tests = []

    for (i,eq) in enumerate(fock.equations.equations)
        cases = fun(i)
        for case in cases
            test_hydrogenic_slater_integrals!(data, tests, eq, case...)
        end
    end
    
    println("Orbital energies pre-optimization:")
    pass = Highlighter((data,i,j) -> j == 8 && data[i,j], bold = true, foreground = :green)
    fail = Highlighter((data,i,j) -> j == 8 && !data[i,j], bold = true, foreground = :red)
    pretty_table(vcat(data...), header=["Orbital", "Term", "Expected", "Expected", "Actual", "Error", "Relative error", "Pass"],
                 highlighters=(pass,fail))
    do_test || return
    for test in tests
        @eval @test $test
    end
end
