for MT in [:Diagonal, :SymTridiagonal, :Tridiagonal]
    @eval begin
        function LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T,
                                                    <:RadialOperator{<:Any, <:Any, <:$MT},
                                                    Source, Dest}) where {T,Source<:RadialOrbital,Dest<:RadialOrbital}
            mul!(ma.C.args[2], ma.A.args[2], ma.B.args[2], ma.α, ma.β)
            ma.C
        end

        function LazyArrays.materialize!(ma::MulAdd{<:Any, <:Any, <:Any, T,
                                                    <:RadialOperator{<:Any, <:Any, <:Adjoint{<:Any, <:$MT}},
                                                    Source, Dest}) where {T,Source<:RadialOrbital,Dest<:RadialOrbital}
            mul!(ma.C.args[2], ma.A.args[2], ma.B.args[2], ma.α, ma.β)
            ma.C
        end

        ArrayLayouts.default_blasmul!(α, A::$MT, B::AbstractVector, β, C::AbstractVector) =
            mul!(C, A, B, α, β)
    end
end
