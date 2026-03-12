module TestGAMS

using Test
import JuMP
using JuMP: Model, set_optimizer_attribute, get_optimizer_attribute, @variable, unsafe_backend
using SquareModels
import GAMS
import MathOptInterface as MOI

# Probe whether the GAMS runtime (not just the Julia package) is installed
const GAMS_AVAILABLE = try
    m = Model(GAMS.Optimizer)
    @variable(m, _x)
    JuMP.@constraint(m, _x == 1)
    JuMP.optimize!(m)
    true
catch
    false
end

if GAMS_AVAILABLE

@testset "GAMS CONOPT solve" begin
    m = Model(GAMS.Optimizer)
    set_optimizer_attribute(m, "NLP", "CONOPT")
    set_optimizer_attribute(m, "LogOption", 0)

    JuMP.@variables m begin
        x
        y
        z
    end

    data = ModelDictionary(m)
    data[x] = 1.0
    data[y] = 2.0
    data[z] = 5.0

    block = @block m begin
        x, x == z * 2
        y, y == z^2
    end
    data[residuals(block)] .= 0.0

    # Verify that _copy_model_config preserves GAMS-specific attributes
    solve_model, _ = SquareModels._build_model(block, data)
    inner = unsafe_backend(solve_model)
    @test MOI.get(inner, MOI.RawOptimizerAttribute("NLP")) == "conopt"
    @test MOI.get(inner, MOI.RawOptimizerAttribute("LogOption")) == 0

    solution = solve(block, data)

    @test solution[x] ≈ 10.0 atol=1e-6
    @test solution[y] ≈ 25.0 atol=1e-6
end

else
    @warn "Skipping GAMS tests: GAMS runtime not available"
end

end # module
