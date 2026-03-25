module TestIntegration

using Test
import JuMP
using JuMP: Model, set_silent, @variable, all_variables
using SquareModels
using Ipopt

@testset "Market Clearing" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    JuMP.@variables m begin
        p >= 0.01
        q >= 0
        α
        β
        γ
        δ
    end

    block = @block m begin
        p, q == α - β * p
        q, q == γ + δ * p
    end

    db = ModelDictionary(m)
    db[α] = 100.0; db[β] = 2.0; db[γ] = 10.0; db[δ] = 3.0
    db[p] = 10.0; db[q] = 10.0
    db[residuals(block)] .= 0.0

    result = solve(block, db)

    expected_p = (100 - 10) / (2 + 3)
    @test result[p] ≈ expected_p atol=1e-6
    @test result[q] ≈ (100 - 2 * expected_p) atol=1e-6
end

@testset "Calibration round-trip" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    JuMP.@variables m begin
        Y >= 0.01
        L >= 0.01
        A
    end

    block = @block m begin
        Y, Y == A * L
        L, L == 100
    end

    db = ModelDictionary(m)
    db[Y] = 150.0; db[L] = 100.0; db[A] = 1.0
    db[residuals(block)] .= 0.0

    @endo_exo_swap! block begin
        A, Y
        residuals(block)[2], L
    end
    calib = solve(block, db)
    @test calib[A] ≈ 1.5 atol=1e-6
end

@testset "Comparative statics" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    JuMP.@variables m begin
        Y >= 0.01
        L >= 0.01
        K >= 0.01
        A
        α
    end

    block = @block m begin
        Y, Y == A * K^α * L^(1 - α)
        L, L == 100
        K, K == 50
    end

    db = ModelDictionary(m)
    db[A] = 1.0; db[α] = 0.3
    db[Y] = 10.0; db[L] = 10.0; db[K] = 10.0
    db[residuals(block)] .= 0.0

    baseline = solve(block, db)

    scenario = copy(baseline)
    scenario[A] = 1.1
    solve!(block, scenario)

    @test scenario[Y] > baseline[Y]
    @test scenario[Y] ≈ 1.1 * baseline[Y] atol=1e-6
end

@testset "Two-good exchange economy" begin
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    JuMP.@variables m begin
        p1 >= 0.01
        p2 >= 0.01
        x1[1:2] >= 0
        x2[1:2] >= 0
        e1[1:2]
        e2[1:2]
        α[1:2]
    end

    block = @block m begin
        p1, p1 == 1
        p2, sum(x2) == sum(e2)
        x1[i ∈ 1:2], x1[i] == α[i] * (p1 * e1[i] + p2 * e2[i]) / p1
        x2[i ∈ 1:2], x2[i] == (1 - α[i]) * (p1 * e1[i] + p2 * e2[i]) / p2
    end

    db = ModelDictionary(m)
    db[e1] .= [10.0, 5.0]; db[e2] .= [5.0, 10.0]
    db[α] .= [0.6, 0.4]
    db[p1] = 1.0; db[p2] = 1.0
    db[x1] .= 1.0; db[x2] .= 1.0
    db[residuals(block)] .= 0.0

    result = solve(block, db)

    @test result[x1[1]] + result[x1[2]] ≈ result[e1[1]] + result[e1[2]] atol=1e-5

    for i in 1:2
        income = result[p1] * result[e1[i]] + result[p2] * result[e2[i]]
        expenditure = result[p1] * result[x1[i]] + result[p2] * result[x2[i]]
        @test income ≈ expenditure atol=1e-5
    end
end

@testset "Quick Example (examples/quick_example.jl)" begin
    include(joinpath(@__DIR__, "..", "examples", "quick_example.jl"))

    @test baseline isa ModelDictionary
    @test scenario isa ModelDictionary
    @test scenario[Y] > baseline[Y]
end

end # module
