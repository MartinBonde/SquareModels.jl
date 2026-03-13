using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, name, value, optimize!, set_silent
using Ipopt
using Random

const _build_model = SquareModels._build_model

"""Time a function over `n` repetitions after a warmup call."""
function bench(f; n=3)
    f()
    times = [(@elapsed f()) for _ in 1:n]
    return minimum(times)
end

"""Time `f(setup())` over `n` repetitions with fresh setup each run."""
function bench_fresh(setup, f; n=3)
    f(setup())
    times = [(@elapsed f(setup())) for _ in 1:n]
    return minimum(times)
end

# ==============================================================================
# Model builder — IO-like structure at realistic scale
# ==============================================================================
# Mimics a simplified Input-Output model: ~50 industries × ~30 demand components
# × ~20 time periods, with sparse (i,d) mapping, sum constraints, and prices.

function build_io_model(; n_i=50, n_d=30, n_t=20, sparsity=0.7, seed=42)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    I = 1:n_i
    D = 1:n_d
    T = 1:n_t
    t1 = 1

    rng = MersenneTwister(seed)
    # Deterministic sparse (i,d) mapping so perf tests are reproducible.
    n_cells = round(Int, n_i * n_d * (1 - sparsity))
    all_pairs = [(i, d) for i in I for d in D]
    shuffle!(rng, all_pairs)
    id_pairs = Set(all_pairs[1:n_cells])

    @variables m begin
        vD[D, T], "Demand by component"
        pD[D, T], "Demand deflator"
        qD[D, T], "Real demand"
        vY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Output by (i,d)"
        pY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Price by (i,d)"
        qY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Quantity by (i,d)"
        vY_i[I, T], "Output by industry"
        pY_i[I, T], "Price by industry"
        qY_i[I, T], "Real output by industry"
        rYM[i=I, d=D, t=T; (i, d) in id_pairs], "Composition share"
        tY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Tax rate"
    end

    return (
        model=m, I=I, D=D, T=T, t1=t1, id_pairs=id_pairs,
        vD=vD, pD=pD, qD=qD,
        vY_id=vY_id, pY_id=pY_id, qY_id=qY_id,
        vY_i=vY_i, pY_i=pY_i, qY_i=qY_i,
        rYM=rYM, tY_id=tY_id,
    )
end

function define_block(io)
    (; model, I, D, T, t1, id_pairs,
     vD, pD, qD, vY_id, pY_id, qY_id, vY_i, pY_i, qY_i, rYM, tY_id) = io

    @block model begin
        # Demand = sum of IO cells
        vD[d = D, t = t1:T[end]],
        vD[d, t] == ∑(vY_id[i, d, t] for i in I if (i, d) in id_pairs)

        # Deflator identity
        pD[d = D, t = t1:T[end]],
        pD[d, t] * qD[d, t] == vD[d, t]

        # IO cell values
        vY_id[i = I, d = D, t = t1:T[end]; (i, d) in id_pairs],
        vY_id[i, d, t] == pY_id[i, d, t] * qY_id[i, d, t]

        # IO cell prices (base price + tax)
        pY_id[i = I, d = D, t = t1:T[end]; (i, d) in id_pairs],
        pY_id[i, d, t] == (1 + tY_id[i, d, t]) * pY_i[i, t]

        # Quantity allocation
        qY_id[i = I, d = D, t = t1:T[end]; (i, d) in id_pairs],
        qY_id[i, d, t] == rYM[i, d, t] * qD[d, t]

        # Industry aggregation
        vY_i[i = I, t = t1:T[end]],
        vY_i[i, t] == ∑(vY_id[i, d, t] for d in D if (i, d) in id_pairs)

        # Industry quantity
        qY_i[i = I, t = t1:T[end]],
        qY_i[i, t] == ∑(qY_id[i, d, t] for d in D if (i, d) in id_pairs)
    end
end

function populate_data(io)
    db = ModelDictionary(io.model)
    db[io.vD] .= 100.0
    db[io.pD] .= 1.0
    db[io.qD] .= 100.0
    db[io.vY_id] .= 10.0
    db[io.pY_id] .= 1.0
    db[io.qY_id] .= 10.0
    db[io.vY_i] .= 50.0
    db[io.pY_i] .= 1.0
    db[io.qY_i] .= 50.0
    db[io.rYM] .= 0.1
    db[io.tY_id] .= 0.05
    return db
end

# ==============================================================================
# Test 1 — ModelDictionary: set/get at scale
# ==============================================================================

@testset "Perf: ModelDictionary set at scale" begin
    io = build_io_model()
    n_vars = length(all_variables(io.model))

    t_populate = bench() do
        populate_data(io)
    end

    println()
    println("  ModelDictionary populate ($n_vars variables)")
    println("  └─ Time: $(round(t_populate * 1000, digits=1)) ms")

    # With the O(n²) add_missing_model_variables! bug, this would be very slow
    # at ~30k variables. Should be well under 1 second.
    @test t_populate < 2.0
end

@testset "Perf: ModelDictionary get at scale" begin
    io = build_io_model()
    db = populate_data(io)

    # Read all variables individually (simulates what _build_model does)
    vars = all_variables(io.model)
    t_get = bench() do
        for v in vars
            db[v]
        end
    end

    println()
    println("  ModelDictionary get ($(length(vars)) reads)")
    println("  └─ Time: $(round(t_get * 1000, digits=1)) ms")

    @test t_get < 2.0
end

# ==============================================================================
# Test 2 — @block definition cost
# ==============================================================================

@testset "Perf: @block definition (IO-scale)" begin
    io = build_io_model()

    t_block = bench_fresh(() -> build_io_model(), define_block)

    println()
    println("  @block definition (IO-scale, $(length(io.id_pairs)) sparse cells × $(length(io.T)) periods)")
    println("  └─ Time: $(round(t_block * 1000, digits=1)) ms")

    @test t_block < 10.0
end

# ==============================================================================
# Test 3 — _build_model cost (the solve model construction)
# ==============================================================================

@testset "Perf: _build_model (IO-scale)" begin
    io = build_io_model()
    db = populate_data(io)
    block = define_block(io)
    db[residuals(block)] .= 0.0

    t_build = bench() do
        _build_model(block, db)
    end

    n_endo = length(endogenous(block))
    println()
    println("  _build_model ($n_endo endogenous variables)")
    println("  └─ Time: $(round(t_build * 1000, digits=1)) ms")

    @test t_build < 10.0
end

# ==============================================================================
# Test 4 — Full solve pipeline
# ==============================================================================

@testset "Perf: solve end-to-end (IO-scale)" begin
    io = build_io_model()
    db = populate_data(io)
    block = define_block(io)
    db[residuals(block)] .= 0.0

    t_solve = bench(n=2) do
        solve(block, db)
    end

    n_endo = length(endogenous(block))
    println()
    println("  solve ($n_endo endogenous variables)")
    println("  └─ Time: $(round(t_solve * 1000, digits=1)) ms")

    # Full solve (build + optimize + extract) should be under 30 seconds
    @test t_solve < 30.0
end

# ==============================================================================
# Test 5 — Repeated solves (re-solve with different data)
# ==============================================================================
# This is the primary use case: calibrate once, then run many scenarios.
# Each scenario calls solve! with different exogenous data.

@testset "Perf: repeated solves" begin
    io = build_io_model()
    db = populate_data(io)
    block = define_block(io)
    db[residuals(block)] .= 0.0

    # First solve (cold)
    baseline = solve(block, db)

    # Subsequent solves with shocked data (warm — should be similar speed)
    scenario = copy(baseline)
    t_resolves = Float64[]
    for shock in 1:3
        scenario[io.rYM] .= 0.1 + 0.01 * shock
        push!(t_resolves, @elapsed solve!(block, scenario))
    end

    t_avg = sum(t_resolves) / length(t_resolves)
    println()
    println("  Repeated solves (3 scenarios)")
    println("  ├─ Times: $(join([round(t*1000, digits=1) for t in t_resolves], ", ")) ms")
    println("  └─ Average: $(round(t_avg * 1000, digits=1)) ms")

    @test t_avg < 30.0
end

# ==============================================================================
# Test 6 — Solve time breakdown (build vs optimize vs extract)
# ==============================================================================

@testset "Perf: solve breakdown (build vs optimize)" begin
    io = build_io_model()
    db = populate_data(io)
    block = define_block(io)
    db[residuals(block)] .= 0.0

    # Measure build time
    t_build = bench(n=2) do
        _build_model(block, db)
    end

    # Measure build + optimize together, then subtract build
    t_build_and_optimize = bench(n=2) do
        sm, vm = _build_model(block, db)
        optimize!(sm)
    end
    t_optimize = t_build_and_optimize - t_build

    # Measure extract time (need a solved model)
    solve_model, var_map = _build_model(block, db)
    optimize!(solve_model)
    t_extract = bench(n=2) do
        for (_, sv) in var_map
            value(sv)
        end
    end

    t_total = t_build + t_optimize + t_extract
    pct_build = round(100 * t_build / t_total, digits=1)
    pct_optimize = round(100 * t_optimize / t_total, digits=1)
    pct_extract = round(100 * t_extract / t_total, digits=1)

    n_endo = length(endogenous(block))
    println()
    println("  Solve breakdown ($n_endo endogenous)")
    println("  ├─ _build_model: $(round(t_build * 1000, digits=1)) ms ($pct_build%)")
    println("  ├─ optimize!:    $(round(t_optimize * 1000, digits=1)) ms ($pct_optimize%)")
    println("  └─ extract:      $(round(t_extract * 1000, digits=1)) ms ($pct_extract%)")

    # For this trivially-linear benchmark, build dominates optimize.
    # For real nonlinear problems, optimize dominates build.
    @test pct_build < 95
end
