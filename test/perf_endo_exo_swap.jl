using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, name
using Ipopt
using Random

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
# Helpers: IO-like model at GREU scale
# ==============================================================================

function build_io_endo_exo_swap_model(; n_i=50, n_d=30, n_t=16, sparsity=0.7, seed=42)
    m = Model()

    I = 1:n_i
    D = 1:n_d
    T = 1:n_t

    rng = MersenneTwister(seed)
    n_cells = round(Int, n_i * n_d * (1 - sparsity))
    all_pairs = [(i, d) for i in I for d in D]
    shuffle!(rng, all_pairs)
    id_pairs = Set(all_pairs[1:n_cells])

    @variables m begin
        vY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Output by (i,d)"
        pY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Price by (i,d)"
        qY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Quantity by (i,d)"
        vD[D, T], "Demand"
        pD[D, T], "Demand deflator"
        qD[D, T], "Real demand"
        rYM[i=I, d=D, t=T; (i, d) in id_pairs], "Composition share"
        tY_id[i=I, d=D, t=T; (i, d) in id_pairs], "Tax rate"
        vY_i[I, T], "Output by industry"
        pY_i[I, T], "Price by industry"
        qY_i[I, T], "Real output by industry"
    end

    n_vars = length(all_variables(m))

    function make_block()
        @block m begin
            vY_id[i = I, d = D, t = T; (i, d) in id_pairs],
            vY_id[i, d, t] == pY_id[i, d, t] * qY_id[i, d, t]

            pY_id[i = I, d = D, t = T; (i, d) in id_pairs],
            pY_id[i, d, t] == (1 + tY_id[i, d, t]) * pY_i[i, t]

            qY_id[i = I, d = D, t = T; (i, d) in id_pairs],
            qY_id[i, d, t] == rYM[i, d, t] * qD[d, t]

            vD[d = D, t = T],
            vD[d, t] == ∑(vY_id[i, d, t] for i in I if (i, d) in id_pairs)

            pD[d = D, t = T],
            pD[d, t] * qD[d, t] == vD[d, t]

            vY_i[i = I, t = T],
            vY_i[i, t] == ∑(vY_id[i, d, t] for d in D if (i, d) in id_pairs)

            qY_i[i = I, t = T],
            qY_i[i, t] == ∑(qY_id[i, d, t] for d in D if (i, d) in id_pairs)
        end
    end

    return (; model=m, n_vars, id_pairs, make_block,
             vY_id, pY_id, qY_id, vD, pD, qD, rYM, tY_id, vY_i, pY_i, qY_i)
end

# ==============================================================================
# Test 1 — _endo_exo_swap! vector swap at simple scale
# ==============================================================================

@testset "Perf: _endo_exo_swap! simple" begin
    n = 5000

    function setup()
        m = Model()
        @variable(m, x[1:n])
        @variable(m, x_exo[1:n])
        @variable(m, param[1:n])
        block = @block m begin
            x[i ∈ 1:n], x[i] + x_exo[i] == param[i]
        end
        (; block, x, x_exo)
    end

    t_swap = bench_fresh(setup, io -> SquareModels._endo_exo_swap!(io.block, io.x_exo, io.x, "bench"))

    println()
    println("  _endo_exo_swap! ($n swaps in one call)")
    println("  └─ Time: $(round(t_swap * 1000, digits=1)) ms")

    @test t_swap < 1.0
end

# ==============================================================================
# Test 2 — Sequential single-pair @endo_exo_swap! calls (the actual GREU pattern)
# ==============================================================================

@testset "Perf: sequential endo_exo_swap swaps" begin
    n = 5000

    function setup()
        m = Model()
        @variable(m, x[1:n])
        @variable(m, x_exo[1:n])
        @variable(m, param[1:n])
        block = @block m begin
            x[i ∈ 1:n], x[i] + x_exo[i] == param[i]
        end
        (; block, x, x_exo)
    end

    t_seq = bench_fresh(setup, function(io)
        for i in 1:n
            SquareModels._endo_exo_swap!(io.block, io.x_exo[i], io.x[i], "bench")
        end
    end)

    println()
    println("  Sequential single-pair swaps ($n)")
    println("  └─ Time: $(round(t_seq * 1000, digits=1)) ms")

    @test t_seq < 2.0
end

# ==============================================================================
# Test 3 — @block definition at IO-scale
# ==============================================================================

@testset "Perf: @block at IO-scale" begin
    io = build_io_endo_exo_swap_model()

    t_block = bench() do
        io.make_block()
    end

    println()
    println("  @block definition (IO-scale, $(length(io.id_pairs)) sparse cells × 16 periods)")
    println("  └─ Time: $(round(t_block * 1000, digits=1)) ms")

    @test t_block < 10.0
end

# ==============================================================================
# Test 4 — _endo_exo_swap! at IO-scale (with realistic block from IO model)
# ==============================================================================

@testset "Perf: endo_exo_swap at IO-scale" begin
    io = build_io_endo_exo_swap_model()

    function setup()
        block = io.make_block()
        endos = endogenous(block)
        resids = residuals(block)
        (; block, endos, resids)
    end

    t_swap = bench_fresh(setup, function(s)
        SquareModels._endo_exo_swap!(s.block, s.resids, s.endos, "bench")
    end)

    block = io.make_block()
    n_endo = length(endogenous(block))

    println()
    println("  _endo_exo_swap! at IO-scale ($n_endo swaps)")
    println("  └─ Time: $(round(t_swap * 1000, digits=1)) ms")

    @test t_swap < 2.0
end

# ==============================================================================
# Test 5 — Sequential endo_exo_swap at IO-scale (the GREU pattern)
# ==============================================================================

@testset "Perf: sequential endo_exo_swap at IO-scale" begin
    io = build_io_endo_exo_swap_model()

    function setup()
        block = io.make_block()
        endos = endogenous(block)
        resids = residuals(block)
        (; block, endos, resids)
    end

    t_seq = bench_fresh(setup, function(s)
        for (endo, resid) in zip(s.endos, s.resids)
            SquareModels._endo_exo_swap!(s.block, resid, endo, "bench")
        end
    end)

    block = io.make_block()
    n_endo = length(endogenous(block))

    println()
    println("  Sequential endo_exo_swap at IO-scale ($n_endo swaps)")
    println("  └─ Time: $(round(t_seq * 1000, digits=1)) ms")

    @test t_seq < 5.0
end
