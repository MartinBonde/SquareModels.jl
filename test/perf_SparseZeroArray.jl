using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, fix, unfix, is_fixed, value, optimize!, AffExpr, name
using JuMP.Containers: SparseAxisArray
using Ipopt

# ==============================================================================
# Helpers
# ==============================================================================

"""Build a JuMP model with a sparse variable over A×B×C where only a fraction of keys exist."""
function build_sparse_model(; dim_size=50, sparsity=0.9)
    m = Model()
    A, B, C = 1:dim_size, 1:dim_size, 1:dim_size

    n_existing = round(Int, dim_size^3 * (1 - sparsity))
    existing = Set{Tuple{Int,Int,Int}}()
    while length(existing) < n_existing
        push!(existing, (rand(A), rand(B), rand(C)))
    end

    @variable(m, x[a=A, b=B, c=C; (a, b, c) in existing])
    domain = (Set(A), Set(B), Set(C))
    sz = SparseZeroArray(x, domain)

    return m, x, sz, A, B, C, existing
end

"""Time a function over `n` repetitions after a warmup call."""
function bench(f; n=3)
    f()  # warmup
    times = [(@elapsed f()) for _ in 1:n]
    return minimum(times)
end

# ==============================================================================
# Test 1 — Filtered vs unfiltered sum on high-sparsity SparseZeroArray
# ==============================================================================

@testset "Perf: filtered vs unfiltered sum (90% sparsity)" begin
    m, x, sz, A, B, C, ABC = build_sparse_model(dim_size=50, sparsity=0.9)

    # --- Filtered sum: only iterate over existing keys ---
    t_filtered = bench() do
        for b in B, c in C
            sum((sz[a, b, c] for a in A if (a, b, c) in ABC), init=SquareModels.Zero())
        end
    end

    # --- Unfiltered sum: rely on Zero() default for missing keys ---
    t_unfiltered = bench() do
        for b in B, c in C
            ∑(sz[a, b, c] for a in A)
        end
    end

    ratio = t_unfiltered / t_filtered

    println()
    println("  Sum benchmark (dim=50, sparsity=90%, $(length(ABC)) stored keys)")
    println("  ├─ Filtered sum (if in ABC):  $(round(t_filtered * 1000, digits=1)) ms")
    println("  ├─ Unfiltered sum (Zero()):    $(round(t_unfiltered * 1000, digits=1)) ms")
    println("  └─ Ratio (unfiltered/filtered): $(round(ratio, digits=2))x")

    # The unfiltered path should not be catastrophically slower.
    # A ratio under 5× is acceptable since it avoids set-lookup overhead per element.
    @test ratio < 10
end

@testset "Perf: filtered vs unfiltered sum — scaling with sparsity" begin
    for sparsity in (0.5, 0.9, 0.99)
        m, x, sz, A, B, C, ABC = build_sparse_model(dim_size=30, sparsity=sparsity)

        t_filtered = bench() do
            for b in B, c in C
                sum((sz[a, b, c] for a in A if (a, b, c) in ABC), init=SquareModels.Zero())
            end
        end

        t_unfiltered = bench() do
            for b in B, c in C
                ∑(sz[a, b, c] for a in A)
            end
        end

        ratio = t_unfiltered / t_filtered
        println("  sparsity=$(Int(sparsity*100))%: filtered=$(round(t_filtered*1000, digits=1))ms  unfiltered=$(round(t_unfiltered*1000, digits=1))ms  ratio=$(round(ratio, digits=2))x")

        # At higher sparsity, the unfiltered path visits more missing keys.
        # The ratio scales roughly as 1/(1-sparsity), so we set a generous ceiling.
        max_ratio = 3 / (1 - sparsity)
        @test ratio < max_ratio
    end
end

# ==============================================================================
# Test 2 — variables(block) with sums over sparse arrays
# ==============================================================================

@testset "Perf: variables(block) with ∑ over SparseZeroArray" begin
    dim = 40
    m = Model()
    A, B = 1:dim, 1:dim

    n_existing = round(Int, dim^2 * 0.1)
    existing = Set{Tuple{Int,Int}}()
    for i in A  # guarantee at least one key per row
        push!(existing, (i, rand(B)))
    end
    while length(existing) < n_existing
        push!(existing, (rand(A), rand(B)))
    end

    @variables m begin
        x[a=A, b=B; (a, b) in existing], "sparse param"
        y[A], "dense endogenous"
    end
    @test x isa SparseZeroArray

    b = @block m begin
        y[i ∈ A], y[i] == ∑(x[i, j] for j in B)
    end

    block_vars = variables(b)
    endo = endogenous(b)
    exo = exogenous(b)

    # All y variables should be endogenous
    @test length(endo) == dim
    @test all(y[i] in endo for i in A)

    # Exogenous should contain only the x variables that actually appear in
    # some constraint — i.e. x[i,j] for (i,j) in existing. Zero() entries
    # must NOT appear.
    x_vars_in_block = filter(v -> startswith(name(v), "x"), block_vars)
    @test length(x_vars_in_block) == length(existing)
    @test all(x.data[k...] in block_vars for k in existing)

    # No Zero() sentinel leaked into the variable set
    @test all(v isa VariableRef for v in block_vars)

    # Performance: variables() should be fast even with large sparse sums
    t_vars = bench() do
        variables(b)
    end
    println()
    println("  variables(block) with ∑ over SparseZeroArray ($(dim)×$(dim), $(length(existing)) stored keys)")
    println("  ├─ endogenous: $(length(endo)),  exogenous: $(length(exo)),  total: $(length(block_vars))")
    println("  └─ variables(b) time: $(round(t_vars * 1e6, digits=1)) μs")
    @test t_vars < 1
end

@testset "Perf: variables(block) — correctness at varying sparsity" begin
    for sparsity in (0.5, 0.9, 0.99)
        dim = 20
        m = Model()
        A, B = 1:dim, 1:dim

        n_existing = max(dim, round(Int, dim^2 * (1 - sparsity)))
        existing = Set{Tuple{Int,Int}}()
        for i in A
            push!(existing, (i, rand(B)))
        end
        while length(existing) < n_existing
            push!(existing, (rand(A), rand(B)))
        end

        @variables m begin
            x[a=A, b=B; (a, b) in existing], "sparse"
            y[A], "dense"
        end

        b = @block m begin
            y[i ∈ A], y[i] == ∑(x[i, j] for j in B)
        end

        block_vars = variables(b)
        x_in_block = filter(v -> startswith(name(v), "x"), block_vars)

        @test length(x_in_block) == length(existing)
        println("  sparsity=$(Int(sparsity*100))%: $(length(existing)) x-vars stored, $(length(x_in_block)) found in variables(block) ✓")
    end
end

@testset "∑ with Zero() terms inside @block constraint" begin
    m = Model(Ipopt.Optimizer)
    JuMP.set_silent(m)
    A, B = 1:5, 1:5
    existing = Set([(1, 1)])  # only one key — rows 2-5 are entirely empty

    @variables m begin
        x[a=A, b=B; (a, b) in existing], "sparse"
        y[A], "dense"
    end

    # ∑ generators that yield Zero() terms inside @block should work
    b = @block m begin
        y[i ∈ A], y[i] == ∑(x[i, j] for j in B)
    end
    @test length(b) == 5

    # variables(block) should only contain the one real x variable, not Zero()s
    block_vars = variables(b)
    x_in_block = filter(v -> startswith(name(v), "x"), block_vars)
    @test length(x_in_block) == 1
    @test x.data[1, 1] in block_vars

    # Solve: fix x[1,1]=3 → y[1]=3, y[2..5]=0
    fix(x.data[1, 1], 3.0, force=true)
    fix.(residuals(b), 0.0, force=true)
    optimize!(m)
    @test value(y[1]) ≈ 3.0 atol=1e-6
    @test value(y[2]) ≈ 0.0 atol=1e-6
    @test value(y[5]) ≈ 0.0 atol=1e-6
end

# ==============================================================================
# Test 3 — FIX / UNFIX performance with SparseZeroArray blocks
# ==============================================================================

"""Build a model with sparse blocks for fix/unfix benchmarking."""
function build_block_model(; dim_size=30, sparsity=0.9)
    m = Model(Ipopt.Optimizer)
    JuMP.set_silent(m)
    A, B = 1:dim_size, 1:dim_size

    n_existing = round(Int, dim_size^2 * (1 - sparsity))
    existing = Set{Tuple{Int,Int}}()
    while length(existing) < n_existing
        push!(existing, (rand(A), rand(B)))
    end

    @variables m begin
        x[a=A, b=B; (a, b) in existing], "endogenous"
        p[a=A, b=B; (a, b) in existing], "parameter"
    end

    b = @block m begin
        x[a ∈ A, b ∈ B; (a, b) in existing], x[a, b] == p[a, b] + 1
    end

    return m, x, p, b, existing
end

@testset "Perf: block creation with SparseZeroArray" begin
    # Measure block creation time for a moderately large sparse model
    t_create = bench(n=3) do
        build_block_model(dim_size=30, sparsity=0.9)
    end
    println()
    println("  Block creation (30×30, 90% sparsity): $(round(t_create * 1000, digits=1)) ms")
    @test t_create < 30  # should be well under 30 seconds
end

@testset "Perf: fix / unfix on SparseZeroArray block" begin
    m, x, p, b, existing = build_block_model(dim_size=30, sparsity=0.9)
    n_vars = length(b)

    # --- fix.(b, value) ---
    t_fix_broadcast = bench() do
        fix.(b, 1.0, force=true)
    end

    # --- unfix(b) ---
    fix.(b, 1.0, force=true)  # ensure fixed state
    t_unfix = bench() do
        unfix(b)
    end

    # --- fix individual variables via .data ---
    t_fix_individual = bench() do
        for (a, bb) in existing
            fix(x.data[a, bb], Float64(a + bb), force=true)
        end
    end

    # --- unfix individual variables ---
    for (a, bb) in existing
        fix(x.data[a, bb], 1.0, force=true)
    end
    t_unfix_individual = bench() do
        for (a, bb) in existing
            if is_fixed(x.data[a, bb])
                unfix(x.data[a, bb])
            end
        end
    end

    println()
    println("  Fix/Unfix benchmark ($n_vars variables)")
    println("  ├─ fix.(block, val):    $(round(t_fix_broadcast * 1000, digits=2)) ms")
    println("  ├─ unfix(block):        $(round(t_unfix * 1000, digits=2)) ms")
    println("  ├─ fix individual vars: $(round(t_fix_individual * 1000, digits=2)) ms")
    println("  └─ unfix individual:    $(round(t_unfix_individual * 1000, digits=2)) ms")

    @test t_fix_broadcast < 5
    @test t_unfix < 5
end

@testset "Perf: fix/unfix + solve cycle with SparseZeroArray" begin
    m, x, p, b, existing = build_block_model(dim_size=20, sparsity=0.9)

    # Fix parameters, unfix residuals, solve — a typical workflow
    for (a, bb) in existing
        fix(p.data[a, bb], Float64(a + bb), force=true)
    end

    t_cycle = bench(n=3) do
        # Fix endogenous → unfix residuals → solve → read values
        fix.(b, 0.0, force=true)
        unfix.(residuals(b))
        optimize!(m)
        [value(r) for r in residuals(b)]

        # Restore: fix residuals → unfix endogenous
        fix.(residuals(b), 0.0, force=true)
        unfix(b)
    end

    println()
    println("  Full fix→solve→unfix cycle: $(round(t_cycle * 1000, digits=1)) ms")
    @test t_cycle < 30
end

@testset "Perf: SparseZeroArray vs plain SparseAxisArray — fix/unfix" begin
    dim_size = 30
    sparsity = 0.9
    A, B = 1:dim_size, 1:dim_size

    n_existing = round(Int, dim_size^2 * (1 - sparsity))
    existing = Set{Tuple{Int,Int}}()
    while length(existing) < n_existing
        push!(existing, (rand(A), rand(B)))
    end

    # --- SparseZeroArray path ---
    m1 = Model(Ipopt.Optimizer)
    JuMP.set_silent(m1)
    @variables m1 begin
        x1[a=A, b=B; (a, b) in existing], "endo"
        p1[a=A, b=B; (a, b) in existing], "param"
    end
    b1 = @block m1 begin
        x1[a ∈ A, b ∈ B; (a, b) in existing], x1[a, b] == p1[a, b] + 1
    end

    # --- Plain SparseAxisArray path (no SparseZeroArray wrapping) ---
    m2 = Model(Ipopt.Optimizer)
    JuMP.set_silent(m2)
    use_sparse_zero_array!(false)
    @variables m2 begin
        x2[a=A, b=B; (a, b) in existing], "endo"
        p2[a=A, b=B; (a, b) in existing], "param"
    end
    use_sparse_zero_array!(true)
    b2 = @block m2 begin
        x2[a ∈ A, b ∈ B; (a, b) in existing], x2[a, b] == p2[a, b] + 1
    end

    @test x1 isa SparseZeroArray
    @test x2 isa SparseAxisArray

    # Fix/unfix with SparseZeroArray block
    t_sza = bench() do
        fix.(b1, 1.0, force=true)
        unfix(b1)
    end

    # Fix/unfix with plain SparseAxisArray block
    t_plain = bench() do
        fix.(b2, 1.0, force=true)
        unfix(b2)
    end

    ratio = t_sza / t_plain
    println()
    println("  Fix/unfix comparison ($(length(b1)) vars)")
    println("  ├─ SparseZeroArray block: $(round(t_sza * 1000, digits=2)) ms")
    println("  ├─ Plain SparseAxisArray: $(round(t_plain * 1000, digits=2)) ms")
    println("  └─ Ratio (SZA/plain):     $(round(ratio, digits=2))x")

    # SparseZeroArray should not add significant overhead to fix/unfix
    @test ratio < 3
end

# ==============================================================================
# Test 4 — Solver (Ipopt) performance: filtered vs unfiltered constraints
# ==============================================================================

@testset "Perf: Ipopt — filtered vs unfiltered sparse sums" begin
    dim = 50
    sparsity = 0.9
    A, B = 1:dim, 1:dim

    n_existing = max(dim, round(Int, dim^2 * (1 - sparsity)))
    existing = Set{Tuple{Int,Int}}()
    for a in A
        push!(existing, (a, rand(B)))
    end
    while length(existing) < n_existing
        push!(existing, (rand(A), rand(B)))
    end

    function build_unfiltered()
        m = Model(Ipopt.Optimizer)
        JuMP.set_silent(m)
        @variables m begin
            x[a=A, b=B; (a, b) in existing], "sparse endo"
            y[A], "aggregation endo"
        end
        bx = @block m begin
            x[a ∈ A, b ∈ B; (a, b) in existing], x[a, b] == a * b
        end
        by = @block m begin
            y[a ∈ A], y[a] == ∑(x[a, b] for b in B)
        end
        b = bx + by
        fix.(b, 0.0, force=true)
        unfix.(residuals(b))
        return m, y, b
    end

    function build_filtered()
        m = Model(Ipopt.Optimizer)
        JuMP.set_silent(m)
        use_sparse_zero_array!(false)
        @variables m begin
            x[a=A, b=B; (a, b) in existing], "sparse endo"
            y[A], "aggregation endo"
        end
        use_sparse_zero_array!(true)
        bx = @block m begin
            x[a ∈ A, b ∈ B; (a, b) in existing], x[a, b] == a * b
        end
        by = @block m begin
            y[a ∈ A], y[a] == sum(x[a, b] for b in B if (a, b) in existing)
        end
        b = bx + by
        fix.(b, 0.0, force=true)
        unfix.(residuals(b))
        return m, y, b
    end

    # Warmup
    m_u, _, _ = build_unfiltered(); optimize!(m_u)
    m_f, _, _ = build_filtered(); optimize!(m_f)

    n_reps = 5

    # --- Build times ---
    t_build_u = minimum((@elapsed build_unfiltered()) for _ in 1:n_reps)
    t_build_f = minimum((@elapsed build_filtered()) for _ in 1:n_reps)

    # --- Solve times (on pre-built models) ---
    solve_times_u = Float64[]
    solve_times_f = Float64[]
    for _ in 1:n_reps
        m_u, _, _ = build_unfiltered()
        push!(solve_times_u, @elapsed optimize!(m_u))
        m_f, _, _ = build_filtered()
        push!(solve_times_f, @elapsed optimize!(m_f))
    end
    t_solve_u = minimum(solve_times_u)
    t_solve_f = minimum(solve_times_f)

    # Verify both produce the same solution
    m_u, y_u, _ = build_unfiltered(); optimize!(m_u)
    m_f, y_f, _ = build_filtered(); optimize!(m_f)
    for a in A
        @test value(y_u[a]) ≈ value(y_f[a]) atol=1e-4
    end

    r_build = t_build_u / t_build_f
    r_solve = t_solve_u / t_solve_f
    r_total = (t_build_u + t_solve_u) / (t_build_f + t_solve_f)

    println()
    println("  Ipopt benchmark ($(dim)×$(dim), $(Int(sparsity*100))% sparsity, $(length(existing)) stored keys)")
    println("  ├─ Build  unfiltered: $(round(t_build_u*1000, digits=1))ms  filtered: $(round(t_build_f*1000, digits=1))ms  ratio: $(round(r_build, digits=2))x")
    println("  ├─ Solve  unfiltered: $(round(t_solve_u*1000, digits=1))ms  filtered: $(round(t_solve_f*1000, digits=1))ms  ratio: $(round(r_solve, digits=2))x")
    println("  └─ Total  unfiltered: $(round((t_build_u+t_solve_u)*1000, digits=1))ms  filtered: $(round((t_build_f+t_solve_f)*1000, digits=1))ms  ratio: $(round(r_total, digits=2))x")

    @test r_build < 5
    @test r_solve < 2  # solve should be near-identical (same model structure)
end

@testset "Perf: Ipopt build vs solve — scaling with sparsity" begin
    dim = 40

    for sparsity in (0.5, 0.9, 0.99)
        A, B = 1:dim, 1:dim
        n_existing = max(dim, round(Int, dim^2 * (1 - sparsity)))
        existing = Set{Tuple{Int,Int}}()
        for a in A
            push!(existing, (a, rand(B)))
        end
        while length(existing) < n_existing
            push!(existing, (rand(A), rand(B)))
        end

        function _build_unfiltered()
            m = Model(Ipopt.Optimizer)
            JuMP.set_silent(m)
            @variables m begin
                x[a=A, b=B; (a, b) in existing], "x"
                y[A], "y"
            end
            bx = @block m begin
                x[a ∈ A, b ∈ B; (a, b) in existing], x[a, b] == a * b
            end
            by = @block m begin
                y[a ∈ A], y[a] == ∑(x[a, b] for b in B)
            end
            b = bx + by
            fix.(b, 0.0, force=true)
            unfix.(residuals(b))
            return m
        end

        function _build_filtered()
            m = Model(Ipopt.Optimizer)
            JuMP.set_silent(m)
            use_sparse_zero_array!(false)
            @variables m begin
                x[a=A, b=B; (a, b) in existing], "x"
                y[A], "y"
            end
            use_sparse_zero_array!(true)
            bx = @block m begin
                x[a ∈ A, b ∈ B; (a, b) in existing], x[a, b] == a * b
            end
            by = @block m begin
                y[a ∈ A], y[a] == sum(x[a, b] for b in B if (a, b) in existing)
            end
            b = bx + by
            fix.(b, 0.0, force=true)
            unfix.(residuals(b))
            return m
        end

        # Warmup
        optimize!(_build_unfiltered())
        optimize!(_build_filtered())

        t_build_u = minimum((@elapsed _build_unfiltered()) for _ in 1:3)
        t_build_f = minimum((@elapsed _build_filtered()) for _ in 1:3)

        solve_u = Float64[]
        solve_f = Float64[]
        for _ in 1:3
            m = _build_unfiltered(); push!(solve_u, @elapsed optimize!(m))
            m = _build_filtered(); push!(solve_f, @elapsed optimize!(m))
        end
        t_solve_u = minimum(solve_u)
        t_solve_f = minimum(solve_f)

        r_build = t_build_u / t_build_f
        r_solve = t_solve_u / t_solve_f
        println("  sparsity=$(Int(sparsity*100))%: build $(round(t_build_u*1000, digits=1))/$(round(t_build_f*1000, digits=1))ms ($(round(r_build, digits=2))x)  solve $(round(t_solve_u*1000, digits=1))/$(round(t_solve_f*1000, digits=1))ms ($(round(r_solve, digits=2))x)")
        @test r_build < 5
        @test r_solve < 2
    end
end
