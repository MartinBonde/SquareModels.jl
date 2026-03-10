using Test
using SquareModels
import JuMP
using JuMP: Model, @variable, all_variables, fix, unfix, is_fixed, value, optimize!, set_start_value, AffExpr, name
using JuMP.Containers: SparseAxisArray
using Ipopt

@testset "Zero sentinel" begin
    z = SquareModels.Zero()
    @test z + 1 === 1
    @test 1 + z === 1
    @test z + z === z
    @test z * 5 === z
    @test 5 * z === z
    @test -z === z
    @test 3 - z === 3
    @test iszero(z)

    # Works with JuMP expressions
    m = Model()
    @variable(m, x)
    @test x + z === x
    @test z + x === x
    expr = AffExpr(1.0, x => 2.0)
    @test expr + z === expr
    @test z + expr === expr
end

@testset "SparseZeroArray construction" begin
    m = Model()
    @variable(m, x[i=1:5, j=[:a, :b]; i <= 3])
    @test x isa SparseAxisArray

    domain = (Set(1:5), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)
    @test sz isa SparseZeroArray
    @test length(sz) == length(x)
end

@testset "SparseZeroArray scalar indexing" begin
    m = Model()
    @variable(m, x[i=1:5, j=[:a, :b]; i <= 3])
    domain = (Set(1:5), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)

    # Existing key returns the variable
    @test sz[1, :a] isa VariableRef
    @test sz[1, :a] === x[1, :a]

    # Missing key within domain returns Zero()
    @test sz[4, :a] isa SquareModels.Zero
    @test sz[5, :b] isa SquareModels.Zero

    # Out-of-domain key throws
    @test_throws ErrorException sz[6, :a]
    @test_throws ErrorException sz[1, :c]
end

@testset "SparseZeroArray slicing" begin
    m = Model()
    @variable(m, x[i=1:3, j=[:a, :b]; j == :a || i <= 2])
    domain = (Set(1:3), Set([:a, :b]))
    sz = SparseZeroArray(x, domain)

    # Colon slicing delegates to underlying SparseAxisArray
    slice = sz[:, :a]
    @test length(slice) == 3
end

@testset "SparseZeroArray forwarded methods" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    @test length(sz) == 6
    @test haskey(sz, (1, 2))
    @test !haskey(sz, (1, 1))
    @test first(sz) isa VariableRef
    @test eltype(typeof(sz)) == VariableRef
    vals = VariableRef[]
    for v in sz
        push!(vals, v)
    end
    @test vals isa Vector{VariableRef}
    @test length(vals) == length(sz)
end

@testset "SparseZeroArray sum with ∑" begin
    m = Model()
    @variable(m, x[i=1:5, j=1:5; i <= 2])
    domain = (Set(1:5), Set(1:5))
    sz = SparseZeroArray(x, domain)

    # Sum over all i for a given j — most i's are missing, should get Zero()
    result = ∑(sz[i, 3] for i in 1:5)
    # Only i=1 and i=2 exist, the rest are Zero()
    @test result isa AffExpr
end

@testset "_all_keys for SparseZeroArray" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    ks = SquareModels._all_keys(sz)
    @test length(ks) == 6
    @test (1, 2) in ks
    @test (1, 1) ∉ ks
end

@testset "copy_variable for SparseZeroArray" begin
    m = Model()
    @variable(m, x[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(x, domain)

    copied = SquareModels.copy_variable("x_copy", sz)
    @test copied isa SparseAxisArray
    @test length(copied) == length(sz)
    @test haskey(m, :x_copy)
end

@testset "base_name for SparseZeroArray" begin
    m = Model()
    @variable(m, my_var[i=1:3, j=1:3; i != j])
    domain = (Set(1:3), Set(1:3))
    sz = SparseZeroArray(my_var, domain)

    @test SquareModels.base_name(sz) == "my_var"
end

@testset "@variables auto-wrapping" begin
    m = Model()
    t = 1:3

    @variables m begin
        sparse_var[i=1:5, j=[:a, :b]; i <= 3], "A sparse variable"
        dense_var[t], "A dense variable"
        scalar_var, "A scalar"
    end

    @test sparse_var isa SparseZeroArray
    @test dense_var isa JuMP.Containers.DenseAxisArray
    @test scalar_var isa VariableRef

    # Auto-wrapped SparseZeroArray works with zero-default
    @test sparse_var[1, :a] isa VariableRef
    @test sparse_var[4, :a] isa SquareModels.Zero

    # Domain checking from index sets
    @test_throws ErrorException sparse_var[6, :a]
    @test_throws ErrorException sparse_var[1, :c]
end

@testset "@variables with tuple destructuring" begin
    m = Model()
    pairs = [(:a, :b), (:c, :d)]
    pairs_set = Set(pairs)
    t = 1:3

    @variables m begin
        x[i=[:a, :c], d=[:b, :d], tt=t; (i, d) in pairs_set], "Sparse with condition"
    end

    @test x isa SparseZeroArray
    @test x[:a, :b, 1] isa VariableRef
    # Valid domain but missing combination
    @test x[:a, :d, 1] isa SquareModels.Zero
    # Out of domain
    @test_throws ErrorException x[:z, :b, 1]
end

@testset "ModelDictionary with SparseZeroArray" begin
    m = Model()
    @variables m begin
        x[i=1:3, j=1:3; i != j], "Sparse var"
    end
    @test x isa SparseZeroArray

    d = ModelDictionary(m)
    # getindex delegates to underlying SparseAxisArray
    w = d[x]
    @test w isa SquareModels.Window

    # setindex! delegates
    d[x] .= 1.0
    @test all(d[v] == 1.0 for v in x)
end

@testset "@block with SparseZeroArray" begin
    m = Model(Ipopt.Optimizer)

    @variables m begin
        x[i=1:3, j=1:3; i != j], "Sparse var"
        param[i=1:3, j=1:3; i != j], "Parameter"
    end

    @test x isa SparseZeroArray
    @test param isa SparseZeroArray

    b = @block m begin
        x[i ∈ 1:3, j ∈ 1:3; i != j], x[i, j] == param[i, j] + 1
    end

    @test length(b) == 6
    @test all(is_endogenous(x.data[i, j], b) for i in 1:3, j in 1:3 if i != j)
end

@testset "@block with SparseZeroArray and ∑" begin
    m = Model(Ipopt.Optimizer)

    @variables m begin
        x[i=1:3, j=1:3; i <= j], "Sparse"
        y[1:3], "Dense"
    end

    @test x isa SparseZeroArray

    # Use ∑ to sum over sparse variable without filter clause
    b = @block m begin
        y[i ∈ 1:3], y[i] == ∑(x[i, j] for j in 1:3)
    end

    @test length(b) == 3

    # Fix x values and solve
    for i in 1:3, j in 1:3
        if i <= j
            fix(x.data[i, j], Float64(i + j), force=true)
        end
    end
    fix.(residuals(b), 0.0, force=true)
    optimize!(m)

    # y[1] = x[1,1] + x[1,2] + x[1,3] = 2 + 3 + 4 = 9
    @test value(y[1]) ≈ 9.0 atol=1e-6
    # y[2] = Zero() + x[2,2] + x[2,3] = 0 + 4 + 5 = 9
    @test value(y[2]) ≈ 9.0 atol=1e-6
    # y[3] = Zero() + Zero() + x[3,3] = 0 + 0 + 6 = 6
    @test value(y[3]) ≈ 6.0 atol=1e-6
end
