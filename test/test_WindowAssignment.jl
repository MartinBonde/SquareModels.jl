module TestWindowAssignment

using Test
using JuMP
using SquareModels

function window(storage, slots, shape=(length(slots),))
    indices = reshape(collect(1:length(slots)), shape)
    return SquareModels.Window(view(storage, slots), indices, nothing)
end

fused_assign!(w, a, b) = (w .= a .* b .+ 3.0; nothing)

@testset "Window assignment evaluates into destination" begin
    storage = zeros(6)
    w = window(storage, [6, 2, 5, 1, 4, 3], (2, 3))
    a = reshape(collect(1.0:6.0), 2, 3)
    b = [2.0, 5.0]
    fused_assign!(w, a, b)
    @test collect(w) == a .* b .+ 3.0
    @test storage == [23, 13, 33, 13, 9, 5]

    # Plain sources are positional and flattened, regardless of labelled shape.
    w .= [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    @test collect(w) == [10 30 50; 20 40 60]
    @test storage == [40, 20, 60, 50, 30, 10]
    w .= [9.0]
    @test all(==(9.0), storage)
    w .= 4.0
    @test all(==(4.0), storage)

    @test_throws DimensionMismatch w .= [1.0, 2.0]
    @test all(==(4.0), storage)
    @test_throws DimensionMismatch w .= zeros(2, 3) .+ zeros(4)
    @test all(==(4.0), storage)

    empty = window(Float64[], Int[], (0, 2))
    empty .= zeros(0, 2) .+ 2.0
    empty .= 1.0
    @test isempty(collect(empty))
end

@testset "Window assignment snapshots overlapping reads" begin
    storage = collect(1.0:6.0)
    destination = window(storage, [2, 3, 4, 5, 6])
    source = window(storage, [1, 2, 3, 4, 5])
    destination .= source .+ 10.0
    @test storage == [1, 11, 12, 13, 14, 15]

    storage .= 1.0:6.0
    destination = window(storage, [6, 5, 4, 3, 2, 1])
    source = window(storage, [1, 2, 3, 4, 5, 6])
    destination .= 2.0 .* source
    @test storage == [12, 10, 8, 6, 4, 2]

    storage .= 1.0:6.0
    repeated = window(storage, [1, 1, 2])
    repeated .= repeated .+ [10.0, 20.0, 30.0]
    @test storage == [21, 32, 3, 4, 5, 6]
    repeated .= repeated .+ 1.0
    @test storage == [22, 33, 3, 4, 5, 6]

    # Direct array assignment also needs a snapshot for an identical repeated view.
    SquareModels._set_window!(repeated, repeated.data_view)
    @test storage == [22, 33, 3, 4, 5, 6]
end

@testset "Window assignment preserves keyed replacement" begin
    m = Model()
    SquareModels.@variables m begin
        dense[i = [:a, :b], t = 1:2]
        sparse[i = [:a, :b], t = 1:2; (i, t) != (:a, 2)]
    end
    d = ModelDictionary(m)
    d[dense] .= 100.0
    d[dense] .= KeyedData([(:b, 2) => 22.0, (:a, 1) => 11.0])
    @test d[dense[:a, 1]] == 11.0
    @test d[dense[:b, 2]] == 22.0
    @test d[dense[:b, 1]] === nothing
    @test d[dense[:a, 2]] === nothing

    OrderedDict = JuMP.Containers.OrderedCollections.OrderedDict
    source = JuMP.Containers.SparseAxisArray(OrderedDict(
        [(:b, 2) => 202.0, (:a, 1) => 101.0],
    ))
    d[sparse] .= 100.0
    d[sparse] .= source
    @test d[sparse[:a, 1]] == 101.0
    @test d[sparse[:b, 2]] == 202.0
    @test d[sparse[:b, 1]] === nothing
    d[dense] .= d[sparse]
    @test d[dense[:a, 1]] == 101.0
    @test d[dense[:b, 2]] == 202.0
    @test d[dense[:b, 1]] === nothing
    @test d[dense[:a, 2]] === nothing
end

@testset "Repeated keyed assignment across datasets" begin
    m = Model()
    @variable(m, x[i = [:a, :b], t = 2025:2026])
    @variable(m, unrelated)
    baseline, scenario = ModelDictionary(m, 91.0), ModelDictionary(m, 92.0)
    selection = prepare_selection(baseline, x)
    source = KeyedData(Dict((:a, 2025) => 0.0, (:b, 2026) => 26.0, (:outside, 2025) => 999.0))
    baseline[x] = source
    @test baseline[x[:a, 2025]] === 0.0
    @test baseline[x[:b, 2026]] === 26.0
    @test isnothing(baseline[x[:b, 2025]])
    @test isnothing(baseline[x[:a, 2026]])

    source.data[(:a, 2025)] = 15.0
    source.data[(:b, 2025)] = 35.0
    delete!(source.data, (:b, 2026))
    baseline[x] = source
    scenario[selection] = KeyedData(Dict(reverse(collect(pairs(source.data)))))
    for d in (baseline, scenario)
        @test d[x[:a, 2025]] === 15.0
        @test d[x[:b, 2025]] === 35.0
        @test isnothing(d[x[:b, 2026]])
        @test isnothing(d[x[:a, 2026]])
    end
    @test baseline[unrelated] === 91.0
    @test scenario[unrelated] === 92.0

    scenario[selection] = SparseZeroArray(Dict((:a, 2025) => 0.0))
    @test scenario[x[:a, 2025]] === 0.0
    @test isnothing(scenario[x[:b, 2025]])
    empty_source = KeyedData(Dict{Tuple{Symbol,Int},Float64}())
    baseline[selection] = empty_source
    @test all(isnothing, baseline[x])
    baseline[x[Symbol[], :]] = empty_source
    @test baseline[unrelated] === 91.0
end

@testset "Nonaliasing Window broadcast avoids a result-sized allocation" begin
    n = 20_000
    storage = zeros(n)
    w = window(storage, collect(n:-1:1))
    a = collect(1.0:n)
    b = fill(2.0, n)
    fused_assign!(w, a, b)
    bytes = @allocated fused_assign!(w, a, b)
    @test storage[1] == 2n + 3
    @test storage[end] == 5.0
    # A Float64 result alone needs 160 kB; allow small fixed broadcast metadata.
    @test bytes < n * sizeof(Float64) ÷ 4
end

@testset "Bulk Window operations validate stale storage including empty windows" begin
    m = Model()
    @variable(m, x[1:2])
    @variable(m, empty[1:0])
    SquareModels.@variables m begin
        sparse[i = 1:2, j = 1:2; i == j]
    end
    d = ModelDictionary(m)
    w = d[x]
    empty_window = d[empty]
    sparse_window = d[sparse]
    refresh_model_layout!(m)
    @test_throws ArgumentError w[1]
    @test_throws ArgumentError w .= 1.0
    @test_throws ArgumentError w .= zeros(2)
    @test_throws ArgumentError collect(w)
    @test_throws ArgumentError iterate(w)
    @test_throws ArgumentError empty_window .= 1.0
    @test_throws ArgumentError empty_window .= Float64[]
    @test_throws ArgumentError collect(empty_window)
    @test_throws ArgumentError iterate(empty_window)
    @test_throws ArgumentError sparse_window[1, 2]
end

@testset "Saved Window views and lazy broadcasts retain stale-storage checks" begin
    m = Model()
    @variable(m, x[1:2, 1:2])
    d = ModelDictionary(m)
    d[x] .= [1.0, 2.0, 3.0, 4.0]
    w = d[x]
    shaped = w.shaped_view
    saved = Base.broadcasted(identity, w)
    nested = Base.broadcasted(+, w, ones(2, 2))
    refresh_model_layout!(m)
    d[x] .= 9.0
    @test_throws ArgumentError shaped[1, 1]
    @test_throws ArgumentError Base.materialize(saved)
    @test_throws ArgumentError Base.materialize(nested)
    @test_throws ArgumentError Base.materialize!(d[x], saved)
    @test collect(d[x]) == fill(9.0, 2, 2)
end

@testset "Executing Window broadcasts unwrap checked operands without copying" begin
    m = Model()
    @variable(m, x[1:100, 1:200])
    a, b, destination = ModelDictionary(m), ModelDictionary(m), ModelDictionary(m)
    a[x] .= collect(1.0:20_000.0)
    b[x] .= 2.0
    source_a, source_b, target = a[x], b[x], destination[x]
    fused_assign!(target, source_a, source_b)
    bytes = @allocated fused_assign!(target, source_a, source_b)
    @test destination[x[1, 1]] == 5.0
    @test destination[x[100, 200]] == 40_003.0
    @test bytes < 20_000 * sizeof(Float64) ÷ 4
end

@testset "DenseAxisArray assignment uses positional values" begin
    m = Model()
    @variable(m, x[[:a, :b], 2020:2021])
    d = ModelDictionary(m)
    DenseAxisArray = JuMP.Containers.DenseAxisArray
    # Labels intentionally differ in order; dense assignment is positional.
    source = DenseAxisArray([1.0 3.0; 2.0 4.0], [:b, :a], 2021:-1:2020)
    d[x] = source
    @test collect(d[x]) == [1.0 3.0; 2.0 4.0]
    d[x] .= source
    @test d[x[:a, 2020]] == 1.0
    @test d[x[:b, 2021]] == 4.0
    d[x] .= source .+ 1.0
    @test collect(d[x]) == [2.0 4.0; 3.0 5.0]
    d[x] = source .* 2.0
    @test collect(d[x]) == [2.0 6.0; 4.0 8.0]

    flat = DenseAxisArray([10.0, 20.0, 30.0, 40.0], [:w, :x, :y, :z])
    d[x] = flat
    @test collect(d[x]) == [10.0 30.0; 20.0 40.0]
    d[x] .= flat
    @test collect(d[x]) == [10.0 30.0; 20.0 40.0]
    singleton = DenseAxisArray(fill(9.0, 1, 1), [:ignored], [1000])
    d[x] = singleton
    @test collect(d[x]) == fill(9.0, 2, 2)
    d[x] .= singleton
    @test collect(d[x]) == fill(9.0, 2, 2)
    wrong = DenseAxisArray([1.0, 2.0], [:a, :b])
    @test_throws DimensionMismatch d[x] = wrong
    @test_throws DimensionMismatch d[x] .= wrong
    @test_throws DimensionMismatch d[x] .= wrong .+ 1.0
    @test collect(d[x]) == fill(9.0, 2, 2)

    # A labelled source can share a dataset's buffer. Reordered writes must
    # snapshot it before the first destination cell is changed.
    d[x] .= [1.0, 2.0, 3.0, 4.0]
    aliased = DenseAxisArray(reshape(d.dictionary.values, 2, 2), [:a, :b], 2020:2021)
    destination = [x[:b, 2021], x[:a, 2021], x[:b, 2020], x[:a, 2020]]
    d[destination] .= aliased
    @test collect(d[x]) == [4.0 2.0; 3.0 1.0]
end

end
