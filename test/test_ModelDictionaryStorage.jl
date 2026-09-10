module TestModelDictionaryStorage

using Test, JuMP, SquareModels, Dictionaries
using DataFrames, Parquet2

@testset "Typed independent datasets share model metadata" begin
    m = Model()
    @variable(m, x[1:4])
    a, b = ModelDictionary(m), ModelDictionary(m)
    a[x] = [1, 2, 3, 4]
    @test a[x[1]] === 1.0
    @test eltype(values(a)) == Union{Nothing,Float64}
    @test keys(a) === keys(b)
    @test all(isnothing, b[x])
    c = copy(a)
    @test keys(c) === keys(a)
    c[x[1]] = 99
    @test a[x[1]] == 1
    mixed = ModelDictionary{Number}(m)
    mixed[x] = Number[1, 1//3, big"2.123456789123456789", 4.0]
    @test mixed[x[1]] === 1
    @test mixed[x[2]] === 1//3
    @test mixed[x[3]] isa BigFloat
    integer = ModelDictionary{Int}(m, [1,2,3,4])
    @test integer[x[2]] === 2
    @test_throws InexactError integer[x[1]] = 0.5
    @test_throws DimensionMismatch ModelDictionary{Float64}(m, [1.0])
    imported = ModelDictionary(m, Dictionary(["x[1]"], Any[1//3]))
    @test imported[x[1]] === 1//3
    @test_throws MethodError ModelDictionary(m, Dictionary(["x[1]"], Any["invalid"]))
    @test_throws KeyError a[["not_a_variable"]]
    other = Model()
    @variable(other, x[1:4])
    @test_throws ArgumentError a[x[1]]
    @test_throws ArgumentError a[x[1]] = 7
    @test !(x[1] in a)
    anonymous_model = Model()
    anonymous = @variable(anonymous_model)
    anonymous_data = ModelDictionary(anonymous_model)
    anonymous_data[[""]] .= 5
    @test anonymous_data[anonymous] == anonymous_data[""] == 5
    @test collect(anonymous_data[[""]]) == [5]
end

@testset "Selections preserve coordinates, order, duplicates and ownership" begin
    m = Model()
    @variable(m, x[[:a,:b], 2020:2022])
    a, b = ModelDictionary(m), ModelDictionary(m)
    a[x] .= collect(1.0:6.0)
    selection = prepare_selection(a, x)
    b[selection] = collect(11.0:16.0)
    @test [a[x[k...]] for k in [(:a,2020),(:b,2021),(:a,2022)]] == [1,4,5]
    @test b[selection][:b,2021] == 14
    @test a[selection].indices === b[selection].indices
    @test a[x].indices === b[x].indices
    cache_size = length(a._layout.selection_cache)
    for _ in 1:5
        a[[x[:b,2022], x[:a,2020]]]
    end
    @test length(a._layout.selection_cache) == cache_size
    repeated = prepare_selection(a, [x[:b,2022], x[:a,2020], x[:b,2022]])
    a[repeated] .= [60.0,10.0,61.0]
    @test a[x[:b,2022]] == 61
    @test a[x[:a,2020]] == 10
    @test collect(a[repeated]) == [61,10,61]
    temporary = [x[:a,2020], x[:b,2021]]
    frozen = prepare_selection(a, temporary)
    reverse!(temporary)
    @test collect(a[frozen]) == [10,4]

    labels = [:first, :second]
    labelled = JuMP.Containers.DenseAxisArray(temporary, labels)
    frozen_labels = prepare_selection(a, labelled)
    reverse!(labels)
    @test axes(a[frozen_labels].indices) == ([:first, :second],)
    @test a[frozen_labels][:first] == 4
    @test a[frozen_labels][:second] == 10
    source = KeyedData(Dict((:first,) => 7.0))
    a[frozen_labels] = source
    @test a[x[:b,2021]] == 7
    @test isnothing(a[x[:a,2020]])
    foreign = Model()
    @variable(foreign, y)
    @test_throws ArgumentError ModelDictionary(foreign)[selection]
    empty_selection = prepare_selection(a, VariableRef[])
    @test isempty(a[empty_selection])
end

@testset "Layout refresh preserves values by identity and invalidates views" begin
    m = Model()
    @variable(m, x[1:3])
    a = ModelDictionary(m, [11.0,22.0,33.0])
    b = copy(a)
    selection, old_window = prepare_selection(a, x), a[x]
    set_name(x[2], "renamed")
    refresh_model_layout!(m)
    @test a["renamed"] == 22
    @test b[x[2]] == 22
    @test keys(a) === keys(b)
    @test_throws KeyError a["x[2]"]
    @test_throws ArgumentError a[selection]
    @test_throws ArgumentError old_window[1]
    @test_throws ArgumentError old_window .= 1
    @test_throws ArgumentError collect(old_window)
    delete(m, x[1])
    @variable(m, z)
    refresh_model_layout!(m)
    @test isnothing(a[z])
    @test a[x[3]] == 33
    @test_throws KeyError a[x[1]]
    @test length(a) == 3
    @variable(m, extra)
    @test isnothing(a[extra])
    @test length(a) == 4
    @test a[x[3]] == 33
    current_window = a[[x[3]]]
    current_selection = prepare_selection(a, [x[3]])
    empty!(m)
    @variable(m, replacement[1:4])
    fresh = ModelDictionary(m)
    @test all(isnothing, fresh[replacement])
    @test_throws ArgumentError a[replacement[1]]
    @test_throws ArgumentError a[current_selection]
    @test_throws ArgumentError current_window[1]
    @test_throws ArgumentError keys(a)
    @test_throws ArgumentError values(a)
    @test_throws ArgumentError collect(a)
end

@testset "Export synchronizes refreshed names and surviving variables" begin
    m = Model()
    @variable(m, x[1:4])
    full = ModelDictionary(m, [10.0,20.0,30.0,40.0])
    subset = ModelDictionary(m, Dictionary(["x[4]","x[2]"], [44.0,22.0]))
    set_name(x[2], "renamed[2025]")
    delete(m, x[1])
    delete(m, x[4])
    @variable(m, added)
    refresh_model_layout!(m)

    mktempdir() do directory
        # Export before any dictionary access can synchronize either dataset.
        full_path = joinpath(directory, "full.parquet")
        subset_path = joinpath(directory, "subset.parquet")
        unload(full_path, full)
        unload(subset_path, subset)
        # Reading through an IOBuffer avoids memory-mapped file handles on Windows.
        full_rows = DataFrame(Parquet2.Dataset(IOBuffer(read(full_path))))
        subset_rows = DataFrame(Parquet2.Dataset(IOBuffer(read(subset_path))))
        @test Dict((r.variable, r.indices) => r.value for r in eachrow(full_rows)) ==
            Dict(("renamed", "2025") => 20.0, ("x", "3") => 30.0)
        @test Dict((r.variable, r.indices) => r.value for r in eachrow(subset_rows)) ==
            Dict(("renamed", "2025") => 22.0)
        @test collect(keys(subset)) == ["renamed[2025]"]

        empty!(m)
        stale_path = joinpath(directory, "stale.parquet")
        @test_throws ArgumentError unload(stale_path, full)
        @test !isfile(stale_path)
    end
end

@testset "Filtered datasets stay independent selections on read" begin
    m = Model()
    @variable(m, x[1:4])
    full = ModelDictionary(m, [10.0,20.0,30.0,40.0])
    subset = full[full .> 20]
    @test length(subset) == 2
    selection = prepare_selection(full, x[3:4])
    @test collect(subset[selection]) == [30,40]
    subset[selection] .= [31,41]
    @test full[x[3]] == 30
    @test_throws KeyError subset[x[1]]
    @test length(subset) == 2
    @variable(m, z)
    @test subset[x[3]] == 31
    @test length(subset) == 2
    @test_throws KeyError subset[z]
    add_missing_model_variables!(subset)
    @test length(subset) == 5
    @test subset[x[3]] == 31
    @test isnothing(subset[x[1]])
    @test isnothing(subset[z])
    @test keys(subset) === keys(ModelDictionary(m))
    reordered = ModelDictionary(m, Dictionary(["x[4]","x[2]"], [44.0,22.0]))
    @test collect(reordered[prepare_selection(full, x[2:2:4])]) == [22,44]
end

end
