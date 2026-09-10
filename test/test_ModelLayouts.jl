module TestModelLayouts

using Test
using JuMP
using SquareModels
using Dictionaries: Dictionary

mutable struct CountingModel <: JuMP.AbstractModel
    inner::JuMP.Model
    scans::Int
    ext::Dict{Symbol,Any}
end
JuMP.num_variables(model::CountingModel) = JuMP.num_variables(model.inner)
function JuMP.all_variables(model::CountingModel)
    model.scans += 1
    return JuMP.all_variables(model.inner)
end

@testset "Repeated name misses reuse the synchronized name index" begin
    inner = Model()
    JuMP.@variable(inner, x[1:100])
    model = CountingModel(inner, 0, Dict{Symbol,Any}())
    @test SquareModels.variable_by_name(model, "x[25]") == x[25]
    @test model.scans == 1
    for _ in 1:100
        @test SquareModels.variable_by_name(model, "absent") === nothing
    end
    @test model.scans == 1
    JuMP.@variable(inner, added)
    @test SquareModels.variable_by_name(model, "added") == added
    @test model.scans == 2
    refresh_model_layout!(model)
    @test model.scans == 3
end

@testset "Shared model layout and complete name lookup" begin
    model = Model()
    JuMP.@variable(model, x[1:3])
    layout = SquareModels._model_layout(model)
    @test layout === SquareModels._model_layout(model)
    @test layout.names == name.(x)
    @test layout.variables == x
    @test SquareModels._layout_slot(layout, x[2]) == 2
    @test SquareModels._layout_slot(layout, "x[3]") == 3
    @test collect(layout.name_indices) == name.(x)
    @test SquareModels.variable_by_name(model, "x[2]") == x[2]
    @test_throws KeyError SquareModels._layout_slot(layout, "missing")

    # Missing lookups reuse the completed index and never grow a negative cache.
    original_names = layout.name_to_slot
    original_variables = layout.variables
    original_indices = layout.name_indices
    layout.selection_cache[x] = :prepared
    revision = layout.revision
    for i in 1:100
        @test SquareModels.variable_by_name(model, "missing[$i]") === nothing
    end
    @test layout.name_to_slot === original_names
    @test layout.variables === original_variables
    @test layout.revision == revision
    @test length(layout.name_to_slot) == 3

    JuMP.@variable(model, y)
    @test SquareModels.variable_by_name(model, "y") == y
    @test layout === SquareModels._model_layout(model)
    @test layout.revision > revision
    @test length(original_variables) == 3
    @test length(layout.variables) == 4
    @test layout.name_indices !== original_indices
    @test collect(original_indices) == name.(x)
    @test isempty(layout.selection_cache)

    # A deletion leaves nonconsecutive MOI indices. Slots remain dense.
    JuMP.delete(model, x[2])
    refresh_model_layout!(model)
    @test SquareModels.variable_by_name(model, "x[2]") === nothing
    @test SquareModels._layout_slot(layout, x[3]) == 2
    @test SquareModels._layout_slot(layout, y) == 3
    @test_throws KeyError SquareModels._layout_slot(layout, x[2])
    other = Model()
    JuMP.@variable(other, foreign)
    @test_throws ArgumentError SquareModels._layout_slot(layout, foreign)
end

@testset "Explicit refresh handles renames and count-neutral changes" begin
    model = Model()
    JuMP.@variable(model, x)
    layout = SquareModels._model_layout(model)
    old_names = layout.names
    JuMP.set_name(x, "renamed")
    refresh_model_layout!(model)
    @test old_names == ["x"]
    @test SquareModels.variable_by_name(model, "x") === nothing
    @test SquareModels.variable_by_name(model, "renamed") == x
    @test layout.names == ["renamed"]

    JuMP.delete(model, x)
    y = JuMP.@variable(model, base_name = "replacement")
    refresh_model_layout!(model)
    @test SquareModels.variable_by_name(model, "renamed") === nothing
    @test SquareModels.variable_by_name(model, "replacement") == y
    @test_throws KeyError SquareModels._layout_slot(layout, x)
end

@testset "Duplicate and anonymous names match JuMP" begin
    model = Model()
    first = JuMP.@variable(model, base_name = "duplicate")
    second = JuMP.@variable(model, base_name = "duplicate")
    unnamed = JuMP.@variable(model)
    @test_throws ErrorException SquareModels.variable_by_name(model, "duplicate")
    @test SquareModels.variable_by_name(model, "") === nothing
    layout = SquareModels._model_layout(model)
    @test layout.name_indices === nothing
    @test SquareModels._layout_slot(layout, first) != SquareModels._layout_slot(layout, second)
    @test SquareModels._layout_slot(layout, unnamed) > 0
    JuMP.set_name(second, "unique")
    refresh_model_layout!(model)
    @test SquareModels.variable_by_name(model, "duplicate") == first
    @test SquareModels.variable_by_name(model, "unique") == second
    @test layout.name_indices !== nothing
    JuMP.@variable(model)
    @test SquareModels._model_layout(model).name_indices === nothing
end

@testset "Copying a model owns a separate layout" begin
    model = Model()
    JuMP.@variable(model, x)
    layout = SquareModels._model_layout(model)
    copied, references = JuMP.copy_model(model)
    copied_layout = SquareModels._model_layout(copied)
    @test copied_layout !== layout
    @test copied_layout.model === copied
    @test SquareModels.variable_by_name(copied, "x") == references[x]
    @test JuMP.owner_model(only(copied_layout.variables)) === copied
end

@testset "Emptying a model invalidates its old layout" begin
    model = Model()
    JuMP.@variable(model, old[1:2])
    layout = SquareModels._model_layout(model)
    @test SquareModels._check_model_layout(layout) === nothing
    empty!(model)
    JuMP.@variable(model, replacement[1:2])
    @test_throws ArgumentError SquareModels._check_model_layout(layout)
    @test_throws ArgumentError SquareModels._ensure_model_layout!(layout)
    refreshed = refresh_model_layout!(model)
    @test refreshed !== layout
    @test SquareModels.variable_by_name(model, "replacement[1]") == replacement[1]
    @test_throws ArgumentError SquareModels._ensure_model_layout!(layout)
end

@testset "Cached JuMP growth stamps track additions without counting variables" begin
    model = Model()
    JuMP.@variable(model, x[1:3])
    layout = SquareModels._model_layout(model)
    @test SquareModels._model_growth_stamp(model) == 3
    @test layout.growth_stamp == 3
    original_names = layout.name_to_slot
    @test SquareModels._ensure_model_layout!(layout) === layout
    @test layout.name_to_slot === original_names
    JuMP.delete(model, x[2])
    @test SquareModels._model_growth_stamp(model) == 3
    @test JuMP.num_variables(model) == 2
    # Deletions alone need explicit refresh, which retains the growth stamp.
    refresh_model_layout!(model)
    @test layout.n_variables == 2
    @test layout.growth_stamp == 3
    @test SquareModels.variable_by_name(model, "x[2]") === nothing
    JuMP.delete(model, x[1])
    JuMP.@variable(model, added)
    @test JuMP.num_variables(model) == 2
    @test SquareModels._model_growth_stamp(model) == 4
    # Even a count-neutral delete/add is automatically detected by growth.
    @test SquareModels.variable_by_name(model, "added") == added
    @test SquareModels.variable_by_name(model, "x[1]") === nothing
    @test layout.growth_stamp == 4
    @test SquareModels._layout_slot(layout, x[3]) == 1
    custom = CountingModel(Model(), 0, Dict{Symbol,Any}())
    @test SquareModels._model_growth_stamp(custom) === nothing
end

@testset "Foreign extension layouts are rebuilt for their owner" begin
    original = Model()
    JuMP.@variable(original, x)
    original_layout = SquareModels._model_layout(original)
    other = Model()
    JuMP.@variable(other, y)
    other.ext[SquareModels._MODEL_LAYOUT_KEY] = original_layout
    other_layout = SquareModels._model_layout(other)
    @test other_layout !== original_layout
    @test other_layout.model === other
    @test SquareModels.variable_by_name(other, "y") == y
    @test SquareModels.variable_by_name(other, "x") === nothing
    @test SquareModels._check_model_layout(original_layout) === nothing
end

@testset "Generic JuMP references retain concrete storage types" begin
    for T in (Float32, BigFloat)
        @testset "$T" begin
            model = GenericModel{T}()
            data = ModelDictionary{T}(model)
            ref_type = JuMP.variable_ref_type(model)
            # The empty model must keep its reference type for later additions.
            @test eltype(data._layout.variables) === ref_type
            @test eltype(data._variables) === ref_type
            @test fieldtype(typeof(data._layout), :variables) === Vector{ref_type}
            @test fieldtype(typeof(data), :_variables) === Vector{ref_type}
            @test isempty(data)
            JuMP.@variable(model, scalar)
            JuMP.@variable(model, dense[[:a, :b], 2025:2026])
            JuMP.@variable(model, sparse[i = 1:2, j = 1:2; i == j])
            data[scalar] = T(1.25)
            data[dense] = T[2, 3, 4, 5]
            data[sparse] .= T[6, 7]
            @test data[scalar] == T(1.25)
            @test data[scalar] isa T
            @test data[dense[:b, 2025]] == T(3)
            @test data[sparse[2, 2]] == T(7)
            @test collect(data[dense]) == T[2 4; 3 5]
            @test SquareModels.variable_by_name(model, "scalar") == scalar
            @test SquareModels._layout_slot(data._layout, scalar) == 1
            @test eltype(data._variables) === ref_type

            # Numeric storage and JuMP's coefficient/reference type are separate.
            defaults = ModelDictionary(model)
            defaults[scalar] = 8.5
            @test defaults[scalar] === 8.5
            @test eltype(defaults._variables) === ref_type
            selection = prepare_selection(data, dense)
            copied = copy(data)
            copied[selection] .= T[10, 11, 12, 13]
            @test data[dense[:a, 2025]] == T(2)
            @test copied[dense[:a, 2025]] == T(10)
            @test eltype(copied._variables) === ref_type

            subset = data[data .> T(5)]
            @test subset[sparse[1, 1]] == T(6)
            @test eltype(subset._variables) === ref_type
            imported = ModelDictionary(model, Dictionary(["scalar"], T[9]))
            @test imported[scalar] == T(9)
            @test eltype(imported._variables) === ref_type

            JuMP.set_name(scalar, "renamed")
            refresh_model_layout!(model)
            @test data["renamed"] == T(1.25)
            @test_throws ArgumentError data[selection]
            @test eltype(data._variables) === ref_type
            copied_model, references = JuMP.copy_model(model)
            copied_layout = SquareModels._model_layout(copied_model)
            @test eltype(copied_layout.variables) === ref_type
            @test SquareModels.variable_by_name(copied_model, "renamed") == references[scalar]
            JuMP.delete(model, dense[:a, 2025])
            refresh_model_layout!(model)
            @test data[dense[:b, 2025]] == T(3)
            @test_throws KeyError data[dense[:a, 2025]]
            @test eltype(data._variables) === ref_type

            other = GenericModel{T}()
            JuMP.@variable(other, foreign)
            @test_throws ArgumentError data[foreign]
        end
    end
end

end
