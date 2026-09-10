module TestLoadPreparation

using Test
using JuMP
using SquareModels
using DataFrames

@testset "Simple loading reuses canonical model keys" begin
    model = Model()
    @variable(model, C[t = 2025:2026])
    @variable(model, N[t = 2025:2026])
    @variable(model, scalar)
    frame = DataFrame(
        variable = ["C", "vC", "vC", "nPop", "scalar", "outside"],
        indices = ["2025", "cTot,2025", "cTot,2026", "2025", "", "2025"],
        value = [1.0, 25.0, 26.0, 125.0, 50.0, 999.0],
    )
    no_renames = Dict{String,String}()
    no_slices = Dict{String,Tuple{String,Vector{String},Vector{Int}}}()
    first = SquareModels._load_simple(frame, model, no_renames, no_slices)
    @test first[C[2025]] === 1.0
    @test isnothing(first[C[2026]])
    @test isnothing(first[N[2025]])
    @test first[scalar] === 50.0
    prepared = first._layout.selection_cache[:SquareModels_load_keys]
    @test prepared isa Vector{Tuple{String,String}}

    renames = Dict("N" => "nPop")
    renamed = SquareModels._load_simple(frame, model, renames, no_slices)
    @test renamed._layout.selection_cache[:SquareModels_load_keys] === prepared
    @test renamed[N[2025]] === 125.0
    @test isnothing(renamed[N[2026]])
    @test renamed[C[2025]] === 1.0
    slices = Dict("C" => ("vC", ["cTot"], [2]))
    sliced = SquareModels._load_simple(frame, model, renames, slices)
    @test sliced._layout.selection_cache[:SquareModels_load_keys] === prepared
    @test sliced[C[2025]] === 25.0
    @test sliced[C[2026]] === 26.0
    @test sliced[N[2025]] === 125.0

    # Cached entries contain model coordinates only, never values from a prior
    # source. Changes to the frame, duplicate rows, and missing observations
    # use the last row for duplicate keys and clear absent observations.
    changed_frame = DataFrame(variable = ["C", "C"], indices = ["2025", "2025"], value = [2.0, 3.0])
    changed = SquareModels._load_simple(changed_frame, model, no_renames, no_slices)
    @test changed._layout.selection_cache[:SquareModels_load_keys] === prepared
    @test changed[C[2025]] === 3.0
    @test isnothing(changed[scalar])
    @test first[C[2025]] === 1.0

    # A rename leaves the variable count unchanged. Explicit refresh must
    # invalidate the prepared names before the next load.
    JuMP.set_name(scalar, "renamed_scalar")
    refresh_model_layout!(model)
    @test !haskey(first._layout.selection_cache, :SquareModels_load_keys)
    renamed_frame = DataFrame(variable = ["renamed_scalar"], indices = [""], value = [60.0])
    refreshed = SquareModels._load_simple(renamed_frame, model, no_renames, no_slices)
    @test refreshed._layout.selection_cache[:SquareModels_load_keys] !== prepared
    @test refreshed[scalar] === 60.0
    after_rename = refreshed._layout.selection_cache[:SquareModels_load_keys]

    # A variable-count change refreshes the layout automatically.
    @variable(model, added)
    added_frame = DataFrame(variable = ["added"], indices = [""], value = [70.0])
    expanded = SquareModels._load_simple(added_frame, model, no_renames, no_slices)
    @test expanded._layout.selection_cache[:SquareModels_load_keys] !== after_rename
    @test expanded[added] === 70.0
    @test isnothing(expanded[scalar])
end

end
