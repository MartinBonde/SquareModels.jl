module TestLoadPreparation

using Test
using JuMP
using SquareModels
using DataFrames

@testset "Simple loading maps rows to model variables" begin
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

    renames = Dict("N" => "nPop")
    renamed = SquareModels._load_simple(frame, model, renames, no_slices)
    @test renamed[N[2025]] === 125.0
    @test isnothing(renamed[N[2026]])
    @test renamed[C[2025]] === 1.0
    slices = Dict("C" => ("vC", ["cTot"], [2]))
    sliced = SquareModels._load_simple(frame, model, renames, slices)
    @test sliced[C[2025]] === 25.0
    @test sliced[C[2026]] === 26.0
    @test sliced[N[2025]] === 125.0

    # A renamed or sliced base does not read rows under its own name. Two model
    # bases can read the same data symbol.
    own_rows = DataFrame(variable = ["N", "C", "C"], indices = ["2025", "2025", "cTot,2025"], value = [1.0, 2.0, 3.0])
    shared = SquareModels._load_simple(own_rows, model, Dict("N" => "C", "C" => "N"), no_slices)
    @test shared[N[2025]] === 2.0
    @test shared[C[2025]] === 1.0
    both = SquareModels._load_simple(own_rows, model, Dict("N" => "C", "C" => "N"), Dict("C" => ("C", ["cTot"], [2])))
    @test both[N[2025]] === 2.0
    @test both[C[2025]] === 3.0

    # Later rows win for duplicate keys. Absent observations stay unset.
    changed_frame = DataFrame(variable = ["C", "C"], indices = ["2025", "2025"], value = [2.0, 3.0])
    changed = SquareModels._load_simple(changed_frame, model, no_renames, no_slices)
    @test changed[C[2025]] === 3.0
    @test isnothing(changed[scalar])
    @test first[C[2025]] === 1.0

    # A row matches only the variable whose name parses back to the row's key.
    @variable(model, odd, base_name = "x[a]")
    odd_frame = DataFrame(variable = ["x[a]", "x", "C", "C[2025]"], indices = ["", "a", "2025", ""], value = [1.0, 2.0, 4.0, 3.0])
    odd_loaded = SquareModels._load_simple(odd_frame, model, no_renames, no_slices)
    @test odd_loaded[odd] === 2.0
    @test odd_loaded[C[2025]] === 4.0

    # Explicit refresh after a rename makes the new name loadable.
    JuMP.set_name(scalar, "renamed_scalar")
    refresh_model_layout!(model)
    renamed_frame = DataFrame(variable = ["renamed_scalar", "scalar"], indices = ["", ""], value = [60.0, 61.0])
    refreshed = SquareModels._load_simple(renamed_frame, model, no_renames, no_slices)
    @test refreshed[scalar] === 60.0

    # A variable-count change refreshes the layout automatically.
    @variable(model, added)
    added_frame = DataFrame(variable = ["added"], indices = [""], value = [70.0])
    expanded = SquareModels._load_simple(added_frame, model, no_renames, no_slices)
    @test expanded[added] === 70.0
    @test isnothing(expanded[scalar])
end

end
