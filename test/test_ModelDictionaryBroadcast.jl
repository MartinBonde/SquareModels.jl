module TestModelDictionaryBroadcast

using Test
using JuMP
using SquareModels
using Dictionaries

fused_dictionary_assign!(destination, a, b) = (destination .= 2.0 .* a .- b .+ 3.0; nothing)

@testset "Dictionary broadcasting uses aligned value buffers" begin
    m = Model()
    @variable(m, x[1:4])
    a = ModelDictionary(m)
    b = ModelDictionary(m)
    a[x] .= [1.0, 2.0, nothing, 4.0]
    b[x] .= [10.0, nothing, 30.0, 40.0]

    result = 2.0 .* a .- b .+ 3.0
    @test isequal(collect(result[x]), [-5.0, nothing, nothing, -29.0])
    @test keys(result.dictionary) === keys(a.dictionary)
    @test result.dictionary.values !== a.dictionary.values
    @test result.dictionary.values !== b.dictionary.values
    boolean = a .> 2.0
    @test isequal(collect(boolean[x]), [false, false, nothing, true])
    @test eltype(boolean.dictionary.values) == Union{Nothing,Bool}

    destination = ModelDictionary(m)
    fused_dictionary_assign!(destination, a, b)
    @test isequal(collect(destination[x]), collect(result[x]))
    a .= a .+ 10.0
    @test isequal(collect(a[x]), [11.0, 12.0, nothing, 14.0])
    a .= 9.0
    @test collect(a[x]) == fill(9.0, 4)
    a .= nothing
    @test all(isnothing, a[x])
end

@testset "Dictionary broadcasting rejects misaligned names before writing" begin
    m = Model()
    @variable(m, x[1:3])
    a = ModelDictionary(m)
    a[x] .= [1.0, 2.0, 3.0]
    names = reverse(name.(x))
    reordered = ModelDictionary(m, Dictionary(names, [30.0, 20.0, 10.0]))
    @test_throws DimensionMismatch a .+ reordered
    @test_throws DimensionMismatch a .+ 2.0 .* reordered
    @test_throws DimensionMismatch a .= 2.0 .* a .- reordered
    @test collect(a[x]) == [1.0, 2.0, 3.0]

    other_model = Model()
    @variable(other_model, y[1:3])
    other = ModelDictionary(other_model)
    @test_throws DimensionMismatch a .+ other

    # Explicit matching ordered names support data imported from another model.
    matching_model = Model()
    @variable(matching_model, x[1:3])
    matching = ModelDictionary(matching_model)
    matching[x] .= [10.0, 20.0, 30.0]
    @test collect((a .+ matching)[name.(x)]) == [11.0, 22.0, 33.0]
end

@testset "In-place dictionary broadcast avoids copying its input buffers" begin
    n = 20_000
    m = Model()
    @variable(m, x[1:n])
    a = ModelDictionary(m)
    b = ModelDictionary(m)
    destination = ModelDictionary(m)
    a[x] .= collect(1.0:n)
    b[x] .= 2.0
    fused_dictionary_assign!(destination, a, b)
    bytes = @allocated fused_dictionary_assign!(destination, a, b)
    @test destination[x[1]] == 3.0
    @test destination[x[end]] == 2n + 1
    # Even one copied Float64 input would need 160 kB.
    @test bytes < n * sizeof(Float64) ÷ 4
end

end
