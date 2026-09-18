module TestParquetIO

using Test
using JuMP
using SquareModels
using DataFrames
using Parquet2

@testset "Parquet readers release input files" begin
	model = Model()
	@variable(model, x[1:2])
	data = DataFrame(variable=["x", "x"], indices=["1", "2"], value=[1.5, 3.0])
	readers = (
		("load", path -> load(path, model)[x[1]], 1.5),
		("read_indices", read_indices, [1, 2]),
		("read_sparse_array", path -> read_sparse_array(path)[1], 1.5),
		("read_variable", path -> read_variable(path, x)[1], 1.5),
	)
	for (label, reader, expected) in readers
		@testset "$label" begin
			mktempdir() do dir
				path = joinpath(dir, "data.parquet")
				Parquet2.writefile(path, data)
				# An incidental collection must not hide a leaked memory mapping.
				gc_enabled = GC.enable(false)
				try
					result = reader(path)
					@test isnothing(rm(path))
					@test !isfile(path)
					@test result == expected
				finally
					GC.enable(gc_enabled)
					GC.gc() # Release mappings after a failed regression test, for cleanup.
				end
			end
		end
	end
end

@testset "Parquet readers release files after invalid input" begin
	model = Model()
	for (label, reader) in (("load", path -> load(path, model)), ("read_indices", read_indices))
		@testset "$label" begin
			mktempdir() do dir
				path = joinpath(dir, "invalid.parquet")
				Parquet2.writefile(path, DataFrame(unexpected=[1.0]))
				gc_enabled = GC.enable(false)
				try
					@test_throws ErrorException reader(path)
					@test isnothing(rm(path))
					@test !isfile(path)
				finally
					GC.enable(gc_enabled)
					GC.gc()
				end
			end
		end
	end
end

end
