# Check console printing and table export for model expressions.
# Keep evaluation, labels, and sparse gaps consistent across both outputs.
module TestModelExpressions

using Test, Dates
using JuMP, SquareModels
using Tables, CSV, DataFrames

function capture_stdout(f)
	return mktemp() do path, io
		result = redirect_stdout(f, io)
		seekstart(io)
		return read(io, String), result
	end
end

@testset "Console printing" begin
	model = Model()
	SquareModels.@variables model begin
		x[2020:2022]
		y[2020:2022]
		k[s=[:a, :b], t=2020:2022; true]
	end
	baseline = ModelDictionary(model)
	baseline[x] = [10, 20, 30]
	baseline[k] .= 2.0
	scenario = copy(baseline)
	scenario[x] = [11, 22, 33]
	set_default_source!(baseline)
	set_default_periods!(2021:2022)
	set_default_operator!(:n)
	try
		stock = reduce(.+, (k[s,:] for s in [:a, :b]))
		@test collect(@evalexpr(stock)) == [4, 4]
		@test (@evalexpr (stock, x)).names == ["stock", "x"]
		baseline[k[:a,2021]] = 3.0
		@test collect(@evalexpr(stock)) == [5, 4]
		text, result = capture_stdout(() -> @prt x)
		@test result === nothing
		@test occursin("2021", text) && occursin("2022", text)
		@test !occursin("2020", text)
		@test count("year", text) == 1
		@test text == sprint(show, MIME"text/plain"(), @evalexpr(x)) * "\n"
		quiet, evaluated = capture_stdout(() -> @evalexpr (x, y))
		@test isempty(quiet)
		@test evaluated isa MultiVarResult
		printed, result = capture_stdout(() -> @prt $evaluated)
		@test result === nothing
		@test occursin("nothing", printed)
		@test count("year", printed) == 1
		@test all(isnothing, evaluated[2])
		unset_array = @evalexpr y
		printed, result = capture_stdout(() -> @prt $unset_array)
		@test result === nothing
		@test occursin("nothing", printed)
		scalar, result = capture_stdout(() -> @prt x[2020])
		@test result === nothing
		@test scalar == "10.0\n"
		unset, result = capture_stdout(() -> @prt y[2020])
		@test result === nothing
		@test unset == "nothing\n"
		comparison, result = capture_stdout(() -> @prt :q baseline=>scenario x)
		@test result === nothing
		@test comparison == sprint(show, MIME"text/plain"(), @evalexpr(:q, baseline=>scenario, x)) * "\n"
		group, result = capture_stdout(() -> @prt :n (baseline, scenario) x)
		@test result === nothing
		@test occursin("baseline", group) && occursin("scenario", group)
	finally
		reset_print_defaults!()
	end
end

@testset "Table export" begin
	a = LabeledArray([10.0 20.0; 30.0 40.0], ([:a, :b], 2020:2021), "stock")
	columns = Tables.columntable(a)
	@test propertynames(columns) == (:year, Symbol("stock[a]"), Symbol("stock[b]"))
	@test columns.year == [2020, 2021]
	@test columns[2] == [10, 20]
	@test columns[3] == [30, 40]
	@test DataFrame(a).year == [2020, 2021]
	@test Tables.columntable(LabeledArray([1.0], ([2020],))).value == [1]
	@test isempty(Tables.columntable(LabeledArray(Float64[], (Int[],), "empty")).empty)
	@test Tables.columntable(LabeledArray([1.0], ([Date(2020)],), "x")).year == [Date(2020)]

	large_integer = 9_007_199_254_740_993
	mixed = MultiVarResult(["integer", "float"],
		(LabeledArray([large_integer], ([2020],)), LabeledArray([1.5], ([2020],))))
	@test Tables.columntable(mixed).integer[1] === large_integer
	@test DataFrame(mixed).integer[1] === large_integer
	@test occursin(string(large_integer), sprint(show, MIME"text/plain"(), mixed))
	mktemp() do path, io
		CSV.write(io, mixed)
		seekstart(io)
		@test CSV.read(io, DataFrame).integer[1] === large_integer
	end

	OrderedDict = JuMP.Containers.OrderedCollections.OrderedDict
	sparse = JuMP.Containers.SparseAxisArray(OrderedDict([(:a, 2022) => 3.0, (:b, 2020) => 4.0]))
	dense = LabeledArray([2.0, nothing], ([2021, 2020],), "dense")
	for data in (sparse, SparseZeroArray(sparse, (Set([:a, :b]), Set(2020:2022))))
		result = MultiVarResult(["dense", "sparse"], (dense, LabeledArray(data, nothing)))
		columns = Tables.columntable(result)
		@test columns.year == [2020, 2021, 2022]
		@test isequal(columns.dense, [missing, 2, missing])
		@test isequal(columns[3], [missing, missing, 3])
		@test isequal(columns[4], [4, missing, missing])
		@test dense.data[2] === nothing
		@test length(data) == 2
		text = sprint(show, MIME"text/plain"(), result)
		@test !occursin("missing", text)
		@test occursin("nothing", text)
		mktemp() do path, io
			CSV.write(io, result)
			seekstart(io)
			frame = CSV.read(io, DataFrame)
			@test isequal(frame, DataFrame(result))
		end
	end

	scalars = MultiVarResult(["a,\"b\"\nc", "unset"], (2.5, nothing))
	@test isequal(Tables.columntable(scalars).unset, [missing])
	@test size(DataFrame(scalars)) == (1, 2)
	mktemp() do path, io
		CSV.write(io, scalars)
		seekstart(io)
		@test isequal(CSV.read(io, DataFrame), DataFrame(scalars))
	end
	@test Tables.columntable(LabeledArray([""], ([2020],), "text")).text == [""]
	@test_throws ArgumentError Tables.columns(MultiVarResult(["x", "x"], (1, 2)))
	@test_throws ArgumentError Tables.columns(LabeledArray([1], ([2020],), "year"))
	@test_throws ArgumentError Tables.columns(MultiVarResult(["x", "y"], (1, dense)))
	@test_throws ArgumentError Tables.columns(MultiVarResult(["group"], (scalars,)))
end

@testset "Display limits" begin
	wide = MultiVarResult(["column$i" for i in 1:20], Tuple(LabeledArray(1:30, (2000:2029,)) for i in 1:20))
	unlimited = sprint(show, MIME"text/plain"(), wide; context=(:limit => false, :displaysize => (8, 40)))
	limited = sprint(show, MIME"text/plain"(), wide; context=(:limit => true, :displaysize => (8, 40)))
	@test occursin("column20", unlimited)
	@test occursin("2029", unlimited)
	@test !occursin("omitted", unlimited)
	@test occursin("omitted", limited)
end

end # module
