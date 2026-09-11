module TestErrors

using Test, JuMP, SquareModels

@testset "Model error tables" begin
	model = Model()
	JuMP.@variables model begin
		x
		x_J
	end
	data = ModelDictionary(model, 0.0)
	data[x_J] = 2.0
	reference = copy(data)
	reference[x] = 3.0
	block = @block model begin
		@test_constraint("balance check")
		x, x == 1
	end

	# Exercise the errors thrown by the public assertions, as well as the
	# structural error's three diagnostic tables.
	cases = [
		(() -> assert_residuals_small(data), ResidualError, ["residual", "x_J", "|value|", "tolerance"]),
		(() -> assert_no_diff(data, reference), ToleranceError, ["variable", "abs diff", "rel diff", "reference"]),
		(() -> assert_test_constraints(block, data), TestConstraintError, ["distance", "tolerance", "balance check"]),
		(() -> throw(NonSquareError("Duplicate equations", [x => only(test_constraints(block)).equation])), NonSquareError, ["endogenous variable", "equation expression"]),
		(() -> throw(NonSquareError("Not square"; trivial=[("x", 2.0)], orphans=["x"])), NonSquareError, ["constant", "status", "infeasible", "orphan"]),
	]
	for (run, error_type, labels) in cases
		@testset "$error_type: $(first(labels))" begin
			err = try
				run()
			catch e
				e
			end
			@test err isa error_type
			for context in ((), (:limit => true, :displaysize => (24, 120)))
				output = sprint(showerror, err; context)
				@test occursin('┌', output)
				@test all(label -> occursin(label, output), labels)
				@test sprint(show, MIME"text/plain"(), err; context) == output
			end
			# Compact representations embedded in logs must remain on one line.
			@test !occursin('\n', sprint(show, err))
		end
	end
end

end # module
