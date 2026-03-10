# Optimization Example - Minimum Distance Estimation
#
# This example shows how to combine SquareModels equations with JuMP
# optimization to estimate model parameters.
#
# A SquareModels `@block` defines structural equations as JuMP constraints
# on `data.model`. By treating a parameter as a free variable and adding
# a JuMP objective, the system becomes non-square — an optimization problem
# rather than a system of equations.
#
# Use case: we observe wages across labor types and want to estimate the
# substitution elasticity σ that best explains the data.

import JuMP
using JuMP: Model, set_silent, @objective, optimize!, set_lower_bound, set_upper_bound
using Ipopt, SquareModels

# ==============================================================================
# Model setup
# ==============================================================================
data = ModelDictionary(Model(Ipopt.Optimizer))
set_silent(data.model)

j = 1:3

@variables data.model begin
	L[j], "Labor demand"
	w[j], "Wage"
	Y, "Output"
	p, "Price"

	N[j], "Labor supply (exogenous)"
	μ[j], "CES share parameter (exogenous)"
	ρ[j], "Productivity (exogenous)"
	σ, "Elasticity of substitution (to estimate)"
end

# Structural equations — a square system with 7 equations and 7 endogenous variables
model_eq = @block data begin
	L[j ∈ j], L[j] == μ[j] * (w[j] / p)^(-σ) * Y   # CES labor demand
	w[j ∈ j], L[j] == ρ[j] * N[j]                    # Labor market clearing
	Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)      # Zero profit
	p,        p == 1                                   # Numeraire
end

# ==============================================================================
# Exogenous data
# ==============================================================================
data[N] = [3200, 500, 800]
data[μ] = [0.55, 0.25, 0.20]
data[ρ] = [0.25, 0.40, 0.30]

# Starting values for endogenous variables
data[L] = [800, 200, 240]
data[w] .= 1.0
data[Y] = 1240
data[p] = 1.0

# ==============================================================================
# Generate observed data
# ==============================================================================
# Solve the square system with the "true" σ to create synthetic observations
data[σ] = 1.8
truth = solve(model_eq, data; replace_nothing=1.0)

w_observed = [truth[w[jj]] for jj in j]
println("Observed wages: ", round.(w_observed, digits=4))

# ==============================================================================
# Estimation: find σ by minimizing distance to observed wages
# ==============================================================================
# Solve the model at the starting σ to get consistent initial values
data[σ] = 3.0
initial = solve(model_eq, data; replace_nothing=1.0)

# Fix ALL variables to their values, then selectively unfix
fix(initial)

# Unfix the endogenous variables — they are determined by the structural equations
unfix(model_eq)

# Also unfix σ — this creates one extra degree of freedom.
# The system now has 7 constraints and 8 free variables (7 endogenous + σ),
# so we need an objective to make the problem well-defined.
unfix(σ)
set_lower_bound(σ, 0.1)
set_upper_bound(σ, 10.0)

@objective(data.model, Min, sum((w[jj] - w_observed[jj])^2 for jj in j))

set_start_value(initial)
optimize!(data.model)

# ==============================================================================
# Results
# ==============================================================================
println("\nEstimated σ = ", round(value(σ), digits=4), "  (true value: 1.8)")
println("\nWage comparison:")
for jj in j
	println("  Sector $jj: model = ", round(value(w[jj]), digits=4),
	        ",  observed = ", round(w_observed[jj], digits=4))
end

# Recover solution as a ModelDictionary for use with SquareModels workflows
estimated = value_dict(data.model)
println("\nEstimated output Y = ", round(estimated[Y], digits=4),
        "  (true: ", round(truth[Y], digits=4), ")")
