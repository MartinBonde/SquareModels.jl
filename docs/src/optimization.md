# Optimization

`@block` stores structural equations as SquareModels equations rather than
solving immediately. You can still combine those equations with ordinary JuMP
variables, objectives, and constraints when you need an optimization problem
instead of a square system.

A common use case is minimum-distance estimation:

1. Use a square model to describe the structural relationships.
2. Treat one or more parameters as JuMP decision variables.
3. Add an objective that penalizes distance between model predictions and data.
4. Optimize with the usual JuMP workflow.

The full example is available in `examples/optimization_example.jl`.

```julia
import JuMP
using JuMP: @objective, @variable, optimize!
using Ipopt
using SquareModels

est_model = JuMP.Model(Ipopt.Optimizer)

est_L = @variable(est_model, [j], lower_bound=0.01)
est_w = @variable(est_model, [j], lower_bound=0.01)
est_Y = @variable(est_model, lower_bound=0.01)
est_p = @variable(est_model, lower_bound=0.01)
est_σ = @variable(est_model, lower_bound=0.1, upper_bound=10.0)

for jj in j
    JuMP.@constraint(est_model, est_L[jj] == μ[jj] * (est_w[jj] / est_p)^(-est_σ) * est_Y)
    JuMP.@constraint(est_model, est_L[jj] == ρ[jj] * N[jj])
end

JuMP.@constraint(est_model, est_p * est_Y == sum(est_w[jj] * est_L[jj] for jj in j))
JuMP.@constraint(est_model, est_p == 1)

@objective(est_model, Min, sum((est_w[jj] - w_observed[jj])^2 for jj in j))
optimize!(est_model)
```

Use this pattern when the system is intentionally non-square. Use [`solve`](@ref)
and [`solve!`](@ref) when the system should remain square and you want
SquareModels to substitute exogenous data automatically.
