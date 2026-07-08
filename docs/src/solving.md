# Solving

SquareModels solves a [`Block`](@ref) by building an intermediate JuMP model with
the block's endogenous variables. Exogenous variables are substituted from a
[`ModelDictionary`](@ref), the intermediate model is optimized, and the solution
is written back to a dictionary.

Use [`solve`](@ref) when you want a new dictionary:

```@example solving
import JuMP
using JuMP: set_silent
using Ipopt
using SquareModels

model = square_model(Ipopt.Optimizer)
set_silent(model)

@variables model begin
    x, "First endogenous variable"
    y, "Second endogenous variable"
    a, "Exogenous shift"
end

data = ModelDictionary(model)
data[x] = 1.0
data[y] = 1.0
data[a] = 3.0

block = @block data begin
    x, x == 2
    y, y == a + x
end

solution = solve(block, data)
round(solution[y], digits=4)
```

Use [`solve!`](@ref) when you want to update the existing dictionary in place:

```@example solving
data[a] = 8.0
solve!(block, data)
round(data[y], digits=4)
```

## Calibration

Calibration is the same solve pipeline with a different endogenous set. Start
from a copy of the behavioral block and use [`@endo_exo_swap!`](@ref):

```@example solving
calibration = copy(block)
@endo_exo_swap! calibration begin
    a, y
end

data[y] = 20.0
calibrated = solve(calibration, data)
round(calibrated[a], digits=4)
```

## Starting Values

`solve` and `solve!` take optional `start_values` and `replace_nothing` keywords:

```julia
solution = solve(block, data; start_values=baseline, replace_nothing=1.0)
```

`replace_nothing` is useful during early calibration when a variable exists in
the model but has no data value yet.

## Diagnostics

Before solving, SquareModels checks whether substituting exogenous data makes the
system effectively non-square. For example, equations that collapse to constants
or endogenous variables that disappear from every equation are reported before
the optimizer runs.

Call [`diagnose`](@ref) directly when you want to inspect those structural issues:

```julia
trivial, orphans = diagnose(block, data)
```

Set `skip_diagnostics=true` only when you have already validated the block and
need to avoid the extra pre-solve pass.
