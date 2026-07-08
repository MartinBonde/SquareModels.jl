# Examples

The repository includes runnable examples under `examples/`.

## Quick Example

`examples/quick_example.jl` is the smallest complete workflow:

- define a labor-market block,
- calibrate productivity and scale parameters,
- solve a counterfactual population shock,
- compare scenario values with the baseline.

This is the source for the [Getting Started](@ref) walkthrough.

## Modular Example

`examples/modular_example.jl` demonstrates a two-sector dynamic model organized
with Julia modules. It shows how to define variables near the equations that own
them, assemble blocks from submodels, and change the terminal period before a
scenario solve.

## Optimization Example

`examples/optimization_example.jl` shows how to combine SquareModels-style
structural equations with a JuMP objective. The example estimates a substitution
elasticity by minimizing the distance between model wages and observed wages.

## Running Examples

From the package root:

```julia
julia --project=. examples/quick_example.jl
julia --project=. examples/modular_example.jl
julia --project=. examples/optimization_example.jl
```

The examples use Ipopt, so make sure it is available in your active environment.
