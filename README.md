# SquareModels

A JuMP extension for writing modular models with **square systems of equations** — systems where the number of constraints equals the number of endogenous variables.

## Motivation
Large-scale macroeconomic models are typically "square" — each equation determines one endogenous variable. This package provides tools to:

- **Map constraints to endogenous variables** — Each constraint is explicitly paired with the variable it determines
- **Build models modularly** — Define separate `Block`s of equations that can be combined
- **Swap endogenous/exogenous variables** — Use `@endo_exo!` to change which variables are endogenous for calibration or counterfactual scenarios

## Quick Example

> 📄 Full runnable version: [`examples/quick_example.jl`](examples/quick_example.jl)

```julia
import JuMP
using JuMP: Model, set_silent
using Ipopt, SquareModels

data = ModelDictionary(Model(Ipopt.Optimizer))
set_silent(data.model)

j = 1:2  # Types of labor

@variables data.model begin
    L[j], "Labor demand"
    w[j], "Wage"
    Y, "Output"
    C, "Consumption"
    p, "Price"

    N[j], "Labor force (exogenous)"
    ρ[j], "Productivity (calibrated)"
    μ[j], "Scale parameter (calibrated)"
    σ, "Substitution elasticity (exogenous)"
end

# Define a Block: each line pairs an endogenous variable with its equation
model_block = @block data begin
    L[j ∈ j], L[j] == μ[j] * (w[j] / p)^-σ * Y   # Labor demand
    w[j ∈ j], L[j] == ρ[j] * N[j]                 # Labor market clearing
    Y,        p * Y == ∑(w[j] * L[j] for j ∈ j)   # Zero profit
    C,        C == ∑(w[j] * ρ[j] * N[j] for j ∈ j) / p  # Budget constraint
    p,        p == 1                               # Numeraire
end

# Set data values
data[σ] = 2.0
data[w] .= 1
data[N] = [3200, 500]
data[L] = [800, 200]

# Calibration: swap observed values with parameters to be calibrated
calibration = copy(model_block)
@endo_exo! calibration begin
    μ, L
    ρ, w
end

baseline = solve(calibration, data; replace_nothing=1.0)

# Counterfactual: shock exogenous variables
scenario = copy(baseline)
scenario[N] .= [2700, 1000]  # Population shock
solve!(model_block, scenario)

println("Multipliers: ", scenario ./ baseline .- 1)
```

## Modular Models

For larger models, organize code into Julia modules with explicit cross-module imports:

> 📄 Full runnable version: [`examples/modular_example.jl`](examples/modular_example.jl)

```julia
import JuMP
using JuMP: Model, set_silent
using Ipopt, SquareModels

data = ModelDictionary(Model(Ipopt.Optimizer))

module Production
    import JuMP
    using SquareModels
    import ..data

    const s = [:agri, :manuf]  # Sectors defined here

    @variables data.model begin
        Y[s], "Output by sector"
        p[s], "Price by sector"
        A[s], "Productivity (calibrated)"
    end

    function define_equations()
        @block data begin
            Y[s = s], Y[s] == A[s] * K[s]
            p[s = [:agri]], p[s] == 1  # Numeraire
        end
    end

    function define_calibration()
        block = define_equations()
        @endo_exo! block begin
            A, Y  # Calibrate productivity to match output
        end
        block
    end
end

module HouseHolds
    import JuMP
    using SquareModels
    import ..data

    s = Main.Production.s  # Import sectors from Production
    p = Main.Production.p  # Import prices from Production

    @variables data.model begin
        C[s], "Consumption by sector"
        α[s], "Consumption shares (calibrated)"
    end

    function define_equations()
        @block data begin
            C[s = s], p[s] * C[s] == α[s] * I
        end
    end
end

# Assemble and solve
submodels = [Production, HouseHolds]
base = sum(m.define_equations() for m in submodels)
calibration = sum(m.define_calibration() for m in submodels)
baseline = solve(calibration, data; replace_nothing=1.0)
```

## Key Concepts

### Blocks

A `Block` is a collection of constraints paired with their endogenous variables:

```julia
block = @block data begin
  x,           x == a + b
  y[i ∈ 1:3],  y[i] == i * z
end
```

When using `@block data` (with a `ModelDictionary`), residual values are automatically initialized to zero.

Blocks can be combined with `+`:
```julia
full_model = consumers + production + government
```

### Endo-Exo Swapping

During calibration, you often want to treat normally-endogenous variables as exogenous (data) and solve for parameters instead. The `@endo_exo!` macro swaps variable roles within a block:

```julia
calibration_block = copy(base_block)
@endo_exo! calibration_block begin
  μ,  Y      # Solve for μ given Y (instead of Y given μ)
  δ,  K[t₁]  # Solve for δ given initial capital
end
```

### Solving

Use `solve` to solve a block and return a new `ModelDictionary` with the solution:

```julia
solution = solve(block, data; replace_nothing=1.0)
```

The `replace_nothing` parameter replaces any `nothing` start values with the specified number.

Use `solve!` to update the data in-place:

```julia
solve!(block, data)
```

### Variable Tags and Descriptions

Variables can have descriptions and tags for documentation and programmatic grouping.
Tags use Julia's `::` syntax, following the Holy trait pattern:

```julia
# Define tags as trait markers
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)

# Block-level tags apply to ALL variables in the block
@variables data.model :: GrowthAdjusted begin
    qGDP[t], "Real GDP"
    qC[t], "Real consumption"
end

# Variable-level tags for individual variables
@variables data.model begin
    pGDP[t] :: InflationAdjusted, "GDP deflator"
    σ, "Substitution elasticity"
end

# Combine block-level and variable-level tags (they accumulate)
@variables data.model :: GrowthAdjusted begin
    vGDP[t] :: InflationAdjusted, "Nominal GDP"  # Has both tags
end

# Query metadata
description(:vGDP)              # "Nominal GDP"
tags(:vGDP)                     # Set([GrowthAdjusted, InflationAdjusted])
has_tag(:vGDP, GrowthAdjusted)  # true
tagged(GrowthAdjusted)          # [:qGDP, :qC, :vGDP]
```

Note: SquareModels provides its own `@variables` macro that extends JuMP's version with tags and descriptions. If you need JuMP's original macro, use `JuMP.@variables`.

## Project Structure

```
SquareModels/
├── src/
│   ├── SquareModels.jl       # Main module: Block, @block, @endo_exo!
│   ├── endo_exo.jl           # Endo-exo swap implementation
│   ├── ModelDictionaries.jl  # Variable-value mappings
│   ├── solve.jl              # Solve functions
│   ├── tagged_variables.jl   # @variables macro with tags/descriptions
│   └── utils.jl              # Helper functions
├── examples/
│   ├── quick_example.jl      # Simple labor market model
│   └── modular_example.jl    # Modular CGE model example
└── test/
    └── runtests.jl
```

## License
This project is licensed under an MIT license — see [LICENSE](LICENSE) for details.

## Acknowledgments
This work is part of the [DREAM](https://dreamgruppen.dk/) group's effort to modernize economic modeling tools in Denmark and the rest of the world.
