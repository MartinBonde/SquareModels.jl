# SquareModels

A JuMP extension for writing modular models with **square systems of equations** — systems where the number of constraints equals the number of endogenous variables.

## Motivation
Large-scale macroeconomic models are typically "square" — each equation determines one endogenous variable. This package provides tools to:

- **Map constraints to endogenous variables** — Each constraint is explicitly paired with the variable it determines
- **Build models modularly** — Define separate `Block`s of equations that can be combined
- **Swap endogenous/exogenous variables** — Use `@endo_exo_swap!` to change which variables are endogenous for calibration or counterfactual scenarios

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
@endo_exo_swap! calibration begin
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
        @endo_exo_swap! block begin
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

## Optimization

Since `@block` adds constraints directly to the JuMP model, you can use standard JuMP features — objectives, extra variables, additional constraints — to build non-square optimization problems on top of a square model. A typical use case is minimum distance estimation: treat a parameter as a free variable and minimize the distance between model predictions and observed data.

> 📄 Full example: [`examples/optimization_example.jl`](examples/optimization_example.jl)

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

During calibration, you often want to treat normally-endogenous variables as exogenous (data) and solve for parameters instead. The `@endo_exo_swap!` macro swaps variable roles within a block:

```julia
calibration_block = copy(base_block)
@endo_exo_swap! calibration_block begin
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

### ModelDictionary Indexing

`ModelDictionary` supports indexing with JuMP variable containers, including slices with vectors or ranges. This eliminates the need for explicit loops over time periods or sets:

```julia
# Single value
data[x[2025]]          # scalar

# Vector of variable references — returns a Window (a view into the dictionary)
data[x[2025:2060]]     # all periods
data[x[[2025, 2030]]]  # selected periods

# Multi-dimensional variables
data[y[:electric, 2025:2060]]  # one fuel type, all periods
data[y[:, 2025]]               # all fuel types, one period

# Assignment works the same way
data[x[2025:2060]] .= 1.0
data[y[:electric, 2025:2060]] .= 0.8
```

The returned `Window` supports broadcasting (`.=`, `.*`, etc.) and iteration, but external libraries (e.g. Makie) may require `collect` or `Float64.()` to convert to a plain `Vector`. The `ModelPlotting` submodule (below) handles this conversion for you.

### Plotting

Plotting is delegated to [Makie](https://docs.makie.org): the package extension teaches Makie how to turn an `AbstractSeries` (a `Window` or a `LabeledSeries`) into x/y data via `Makie.convert_arguments`, so all the usual Makie verbs accept model data directly. It handles the `Window` → numeric conversion, axis labels (e.g. years parsed from the variable index), and legend labels (the variable name). A multi-dimensional variable fans out into one line per leading-index combination — the last dimension is the x-axis, so `y[region, year]` plots one line per region over the years.

The extension loads once any Makie backend is present, so `Makie`/`CairoMakie` is an optional dependency — add a backend to your own project and load it before plotting:

```julia
using CairoMakie               # loads any Makie backend → loads the extension (headless)
using SquareModels

# Any Makie verb works on a slice — convert_arguments does the rest:
lines(data.qGDP)
scatter!(data.qC[2020:2060])

# Batteries-included single figure (title and legend default to the variable name):
fig = plotvar(data, vGDP[2020:2060]; ylabel="Million EUR")
save("gdp.png", fig)           # Makie's save; format inferred from extension

# Plot expressions of variables; each series is labelled with its source text.
# Bare names resolve against `data`; arithmetic is applied elementwise.
@plot data qGDP / qGDP[2019]
@plot data [qGDP * pGDP, qGDP / qGDP[2019]]

# Multi-dimensional variables fan out into one line per leading index:
@plot data emissions          # emissions[region, year] → one line per region

# Programmatic construction without the macro — build LabeledSeries explicitly:
plotseries([labeled(data[v] .* data.pGDP, "$v * pGDP") for v in (:qGDP, :qC)])
```

`CairoMakie` renders headlessly (no display required), so this works for generating report figures in scripts or CI.

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

### SparseZeroArray

When `@variables` creates a `SparseAxisArray` (i.e. a variable with conditional indices), it is automatically wrapped in a `SparseZeroArray`. Missing index combinations return a no-op `Zero()` sentinel instead of throwing a `KeyError`:

```julia
@variables data.model begin
    x[i=1:5, j=1:5; i <= j], "Upper triangular variable"
end

x[1, 2]  # VariableRef (exists)
x[3, 1]  # Zero() (missing but within domain — safe to use in sums)
x[6, 1]  # Error (out of domain — catches typos)
```

This eliminates the need for filter clauses when summing over sparse dimensions:

```julia
# Without SparseZeroArray — requires filtering
total = sum(x[i, j] for i in 1:5, j in 1:5 if haskey(x, (i, j)))

# With SparseZeroArray — just sum directly
total = ∑(x[i, j] for i in 1:5, j in 1:5)
```

The `∑` function is a convenience wrapper around `sum` that uses `Zero()` as the initial value, so summing over empty or all-missing dimensions works without errors.

To access the underlying `SparseAxisArray`, use `x.data`.

#### Disabling SparseZeroArray

If you prefer standard JuMP `SparseAxisArray` behavior, disable the auto-wrapping:

```julia
use_sparse_zero_array!(false)  # @variables will create plain SparseAxisArrays
use_sparse_zero_array!(true)   # Re-enable (default)
```

## Project Structure

```
SquareModels/
├── src/
│   ├── SquareModels.jl       # Main module: Block, @block, @endo_exo_swap!
│   ├── SparseZeroArrays.jl   # Domain-aware sparse arrays with zero default
│   ├── endo_exo_swap.jl      # Endo-exo swap implementation
│   ├── ModelDictionaries.jl  # Variable-value mappings
│   ├── solve.jl              # Solve functions
│   ├── tagged_variables.jl   # @variables macro with tags/descriptions
│   ├── ModelPlotting.jl      # Plotting interface (functions defined by extension)
│   └── utils.jl              # Helper functions
├── ext/
│   ├── SquareModelsMakieExt.jl      # Plotting via Makie (optional dep)
│   └── SquareModelsGAMSExt.jl       # GAMS/GDX support (optional dep)
├── examples/
│   ├── quick_example.jl      # Simple labor market model
│   ├── modular_example.jl    # Modular CGE model example
│   └── optimization_example.jl # Minimum distance estimation
└── test/
    └── runtests.jl
```

## License
This project is licensed under an MIT license — see [LICENSE](LICENSE) for details.

## Acknowledgments
This work is part of the [DREAM](https://dreamgruppen.dk/) group's effort to modernize economic modeling tools in Denmark and the rest of the world.
