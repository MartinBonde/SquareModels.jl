# Core Concepts

## Blocks

A [`Block`](@ref) stores a square mapping from endogenous variables to equations.
The mapping is explicit: each line in `@block` starts with the variable that the
equation determines.

```julia
block = @block data begin
    x,          x == a + b
    y[i ∈ I],   y[i] == z[i] + 1
end
```

Blocks can be combined with `+` as long as they belong to the same JuMP model and
do not determine the same endogenous variable twice:

```julia
full_model = households + production + government
```

Use [`endogenous`](@ref), [`exogenous`](@ref), [`variables`](@ref), and
[`residuals`](@ref) to inspect the block.

## Endo-Exo Swapping

Calibration often means solving for parameters that are normally exogenous while
holding observed endogenous variables fixed. [`@endo_exo_swap!`](@ref) changes
which variables are endogenous in an existing block:

```julia
calibration = copy(model_block)
@endo_exo_swap! calibration begin
    μ, Y
    δ, K[t₀]
end
```

The left-hand variable becomes endogenous. The right-hand variable must already
be endogenous in the block and becomes exogenous data.

## Model Dictionaries

[`ModelDictionary`](@ref) maps JuMP variables to values and supports scalar,
container, and slice indexing:

```julia
data[x] = 1.0
data[y] = [1.0, 2.0, 3.0]
data[y[2020:2030]] .= 1.0
```

Indexing a variable container returns a `Window`, which behaves like a view into
the dictionary and keeps the original model indices. That is what makes slices
usable for printing and plotting.

## Variable Metadata

SquareModels extends JuMP's `@variables` syntax with descriptions and tags:

```julia
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)

@variables data.model :: GrowthAdjusted begin
    qGDP[t], "Real GDP"
    vGDP[t] :: InflationAdjusted, "Nominal GDP"
end
```

Use [`description`](@ref), [`tags`](@ref), [`has_tag`](@ref), and [`tagged`](@ref)
to query the metadata.

## SparseZeroArray

Conditional JuMP containers are wrapped in [`SparseZeroArray`](@ref) by default.
Missing combinations inside the declared domain return `Zero()`, which behaves
like an additive zero in JuMP expressions:

```julia
@variables data.model begin
    x[i=1:5, j=1:5; i <= j], "Upper triangular variable"
end

x[1, 2]  # VariableRef
x[3, 1]  # Zero()
```

This lets you write sums over sparse domains without filtering every access:

```julia
total = ∑(x[i, j] for i in 1:5, j in 1:5)
```

Disable the wrapper with [`use_sparse_zero_array!`](@ref) if you prefer standard
JuMP `SparseAxisArray` behavior.
