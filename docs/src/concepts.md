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

Indexing a variable container returns a `Window`, which behaves like a view into
the dictionary and keeps the original model indices. That is what makes slices
usable for printing and plotting. A `Window` supports broadcasting (`.=`, `.*`,
etc.) and iteration, but external libraries may require `collect` or
`Float64.()` to convert to a plain `Vector`. At the REPL, a multi-dimensional
`Window` displays as a table (rows for the leading indices, columns for the last
dimension) via PrettyTables.jl.

## Loading and Saving Data

[`unload`](@ref) saves a `ModelDictionary` to Parquet in a simple tabular format
(`variable`, `indices`, `value`). Entries with `nothing` are omitted. [`load`](@ref)
reads Parquet, CSV, or — with `using GDXInterface` — GDX files and matches rows to
the model's variables by base name and indices.

```julia
unload("solution.parquet", baseline)
baseline = load("solution.parquet", model)
```

**Index matching:** only indices that exist in both the file and the model are loaded.
Extra data indices are ignored; model indices missing from the file stay `nothing`.

**Renames** map model variable base names to differently named data symbols (like GAMS
`$LOAD`). Pass keyword arguments or `Pair`s:

```julia
d = load("data.parquet", model; N_a = "nPop", Y => "OtherY")
```

**Slices** extract a lower-dimensional symbol from higher-dimensional data. Use `:`
for positions filled from the model variable's indices:

```julia
d = load("data.gdx", model;
    N = "nPop",              # simple rename
    C = "vC[:cTot,:]",        # C[t] ← vC[:cTot, t]
    K = "vK[:iTot,:tot,:]",   # K[t] ← vK[:iTot, :tot, t]
)
```

For reading a single variable without building a full dictionary, use
[`read_variable`](@ref), [`read_sparse_array`](@ref), or [`read_indices`](@ref) on
simple-format CSV/Parquet files.

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
