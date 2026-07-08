# SquareModels.jl

SquareModels is a JuMP extension for building modular models with square systems
of equations: systems where each equation determines one endogenous variable.

It is designed for large economic models where you want to:

- map each equation explicitly to the variable it determines,
- build a model from independently maintained blocks,
- calibrate by swapping endogenous and exogenous variables,
- solve scenarios into reusable data dictionaries,
- inspect, print, and plot model results.

## Installation

Add SquareModels to your Julia environment and load it together with JuMP and a
solver:

```julia
using JuMP
using Ipopt
using SquareModels
```

Plotting is optional and loads through Julia's package extension system. Add and
load a Makie backend when you need figures:

```julia
using CairoMakie
using SquareModels
```

## Documentation Map

- [Getting Started](@ref) introduces a compact calibration and scenario workflow.
- [Core Concepts](@ref) explains blocks, model dictionaries, metadata, and sparse arrays.
- [Solving](@ref) covers `solve`, `solve!`, calibration, and diagnostics.
- [Plotting and Printing](@ref) covers `plotvar`, `@plot`, `@prt`, and expression evaluation.
- [Modular Models](@ref) shows how to organize larger models into Julia modules.
- [Optimization](@ref) explains how square systems can be embedded in JuMP optimization problems.
- [API Reference](@ref) lists exported docstrings.

## Package Structure

The main package lives in `src/SquareModels.jl`. Optional plotting support is
implemented in `ext/SquareModelsMakieExt.jl`, which loads when a Makie backend is
available. Runnable examples are available in the repository's `examples/`
directory.
