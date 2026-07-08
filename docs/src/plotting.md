# Plotting and Printing

SquareModels delegates plotting to Makie. Load a backend such as CairoMakie to
activate the package extension:

```julia
using CairoMakie
using SquareModels
```

The extension teaches Makie how to plot `Window` and `LabeledSeries` values, so
model slices keep their axes and labels.

## Plotting Variables

`plotvar` builds a complete figure for one model variable or slice:

```@example plotting
using CairoMakie
import JuMP
using JuMP: Model
using SquareModels

model = Model()
t = 2020:2024
regions = [:north, :south]

@variables model begin
    qGDP[t], "Real GDP"
    qC[t], "Real consumption"
    emissions[regions, t], "Emissions"
end

data = ModelDictionary(model)
data[qGDP] = [100, 103, 106, 110, 114]
data[qC] = [80, 82, 84, 87, 90]
data[emissions] = [10 9 8 8 7; 6 6 5 5 4]

fig = plotvar(data, qGDP[t]; ylabel="Index")
save("qgdp.png", fig)
nothing
```

![](qgdp.png)

Multi-dimensional variables fan out into one line per leading-index combination;
the last dimension is used as the x-axis.

```@example plotting
fig = plotvar(data, emissions[regions, t]; ylabel="Mt CO2")
save("emissions.png", fig)
nothing
```

![](emissions.png)

## Plotting Expressions

[`@plot`](@ref) resolves bare variable names against a model dictionary and labels
each plotted expression with the source text:

```@example plotting
fig = @plot data [qGDP / qGDP[2020], qC / qC[2020]]
save("normalised.png", fig)
nothing
```

![](normalised.png)

Use explicit dots for elementwise function calls:

```julia
@plot data log.(qGDP)
```

For programmatic workflows, build series explicitly with [`labeled`](@ref) and
draw them with `plotseries`:

```julia
plotseries([
    labeled(data[qGDP] ./ data[qGDP[2020]], "qGDP / qGDP[2020]"),
    labeled(data[qC] ./ data[qC[2020]], "qC / qC[2020]"),
])
```

## Printing and Expression Evaluation

[`@evalexpr`](@ref) evaluates expressions without plotting:

```@example plotting
normalised_gdp = @evalexpr data qGDP / qGDP[2020]
round.(normalised_gdp, digits=3)
```

[`@prt`](@ref) prints values and transformations in a table-oriented format:

```julia
@prt data qGDP
@prt :p data qGDP
@prt :q baseline=>scenario qGDP
```

For interactive work, set defaults once and omit the source:

```julia
set_default_source!(baseline => scenario)
set_default_operator!(:q)
set_default_periods!(2020:2030)
@prt qGDP
@plot qGDP
reset_print_defaults!()
```
