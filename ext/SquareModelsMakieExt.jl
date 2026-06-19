module SquareModelsMakieExt

using Makie
using SquareModels: ModelPlotting, ModelDictionary, Window, AbstractSeries
using SquareModels.ModelPlotting: LabeledSeries, _to_float, _coerce_axis

# ----------------------------------------------------------------------------------------------------------------------
# Model-specific glue: Window -> lines
# ----------------------------------------------------------------------------------------------------------------------
# Per-dimension axis keys; for a DenseAxisArray these are the user-facing keys
# (e.g. `([:a, :b, :c], 2020:2024)`), for a plain Array they are 1-based ranges.
_dim_keys(w::Window) = axes(w.indices)

# x-axis = the last dimension's keys (years coerce to a numeric axis).
ModelPlotting.axis_of(w::Window) = _coerce_axis(collect(_dim_keys(w)[end]))

_line_label(name, combo) = isempty(combo) ? name :
	(isempty(name) ? join(combo, ", ") : "$name[$(join(combo, ", "))]")

"""Split a Window into one line per leading-index combination (last dim = x-axis).
A 1-D window yields a single line; `y[region, year]` yields one line per region."""
function ModelPlotting.expand(w::Window)
	dk = _dim_keys(w)
	xaxis = collect(dk[end])
	x = _coerce_axis(xaxis)
	name = w.varname === nothing ? "" : String(w.varname)
	out = LabeledSeries[]
	for combo in Iterators.product(dk[1:end-1]...)
		y = Float64[_to_float(w[combo..., t]) for t in xaxis]
		push!(out, LabeledSeries(collect(x), y, _line_label(name, combo)))
	end
	return out
end

ModelPlotting.to_series(w::Window) = (s = only(ModelPlotting.expand(w)); (s.x, s.y))

# ----------------------------------------------------------------------------------------------------------------------
# Makie integration: teach Makie to accept any AbstractSeries (not piracy — we own these types)
# ----------------------------------------------------------------------------------------------------------------------
function Makie.convert_arguments(P::Makie.PointBased, s::AbstractSeries)
	x, y = ModelPlotting.to_series(s)
	return convert_arguments(P, x, y)
end

Makie.plottype(::AbstractSeries) = Makie.Lines

# ----------------------------------------------------------------------------------------------------------------------
# Batteries-included figure builders
# ----------------------------------------------------------------------------------------------------------------------
function ModelPlotting.plotvar(
	w::Window;
	label=nothing,
	title=nothing,
	xlabel="Year",
	ylabel="Value",
	figure=(;),
	axis=(;),
	kwargs...,
)
	ls = ModelPlotting.expand(w)
	name = w.varname === nothing ? "" : String(w.varname)
	fig = Figure(; size=(800, 450), figure...)
	ax = Axis(fig[1, 1]; title=something(title, name), xlabel, ylabel, axis...)
	for s in ls
		lbl = length(ls) == 1 ? something(label, s.label) : s.label
		lines!(ax, s; label=lbl, kwargs...)
	end
	axislegend(ax; position=:rb)
	return fig
end

ModelPlotting.plotvar(db::ModelDictionary, slice; kwargs...) = ModelPlotting.plotvar(db[slice]; kwargs...)

function ModelPlotting.plotseries(
	series::AbstractVector{<:AbstractSeries};
	title="",
	xlabel="Year",
	ylabel="Value",
	figure=(;),
	axis=(;),
	kwargs...,
)
	fig = Figure(; size=(800, 450), figure...)
	ax = Axis(fig[1, 1]; title, xlabel, ylabel, axis...)
	for s0 in series, s in ModelPlotting.expand(s0)
		lines!(ax, s; label=s.label, kwargs...)
	end
	axislegend(ax; position=:rb)
	return fig
end

ModelPlotting.plotseries(s::AbstractSeries; kwargs...) = ModelPlotting.plotseries([s]; kwargs...)

end
