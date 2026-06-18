module SquareModelsCairoMakieExt

using CairoMakie
using SquareModels: ModelPlotting, ModelDictionary, Window, description, _key_to_tuple

_to_float(x) = x === nothing ? NaN : Float64(x)

# Every entry point funnels the slice through a Window first, so labels and
# descriptions only ever need to handle Windows.
_as_window(::ModelDictionary, slice::Window) = slice
_as_window(db::ModelDictionary, slice) = db[slice]

"""Extract axis labels from a Window's index keys."""
function _axis_labels(w::Window)
	labels = Any[]
	for k in keys(w.indices)
		kt = _key_to_tuple(k)
		push!(labels, length(kt) == 1 ? only(kt) : kt)
	end
	return labels
end

"""Prefer numeric axes when all labels are numbers (e.g. years)."""
function _coerce_axis(labels)
	all(l -> l isa Integer, labels) && return collect(Int, labels)
	all(l -> l isa Real, labels) && return collect(Float64, labels)
	return labels
end

function ModelPlotting.series(db::ModelDictionary, slice)
	w = _as_window(db, slice)
	y = _to_float.(collect(w))
	return _coerce_axis(_axis_labels(w)), y
end

_default_label(w::Window) = isnothing(w.varname) ? "" : String(w.varname)
_series_description(w::Window) = isnothing(w.varname) ? "" : description(Symbol(w.varname))

function ModelPlotting.plot_series!(ax, db::ModelDictionary, slice; label=nothing, kwargs...)
	w = _as_window(db, slice)
	x, y = ModelPlotting.series(db, w)
	label = something(label, _default_label(w))
	desc = _series_description(w)
	isempty(desc) || (label = "$label ($desc)")
	return lines!(ax, x, y; label, kwargs...)
end

function ModelPlotting.plot_variable(
	db::ModelDictionary,
	slice;
	title=nothing,
	xlabel="Year",
	ylabel="Value",
	kwargs...,
)
	w = _as_window(db, slice)
	title = something(title, _series_description(w))
	fig = Figure(size=(800, 450))
	ax = Axis(fig[1, 1]; title, xlabel, ylabel)
	ModelPlotting.plot_series!(ax, db, w; kwargs...)
	axislegend(ax; position=:rb)
	return fig
end

ModelPlotting.save_figure(fig, path::AbstractString) = save(path, fig)

end
