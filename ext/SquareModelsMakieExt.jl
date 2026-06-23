module SquareModelsMakieExt

using Makie
using SquareModels: ModelPlotting, ModelDictionary, Window, AbstractSeries

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
