module SquareModelsMakieExt

using Makie
using SquareModels: ModelPlotting, ModelDictionary, Window, AbstractSeries
using SquareModels.ModelPlotting: plot_finalize

const _LINESTYLES = [:solid, :dot, :dash, :dashdot]

# "qGDP <q>" and "qGDP <r>" are the same variable under different operators.
_base_label(s::AbstractSeries) = replace(s.label, r" <[^>]+>$" => "")

function _palette_colors()
	pal = Makie.to_value(Makie.theme(:palette))
	haskey(pal, :color) ? Makie.to_value(pal[:color]) : Makie.wong_colors()
end

"""
Style lines so that series with the same base label (the same variable plotted
for multiple sources, or value/reference pairs like `:an`) share a color and are
distinguished by linestyle instead (solid, dot, dash, ...). Applied
automatically when duplicate base labels are present; pass `alternating_dash=false`
to disable or `alternating_dash=true` to force pairing of consecutive lines.
"""
function ModelPlotting.alternating_dash!(ax, series)
	labels = _base_label.(series)
	groups = unique(labels)
	if length(groups) == length(labels)
		# No duplicates: fall back to pairing consecutive lines (Plotly's alternating_dash).
		byplot = [(i - 1) ÷ 2 + 1 for i in eachindex(labels)]
		groups = 1:maximum(byplot; init=0)
	else
		byplot = [findfirst(==(l), groups) for l in labels]
	end
	colors = _palette_colors()
	seen = zeros(Int, length(groups))
	plots = [p for p in ax.scene.plots if p isa Makie.Lines]
	for (p, g) in zip(plots, byplot)
		seen[g] += 1
		p.color = colors[mod1(g, length(colors))]
		p.linestyle = _LINESTYLES[mod1(seen[g], length(_LINESTYLES))]
	end
	return ax
end

function _apply_alternating_dash!(ax, series, alternating_dash)
	alternating_dash === false && return
	auto = alternating_dash === nothing
	if auto
		labels = _base_label.(series)
		length(unique(labels)) == length(labels) && return
	end
	ModelPlotting.alternating_dash!(ax, series)
end

# The global finalize hook (see `set_plot_finalize!`) is the theming entry point:
# when set, it takes over legend placement etc., so the default native legend is
# skipped unless explicitly requested with `legend=true` or a NamedTuple.
function _finish(fig, ax, series, legend, alternating_dash)
	_apply_alternating_dash!(ax, series, alternating_dash)
	f = plot_finalize()
	if legend === true
		axislegend(ax)
	elseif legend isa Union{NamedTuple,AbstractDict}
		axislegend(ax; legend...)
	elseif legend === nothing && f === nothing
		axislegend(ax; position=:rb)
	end
	f === nothing && return fig
	return f(fig, ax, series)
end

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
	xlabel="",
	ylabel="Value",
	figure=(;),
	axis=(;),
	legend=nothing,
	alternating_dash=nothing,
	kwargs...,
)
	ls = ModelPlotting.expand(w)
	name = w.varname === nothing ? "" : String(w.varname)
	fig = Figure(; figure...)
	ax = Axis(fig[1, 1]; title=something(title, name), xlabel, ylabel, axis...)
	for s in ls
		lbl = length(ls) == 1 ? something(label, s.label) : s.label
		lines!(ax, s; label=lbl, kwargs...)
	end
	return _finish(fig, ax, ls, legend, alternating_dash)
end

ModelPlotting.plotvar(db::ModelDictionary, slice; kwargs...) = ModelPlotting.plotvar(db[slice]; kwargs...)

function ModelPlotting.plotseries(
	series::AbstractVector{<:AbstractSeries};
	title="",
	xlabel="",
	ylabel="Value",
	figure=(;),
	axis=(;),
	legend=nothing,
	alternating_dash=nothing,
	kwargs...,
)
	expanded = AbstractSeries[]
	for s0 in series, s in ModelPlotting.expand(s0)
		push!(expanded, s)
	end
	fig = Figure(; figure...)
	ax = Axis(fig[1, 1]; title, xlabel, ylabel, axis...)
	for s in expanded
		lines!(ax, s; label=s.label, kwargs...)
	end
	return _finish(fig, ax, expanded, legend, alternating_dash)
end

ModelPlotting.plotseries(s::AbstractSeries; kwargs...) = ModelPlotting.plotseries([s]; kwargs...)

end
