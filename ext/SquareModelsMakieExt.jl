# Draw model series and trellis panels with the active Makie theme.
# Keep model expressions and data transformations in SquareModels.
module SquareModelsMakieExt

using Makie
using SquareModels: ModelPlotting, ModelDictionary, Window, AbstractSeries
using SquareModels.ModelPlotting: plot_finalize

const _LINESTYLES = [:solid, :dot, :dash, :dashdot]

_base_label(s::AbstractSeries) = s.label
_op(s::AbstractSeries) = :n
_op(s::ModelPlotting.LabeledSeries) = s.op

# When several operators are plotted for the same base label (e.g. `:a` expands
# to :n, :p, :r, :rp), the label alone can't tell the lines apart, so tag those
# (and only those) with the operator, e.g. "qGDP <p>".
function _legend_labels(series)
	groups = Dict{String,Set{Symbol}}()
	for s in series
		push!(get!(groups, _base_label(s), Set{Symbol}()), _op(s))
	end
	return [length(groups[_base_label(s)]) > 1 ? "$(s.label) <$(_op(s))>" : s.label for s in series]
end

"""Default y-axis label for a set of series: the shared operator's label (e.g.
`:m => "Difference from baseline"`) when every series uses the same operator,
otherwise falls back to `"Value"`."""
function _default_ylabel(series)
	ops = unique(_op.(series))
	length(ops) == 1 && return something(ModelPlotting._op_axis_label(only(ops)), "Value")
	return "Value"
end

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

# An explicit legend choice overrides the organisation's default legend.
_legend!(::Nothing, fig, ax, series) =
	_legend!(something(plot_finalize(), (position=:rb,)), fig, ax, series)
_legend!(enabled::Bool, fig, ax, series) = enabled ? axislegend(ax) : nothing
_legend!(options::Union{NamedTuple,AbstractDict}, fig, ax, series) = axislegend(ax; options...)
_legend!(f::Function, fig, ax, series) = f(fig, ax, series)

function _finish(fig, ax, series, legend, decorate)
	decorate === nothing || decorate(ax, series)
	_legend!(legend, fig, ax, series)
	return fig
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
	kwargs...,
)
	ls = ModelPlotting.expand(w)
	name = w.varname === nothing ? "" : String(w.varname)
	labels = label === nothing ? nothing : [label]
	return ModelPlotting.plotseries(ls; title=something(title, name), labels, kwargs...)
end

ModelPlotting.plotvar(db::ModelDictionary, slice; kwargs...) = ModelPlotting.plotvar(db[slice]; kwargs...)

_expand(series) = reduce(vcat, ModelPlotting.expand.(series); init=ModelPlotting.LabeledSeries[])

function ModelPlotting.plotseries!(
	ax::Axis, series::AbstractVector{<:AbstractSeries};
	labels=nothing, styles=nothing, alternating_dash=nothing, kwargs...,
)
	expanded = _expand(series)
	labels = labels === nothing ? _legend_labels(expanded) : labels
	styles = styles === nothing ? fill((;), length(expanded)) : styles
	@assert length(labels) == length(styles) == length(expanded) "Supply one label and style per line."
	plots = [lines!(ax, s; label, kwargs...) for (s, label) in zip(expanded, labels)]
	_apply_alternating_dash!(ax, expanded, alternating_dash)
	# Explicit per-series styles take precedence over the default dash cycle.
	for (plot, style) in zip(plots, styles), (key, value) in pairs(style)
		plot[key] = value
	end
	return plots
end

ModelPlotting.plotseries!(ax::Axis, series::AbstractSeries; kwargs...) =
	ModelPlotting.plotseries!(ax, [series]; kwargs...)

function ModelPlotting.plotseries(series::AbstractVector{<:AbstractSeries};
	figure=(;), position=nothing, layout=:overlay, columns::Integer=3, kwargs...,
)
	@assert layout in (:overlay, :trellis) "Layout must be :overlay or :trellis."
	@assert columns > 0 "Trellis columns must be positive."
	expanded = _expand(series)
	if position === nothing
		panel_count = length(unique(s.panel for s in expanded))
		ncols = min(columns, panel_count)
		default_size = to_value(Makie.theme(:size))
		size_options = layout == :trellis && panel_count > 0 ?
			(size=(ncols * default_size[1], cld(panel_count, ncols) * default_size[2]),) : (;)
		fig = Figure(; size_options..., figure...)
		position = fig[1, 1]
	end
	return ModelPlotting.plotseries(position, expanded; layout, columns, kwargs...)
end

function ModelPlotting.plotseries(
	position::Union{Makie.GridPosition,Makie.GridSubposition},
	series::AbstractVector{<:AbstractSeries};
	title="",
	xlabel="",
	ylabel=nothing,
	axis=(;),
	legend=nothing,
	decorate=nothing,
	layout=:overlay,
	columns::Integer=3,
	panel_titles=nothing,
	linkx::Bool=true,
	linky::Bool=false,
	labels=nothing,
	styles=nothing,
	kwargs...,
)
	@assert layout in (:overlay, :trellis) "Layout must be :overlay or :trellis."
	expanded = _expand(series)
	layout == :trellis && return _trellis(position, expanded;
		title, xlabel, ylabel, axis, legend, decorate, columns, panel_titles, linkx, linky, labels, styles, kwargs...)
	ax = Axis(position; title, xlabel, ylabel=something(ylabel, _default_ylabel(expanded)), axis...)
	fig = ax.parent
	ModelPlotting.plotseries!(ax, expanded; labels, styles, kwargs...)
	return _finish(fig, ax, expanded, legend, decorate)
end

_subset(::Nothing, indices) = nothing
_subset(values, indices) = values[indices]

function _trellis(position, series; columns, title, panel_titles, linkx, linky, labels, styles, kwargs...)
	@assert columns > 0 "Trellis columns must be positive."
	panels = unique(s.panel for s in series)
	@assert !isempty(panels) "A trellis plot needs at least one series."
	titles = panel_titles === nothing ? [ModelPlotting._line_label(panel...) for panel in panels] : panel_titles
	@assert length(titles) == length(panels) "Supply one title per panel."
	labels === nothing || @assert length(labels) == length(series) "Supply one label per line."
	styles === nothing || @assert length(styles) == length(series) "Supply one style per line."
	grid = GridLayout(position)
	fig = Makie.get_top_parent(grid)
	offset = isempty(title) ? 0 : 1
	isempty(title) || Label(grid[1, 1:min(columns, length(panels))], title; font=:bold)
	axes = Axis[]
	for (n, panel) in enumerate(panels)
		indices = findall(s -> s.panel == panel, series)
		# Each panel owns a nested layout, so its legend cannot occupy another panel.
		cell = GridLayout(grid[offset + cld(n, columns), mod1(n, columns)])
		ModelPlotting.plotseries(cell[1, 1], series[indices]; title=titles[n],
			labels=_subset(labels, indices), styles=_subset(styles, indices), kwargs...)
		push!(axes, content(cell[1, 1]))
	end
	linkx && linkxaxes!(axes...)
	linky && linkyaxes!(axes...)
	return fig
end

ModelPlotting.plotseries(s::AbstractSeries; kwargs...) = ModelPlotting.plotseries([s]; kwargs...)
ModelPlotting.plotseries(position::Union{Makie.GridPosition,Makie.GridSubposition}, s::AbstractSeries; kwargs...) =
	ModelPlotting.plotseries(position, [s]; kwargs...)

end
