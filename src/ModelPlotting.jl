# ModelPlotting - Plot ModelDictionary variables with Makie
#
# Architecture:
#   - `Window` and `LabeledSeries` share the `AbstractSeries` supertype, so the
#     Makie glue is written once against it.
#   - `expand(s) -> Vector{LabeledSeries}` is the central operation: it splits a
#     (possibly multi-dimensional) series into individual lines, one per
#     leading-index combination, with the last dimension as the x-axis. The
#     batteries-included builders and `@plot` draw one line per element.
#   - The heavy lifting (drawing) is delegated to Makie. The `SquareModelsMakieExt`
#     extension teaches Makie how to turn an `AbstractSeries` into x/y data via
#     `Makie.convert_arguments`, so `lines`, `scatter!`, ... accept a 1-D `Window`
#     or a `LabeledSeries` directly.
#   - `expand`/`to_series`/`axis_of` are the model-specific glue (year axes,
#     nothing -> NaN, dimension fan-out). They are generic functions; the Window
#     methods live in the extension, the LabeledSeries and plain-array methods here.
#   - `plotvar`/`plotseries` are thin "batteries-included" figure builders
#     (title, legend) implemented in the extension.
#   - `@plot` is syntactic sugar that resolves bare variable names against a
#     ModelDictionary and labels each series with its source text. `labeled` is
#     the explicit escape hatch for programmatic construction.
#
# Drawing functions are supplied by the Makie extension once a backend is loaded
# (`using CairoMakie`).

module ModelPlotting

using Base.Meta: isexpr
import ..AbstractSeries   # shared supertype with `Window` (defined in the parent module)
import ..Window
using ..ModelExpressions: _active_specs, _collect_bases, _db_parts, _default_periods, _expand_dot_macro, _expand_ops, _macro_parts, _need_ref, _op_label, _ref_expr, _ref_value, _rewrite, _transform, _value_expr

export @plot, plotvar, plotseries, labeled, LabeledSeries, alternating_dash!
export set_plot_finalize!, reset_plot_finalize!, plot_finalize

const _plot_finalize = Ref{Union{Nothing,Function}}(nothing)

"""
    set_plot_finalize!(f)
    set_plot_finalize!(nothing)

Register `f(fig, ax, series)` to run after `plotvar`/`plotseries` draw the axis and lines.
Use for custom legends, dash patterns, annotations, etc.
"""
set_plot_finalize!(f::Function) = (_plot_finalize[] = f; f)
set_plot_finalize!(::Nothing) = (_plot_finalize[] = nothing; nothing)
plot_finalize() = _plot_finalize[]
reset_plot_finalize!() = set_plot_finalize!(nothing)

# Implemented in the Makie extension; nothing to draw without it.
function _plotting_error(name, args)
	if Base.get_extension(parentmodule(@__MODULE__), :SquareModelsMakieExt) === nothing
		error("`$name` requires Makie. Load a Makie backend first, e.g. `using CairoMakie`, then call `$name` again.")
	end
	types = join(string.(typeof.(args)), ", ")
	error("No `$name` method for argument types ($types).")
end

plotvar(args...; kwargs...) = _plotting_error(:plotvar, args)
plotseries(args...; kwargs...) = _plotting_error(:plotseries, args)
alternating_dash!(args...; kwargs...) = _plotting_error(:alternating_dash!, args)

"""Convert a value to Float64, mapping `nothing` to `NaN` (like missing data)."""
_to_float(x) = x === nothing ? NaN : Float64(x)

"""Prefer numeric axes when all labels are numbers (e.g. years)."""
function _coerce_axis(labels)
	all(l -> l isa Integer, labels) && return collect(Int, labels)
	all(l -> l isa Real, labels) && return collect(Float64, labels)
	return labels
end

# `to_series(x) -> (xaxis_or_nothing, yvalues)` and `axis_of(x) -> xaxis_or_nothing`
# are extended for `Window` in the Makie extension. These fallbacks handle plain
# numeric data so `labeled` works on raw arrays too.
function to_series end
function axis_of end

to_series(y::AbstractArray) = (nothing, [_to_float(v) for v in vec(Array(y))])
to_series(y::Number) = (nothing, [_to_float(y)])
axis_of(::Any) = nothing

# `expand(s) -> Vector{LabeledSeries}` splits a series into individual lines. The
# `Window` method keeps model dimensions intact before any plotting backend is loaded.
function expand end

"""
    LabeledSeries(x, y, label)

A plottable series carrying its own x-axis, y-values, and legend label. This is
the common currency between `@plot`, `labeled`, and Makie (`convert_arguments`).
Unlike a `Window` (a view onto model data), it holds eager, computed values; both
share the [`AbstractSeries`](@ref) supertype.
"""
struct LabeledSeries <: AbstractSeries
	x::Vector
	y::Vector{Float64}
	label::String
end

to_series(s::LabeledSeries) = (s.x, s.y)
axis_of(s::LabeledSeries) = s.x
expand(s::LabeledSeries) = [s]

_dim_keys(w::Window) = axes(w.indices)

axis_of(w::Window) = _coerce_axis(collect(_dim_keys(w)[end]))

_line_label(name, combo) = isempty(combo) ? name :
	(isempty(name) ? join(combo, ", ") : "$name[$(join(combo, ", "))]")

function expand(w::Window)
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

to_series(w::Window) = (s = only(expand(w)); (s.x, s.y))

"""
    labeled(values, label; xfrom=())

Build a `LabeledSeries` from `values` (a `Window`, array, or number).

`label` becomes the legend entry. The x-axis is taken from `values` when it is a
`Window`; otherwise the first matching-length `Window` in `xfrom` supplies it
(this is how `@plot` reattaches a year axis to an arithmetic expression). Falls
back to `1:length` when no axis is available.

Use directly for programmatic plotting:
```julia
plot([labeled(db[v] .* db[other], "\$v*\$other") for v in vars])
```
"""
function labeled(values, label; xfrom=())
	x, y = to_series(values)
	if x === nothing
		for c in xfrom
			ax = axis_of(c)
			if ax !== nothing && length(ax) == length(y)
				x = ax
				break
			end
		end
	end
	x === nothing && (x = collect(1:length(y)))
	return LabeledSeries(collect(x), y, string(label))
end

# Resolve one `@plot` item to its lines. A series (e.g. a multi-dimensional
# `Window`) is fanned out with its own labels; anything else becomes a single
# labelled line using the expression's source text.
_lines(v::AbstractSeries, label, xfrom) = expand(v)
_lines(v::AbstractArray, label, xfrom) = _array_lines(v, label, xfrom)
_lines(v, label, xfrom) = [labeled(v, label; xfrom)]

function _array_lines(v::AbstractArray, label, xfrom)
	dims = axes(v)
	ndims(v) == 0 && return [labeled(only(v), label; xfrom)]
	ndims(v) <= 1 && return [LabeledSeries(_coerce_axis(collect(dims[end])), Float64[_to_float(y) for y in Array(v)], string(label))]
	idx_dims = axes(v)
	x = _coerce_axis(collect(dims[end]))
	out = LabeledSeries[]
	for combo in Iterators.product(idx_dims[1:end-1]...)
		y = Float64[_to_float(v[combo..., t]) for t in idx_dims[end]]
		push!(out, LabeledSeries(collect(x), y, _line_label(label, combo)))
	end
	return out
end

function _filter_periods(s::LabeledSeries, periods)
	periods === nothing && return s
	keep = [_period_match(x, periods) for x in s.x]
	any(keep) || return s
	return LabeledSeries(s.x[keep], s.y[keep], s.label)
end

_period_match(x, periods) = periods isa Union{AbstractArray,Tuple,AbstractRange} ? x in periods : x == periods

# Label each series with the source text the user wrote (with any `@.` expanded).
_label_text(ex) = string(_expand_dot_macro(ex))

function _op_lines(ops, x::AbstractSeries, ref, label, xfrom, periods)
	out = LabeledSeries[]
	for op in _expand_ops(ops)
		xlines = expand(x)
		reflines = _need_ref(op) ? _ref_lines(ref, op) : nothing
		for (i, s) in enumerate(xlines)
			r = reflines === nothing ? ref : reflines[i].y
			line_label = length(xlines) == 1 ? label : s.label
			push!(out, _filter_periods(LabeledSeries(s.x, _transform(op, s.y, r), _op_label(line_label, op)), periods))
		end
	end
	return out
end

function _ref_lines(ref, op)
	r = _ref_value(ref, op)
	return r isa AbstractSeries ? expand(r) : nothing
end

function _op_lines(ops, x::AbstractArray, ref, label, xfrom, periods)
	out = LabeledSeries[]
	for op in _expand_ops(ops)
		xlines = _lines(x, label, xfrom)
		reflines = _need_ref(op) ? _lines(_ref_value(ref, op), label, xfrom) : nothing
		for (i, s) in enumerate(xlines)
			r = reflines === nothing ? ref : reflines[i].y
			line_label = length(xlines) == 1 ? label : s.label
			push!(out, _filter_periods(LabeledSeries(s.x, _transform(op, s.y, r), _op_label(line_label, op)), periods))
		end
	end
	return out
end

function _op_lines(ops, x, ref, label, xfrom, periods)
	out = LabeledSeries[]
	for op in _expand_ops(ops)
		append!(out, _filter_periods.(_lines(_transform(op, x, ref), _op_label(label, op), xfrom), Ref(periods)))
	end
	return out
end

# ----------------------------------------------------------------------------------------------------------------------
# @plot macro
# ----------------------------------------------------------------------------------------------------------------------

_series_expr(item, dbv, refv, periodv, ops, oplines_ref) = begin
	bases = _collect_bases(item)
	cands = Expr(:tuple, Any[_rewrite(b, dbv) for b in bases]...)
	ref = _ref_expr(item, refv, periodv)
	:($oplines_ref($ops, $(_value_expr(item, dbv, periodv)), $ref, $(_label_text(item)), $cands, $periodv))
end

function _series_arg(expr, dbv, refv, periodv, ops, oplines_ref)
	vcat_ref = GlobalRef(Base, :vcat)
	isexpr(expr, :vect) || return _series_expr(expr, dbv, refv, periodv, ops, oplines_ref)
	items = Any[_series_expr(it, dbv, refv, periodv, ops, oplines_ref) for it in expr.args]
	return Expr(:call, vcat_ref, items...)
end

"""
    @plot db expr
    @plot op db expr
    @plot periods expr
    @plot op periods expr
    @plot ops db [expr1, expr2, ...]

Plot one or more expressions of model variables, resolving bare names against the
ModelDictionary `db` and labelling each series with its source text.

```julia
@plot db qGDP                       # single variable
@plot :p db qGDP                    # percentage growth
@plot :q baseline=>shock qGDP       # percent deviation from baseline
@plot 2020:2060 qGDP                # default source, limited to periods
@plot db qGDP / qGDP[2019]          # normalised, label "qGDP / qGDP[2019]"
@plot db [qGDP * pGDP, qGDP / qGDP[2019]]   # multiple series on one axis
@plot db y                          # multi-dim y[region, year] → one line per region
```

A multi-dimensional variable fans out into one line per leading-index combination
(the last dimension is the x-axis), each labelled `name[index...]`.

Bare identifiers are treated as variables of `db`; use `\$(value)` to inject
values from the surrounding scope (e.g. `@plot db qGDP / \$base`). Arithmetic
operators are broadcast implicitly; named calls (e.g. `sum`, `log`) are left as
written, so use explicit dots like `log.(x)` for elementwise functions.
"""
macro plot(args...)
	ops, db, ref, expr, use_defaults, periods = _macro_parts(args)
	dbv = gensym(:db)
	refv = gensym(:ref)
	periodv = gensym(:periods)
	oplines_ref = GlobalRef(@__MODULE__, :_op_lines)
	plotseries_ref = GlobalRef(@__MODULE__, :plotseries)
	default_periods_ref = GlobalRef(@__MODULE__, :_default_periods)
	period_arg = periods === nothing ? :($default_periods_ref()) : esc(periods)
	if use_defaults
		specsv = gensym(:defaults)
		specv = gensym(:spec)
		linesv = gensym(:lines)
		default_specs_ref = GlobalRef(@__MODULE__, :_active_specs)
		append_ref = GlobalRef(Base, :append!)
		arg = _series_arg(expr, dbv, refv, periodv, ops, oplines_ref)
		return quote
			let $specsv = $default_specs_ref(), $linesv = LabeledSeries[], $(esc(periodv)) = $period_arg
				for $specv in $specsv
					let $(esc(dbv)) = getproperty($specv, :source), $(esc(refv)) = getproperty($specv, :reference)
						$append_ref($linesv, $(esc(arg)))
					end
				end
				$plotseries_ref($linesv)
			end
		end
	end
	primary, ref = _db_parts(db, ref)
	refv = ref === nothing ? nothing : refv
	arg = _series_arg(expr, dbv, refv, periodv, ops, oplines_ref)
	body = quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(periodv)) = $period_arg
			$plotseries_ref($(esc(arg)))
		end
	end
	ref === nothing && return body
	return quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(refv)) = $(esc(ref)), $(esc(periodv)) = $period_arg
			$plotseries_ref($(esc(arg)))
		end
	end
end

end
