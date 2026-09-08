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
import .._SparseTableArray, .._table_layout
using ..ModelExpressions: LabeledArray
using ..ModelExpressions: _active_specs, _collect_bases, _db_parts, _default_periods, _expand_dot_macro, _expand_ops, _macro_parts, _need_ref, _op_axis_label, _ref_expr, _ref_value, _rewrite, _transform, _value_expr

export @plot, plotvar, plotseries, plotseries!, labeled, LabeledSeries, alternating_dash!
export set_plot_finalize!, reset_plot_finalize!, plot_finalize

const _plot_finalize = Ref{Union{Nothing,Function}}(nothing)

"""
    set_plot_finalize!(f)
    set_plot_finalize!(nothing)

Register the default legend function `f(fig, ax, series)` for plot builders.
An explicit `legend=false`, `true`, NamedTuple, or function overrides this default.
Use the per-plot `decorate(ax, series)` keyword for annotations.
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

"""
    plotvar(window; kwargs...)
    plotvar(data::ModelDictionary, variable; kwargs...)

Plot one model variable and return its Makie `Figure`.

Load a Makie backend such as CairoMakie before calling this function. Keywords
configure the title, axis, legend, line style, and underlying Makie figure.
"""
plotvar(args...; kwargs...) = _plotting_error(:plotvar, args)

"""
    plotseries(series; kwargs...)
    plotseries(position, series; kwargs...)

Plot one or more [`AbstractSeries`](@ref) values and return their Makie `Figure`.
Pass a Makie grid position, such as `fig[1, 2]`, to add an axis to an existing figure.

Load a Makie backend such as CairoMakie before calling this function. A single
series or a vector of series is accepted.
"""
plotseries(args...; kwargs...) = _plotting_error(:plotseries, args)
"""Draw series on an existing Makie axis and return the created line handles."""
plotseries!(args...; kwargs...) = _plotting_error(:plotseries!, args)
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
    LabeledSeries(x, y, label, op=:n)

A plottable series carrying its own x-axis, y-values, and legend label. This is
the common currency between `@plot`, `labeled`, and Makie (`convert_arguments`).
Unlike a `Window` (a view onto model data), it holds eager, computed values; both
share the [`AbstractSeries`](@ref) supertype.

`op` records the print/plot operator (e.g. `:m`, `:q`) that produced `y`, used to
pick a default y-axis label (see `_op_axis_label`) without cluttering the legend
label itself.
"""
struct LabeledSeries <: AbstractSeries
	x::Vector
	y::Vector{Float64}
	label::String
	op::Symbol
	panel::Tuple{String,Tuple}
end
LabeledSeries(x, y, label, op=:n) = LabeledSeries(x, y, label, op, (label, ()))

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
		push!(out, LabeledSeries(collect(x), y, _line_label(name, combo), :n, (name, combo)))
	end
	return out
end

to_series(w::Window) = (s = only(expand(w)); (s.x, s.y))

# Sparse layouts contain only live leading-index combinations. A gap stays NaN.
function _layout_lines(layout, name)
	x = _coerce_axis(collect(layout.periods))
	return [LabeledSeries(x,
		[isequal(v, "") ? NaN : _to_float(v) for v in layout.data[:, j]],
		_line_label(name, combo), :n, (name, combo))
		for (j, combo) in enumerate(layout.combos)]
end
expand(w::Window{<:Any,<:_SparseTableArray}) = _layout_lines(_table_layout(w), something(w.varname, ""))

"""
    labeled(values, label; xfrom=())

Build a `LabeledSeries` from `values` (a `Window`, array, or number).

`label` becomes the legend entry. The x-axis is taken from `values` when it is a
`Window`; otherwise the first matching-length `Window` in `xfrom` supplies it
(this is how `@plot` reattaches a year axis to an arithmetic expression). Falls
back to `1:length` when no axis is available.

Use directly for programmatic plotting:
```julia
plotseries([labeled(db[v] .* db[other], "\$v*\$other") for v in vars])
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

_plot_axes(v::AbstractArray) = axes(v)
_plot_axes(v::LabeledArray) = v.dims

_array_lines(v::LabeledArray{T,N,A}, label, xfrom) where {T,N,A<:_SparseTableArray} =
	_layout_lines(_table_layout(v.data), label)
_array_lines(v::_SparseTableArray, label, xfrom) = _layout_lines(_table_layout(v), label)

function _array_lines(v::AbstractArray, label, xfrom)
	dims = _plot_axes(v)
	ndims(v) == 0 && return [labeled(only(v), label; xfrom)]
	ndims(v) <= 1 && return [LabeledSeries(_coerce_axis(collect(dims[end])), Float64[_to_float(y) for y in Array(v)], string(label))]
	idx_dims = axes(v)
	x = _coerce_axis(collect(dims[end]))
	out = LabeledSeries[]
	for (combo, labels) in zip(Iterators.product(idx_dims[1:end-1]...), Iterators.product(dims[1:end-1]...))
		y = Float64[_to_float(v[combo..., t]) for t in idx_dims[end]]
		push!(out, LabeledSeries(collect(x), y, _line_label(label, labels), :n, (label, labels)))
	end
	return out
end

function _filter_periods(s::LabeledSeries, periods)
	periods === nothing && return s
	keep = [_period_match(x, periods) for x in s.x]
	any(keep) || return s
	return LabeledSeries(s.x[keep], s.y[keep], s.label, s.op, s.panel)
end

_period_match(x, periods) = periods isa Union{AbstractArray,Tuple,AbstractRange} ? x in periods : x == periods

# Label each series with the source text the user wrote (with any `@.` expanded).
_label_text(ex) = string(_expand_dot_macro(ex))

function _line_transform(op, s, ref, reflines, i)
	if op in (:m, :q, :mp) && reflines !== nothing
		index = findfirst(r -> r.panel[2] == s.panel[2], reflines)
		index === nothing && return fill(NaN, length(s.y))
		r = reflines[index]
		# Keep years until reference operators have aligned the observations.
		return collect(_transform(op, LabeledArray(s.y, (s.x,)), LabeledArray(r.y, (r.x,))))
	end
	return _transform(op, s.y, reflines === nothing ? ref : reflines[i].y)
end

function _op_lines(ops, x::AbstractSeries, ref, label, xfrom, periods)
	out = LabeledSeries[]
	for op in _expand_ops(ops)
		xlines = expand(x)
		reflines = _need_ref(op) ? _ref_lines(ref, op) : nothing
		for (i, s) in enumerate(xlines)
			line_label = length(xlines) == 1 ? label : s.label
			push!(out, _filter_periods(LabeledSeries(s.x, _line_transform(op, s, ref, reflines, i), line_label, op,
				(label, s.panel[2])), periods))
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
			line_label = length(xlines) == 1 ? label : s.label
			push!(out, _filter_periods(LabeledSeries(s.x, _line_transform(op, s, ref, reflines, i), line_label, op,
				(label, s.panel[2])), periods))
		end
	end
	return out
end

_with_op(s::LabeledSeries, op) = LabeledSeries(s.x, s.y, s.label, op, s.panel)

function _op_lines(ops, x, ref, label, xfrom, periods)
	out = LabeledSeries[]
	for op in _expand_ops(ops)
		lines = _with_op.(_lines(_transform(op, x, ref), label, xfrom), op)
		append!(out, _filter_periods.(lines, Ref(periods)))
	end
	return out
end

# ----------------------------------------------------------------------------------------------------------------------
# @plot macro
# ----------------------------------------------------------------------------------------------------------------------

_plot_keyword(ex::Symbol) = Expr(:kw, ex, esc(ex))
_plot_keyword(ex::Expr) = ex.head === :... ? Expr(:..., esc(only(ex.args))) :
	Expr(:kw, ex.args[1], esc(ex.args[2]))

function _plot_call(series, keywords)
	return Expr(:call, GlobalRef(@__MODULE__, :plotseries),
		Expr(:parameters, _plot_keyword.(keywords)...), series)
end

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
    @plot(op, periods, db, expr; kwargs...)

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

Keyword options pass to `plotseries`. Use `position=fig[row, column]` to draw in
an existing figure. In the four-argument form, `op` and `periods` can be local variables.
"""
macro plot(args...)
	keywords = Any[]
	positional = Any[]
	for arg in args
		if isexpr(arg, :parameters)
			append!(keywords, arg.args)
		elseif isexpr(arg, :(=))
			push!(keywords, arg)
		else
			push!(positional, arg)
		end
	end
	ops, db, ref, expr, use_defaults, periods = _macro_parts(positional)
	dbv = gensym(:db)
	refv = gensym(:ref)
	periodv = gensym(:periods)
	oplines_ref = GlobalRef(@__MODULE__, :_op_lines)
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
				$(_plot_call(linesv, keywords))
			end
		end
	end
	primary, ref = _db_parts(db, ref)
	refv = ref === nothing ? nothing : refv
	arg = _series_arg(expr, dbv, refv, periodv, ops, oplines_ref)
	body = quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(periodv)) = $period_arg
			$(_plot_call(esc(arg), keywords))
		end
	end
	ref === nothing && return body
	return quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(refv)) = $(esc(ref)), $(esc(periodv)) = $period_arg
			$(_plot_call(esc(arg), keywords))
		end
	end
end

end
