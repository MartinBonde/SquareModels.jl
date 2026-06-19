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
# Drawing functions are empty generics until a Makie backend is loaded
# (`using CairoMakie`); calling them before that raises a `MethodError`.

module ModelPlotting

using Base.Meta: isexpr
import ..AbstractSeries   # shared supertype with `Window` (defined in the parent module)

export @plot, plotvar, plotseries, labeled, LabeledSeries

# Implemented in the Makie extension for `Window`; nothing to draw without it.
function plotvar end
function plotseries end

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

to_series(y::AbstractArray) = (nothing, [_to_float(v) for v in vec(collect(y))])
to_series(y::Number) = (nothing, [_to_float(y)])
axis_of(::Any) = nothing

# `expand(s) -> Vector{LabeledSeries}` splits a series into individual lines. The
# `Window` method (multi-dimensional fan-out) lives in the Makie extension.
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
_lines(v, label, xfrom) = [labeled(v, label; xfrom)]

# ----------------------------------------------------------------------------------------------------------------------
# @plot macro
# ----------------------------------------------------------------------------------------------------------------------
const _DOTTABLE_OPS = (:+, :-, :*, :/, :^, :%, :\)

"""Rewrite an expression AST: bare variable names become `db[:name]` lookups and
non-dot arithmetic operators become their broadcasting (dotted) form, so e.g.
`qGDP * pGDP` evaluates elementwise as `db[:qGDP] .* db[:pGDP]`. Index positions,
call heads, literals, and `\$(...)` interpolations are left untouched."""
function _rewrite(ex, dbv)
	ex isa Symbol && return :($dbv[$(QuoteNode(ex))])
	ex isa Expr || return ex
	if ex.head === :ref
		base = ex.args[1]
		newbase = base isa Symbol ? :($dbv[$(QuoteNode(base))]) : _rewrite(base, dbv)
		return Expr(:ref, newbase, ex.args[2:end]...)  # indices unchanged
	elseif ex.head === :call
		f = ex.args[1]
		args = Any[_rewrite(a, dbv) for a in ex.args[2:end]]
		f in _DOTTABLE_OPS && return Expr(:call, Symbol(".", f), args...)
		return Expr(:call, f, args...)
	elseif ex.head === :.        # property access (e.g. Time.t) — external, leave as-is
		return ex
	elseif ex.head === :$        # interpolation — inject caller value verbatim
		return ex.args[1]
	else
		return Expr(ex.head, Any[_rewrite(a, dbv) for a in ex.args]...)
	end
end

"""Collect the unique variable base names referenced as values in `ex` (bare
symbols and `:ref` bases), skipping index positions, call heads, and interpolations.
These supply candidate year axes for arithmetic expressions."""
function _collect_bases(ex, acc=Symbol[])
	if ex isa Symbol
		ex in acc || push!(acc, ex)
	elseif ex isa Expr
		if ex.head === :ref
			_collect_bases(ex.args[1], acc)
		elseif ex.head === :call
			for a in ex.args[2:end]
				_collect_bases(a, acc)
			end
		elseif ex.head === :$ || ex.head === :.
			# external / interpolated — not a db variable
		else
			for a in ex.args
				_collect_bases(a, acc)
			end
		end
	end
	return acc
end

_series_expr(item, dbv, lines_ref) = begin
	bases = _collect_bases(item)
	cands = Expr(:tuple, Any[:($dbv[$(QuoteNode(b))]) for b in bases]...)
	:($lines_ref($(_rewrite(item, dbv)), $(string(item)), $cands))
end

"""
    @plot db expr
    @plot db [expr1, expr2, ...]

Plot one or more expressions of model variables, resolving bare names against the
ModelDictionary `db` and labelling each series with its source text.

```julia
@plot db qGDP                       # single variable
@plot db qGDP / qGDP[2019]          # normalised, label "qGDP / qGDP[2019]"
@plot db [qGDP * pGDP, qGDP / qGDP[2019]]   # multiple series on one axis
@plot db y                          # multi-dim y[region, year] → one line per region
```

A multi-dimensional variable fans out into one line per leading-index combination
(the last dimension is the x-axis), each labelled `name[index...]`.

Bare identifiers are treated as variables of `db`; use `\$(value)` to inject
values from the surrounding scope (e.g. `@plot db qGDP / \$base`). Arithmetic
operators are applied elementwise.
"""
macro plot(db, expr)
	dbv = gensym(:db)
	lines_ref = GlobalRef(@__MODULE__, :_lines)
	plotseries_ref = GlobalRef(@__MODULE__, :plotseries)
	vcat_ref = GlobalRef(Base, :vcat)
	if isexpr(expr, :vect)
		items = Any[_series_expr(it, dbv, lines_ref) for it in expr.args]
		arg = Expr(:call, vcat_ref, items...)
	else
		arg = _series_expr(expr, dbv, lines_ref)
	end
	return quote
		let $(esc(dbv)) = $(esc(db))
			$plotseries_ref($(esc(arg)))
		end
	end
end

end
