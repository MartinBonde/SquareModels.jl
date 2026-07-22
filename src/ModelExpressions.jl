# ModelExpressions - Evaluate ModelDictionary expressions without plotting

module ModelExpressions

using Base.Meta: isexpr
using JuMP: JuMP
using PrettyTables: pretty_table
import ..Window

export @evalexpr, @prt, LabeledArray, MultiVarResult
export set_default_source!, set_default_operator!, set_default_periods!, set_column_label_total_width!, reset_print_defaults!

"""
    LabeledArray(data, dims)

A thin `AbstractArray` wrapper that behaves exactly like `data` for indexing,
equality, and `Array` conversion, but remembers the per-dimension labels in
`dims` (e.g. `([:hh, :firm], 2020:2021)`) and an optional expression `name`
purely for display. `@prt`/`@evalexpr` wrap transformed multi-dimensional
results in one of these so they still print as a table (rows from the leading
dimensions, columns from the last) instead of a bare, unlabelled matrix.
"""
struct LabeledArray{T, N} <: AbstractArray{T, N}
	data::Array{T, N}
	dims::NTuple{N, Any}
	name::String
end
LabeledArray(data::AbstractArray, dims, name="") = LabeledArray(Array(data), Tuple(dims), name)

Base.size(a::LabeledArray) = size(a.data)
Base.getindex(a::LabeledArray, i...) = getindex(a.data, i...)

Base.show(io::IO, ::MIME"text/plain", a::LabeledArray) = _period_row_table(io, a.data, a.dims, a.name)
Base.show(io::IO, a::LabeledArray) = show(io, MIME"text/plain"(), a)

"""Axis label collections for `x`, or `nothing` when `x` carries no labels."""
_axis_labels(x::JuMP.Containers.DenseAxisArray) = axes(x)
_axis_labels(x::Window) = axes(x.indices)
_axis_labels(_) = nothing

"""Wrap `result` in a [`LabeledArray`](@ref) using `x`'s axis labels, when available."""
function _relabel(result::AbstractArray, x, name="")
	dims = _axis_labels(x)
	(dims === nothing || ndims(result) != length(dims)) && return result
	return LabeledArray(result, dims, name)
end
_relabel(result, x, name="") = result

"""
    MultiVarResult(names, values)

Result of printing several expressions together, e.g. `@prt data (qGDP, pGDP)`.
Behaves like the underlying `values` tuple for equality, iteration, and
indexing, but displays as a single PrettyTables.jl table with one column per
name when every value is a scalar or carries the same final-dimension labels;
falls back to printing each value under its own heading otherwise.
"""
struct MultiVarResult{T<:Tuple}
	names::Vector{String}
	values::T
end

Base.:(==)(a::MultiVarResult, b) = a.values == b
Base.:(==)(a, b::MultiVarResult) = a == b.values
Base.:(==)(a::MultiVarResult, b::MultiVarResult) = a.values == b.values
Base.:(==)(a::MultiVarResult, b::AbstractVector) = collect(a.values) == b
Base.:(==)(a::AbstractVector, b::MultiVarResult) = a == collect(b.values)
Base.length(a::MultiVarResult) = length(a.values)
Base.iterate(a::MultiVarResult, state...) = iterate(a.values, state...)
Base.getindex(a::MultiVarResult, i) = a.values[i]

_dims_of(x::LabeledArray) = x.dims
_dims_of(::Number) = ()
_dims_of(x) = _axis_labels(x)

_data_of(x::LabeledArray) = x.data
_data_of(x::Window) = x.shaped_view
_data_of(x::AbstractArray) = Array(x)
_data_of(x::Number) = [x]

const _COLUMN_LABEL_MIN_WIDTH = 10
const DEFAULT_COLUMN_LABEL_TOTAL_WIDTH = Ref(72)

_table_label(source, expr) = string(source, '\n', _expr_label(expr))

_leading_combos(dims) = length(dims) == 1 ? [()] : vec(collect(Iterators.product(dims[1:end-1]...)))
_column_label(name, combo) = isempty(combo) ? name : (isempty(name) ? join(combo, ", ") : "$name[$(join(combo, ", "))]")

function _source_expr_parts(name)
	parts = split(name, '\n', limit=2)
	length(parts) == 1 && return nothing
	return (parts[1], parts[2])
end

_column_label_width(ncols) = max(_COLUMN_LABEL_MIN_WIDTH, DEFAULT_COLUMN_LABEL_TOTAL_WIDTH[] ÷ max(ncols, 1))

function _wrap_label(label, width)
	isempty(label) && return [""]
	n = length(label)
	n <= width && return [label]
	return [label[i:min(i + width - 1, n)] for i in 1:width:n]
end

# PrettyTables `auto_wrap` crops column labels with an ellipsis rather than
# adding label rows, and `line_breaks` does not split `\n` inside labels.
function _column_labels(labels)
	splits = _source_expr_parts.(labels)
	width = _column_label_width(length(labels))
	wrapped = [_wrap_label(isnothing(s) ? label : s[2], width) for (label, s) in zip(labels, splits)]
	n = maximum(length, wrapped)
	rows = [[i <= length(w) ? w[i] : "" for w in wrapped] for i in 1:n]
	any(!isnothing, splits) && pushfirst!(rows, [isnothing(s) ? "" : s[1] for s in splits])
	return rows
end

function _period_row_table(io::IO, data, dims, name="")
	periods = collect(dims[end])
	combos = _leading_combos(dims)
	mat = permutedims(reshape(data, length(combos), length(periods)))
	pretty_table(io, mat;
		column_labels=_column_labels([_column_label(name, c) for c in combos]),
		row_labels=string.(periods),
		stubhead_label="year",
		fit_table_in_display_vertically=false)
end

function Base.show(io::IO, ::MIME"text/plain", r::MultiVarResult)
	dims = _dims_of.(r.values)
	if all(==(()), dims)
		mat = reduce(hcat, vec(_data_of(v)) for v in r.values)
		pretty_table(io, mat;
			column_labels=_column_labels(r.names),
			fit_table_in_display_vertically=false)
	elseif all(d -> d !== nothing && !isempty(d), dims) && all(d -> collect(d[end]) == collect(first(dims)[end]), dims)
		periods = collect(first(dims)[end])
		combos = _leading_combos.(dims)
		mats = [permutedims(reshape(_data_of(v), length(c), length(periods))) for (v, c) in zip(r.values, combos)]
		labels = [_column_label(name, c) for (name, cs) in zip(r.names, combos) for c in cs]
		pretty_table(io, reduce(hcat, mats);
			column_labels=_column_labels(labels),
			row_labels=string.(periods),
			stubhead_label="year",
			fit_table_in_display_vertically=false)
	else
		for (i, (name, v)) in enumerate(zip(r.names, r.values))
			i > 1 && println(io)
			println(io, name)
			show(io, MIME"text/plain"(), v)
		end
	end
end
Base.show(io::IO, r::MultiVarResult) = show(io, MIME"text/plain"(), r)

"""
    _GroupEntry(source_label, source, reference_label, reference)

One database (or `reference => source` pair) contributed to a `@prt`/`@evalexpr`
call whose `db` argument is a `Pair` or a `Tuple` of sources/pairs, e.g.
`@prt data (baseline=>shock1, baseline=>shock2)`. Carries the raw (untransformed)
values so [`_group_result`](@ref) can decide, once per call, whether the active
operator needs `reference` (and skip computing it if not).
"""
struct _GroupEntry
	source_label::String
	source::Any
	reference_label::Union{String,Nothing}
	reference::Any
end

"""Push `(name, val)` onto `names`/`vals` unless `name` is already present (keeps a shared reference, e.g. a common baseline, from being repeated as a column)."""
function _push_unique!(names, vals, name, val)
	name in names && return nothing
	push!(names, name)
	push!(vals, val)
	return nothing
end

"""
    _group_result(ops, entries::Vector{_GroupEntry})

Turn the [`_GroupEntry`](@ref)s collected from a `Pair`/`Tuple` `db` argument
into a display value. When `ops` needs a reference (e.g. `:q`, `:m`), each
entry becomes one column of the transformed source (unwrapped to a bare value
when there is only one); a `reference`-less entry then errors. Otherwise (e.g.
the default `:n`), every distinct source and reference is shown as its own
raw column in a [`MultiVarResult`](@ref), e.g. `baseline:qGDP, shock1:qGDP,
shock2:qGDP` — a reference shared by several entries (like a common baseline)
is only shown once.
"""
function _group_result(ops, entries::Vector{_GroupEntry})
	needs = _group_needs_ref(ops)
	names = String[]
	vals = Any[]
	for e in entries
		if needs
			e.reference === nothing && error("operator requires a reference source for $(e.source_label); use `reference => source`")
			_push_unique!(names, vals, e.source_label, _apply_ops(ops, e.source, () -> e.reference, e.source_label))
		elseif e.reference === nothing
			_push_unique!(names, vals, e.source_label, _apply_ops(ops, e.source, nothing, e.source_label))
		else
			_push_unique!(names, vals, e.reference_label, _apply_ops(ops, e.reference, nothing, e.reference_label))
			_push_unique!(names, vals, e.source_label, _apply_ops(ops, e.source, nothing, e.source_label))
		end
	end
	needs && length(vals) == 1 && return only(vals)
	return MultiVarResult(names, Tuple(vals))
end

const DEFAULT_SPECS = Ref{Any}(nothing)
const DEFAULT_OPERATOR = Ref{Any}(:n)
const DEFAULT_PERIODS = Ref{Any}(nothing)

"""
    set_default_source!(sources...)

Set one or more default `ModelDictionary` sources used by `@prt`, `@plot`, and
`@evalexpr` when the source argument is omitted.

Plain sources use themselves as references. Use `baseline => source` or
another `Pair` to set a separate reference for a source.
"""
function set_default_source!(sources...)
	isempty(sources) && error("expected at least one default source")
	specs = [_source_spec(source, i) for (i, source) in enumerate(sources)]
	DEFAULT_SPECS[] = specs
	return nothing
end

"""Set the default operator used when no operator is given."""
function set_default_operator!(op)
	DEFAULT_OPERATOR[] = op
end

"""Set the default final-dimension periods used by `@prt`, `@evalexpr`, and `@plot`."""
function set_default_periods!(periods)
	DEFAULT_PERIODS[] = periods
end

"""Set the total character budget shared across `@prt` column labels when wrapping."""
function set_column_label_total_width!(width::Integer)
	width < _COLUMN_LABEL_MIN_WIDTH && error("column label total width must be at least $_COLUMN_LABEL_MIN_WIDTH")
	DEFAULT_COLUMN_LABEL_TOTAL_WIDTH[] = width
	return nothing
end

"""Clear interactive print/plot defaults."""
function reset_print_defaults!()
	DEFAULT_SPECS[] = nothing
	DEFAULT_OPERATOR[] = :n
	DEFAULT_PERIODS[] = nothing
	DEFAULT_COLUMN_LABEL_TOTAL_WIDTH[] = 72
	return nothing
end

_source_spec(source, i) = (; source, reference=source, source_label="baseline$i", reference_label="baseline$i")
_source_spec(pair::Pair, i) = (; source=pair.second, reference=pair.first, source_label="s$i", reference_label="baseline$i")
_source_spec(::Union{AbstractVector,Tuple}, i) = error("pass multiple default sources as separate arguments; use `reference => source` when the reference differs")

function _active_specs()
	DEFAULT_SPECS[] === nothing && error("no default source set; use set_default_source!(db) or pass the source explicitly")
	return DEFAULT_SPECS[]
end

_default_operator() = DEFAULT_OPERATOR[]
_default_periods() = DEFAULT_PERIODS[]

# `nothing` marks unassigned entries (distinct from explicit `missing` in read
# data) and must propagate through expression arithmetic: `p * q` with an
# unassigned `p[t]` should yield `nothing` at `t`, not a MethodError. Plain
# `nothing` has no arithmetic, so while JuMP evaluates an expression the
# unassigned entries are represented by the numeric sentinel `_NA`, which
# absorbs every operation; results are converted back to `nothing` afterwards.
struct _Unassigned <: Real end
const _NA = _Unassigned()

Base.promote_rule(::Type{_Unassigned}, ::Type{<:Real}) = _Unassigned
Base.convert(::Type{_Unassigned}, x::_Unassigned) = x
Base.convert(::Type{_Unassigned}, ::Real) = _NA
Base.zero(::Type{_Unassigned}) = _NA
Base.one(::Type{_Unassigned}) = _NA
Base.show(io::IO, ::_Unassigned) = print(io, "nothing")
Base.:(==)(::_Unassigned, ::_Unassigned) = true
Base.:(==)(::_Unassigned, ::Real) = false
Base.:(==)(::Real, ::_Unassigned) = false
Base.isless(::_Unassigned, ::_Unassigned) = false
Base.isless(::_Unassigned, ::Real) = false
Base.isless(::Real, ::_Unassigned) = false
for op in (:+, :-, :*, :/, :\, :^, :%, :min, :max)
	@eval Base.$op(::_Unassigned, ::_Unassigned) = _NA
end
for f in (:+, :-, :abs, :sqrt, :log, :log2, :log10, :log1p, :exp, :expm1, :inv)
	@eval Base.$f(::_Unassigned) = _NA
end

_nothing_to_na(x) = x === nothing ? _NA : x
_na_to_nothing(x) = x === _NA ? nothing : x
_restore_nothing(x) = _na_to_nothing(x)
_restore_nothing(x::AbstractArray) = _na_to_nothing.(x)

_to_float(x) = (x === nothing || x === _NA) ? NaN : Float64(x)

_as_numeric(x::Number) = Float64(x)
_as_numeric(x) = [_to_float(v) for v in Array(x)]

function _lag1(a::AbstractArray)
	out = similar(a, Float64)
	fill!(out, NaN)
	for I in CartesianIndices(a)
		I[ndims(a)] == 1 && continue
		prev = CartesianIndex(ntuple(d -> d == ndims(a) ? I[d] - 1 : I[d], ndims(a)))
		out[I] = a[prev]
	end
	return out
end
_lag1(::Number) = NaN

_dif(x) = (a = _as_numeric(x); a .- _lag1(a))
_pch(x) = (a = _as_numeric(x); (a ./ _lag1(a) .- 1) .* 100)
_gdif(x) = _dif(_pch(x))
_log(x) = log.(_as_numeric(x))
_ldif(x) = _dif(_log(x))

function _need_ref(op)
	op in (:m, :q, :mp, :r, :rn, :rd, :rp, :rdp, :rl, :rdl)
end

function _normalize_ops(ops)
	ops isa Symbol && return (ops,)
	return Tuple(ops)
end

function _expand_ops(ops)
	out = Symbol[]
	for op in _normalize_ops(ops)
		if op == :a
			append!(out, (:n, :p, :r, :rp))
		elseif op == :an
			append!(out, (:n, :r))
		elseif op == :ad
			append!(out, (:d, :rd))
		elseif op == :ap
			append!(out, (:p, :rp))
		elseif op == :adp
			append!(out, (:dp, :rdp))
		elseif op == :al
			append!(out, (:l, :rl))
		elseif op == :adl
			append!(out, (:dl, :rdl))
		else
			push!(out, op)
		end
	end
	return Tuple(out)
end

function _ref_value(ref, op)
	v = ref isa Function ? ref() : ref
	v === nothing && _need_ref(op) && error("operator :$op requires a reference source, e.g. @prt :$op baseline=>shock x")
	return v
end

"""Whether any operator in `ops` (after expanding aliases like `:a`) needs a reference source."""
_group_needs_ref(ops) = any(_need_ref, _expand_ops(ops))

# A `[expr1, expr2]` group evaluates to a MultiVarResult; apply the operator
# per element, pairing each with the matching element of the reference.
function _transform(op::Symbol, x::MultiVarResult, ref=nothing, name="")
	refv = ref isa Function ? ref() : ref
	vals = Tuple(_transform(op, v, refv isa MultiVarResult ? refv.values[i] : refv, x.names[i])
		for (i, v) in enumerate(x.values))
	return MultiVarResult(x.names, vals)
end

function _transform(op::Symbol, x, ref=nothing, name="")
	op in (:n, :abs) && return _relabel(x, x, name)
	op in (:d, :dif) && return _relabel(_dif(x), x, name)
	op in (:p, :pch) && return _relabel(_pch(x), x, name)
	op in (:dp, :gdif) && return _relabel(_gdif(x), x, name)
	op == :l && return _relabel(_log(x), x, name)
	op == :dl && return _relabel(_ldif(x), x, name)
	ref = _ref_value(ref, op)
	op == :m && return _relabel(_as_numeric(x) .- _as_numeric(ref), x, name)
	op == :q && return _relabel((_as_numeric(x) ./ _as_numeric(ref) .- 1) .* 100, x, name)
	op == :mp && return _relabel(_pch(x) .- _pch(ref), x, name)
	op in (:r, :rn) && return ref
	op == :rd && return _relabel(_dif(ref), ref, name)
	op == :rp && return _relabel(_pch(ref), ref, name)
	op == :rdp && return _relabel(_gdif(ref), ref, name)
	op == :rl && return _relabel(_log(ref), ref, name)
	op == :rdl && return _relabel(_ldif(ref), ref, name)
	error("unknown print operator :$op")
end

function _apply_ops(ops, x, ref=nothing, name="")
	os = _expand_ops(ops)
	length(os) == 1 && return _transform(only(os), x, ref, name)
	return Any[_transform(op, x, ref, name) for op in os]
end

"""Default y-axis label for an operator, e.g. `:m => "Difference from baseline"`. `nothing` when the operator doesn't imply a particular unit (e.g. `:n`)."""
const _OP_AXIS_LABELS = Dict(
	:d => "Difference",
	:dif => "Difference",
	:p => "Percent change",
	:pch => "Percent change",
	:dp => "Growth rate difference",
	:gdif => "Growth rate difference",
	:l => "Log",
	:dl => "Log difference",
	:m => "Difference from baseline",
	:q => "Percent deviation from baseline",
	:mp => "Growth rate difference from baseline",
	:rd => "Difference",
	:rp => "Percent change",
	:rdp => "Growth rate difference",
	:rl => "Log",
	:rdl => "Log difference",
)
_op_axis_label(op) = get(_OP_AXIS_LABELS, op, nothing)

function _lookup(db, name::Symbol, fallback, periods=nothing)
	x = if haskey(db.model, name)
		db.model[name]
	elseif haskey(db, String(name))
		db[name]
	else
		return fallback()
	end
	return _with_periods(x, periods)
end
_value(db, x) = _restore_nothing(JuMP.value(v -> _nothing_to_na(db[v]), x))
_value(db, x::AbstractArray{<:Number}) = x
_value(db, x::Tuple) = map(y -> _value(db, y), x)

_with_periods(x, periods) = periods === nothing ? x : _slice_periods(x, periods)
_slice_periods(x, periods) = x
_slice_periods(x::AbstractArray, periods) = x[ntuple(_ -> Colon(), ndims(x) - 1)..., periods]
_slice_periods(x::Window, periods) = x[ntuple(_ -> Colon(), ndims(x) - 1)..., periods]
_period_ref(base, periods, indices...) = periods === nothing ? base[indices...] : base[indices[1:end-1]..., periods]

# Only arithmetic operators broadcast implicitly (`a * b` -> `a .* b`). Named
# calls (`sum`, `log`, ...) are left as written, so reductions stay reductions;
# use explicit dots (e.g. `log.(x)`) for elementwise function application.
const _DOT_OPS = (:+, :-, :*, :/, :^, :%, :\)
_is_dot_macro(ex) = isexpr(ex, :macrocall) && ex.args[1] === Symbol("@__dot__")
_expand_dot_macro(ex) = _is_dot_macro(ex) ? macroexpand(@__MODULE__, ex) : ex

# `string(expr)` parenthesizes nested non-associative operators
# ("((x - x) - x) - x"), so we re-print arithmetic ourselves with minimal parens.
const _UNARY_PREC = Base.operator_precedence(:^) - 1  # unary minus: tighter than :*, looser than :^

_stripdot(op::Symbol) = (s = string(op); length(s) > 1 && s[1] == '.' ? Symbol(s[2:end]) : op)

function _format_index(arg)
	arg isa QuoteNode && return _format_index(arg.value)
	arg isa Symbol && return arg === :(:) ? ":" : ":$arg"
	return _expr_label(arg)
end

_paren(s, prec, min_prec) = prec < min_prec ? "($s)" : s

function _expr_label(ex, min_prec=0)
	ex isa QuoteNode && return _expr_label(ex.value, min_prec)
	ex isa Expr || return string(ex)
	ex = _expand_dot_macro(ex)
	if ex.head === :call
		op, args = ex.args[1], ex.args[2:end]
		base_op = op isa Symbol ? _stripdot(op) : op
		if base_op in (:+, :-) && length(args) == 1
			return _paren(string(op, _expr_label(args[1], _UNARY_PREC)), _UNARY_PREC, min_prec)
		end
		if base_op isa Symbol && length(args) >= 2 && base_op in (:+, :-, :*, :/, :\, :%, :^)
			prec = Base.operator_precedence(base_op)
			rassoc = base_op === :^
			parts = [_expr_label(a, i == (rassoc ? length(args) : 1) ? prec : prec + 1)
				for (i, a) in enumerate(args)]
			s = join(parts, base_op === :^ ? string(op) : " $op ")
			return _paren(s, prec, min_prec)
		end
		return string(_expr_label(op), "(", join(_expr_label.(args), ", "), ")")
	elseif ex.head === :ref
		return string(_expr_label(ex.args[1]), "[", join(_format_index.(ex.args[2:end]), ", "), "]")
	elseif ex.head === :.
		field = ex.args[2] isa QuoteNode ? ex.args[2].value : _expr_label(ex.args[2])
		return string(_expr_label(ex.args[1]), ".", field)
	elseif ex.head === :tuple
		return "(" * join(_expr_label.(ex.args), ", ") * ")"
	elseif ex.head === :vect
		return "[" * join(_expr_label.(ex.args), ", ") * "]"
	end
	return string(ex)
end

function _model_binding_expr(dbv, name::Symbol, periodv=nothing)
	lookup_ref = GlobalRef(@__MODULE__, :_lookup)
	return :($lookup_ref($dbv, $(QuoteNode(name)), () -> $name, $periodv))
end

"""Rewrite an expression AST so bare variable names prefer JuMP variables from `db.model`."""
function _rewrite(ex, dbv, periodv=nothing, bound=())
	ex isa Symbol && return ex in bound ? ex : _model_binding_expr(dbv, ex, periodv)
	ex isa Expr || return ex
	ex = _expand_dot_macro(ex)
	if ex.head === :ref
		base = ex.args[1]
		newbase = base isa Symbol && !(base in bound) ? _model_binding_expr(dbv, base) : _rewrite(base, dbv, periodv, bound)
		indices = Any[ex.args[2:end]...]
		if periodv !== nothing && !isempty(indices) && _is_colon_index(indices[end])
			return :($(GlobalRef(@__MODULE__, :_period_ref))($newbase, $periodv, $(indices...)))
		end
		return Expr(:ref, newbase, indices...)
	elseif ex.head === :call
		f = ex.args[1]
		args = Any[_rewrite(a, dbv, periodv, bound) for a in ex.args[2:end]]
		return f in _DOT_OPS ? Expr(:call, Symbol(".", f), args...) : Expr(:call, f, args...)
	elseif ex.head === :generator
		return _rewrite_generator(ex, dbv, periodv, bound)
	elseif ex.head === :.
		isexpr(ex.args[2], :tuple) || return ex
		return Expr(:., ex.args[1], Expr(:tuple, Any[_rewrite(a, dbv, periodv, bound) for a in ex.args[2].args]...))
	elseif ex.head === :$
		return ex.args[1]
	else
		return Expr(ex.head, Any[_rewrite(a, dbv, periodv, bound) for a in ex.args]...)
	end
end

function _rewrite_generator(ex, dbv, periodv, bound)
	new_args = Any[]
	current_bound = bound
	for arg in ex.args[2:end]
		if isexpr(arg, :(=))
			push!(new_args, Expr(:(=), arg.args[1], _rewrite(arg.args[2], dbv, periodv, current_bound)))
			current_bound = (current_bound..., _binding_symbols(arg.args[1])...)
		else
			push!(new_args, _rewrite(arg, dbv, periodv, current_bound))
		end
	end
	return Expr(:generator, _rewrite(ex.args[1], dbv, periodv, current_bound), new_args...)
end

_binding_symbols(x::Symbol) = (x,)
_binding_symbols(x::Expr) = Tuple(Symbol[s for a in x.args for s in _binding_symbols(a)])
_binding_symbols(_) = ()

_is_colon_index(x) = x === Symbol(":")

function _collect_bases(ex, acc=Symbol[], bound=())
	if ex isa Symbol
		ex in bound && return acc
		ex in acc || push!(acc, ex)
	elseif ex isa Expr
		ex = _expand_dot_macro(ex)
		if ex.head === :ref
			_collect_bases(ex.args[1], acc, bound)
		elseif ex.head === :call
			for a in ex.args[2:end]
				_collect_bases(a, acc, bound)
			end
		elseif ex.head === :generator
			_collect_generator_bases(ex, acc, bound)
		elseif ex.head === :.
			if isexpr(ex.args[2], :tuple)
				for a in ex.args[2].args
					_collect_bases(a, acc, bound)
				end
			end
		elseif ex.head === :$
		else
			for a in ex.args
				_collect_bases(a, acc, bound)
			end
		end
	end
	return acc
end

function _collect_generator_bases(ex, acc, bound)
	current_bound = bound
	for arg in ex.args[2:end]
		if isexpr(arg, :(=))
			_collect_bases(arg.args[2], acc, current_bound)
			current_bound = (current_bound..., _binding_symbols(arg.args[1])...)
		else
			_collect_bases(arg, acc, current_bound)
		end
	end
	return _collect_bases(ex.args[1], acc, current_bound)
end

function _value_expr(item, dbv, periodv)
	if isexpr(item, :vect) || (isexpr(item, :tuple) && length(item.args) > 1)
		items = Any[_value_expr(it, dbv, periodv) for it in item.args]
		names = String[_expr_label(it) for it in item.args]
		return :($(GlobalRef(@__MODULE__, :MultiVarResult))($names, ($(items...),)))
	end
	return :($(GlobalRef(@__MODULE__, :_value))($dbv, $(_rewrite(item, dbv, periodv))))
end

function _ref_expr(item, refv, periodv=nothing)
	refv === nothing && return nothing
	ex = _rewrite(item, refv, periodv)
	return :(() -> $(GlobalRef(@__MODULE__, :_value))($refv, $ex))
end

function _value_arg(expr, dbv, refv, periodv, ops, apply_ref)
	if isexpr(expr, :vect) || (isexpr(expr, :tuple) && length(expr.args) > 1)
		items = Any[_value_arg(it, dbv, refv, periodv, ops, apply_ref) for it in expr.args]
		names = String[_expr_label(it) for it in expr.args]
		multivar_ref = GlobalRef(@__MODULE__, :MultiVarResult)
		return :($multivar_ref($names, ($(items...),)))
	end
	ref = _ref_expr(expr, refv, periodv)
	expr_name = _expr_label(expr)
	return :($apply_ref($ops, $(_value_expr(expr, dbv, periodv)), $ref, $expr_name))
end

_is_op_literal(x::QuoteNode) = x.value isa Symbol
_is_op_literal(x::Expr) = isexpr(x, :vect) && all(_is_op_literal, x.args)
_is_op_literal(_) = false

_is_period_literal(x::Number) = true
_is_period_literal(x::Expr) = (isexpr(x, :call) && x.args[1] === :(:)) || (isexpr(x, :vect) && !_is_op_literal(x))
_is_period_literal(_) = false

function _macro_parts(args)
	default_operator = :(($(GlobalRef(@__MODULE__, :_default_operator))()))
	length(args) == 1 && return (default_operator, nothing, nothing, args[1], true, nothing)
	length(args) == 2 && _is_op_literal(args[1]) && return (args[1], nothing, nothing, args[2], true, nothing)
	length(args) == 2 && _is_period_literal(args[1]) && return (default_operator, nothing, nothing, args[2], true, args[1])
	length(args) == 2 && return (default_operator, args[1], nothing, args[2], false, nothing)
	length(args) == 3 && _is_op_literal(args[1]) && _is_period_literal(args[2]) && return (args[1], nothing, nothing, args[3], true, args[2])
	length(args) == 3 && _is_period_literal(args[1]) && return (default_operator, args[2], nothing, args[3], false, args[1])
	length(args) == 3 && return (args[1], args[2], nothing, args[3], false, nothing)
	length(args) == 4 && _is_op_literal(args[1]) && _is_period_literal(args[2]) && return (args[1], args[3], nothing, args[4], false, args[2])
	error("expected `expr`, `op expr`, `periods expr`, `db expr`, `op db expr`, or `op periods db expr`")
end

"""`reference => source` pair, e.g. `baseline => shock`, as used to supply a reference database."""
_is_pair_expr(x) = isexpr(x, :call) && length(x.args) == 3 && x.args[1] === :(=>)

function _db_parts(db, ref)
	_is_pair_expr(db) && return (db.args[3], db.args[2])
	return (db, ref)
end

"""
    _group_entry_expr(el, expr, periodv)

Build the `(bindings, entry)` pair for one element `el` of a `db` group (either
a plain source expression or a `reference => source` pair): `bindings` are
`let`-bindings for the gensym'd database variable(s), and `entry` constructs
the corresponding [`_GroupEntry`](@ref) evaluating `expr` against them.
"""
function _group_entry_expr(el, expr, periodv)
	entry_ref = GlobalRef(@__MODULE__, :_GroupEntry)
	dbv = gensym(:db)
	if _is_pair_expr(el)
		refexpr, srcexpr = el.args[2], el.args[3]
		refv = gensym(:ref)
		bindings = [:($(esc(dbv)) = $(esc(srcexpr))), :($(esc(refv)) = $(esc(refexpr)))]
		src_label = _table_label(string(srcexpr), expr)
		ref_label = _table_label(string(refexpr), expr)
		entry = esc(:($entry_ref($src_label, $(_value_expr(expr, dbv, periodv)), $ref_label, $(_value_expr(expr, refv, periodv)))))
	else
		bindings = [:($(esc(dbv)) = $(esc(el)))]
		src_label = _table_label(string(el), expr)
		entry = esc(:($entry_ref($src_label, $(_value_expr(expr, dbv, periodv)), nothing, nothing)))
	end
	return bindings, entry
end

"""Build the `@prt`/`@evalexpr` body for a `db` argument that is a `Pair` or a `Tuple` of sources/pairs (see [`_group_result`](@ref))."""
function _group_macro(dbargs, expr, ops, period_arg, periodv)
	bindings = Any[]
	entries = Any[]
	for el in dbargs
		bs, entry = _group_entry_expr(el, expr, periodv)
		append!(bindings, bs)
		push!(entries, entry)
	end
	group_result_ref = GlobalRef(@__MODULE__, :_group_result)
	entry_type_ref = GlobalRef(@__MODULE__, :_GroupEntry)
	return quote
		let $(bindings...), $(esc(periodv)) = $period_arg
			$group_result_ref($(esc(ops)), $entry_type_ref[$(entries...)])
		end
	end
end

function _eval_macro(args)
	ops, db, ref, expr, use_defaults, periods = _macro_parts(args)
	dbv = gensym(:db)
	refv = gensym(:ref)
	periodv = gensym(:periods)
	apply_ref = GlobalRef(@__MODULE__, :_apply_ops)
	default_periods_ref = GlobalRef(@__MODULE__, :_default_periods)
	period_arg = periods === nothing ? :($default_periods_ref()) : esc(periods)
	if use_defaults
		specsv = gensym(:defaults)
		specv = gensym(:spec)
		default_specs_ref = GlobalRef(@__MODULE__, :_active_specs)
		arg = _value_arg(expr, dbv, refv, periodv, ops, apply_ref)
		group_result_ref = GlobalRef(@__MODULE__, :_group_result)
		entry_type_ref = GlobalRef(@__MODULE__, :_GroupEntry)
		expr_label = _expr_label(expr)
		entry = :($entry_type_ref(
			string(getproperty($specv, :source_label), '\n', $expr_label),
			$(esc(_value_expr(expr, dbv, periodv))),
			string(getproperty($specv, :reference_label), '\n', $expr_label),
			$(esc(_value_expr(expr, refv, periodv)))))
		return quote
			let $specsv = $default_specs_ref(), $(esc(periodv)) = $period_arg
				if length($specsv) == 1
					let $specv = only($specsv), $(esc(dbv)) = getproperty($specv, :source), $(esc(refv)) = getproperty($specv, :reference)
						$(esc(arg))
					end
				else
					$group_result_ref($(esc(ops)), $entry_type_ref[
						let $(esc(dbv)) = getproperty($specv, :source), $(esc(refv)) = getproperty($specv, :reference)
							$entry
						end for $specv in $specsv
					])
				end
			end
		end
	end
	if _is_pair_expr(db) || (isexpr(db, :tuple) && length(db.args) >= 2)
		dbargs = isexpr(db, :tuple) ? db.args : Any[db]
		return _group_macro(dbargs, expr, ops, period_arg, periodv)
	end
	primary, ref = _db_parts(db, ref)
	refv = ref === nothing ? nothing : refv
	arg = _value_arg(expr, dbv, refv, periodv, ops, apply_ref)
	body = quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(periodv)) = $period_arg
			$(esc(arg))
		end
	end
	ref === nothing && return body
	return quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(refv)) = $(esc(ref)), $(esc(periodv)) = $period_arg
			$(esc(arg))
		end
	end
end

"""
    @evalexpr db expr
    @evalexpr op db expr
    @evalexpr periods expr
    @evalexpr op periods expr
    @evalexpr ops db [expr1, expr2, ...]
    @evalexpr op reference=>source expr
    @evalexpr op (source1, reference=>source2, ...) expr

Evaluate one or more model expressions, resolving bare names against `db`.

Use `reference => source` in place of `db` to supply a reference for operators
that need one (e.g. `:q`, `:m`); `db` alone (no operator that needs a
reference) or a `Tuple` of sources/pairs instead evaluate `expr` against each
database and return a [`MultiVarResult`](@ref) with one column per database
(a reference shared by several pairs, e.g. a common baseline, is only shown
once).
"""
macro evalexpr(args...)
	return _eval_macro(args)
end

"""
    @prt db expr
    @prt op db expr
    @prt periods expr
    @prt op periods expr
    @prt ops db [expr1, expr2, ...]
    @prt op reference=>source expr
    @prt op (source1, reference=>source2, ...) expr

Evaluate a model expression for display in the REPL. This is an alias for
[`@evalexpr`](@ref), so the returned value is what gets printed by the caller.
"""
macro prt(args...)
	return _eval_macro(args)
end

end
