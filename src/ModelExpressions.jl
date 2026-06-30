# ModelExpressions - Evaluate ModelDictionary expressions without plotting

module ModelExpressions

using Base.Meta: isexpr
using JuMP: JuMP
import ..Window

export @evalexpr, @prt
export set_default_source!, set_default_operator!, set_default_periods!, reset_print_defaults!

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
	specs = _source_spec.(sources)
	DEFAULT_SPECS[] = specs
	return specs
end

"""Set the default operator used when no operator is given."""
function set_default_operator!(op)
	DEFAULT_OPERATOR[] = op
	return op
end

"""Set the default final-dimension periods used by `@prt`, `@evalexpr`, and `@plot`."""
function set_default_periods!(periods)
	DEFAULT_PERIODS[] = periods
	return periods
end

"""Clear interactive print/plot defaults."""
function reset_print_defaults!()
	DEFAULT_SPECS[] = nothing
	DEFAULT_OPERATOR[] = :n
	DEFAULT_PERIODS[] = nothing
	return nothing
end

_source_spec(source) = (; source, reference=source)
_source_spec(pair::Pair) = (; source=pair.second, reference=pair.first)
_source_spec(::Union{AbstractVector,Tuple}) = error("pass multiple default sources as separate arguments; use `reference => source` when the reference differs")

function _active_specs()
	DEFAULT_SPECS[] === nothing && error("no default source set; use set_default_source!(db) or pass the source explicitly")
	return DEFAULT_SPECS[]
end

_default_operator() = DEFAULT_OPERATOR[]
_default_periods() = DEFAULT_PERIODS[]

_to_float(x) = x === nothing ? NaN : Float64(x)

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
	v === nothing && _need_ref(op) && error("operator :$op requires a reference source, e.g. @prt :$op (shock, baseline) x")
	return v
end

function _transform(op::Symbol, x, ref=nothing)
	op in (:n, :abs) && return x
	op in (:d, :dif) && return _dif(x)
	op in (:p, :pch) && return _pch(x)
	op in (:dp, :gdif) && return _gdif(x)
	op == :l && return _log(x)
	op == :dl && return _ldif(x)
	ref = _ref_value(ref, op)
	op == :m && return _as_numeric(x) .- _as_numeric(ref)
	op == :q && return (_as_numeric(x) ./ _as_numeric(ref) .- 1) .* 100
	op == :mp && return _pch(x) .- _pch(ref)
	op in (:r, :rn) && return ref
	op == :rd && return _dif(ref)
	op == :rp && return _pch(ref)
	op == :rdp && return _gdif(ref)
	op == :rl && return _log(ref)
	op == :rdl && return _ldif(ref)
	error("unknown print operator :$op")
end

function _apply_ops(ops, x, ref=nothing)
	os = _expand_ops(ops)
	length(os) == 1 && return _transform(only(os), x, ref)
	return Any[_transform(op, x, ref) for op in os]
end

_op_label(label, op) = op == :n ? label : "$label <$op>"

_lookup(db, name::Symbol, fallback) = haskey(db.model, name) ? db.model[name] : (haskey(db, String(name)) ? db[name] : fallback())
_value(db, x) = JuMP.value(v -> db[v], x)
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

function _model_binding_expr(dbv, name::Symbol, periodv=nothing)
	lookup_ref = GlobalRef(@__MODULE__, :_lookup)
	return _period_expr(:($lookup_ref($dbv, $(QuoteNode(name)), () -> $name)), periodv)
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
_period_expr(ex, ::Nothing) = ex
_period_expr(ex, periodv) = :($(GlobalRef(@__MODULE__, :_with_periods))($ex, $periodv))

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

_value_expr(item, dbv, periodv) = :($(GlobalRef(@__MODULE__, :_value))($dbv, $(_rewrite(item, dbv, periodv))))

function _ref_expr(item, refv, periodv=nothing)
	refv === nothing && return nothing
	ex = _rewrite(item, refv, periodv)
	return :(() -> $(GlobalRef(@__MODULE__, :_value))($refv, $ex))
end

function _value_arg(expr, dbv, refv, periodv, ops, apply_ref)
	if isexpr(expr, :vect)
		items = Any[_value_arg(it, dbv, refv, periodv, ops, apply_ref) for it in expr.args]
		return Expr(:vect, items...)
	end
	ref = _ref_expr(expr, refv, periodv)
	return :($apply_ref($ops, $(_value_expr(expr, dbv, periodv)), $ref))
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

function _db_parts(db, ref)
	isexpr(db, :tuple) && length(db.args) == 2 && return (db.args[1], db.args[2])
	return (db, ref)
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
		return quote
			let $specsv = $default_specs_ref(), $(esc(periodv)) = $period_arg
				if length($specsv) == 1
					let $specv = only($specsv), $(esc(dbv)) = getproperty($specv, :source), $(esc(refv)) = getproperty($specv, :reference)
						$(esc(arg))
					end
				else
					Any[let $(esc(dbv)) = getproperty($specv, :source), $(esc(refv)) = getproperty($specv, :reference)
						$(esc(arg))
					end for $specv in $specsv]
				end
			end
		end
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

Evaluate one or more model expressions, resolving bare names against `db`.
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

Evaluate a model expression for display in the REPL. This is an alias for
[`@evalexpr`](@ref), so the returned value is what gets printed by the caller.
"""
macro prt(args...)
	return _eval_macro(args)
end

end
