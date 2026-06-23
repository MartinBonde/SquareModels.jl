# ModelExpressions - Evaluate ModelDictionary expressions without plotting

module ModelExpressions

using Base.Meta: isexpr

export @evalexpr, @prt
export set_default_source!, set_default_operator!, reset_print_defaults!

const DEFAULT_SPECS = Ref{Any}(nothing)
const DEFAULT_OPERATOR = Ref{Any}(:n)

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

"""Clear interactive print/plot defaults."""
function reset_print_defaults!()
	DEFAULT_SPECS[] = nothing
	DEFAULT_OPERATOR[] = :n
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

_to_float(x) = x === nothing ? NaN : Float64(x)

_as_numeric(x::Number) = Float64(x)
_as_numeric(x) = [_to_float(v) for v in collect(x)]

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

_has_source_binding(db, name::Symbol) = haskey(db, String(name)) || haskey(db.model, name)
_lookup(db, name::Symbol, fallback) = _has_source_binding(db, name) ? db[name] : fallback()

const _DOTTABLE_OPS = (:+, :-, :*, :/, :^, :%, :\)

"""Rewrite an expression AST so bare variable names prefer `db[:name]` lookups."""
function _rewrite(ex, dbv)
	lookup_ref = GlobalRef(@__MODULE__, :_lookup)
	ex isa Symbol && return :($lookup_ref($dbv, $(QuoteNode(ex)), () -> $ex))
	ex isa Expr || return ex
	if ex.head === :ref
		base = ex.args[1]
		newbase = base isa Symbol ? :($lookup_ref($dbv, $(QuoteNode(base)), () -> $base)) : _rewrite(base, dbv)
		return Expr(:ref, newbase, ex.args[2:end]...)
	elseif ex.head === :call
		f = ex.args[1]
		args = Any[_rewrite(a, dbv) for a in ex.args[2:end]]
		f in _DOTTABLE_OPS && return Expr(:call, Symbol(".", f), args...)
		return Expr(:call, f, args...)
	elseif ex.head === :.
		return ex
	elseif ex.head === :$
		return ex.args[1]
	else
		return Expr(ex.head, Any[_rewrite(a, dbv) for a in ex.args]...)
	end
end

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
		else
			for a in ex.args
				_collect_bases(a, acc)
			end
		end
	end
	return acc
end

_value_expr(item, dbv) = _rewrite(item, dbv)

function _ref_expr(item, refv)
	refv === nothing && return nothing
	ex = _rewrite(item, refv)
	return :(() -> $ex)
end

function _value_arg(expr, dbv, refv, ops, apply_ref)
	if isexpr(expr, :vect)
		items = Any[_value_arg(it, dbv, refv, ops, apply_ref) for it in expr.args]
		return Expr(:vect, items...)
	end
	ref = _ref_expr(expr, refv)
	return :($apply_ref($ops, $(_value_expr(expr, dbv)), $ref))
end

_is_op_literal(x::QuoteNode) = x.value isa Symbol
_is_op_literal(x::Expr) = isexpr(x, :vect) && all(_is_op_literal, x.args)
_is_op_literal(_) = false

function _macro_parts(args)
	default_operator = :(($(GlobalRef(@__MODULE__, :_default_operator))()))
	length(args) == 1 && return (default_operator, nothing, nothing, args[1], true)
	length(args) == 2 && _is_op_literal(args[1]) && return (args[1], nothing, nothing, args[2], true)
	length(args) == 2 && return (default_operator, args[1], nothing, args[2], false)
	length(args) == 3 && return (args[1], args[2], nothing, args[3], false)
	error("expected `expr`, `op expr`, `db expr`, or `ops db expr`")
end

function _db_parts(db, ref)
	isexpr(db, :tuple) && length(db.args) == 2 && return (db.args[1], db.args[2])
	return (db, ref)
end

function _eval_macro(args)
	ops, db, ref, expr, use_defaults = _macro_parts(args)
	dbv = gensym(:db)
	refv = gensym(:ref)
	apply_ref = GlobalRef(@__MODULE__, :_apply_ops)
	if use_defaults
		specsv = gensym(:defaults)
		specv = gensym(:spec)
		default_specs_ref = GlobalRef(@__MODULE__, :_active_specs)
		arg = _value_arg(expr, dbv, refv, ops, apply_ref)
		return quote
			let $specsv = $default_specs_ref()
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
	arg = _value_arg(expr, dbv, refv, ops, apply_ref)
	body = quote
		let $(esc(dbv)) = $(esc(primary))
			$(esc(arg))
		end
	end
	ref === nothing && return body
	return quote
		let $(esc(dbv)) = $(esc(primary)), $(esc(refv)) = $(esc(ref))
			$(esc(arg))
		end
	end
end

"""
    @evalexpr db expr
    @evalexpr op db expr
    @evalexpr ops db [expr1, expr2, ...]

Evaluate one or more model expressions, resolving bare names against `db`.
"""
macro evalexpr(args...)
	return _eval_macro(args)
end

"""
    @prt db expr
    @prt op db expr
    @prt ops db [expr1, expr2, ...]

Evaluate a model expression for display in the REPL. This is an alias for
[`@evalexpr`](@ref), so the returned value is what gets printed by the caller.
"""
macro prt(args...)
	return _eval_macro(args)
end

end
