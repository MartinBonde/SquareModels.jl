# Copyright 2022, Martin Kirk Bonde and contributors
# Licensed under the MIT License. See LICENSE.md for details.

"""
SquareModels
A JuMP extension for writing modular models with square systems of equations
"""
module SquareModels

export @block, Block, Equation, @endo_exo_swap!, @variables, add_equation, add_equation!
export endogenous, residuals, residual, variables, exogenous, is_endogenous, overlaps, shared_endogenous
export VariableRef  # Re-exported from JuMP for macro hygiene
export ModelDictionary, fix, unfix, set_start_value, value, value_dict, add_missing_model_variables!
export keys_match, assert_no_diff, assert_residuals_small
export SquareModelError, ResidualError, ToleranceError, NonSquareError
export unload, load, read_indices, read_sparse_array, read_variable
export RESIDUAL_SUFFIX
export solve, solve!, diagnose, annotate_lst!, square_model
export Tag, description, tags, has_tag, tagged, metadata
export SparseZeroArray, ∑, use_sparse_zero_array!
export ModelExpressions, ModelPlotting, @plot, @evalexpr, @prt, plotvar, plotseries, labeled, LabeledSeries, LabeledArray, MultiVarResult, AbstractSeries
export set_default_source!, set_default_operator!, set_default_periods!, set_column_label_total_width!, reset_print_defaults!

RESIDUAL_SUFFIX = "_J"  # Suffix for residual variables (J for "junk" or adjustment)

# ----------------------------------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------------------------------
using Base.Meta: isexpr
using StatsBase: countmap
using Lazy
using JuMP: JuMP, AbstractModel, AbstractVariableRef, VariableRef, Containers
using JuMP: AffExpr, QuadExpr, NonlinearExpr
using JuMP.Containers: DenseAxisArray, SparseAxisArray
using JuMP: @variable
using JuMP: set_name, name, fix, is_fixed, unfix, all_variables, value, set_start_value
import MathOptInterface as MOI
const _name_lookup_cache = WeakKeyDict{AbstractModel, Dict{String, VariableRef}}()

include("errors.jl")
include("utils.jl")
include("SparseZeroArrays.jl")

"""
    AbstractSeries

Supertype for labelled, plottable data series. Concrete subtypes — `Window`
(a view onto model data) and `LabeledSeries` (a single eager, computed line) —
implement `ModelPlotting.expand(s) -> Vector{LabeledSeries}`, which splits the data
into one labelled line per leading-index combination (the last dimension is the
x-axis, e.g. `y[region, year]` becomes one line per region over the years).
Dispatch on `AbstractSeries` to write plotting code that accepts either.
"""
abstract type AbstractSeries end

"""
    Equation

Lightweight storage for a model equation: the expression and its constraint set.
Unlike `ConstraintRef`, this does NOT register with JuMP/MOI, avoiding backend bridging overhead.
"""
struct Equation
	func::Any
	set::MOI.AbstractScalarSet
end

collect_variables!(vars::Set{VariableRef}, eq::Equation) = collect_variables!(vars, eq.func)

"""
    collect_variables!(vars::Set{VariableRef}, expr) → Set{VariableRef}

Recursively collect all VariableRef objects from a JuMP expression.
Works with AffExpr (linear), QuadExpr (quadratic), and NonlinearExpr (nonlinear).
"""
function collect_variables!(vars::Set{VariableRef}, expr)
    if expr isa VariableRef
        push!(vars, expr)
    elseif expr isa AffExpr
        union!(vars, keys(expr.terms))
    elseif expr isa QuadExpr
        union!(vars, keys(expr.aff.terms))
        for (pair, _) in expr.terms
            push!(vars, pair.a)
            push!(vars, pair.b)
        end
    elseif expr isa NonlinearExpr
        for arg in expr.args
            collect_variables!(vars, arg)
        end
    end
    return vars
end
collect_variables!(vars::Set{VariableRef}, ::Union{Number, Zero}) = vars

"""
    Block

A mapping between equations and their associated endogenous variables in a JuMP model.

Blocks represent "square" systems where each equation is paired with exactly one
variable, enabling modular model construction and endo-exo swaps (changing which
variable is determined by which equation).

Blocks store `Equation` objects (lightweight func + set pairs). When using `solve`,
these equations are transformed (substituting exogenous values from the data) and
added to an intermediate solve model.

# Fields
- `model::AbstractModel`: The JuMP model containing the variables
- `endogenous::Vector{VariableRef}`: Vector of endogenous variable references
- `residuals::Vector{VariableRef}`: Vector of residual variable references
- `variables::Set{VariableRef}`: All variables appearing in the block's equations
- `_endogenous_set::Set{VariableRef}`: Set for O(1) membership checking of endogenous variables
- `equations::Vector{Equation}`: Equation expressions (func + set pairs)

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

length(b)  # 4 (one scalar + three indexed)
x ∈ b      # true
```

See also: [`@block`](@ref), [`@endo_exo_swap!`](@ref), [`endogenous`](@ref), [`variables`](@ref), [`solve`](@ref)
"""
struct Block
	model::AbstractModel
	endogenous::Vector{VariableRef}
	residuals::Vector{VariableRef}
	variables::Set{VariableRef}
	_endogenous_set::Set{VariableRef}
	equations::Vector{Equation}

	function Block(
		model::AbstractModel,
		endogenous::Vector{VariableRef},
		residuals::Vector{VariableRef},
		variables::Set{VariableRef},
		equations::Vector{Equation}
	)
		length(equations) == length(endogenous) ||
			error("Block must be square: got $(length(equations)) equations and $(length(endogenous)) endogenous variables")

		endogenous_set = Set{VariableRef}(endogenous)
		if length(endogenous_set) != length(endogenous)
			display(non_unqiue_pairs(endogenous, equations))
			error("Non-unique mapping between endogenous variables and equations in block definition.\n" *
			      "See non-unique mappings above.")
		end

		new(model, endogenous, residuals, variables, endogenous_set, equations)
	end
end

Block(model) = Block(model, VariableRef[], VariableRef[], Set{VariableRef}(), Equation[])

Base.length(b::Block) = length(b.endogenous)
Base.iterate(b::Block) = iterate(b.endogenous)
Base.iterate(b::Block, state) = iterate(b.endogenous, state)
Base.copy(b::Block) = Block(b.model, copy(b.endogenous), copy(b.residuals), copy(b.variables), copy(b.equations))

"""
    is_endogenous(var::VariableRef, b::Block) → Bool

Check if a variable is endogenous in the block (i.e., has an associated constraint).
Uses O(1) set lookup.

# Example
```julia
b = @block model begin
    x, x == 1
end
is_endogenous(x, b)  # true
```
"""
is_endogenous(var::VariableRef, b::Block) = var ∈ b._endogenous_set

"""
    var ∈ block → Bool

Check if a variable appears in the block's constraints (either as endogenous or exogenous).
Uses O(1) set lookup.

See also: [`is_endogenous`](@ref) to check specifically for endogenous variables.
"""
Base.in(var::VariableRef, b::Block) = var ∈ b.variables

"""
    endogenous(b::Block) → Vector{VariableRef}

Return the vector of endogenous variable references in the block.

These are the variables being solved for - each paired with a constraint.

# Arguments
- `b::Block`: The block to get endogenous variables from

# Returns
A vector of `VariableRef` objects representing all endogenous variables in the block.
The order corresponds to the order in which constraints were defined.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

for v in endogenous(b)
    println(name(v))
end
```

See also: [`constraints`](@ref), [`variables`](@ref), [`exogenous`](@ref)
"""
endogenous(b::Block) = b.endogenous

"""
    residuals(b::Block) → Vector{VariableRef}

Return the residual variables corresponding to each endogenous variable in the block.

Residual variables are automatically created when defining blocks and are named
with the suffix defined by `RESIDUAL_SUFFIX` (default "_J"). They are fixed to 0
by default and can be used to:
- Check for data inconsistencies (unfix residual, fix endo, solve, check residual value)
- Temporarily disable equations (exogenize endo, endogenize residual)
- Debug model issues

# Arguments
- `b::Block`: The block to get residuals from

# Returns
A vector of `VariableRef` objects representing the residual variables.
The order corresponds to `endogenous(b)`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

res = residuals(b)
# res[1] is x_J, res[2:4] are y_J[1], y_J[2], y_J[3]
```

See also: [`endogenous`](@ref), [`constraints`](@ref), [`residuals(::AbstractModel)`](@ref)
"""
residuals(b::Block) = b.residuals


"""
    variables(b::Block) → Vector{VariableRef}

Return a vector of all variables that appear in the block's constraints.

This includes both endogenous variables (being solved for) and exogenous variables
(parameters to this block). Only variables that are actually used in the constraint
expressions are included - unused indices are not present.

# Arguments
- `b::Block`: The block to get variables from

# Returns
A `Vector{VariableRef}` of all variables referenced in the block's constraints.

See also: [`endogenous`](@ref), [`exogenous`](@ref)
"""
variables(b::Block) = collect(b.variables)

"""
    exogenous(b::Block) → Vector{VariableRef}

Return a vector of exogenous variables that appear in the block's constraints.

These are variables that are referenced in the constraint expressions but are not
endogenous (not being solved for) within this block. Only variables that are
actually used are included - unused variable indices are not present.

# Arguments
- `b::Block`: The block to get exogenous variables from

# Returns
A `Vector{VariableRef}` of all exogenous variables referenced in the block.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])
@variable(model, z[1:3])

b = @block model begin
    x, x == sum(y[i] for i in 1:3)
    z[i ∈ 1:3], z[i] == y[i] * 2
end

exo = exogenous(b)  # Contains y[1], y[2], y[3]
```

See also: [`endogenous`](@ref), [`variables`](@ref)
"""
exogenous(b::Block) = collect(setdiff(b.variables, b._endogenous_set))

"""
    residuals(model::AbstractModel) → Vector{VariableRef}

Return all residual variables in the model.

Residual variables are identified by their name suffix (defined by `RESIDUAL_SUFFIX`,
default "_J"). This function collects all such variables from the model.

# Arguments
- `model::AbstractModel`: The JuMP model to search for residual variables

# Returns
A vector of `VariableRef` objects representing all residual variables in the model.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

res = residuals(model)
# Returns [x_J, y_J[1], y_J[2], y_J[3]]
```

See also: [`residuals(::Block)`](@ref), [`RESIDUAL_SUFFIX`](@ref)
"""
residuals(model::AbstractModel) = filter(v -> endswith(base_name(v), RESIDUAL_SUFFIX), all_variables(model))

"""
    Base.summary(io::IO, b::Block)

Print a one-line summary of a block showing the number of equations and variables.

# Arguments
- `io::IO`: The IO stream to print to
- `b::Block`: The block to summarize

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

summary(stdout, b)  # prints: "Block with 4 equations over 4 variables"
```

See also: [`Block`](@ref)
"""
function Base.summary(io::IO, b::Block)
	n = length(b)
	print(io, "Block with $n equations over $n variables")
end

"""
    overlaps(a::Block, b::Block) → Bool

Check if two blocks share any common variables.

Returns `true` if any variable appears in both blocks, which may indicate
duplicate equations or intentional variable sharing across model components.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
`true` if the blocks have at least one variable in common, `false` otherwise

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b1 = @block model begin
    x, x == 1
    y[i ∈ 1:2], y[i] == i
end

b2 = @block model begin
    y[i ∈ 2:3], y[i] == i  # y[2] appears in both blocks
end

overlaps(b1, b2)  # true
```

See also: [`shared_endogenous`](@ref), [`Block`](@ref)
"""
overlaps(a::Block, b::Block) = !isempty(intersect(a._endogenous_set, b._endogenous_set))

"""
    shared_endogenous(a::Block, b::Block) → Vector{VariableRef}

Return the endogenous variables that appear in both blocks.

Useful for understanding how blocks are interconnected and for detecting
accidental duplicate equations.

# Arguments
- `a::Block`: First block
- `b::Block`: Second block

# Returns
A vector of `VariableRef` objects that are endogenous in both blocks (may be empty)

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b1 = @block model begin
    x, x == 1
    y[i ∈ 1:2], y[i] == i
end

b2 = @block model begin
    y[i ∈ 2:3], y[i] == i
end

shared = shared_endogenous(b1, b2)  # [y[2]]
y[2] ∈ shared  # true
y[1] ∈ shared  # false
```

See also: [`overlaps`](@ref), [`Block`](@ref)
"""
shared_endogenous(a::Block, b::Block) = collect(intersect(a._endogenous_set, b._endogenous_set))

"""Format variables grouped by base name for readable error messages."""
function format_variables(vars::AbstractVector{VariableRef})
	groups = Dict{String, Vector{VariableRef}}()
	for var in vars
		bn = base_name(var)
		push!(get!(groups, bn, VariableRef[]), var)
	end

	lines = String[]
	for (bn, group) in sort(collect(groups), by=first)
		if length(group) == 1
			push!(lines, "  $bn: $(group[1])")
		else
			examples = string.(group[1:min(3, length(group))])
			examples_str = join(examples, ", ")
			if length(group) > 3
				examples_str *= ", ..."
			end
			push!(lines, "  $bn: $(length(group)) elements (e.g., $examples_str)")
		end
	end
	return join(lines, "\n")
end

function Block(
	model::AbstractModel,
	endogenous::AbstractArray{V},
	residuals::AbstractArray{R},
	variables::Set{VariableRef},
	equations::Vector{Equation}
) where {V<:VariableRef, R<:VariableRef}
	Block(model, VariableRef[endogenous...], VariableRef[residuals...], variables, equations)
end

function Base.:+(a::Block, b::Block)
	a.model == b.model || error("Cannot add $a and $b. Blocks must belong to the same model.")

	if overlaps(a, b)
		shared = shared_endogenous(a, b)
		formatted = format_variables(shared)
		error("Cannot combine blocks: $(length(shared)) endogenous variable(s) appear in both blocks.\n" *
		      "Overlapping endogenous variables:\n$formatted\n" *
		      "This would create a non-square system with more constraints than unique endogenous variables.")
	end

	combined_vars = union(a.variables, b.variables)
	combined_eqs = vcat(a.equations, b.equations)
	Block(a.model, vcat(a.endogenous, b.endogenous), vcat(a.residuals, b.residuals), combined_vars, combined_eqs)
end

function Base.:-(a::Block, b::Block)
	a.model == b.model || error("Cannot subtract $b from $a. Blocks must belong to the same model.")
	mask = [v ∉ b._endogenous_set for v in a.endogenous]
	if !any(mask)
		return Block(a.model)
	end
	filtered_eqs = a.equations[mask]

	all_vars = Set{VariableRef}()
	for eq in filtered_eqs
		collect_variables!(all_vars, eq.func)
	end

	Block(a.model, a.endogenous[mask], a.residuals[mask], all_vars, filtered_eqs)
end

make_residual_name(var) = string(var) * SquareModels.RESIDUAL_SUFFIX

"""Cached version of JuMP.variable_by_name — O(1) after first call per model."""
function variable_by_name(model::AbstractModel, var_name::AbstractString)
	lookup = get!(_name_lookup_cache, model) do
		Dict{String, VariableRef}(name(v) => v for v in all_variables(model))
	end
	key = String(var_name)
	v = get(lookup, key, nothing)
	v !== nothing && return v
	for v in all_variables(model)
		n = name(v)
		haskey(lookup, n) || (lookup[n] = v)
	end
	return get(lookup, key, nothing)
end

"""
    add_equation(model, endo::VariableRef, lhs, rhs=0)

Create a Block with a single endogenous variable and its equation.

Creates a residual variable (fixed to 0) and the equation `lhs + resid == rhs`.
The residual follows the standard naming convention (endo name + RESIDUAL_SUFFIX).

This is the runtime equivalent of a single line in a `@block` macro - use it when you
need to programmatically add endogenous/equation pairs with runtime variable references.

# Arguments
- `model`: The JuMP model (or ModelDictionary)
- `endo::VariableRef`: The variable to make endogenous
- `lhs`: Left-hand side expression
- `rhs`: Right-hand side (defaults to 0)

# Returns
A Block containing the endogenous/equation pair, which can be merged with other blocks using `+`.

# Examples
```julia
block = block + add_equation(model, x[t], x[t], x[t+1])    # x[t] == x[t+1]
block = block + add_equation(model, x[t], x[t] - x[t+1])   # x[t] - x[t+1] == 0
```
"""
function add_equation(model, endo::VariableRef, lhs, rhs=0)
	m = _get_model(model)
	var_name = name(endo)

	resid = @variable(m, base_name = make_residual_name(var_name))
	fix(resid, 0)

	eq = Equation(lhs - rhs + resid, MOI.EqualTo(0.0))

	all_vars = Set{VariableRef}([endo, resid])
	collect_variables!(all_vars, lhs)
	collect_variables!(all_vars, rhs)

	Block(m, [endo], [resid], all_vars, Equation[eq])
end

function add_equation!(block::Block, endo::VariableRef, lhs, rhs=0)
	endo ∉ block._endogenous_set || error("Cannot add equation: $(name(endo)) is already endogenous in this block.")
	eq_block = add_equation(block.model, endo, lhs, rhs)
	append!(block.endogenous, eq_block.endogenous)
	append!(block.residuals, eq_block.residuals)
	append!(block.equations, eq_block.equations)
	union!(block.variables, eq_block.variables)
	push!(block._endogenous_set, endo)
	return block
end

"""Helper function to extract base name from variable reference"""
_get_name(s::Symbol) = s
_get_name(e::Expr) = e.args[1]

_index_symbol(spec::Symbol) = spec
function _index_symbol(spec::Expr)
	if spec.head in (:(=), :kw)
		spec.args[1]
	elseif spec.head == :call && spec.args[1] in (:∈, :in)
		spec.args[2]
	else
		spec
	end
end

"""
Build the substitution target AST from the block's ref_vars specification.
`x[t ∈ 2:3, s ∈ 1:2]` → `:(x[t, s])`, scalar `x` → `:x`.
Tuple destructuring `x[(i,d) = keys, t ∈ 1:3]` → `:(x[i, d, t])` (flattened).
Semicolon conditions `x[i ∈ 1:3; cond]` → `:(x[i])` (condition stripped).
"""
function _substitution_target(ref_vars)
	base_sym = _get_name(ref_vars)
	if isexpr(ref_vars, :ref)
		index_symbols = Any[]
		for spec in ref_vars.args[2:end]
			isexpr(spec, :parameters) && continue
			sym = _index_symbol(spec)
			if isexpr(sym, :tuple)
				append!(index_symbols, sym.args)
			else
				push!(index_symbols, sym)
			end
		end
		Expr(:ref, base_sym, index_symbols...)
	else
		base_sym
	end
end

"""
Replace occurrences of `target` in `expr` with `(target + model[residual_sym][indices])`.
Handles both scalar references like `x` and indexed references like `x[i,j]`.
"""
_substitute_with_residual(expr, target, model_sym, residual_sym::Symbol) = expr

function _substitute_with_residual(expr::Symbol, target::Symbol, model_sym, residual_sym::Symbol)
	expr == target ? :($expr + $model_sym[$(QuoteNode(residual_sym))]) : expr
end

_is_ref_match(expr::Expr, target::Symbol) = expr.head == :ref && expr.args[1] == target
function _is_ref_match(expr::Expr, target::Expr)
	expr.head == :ref || return false
	expr == target && return true
	_flatten_ref(expr) == _flatten_ref(target)
end

"""Flatten tuple indices in a :ref expression for comparison.
`:(x[(i,d), t])` → `:(x[i, d, t])`"""
function _flatten_ref(e::Expr)
	e.head == :ref || return e
	args = Any[e.args[1]]
	for a in e.args[2:end]
		isexpr(a, :tuple) ? append!(args, a.args) : push!(args, a)
	end
	Expr(:ref, args...)
end

function _substitute_with_residual(expr::Expr, target, model_sym, residual_sym::Symbol)
	if _is_ref_match(expr, target)
		indices = expr.args[2:end]
		residual_access = Expr(:ref, :($model_sym[$(QuoteNode(residual_sym))]), indices...)
		:($expr + $residual_access)
	else
		new_args = [_substitute_with_residual(arg, target, model_sym, residual_sym) for arg in expr.args]
		Expr(expr.head, new_args...)
	end
end

"""Collect index tuples from a JuMP container in iteration order."""
_all_keys(c::AbstractArray) = vec(collect(Iterators.product(axes(c)...)))
_all_keys(c::SparseAxisArray) = keys(c.data)
_all_keys(c::SparseZeroArray) = _all_keys(c.data)

"""Flatten nested tuples in a key.
E.g. `((:a, :b), 1)` becomes `(:a, :b, 1)`."""
_flatten_key(k::Tuple) = tuple(Iterators.flatten(map(x -> x isa Tuple ? x : (x,), k))...)

"""Index a variable with a constraint key, flattening only when the variable has more dimensions.
Handles the difference between `x[a,b,t]` (3D) and `y[(a,b),t]` (2D with tuple index)."""
_ndims(v::SparseAxisArray{T,N}) where {T,N} = N
_ndims(v::SparseZeroArray{T,N}) where {T,N} = N
_ndims(v::AbstractArray) = ndims(v)
function _index_var(var, k::Tuple)
	fk = _flatten_key(k)
	fk === k && return var[k...]
	_ndims(var) == length(fk) ? var[fk...] : var[k...]
end

"""Extract the JuMP model from a container (ModelDictionary or Model)"""
_get_model(m::AbstractModel) = m
# _get_model for ModelDictionary is defined after ModelDictionaries.jl is included

"""Split `lhs == rhs` into `(lhs - rhs)` expression at the AST level."""
function _equality_to_diff(expr::Expr)
	if expr.head == :call && expr.args[1] == :(==) && length(expr.args) == 3
		lhs, rhs = expr.args[2], expr.args[3]
		return :($lhs - ($rhs))
	end
	Expr(expr.head, [_equality_to_diff(a) for a in expr.args]...)
end
_equality_to_diff(x) = x

"""Helper macro for Block macro - returns (endogenous, residuals, equations) where equations are vectors parallel to endogenous"""
macro _block(container, ref_vars, constraint, extra...)
	_error(str...) = JuMP._macro_error(:block, (container, ref_vars, constraint, extra...), __source__, str...)
	sm = @__MODULE__
	jump_expression = GlobalRef(JuMP, Symbol("@expression"))
	get_model = GlobalRef(sm, :_get_model)
	copy_variable_ref = GlobalRef(sm, :copy_variable)
	equation_ref = GlobalRef(sm, :Equation)
	all_keys_ref = GlobalRef(sm, :_all_keys)
	index_var_ref = GlobalRef(sm, :_index_var)
	equal_to_ref = GlobalRef(MOI, :EqualTo)
	code = Expr(:block)
	base_sym = _get_name(ref_vars)
	residual_name = make_residual_name(base_sym)
	residual_symbol = Symbol(residual_name)

	model_expr = :($get_model($container))

	push!(code.args, :($copy_variable_ref($residual_name, $base_sym)))

	transformed_constraint = _substitute_with_residual(constraint, _substitution_target(ref_vars), model_expr, residual_symbol)
	diff_expr = _equality_to_diff(transformed_constraint)

	if isa(ref_vars, Symbol)
		expression_call = Expr(:macrocall, jump_expression, __source__, :_m, diff_expr)
		macrocall = quote
			let _m = $model_expr
				_func = $expression_call
				endo = $ref_vars
				resid = _m[$(QuoteNode(residual_symbol))]
				eqs = $equation_ref[$equation_ref(_func, $equal_to_ref(0.0))]
				([endo], [resid], eqs)
			end
		end
	elseif isexpr(ref_vars, :ref)
		indices = ref_vars.args[2:end]
		expression_call = Expr(:macrocall, jump_expression, __source__, :_m, Expr(:vect, indices...), diff_expr)
		macrocall = quote
			let _m = $model_expr
				_exprs = $expression_call
				_ks = $all_keys_ref(_exprs)
				endos = [$index_var_ref($base_sym, k) for k in _ks]
				resids = [$index_var_ref(_m[$(QuoteNode(residual_symbol))], k) for k in _ks]
				eqs = $equation_ref[$equation_ref(_exprs[k...], $equal_to_ref(0.0)) for k in _ks]
				(endos, resids, eqs)
			end
		end
	else
		_error("Reference must be a variable")
	end
	push!(code.args, macrocall)
	return esc(code)
end

"""
    @block model begin ... end

Create a `Block` of equations mapped to their endogenous variables.

Each line in the block body specifies a variable (or indexed variable) followed by
its defining equation. Equations are stored as lightweight `Equation` objects
(expression + set) without registering JuMP constraints.

# Arguments
- `model`: The JuMP model (or ModelDictionary) containing the variables
- `begin ... end`: A block where each line is `variable, equation_expr`

# Returns
A `Block` containing the equation-to-variable mappings.

# Examples
```julia
model = Model()
@variable(model, p)
@variable(model, w[1:3])
@variable(model, L[1:3])
@variable(model, ρ[1:3])
@variable(model, N[1:3])

# Define a block with scalar and indexed constraints
my_block = @block model begin
    p, p == 1
    w[j ∈ 1:3], L[j] == ρ[j] * N[j]
end

# Check block properties
length(my_block)  # 4
p ∈ my_block      # true
w[1] ∈ my_block   # true
```

```julia
# Multi-dimensional indexing
@variable(model, z[1:2, [:a, :b]])

b = @block model begin
    z[i ∈ 1:2, j ∈ [:a, :b]], z[i,j] == i
end
```

See also: [`Block`](@ref), [`@endo_exo_swap!`](@ref), [`constraints`](@ref), [`endogenous`](@ref), [`variables`](@ref)
"""
macro block(model, expr)
	_error(line_number, it, msg) = error(
		"Invalid @block expression at $(line_number.file):$(line_number.line): $msg. Got $(sprint(show, it)).",
	)
	_is_equality(it) = isexpr(it, :call) && length(it.args) == 3 && it.args[1] == :(==)
	_is_continuation(it) = isexpr(it, :call) && length(it.args) == 2 && it.args[1] in (:+, :-)
	sm = @__MODULE__
	block_macro_ref = GlobalRef(sm, Symbol("@_block"))
	get_model_ref = GlobalRef(sm, :_get_model)
	equation_ref = GlobalRef(sm, :Equation)
	collect_variables_ref = GlobalRef(sm, :collect_variables!)
	residuals_ref = GlobalRef(sm, :residuals)
	line_number = expr.args[1]
	@assert isa(line_number, LineNumberNode)
	block_items = Tuple{LineNumberNode,Expr}[]
	last_tuple = nothing
	for it in expr.args
	    if isa(it, LineNumberNode)
	        line_number = it
	    elseif isexpr(it, :tuple) # line with commas
	        length(it.args) == 2 || _error(line_number, it, "Each line must be `variable, equation`")
	        _is_equality(it.args[2]) || _error(line_number, it, "The equation must use `==`")
	        push!(block_items, (line_number, it))
	        last_tuple = it
	    elseif _is_continuation(it) && last_tuple !== nothing
	        eq = last_tuple.args[2]
	        eq.args[3] = Expr(:call, it.args[1], eq.args[3], it.args[2])
	    else
	        _error(line_number, it, "Unexpected code in block body")
	    end
	end
	code = Expr(:tuple)
	for (line_number, it) in block_items
	    macro_call = Expr(
	        :macrocall,
	        block_macro_ref,
	        line_number,
	        model,
	        it.args...,
	    )
	    push!(code.args, esc(macro_call))
	end
	quote
	    _container = $(esc(model))
	    _model = $get_model_ref(_container)
	    results = [$code...]
	    endogenous = Iterators.flatten([r[1] for r in results])
	    residuals = Iterators.flatten([r[2] for r in results])
	    eqs = Iterators.flatten([r[3] for r in results])
	    endo_vec = VariableRef[endogenous...]
	    res_vec = VariableRef[residuals...]
	    eqs_vec = $equation_ref[eqs...]
	    all_vars = Set{VariableRef}()
	    for eq in eqs_vec
	        $collect_variables_ref(all_vars, eq)
	    end
	    _block = Block(_model, endo_vec, res_vec, all_vars, eqs_vec)
	    if _container isa ModelDictionary
	        _container[$residuals_ref(_block)] .= 0.0
	    end
	    _block
	end
end

"""Split full name JuMP variable into base name and indices"""
function split_name(var::AbstractVariableRef)
	parts = split(string(var), "["; limit=2)
	if length(parts) == 1
		return parts[1], ""  # Scalar variable, no indices
	end
	return parts[1], "[" * parts[2]
end

"""Return base name of JuMP variable"""
base_name(var::AbstractVariableRef) = split_name(var)[1]
base_name(var::AbstractArray{T}) where {T<:AbstractVariableRef} = base_name(first(var))

"""
    residual(var)

Return the residual variable or residual container corresponding to an endogenous
variable or variable container.
"""
residual(var) = first(var).model[Symbol(make_residual_name(base_name(var)))]
residual(var::AbstractVariableRef) = var.model[Symbol(make_residual_name(base_name(var)))]

"""
If a variable Symbol(new_name) does not exist, define a new variable with the same indices as an existing variable.
"""
function copy_variable(new_name::String, original::SparseAxisArray)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    d = Dict(k => VariableRef(m) for k in keys(original.data))
	    new = SparseAxisArray(d)
	    for k in keys(original.data)
	        set_name(new[k...], new_name * split_name(original[k...])[2])
	    end
	    m[sym] = new
	    fix.(new, 0)
	end
	return m[sym]
end
function copy_variable(new_name::String, original::AbstractArray)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    data = reshape([VariableRef(m) for _ in _all_keys(original)], length.(axes(original))...)
	    new = DenseAxisArray(data, axes(original)...)
	    for (x, y) in zip(new, original)
	        set_name(x, new_name * split_name(y)[2])
	    end
	    m[sym] = new
	    fix.(new, 0)
	end
	return m[sym]
end
function copy_variable(new_name::String, original::AbstractVariableRef)
	m = first(original).model
	sym = Symbol(new_name)
	if !haskey(m, sym)
	    new = VariableRef(m)
	    set_name(new, new_name)
	    m[sym] = new
	    fix(new, 0)
	end
	return m[sym]
end
copy_variable(new_name::String, original::SparseZeroArray) = copy_variable(new_name, original.data)

base_name(var::SparseZeroArray) = base_name(first(var))

"""
    unfix(b::Block)

Unfix all endogenous variables in a block.

Iterates through all variables in the block and unfixes any that are currently fixed.
Variables that are already unfixed are skipped.

# Arguments
- `b::Block`: The block whose variables should be unfixed

# Returns
`nothing`

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

b = @block model begin
    x, x == 1
    y[i ∈ 1:3], y[i] == i
end

fix.(b, 1.0)      # Fix all variables in block to 1.0
is_fixed(x)       # true
unfix(b)          # Unfix all variables
is_fixed(x)       # false
```

See also: [`Block`](@ref), [`@endo_exo_swap!`](@ref)
"""
function JuMP.unfix(b::Block)
	for var in b
	    if is_fixed(var)
	        unfix(var)
	    end
	end
	return nothing
end

include("endo_exo_swap.jl")
include("tagged_variables.jl")
include("ModelDictionaries.jl")
include("solve.jl")
include("ModelExpressions.jl")
include("ModelPlotting.jl")
using .ModelExpressions: @evalexpr, @prt, LabeledArray, MultiVarResult, set_default_source!, set_default_operator!, set_default_periods!, set_column_label_total_width!, reset_print_defaults!
using .ModelPlotting: @plot, plotvar, plotseries, labeled, LabeledSeries

# Define _get_model for ModelDictionary (after ModelDictionaries.jl is included)
_get_model(md::ModelDictionary) = md.model

end # Module
