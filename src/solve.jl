# solve.jl - Functions for solving blocks

using JuMP: Model, VariableRef, AffExpr, QuadExpr, NonlinearExpr
using JuMP: @variable, @constraint, name
using JuMP: set_start_value, fix, has_lower_bound, has_upper_bound
using JuMP: lower_bound, upper_bound, set_lower_bound, set_upper_bound
using JuMP: all_variables, is_fixed, value, add_to_expression!
using JuMP: optimize!, assert_is_solved_and_feasible
using JuMP: set_silent, unsafe_backend, backend, set_time_limit_sec
using JuMP: FEASIBILITY_SENSE, set_objective_sense, set_optimizer_attribute
import MathOptInterface as MOI

# ============================================================================
# Expression Transformation
# ============================================================================
# Transform JuMP expressions by:
# - Replacing endogenous VariableRefs with their solve model equivalents
# - Replacing exogenous VariableRefs with their data values (constants)

"""
    transform_expr(expr, var_map, data, endo_set) -> transformed_expr

Transform a JuMP expression by substituting variables.
- Endogenous variables are mapped to solve model variables via `var_map`
- Exogenous variables are replaced with their values from `data`
"""
function transform_expr end

# Passthrough for numbers
transform_expr(x::Number, var_map, data, endo_set) = x

# Transform a single VariableRef
function transform_expr(var::VariableRef, var_map, data, endo_set)
    # Check if it's an endogenous variable or residual (both are in var_map)
    if haskey(var_map, var)
        return var_map[var]
    else
        # Exogenous variable: substitute with data value
        val = data[var]
        val === nothing && error("No data value for exogenous variable $(name(var))")
        return val
    end
end

# Transform AffExpr (linear): constant + Σ(coef * var)
function transform_expr(expr::AffExpr, var_map, data, endo_set)
    new_expr = AffExpr(expr.constant)
    for (var, coef) in expr.terms
        if haskey(var_map, var)
            # Endogenous variable or residual: map to solve model variable
            add_to_expression!(new_expr, coef, var_map[var])
        else
            # Exogenous: substitute with data value
            val = data[var]
            val === nothing && error("No data value for exogenous variable $(name(var))")
            add_to_expression!(new_expr, coef * val)
        end
    end
    return new_expr
end

# Transform QuadExpr (quadratic): aff + Σ(coef * var_a * var_b)
function transform_expr(expr::QuadExpr, var_map, data, endo_set)
    # Transform the affine part
    new_aff = transform_expr(expr.aff, var_map, data, endo_set)
    new_expr = QuadExpr(new_aff)

    for (pair, coef) in expr.terms
        var_a, var_b = pair.a, pair.b

        # Get value or mapped variable for each
        val_a = if haskey(var_map, var_a)
            var_map[var_a]
        else
            v = data[var_a]
            v === nothing && error("No data value for exogenous variable $(name(var_a))")
            v
        end

        val_b = if haskey(var_map, var_b)
            var_map[var_b]
        else
            v = data[var_b]
            v === nothing && error("No data value for exogenous variable $(name(var_b))")
            v
        end

        if val_a isa Number && val_b isa Number
            # Both exogenous: becomes constant
            add_to_expression!(new_expr.aff, coef * val_a * val_b)
        elseif val_a isa Number
            # var_a exogenous: becomes linear term
            add_to_expression!(new_expr.aff, coef * val_a, val_b)
        elseif val_b isa Number
            # var_b exogenous: becomes linear term
            add_to_expression!(new_expr.aff, coef * val_b, val_a)
        else
            # Both endogenous: stays quadratic
            add_to_expression!(new_expr, coef, val_a, val_b)
        end
    end
    return new_expr
end

# Transform NonlinearExpr (tree structure with head and args)
function transform_expr(expr::NonlinearExpr, var_map, data, endo_set)
    new_args = Any[]
    for arg in expr.args
        if arg isa VariableRef
            if haskey(var_map, arg)
                push!(new_args, var_map[arg])
            else
                val = data[arg]
                val === nothing && error("No data value for exogenous variable $(name(arg))")
                push!(new_args, val)
            end
        elseif arg isa Union{AffExpr, QuadExpr, NonlinearExpr}
            push!(new_args, transform_expr(arg, var_map, data, endo_set))
        else
            push!(new_args, arg)  # Numbers, symbols, etc.
        end
    end
    # Use splatting to pass args correctly to NonlinearExpr constructor
    return NonlinearExpr(expr.head, Any[new_args...])
end

# ============================================================================
# Expression analysis
# ============================================================================

"""Check if a JuMP expression contains any VariableRef (allocation-free)."""
_has_variables(::Number) = false
_has_variables(::Zero) = false
_has_variables(::VariableRef) = true
_has_variables(expr::AffExpr) = !isempty(expr.terms)
function _has_variables(expr::QuadExpr)
    !isempty(expr.terms) || !isempty(expr.aff.terms)
end
function _has_variables(expr::NonlinearExpr)
    for arg in expr.args
        _has_variables(arg) && return true
    end
    return false
end

"""Extract the constant value from a variable-free expression."""
_constant_value(x::Number) = Float64(x)
_constant_value(expr::AffExpr) = Float64(expr.constant)
_constant_value(expr::QuadExpr) = Float64(expr.aff.constant)
_constant_value(expr::NonlinearExpr) = _is_trivially_zero(expr) ? 0.0 : NaN
_constant_value(_) = NaN

# ============================================================================
# Effective variable analysis (zero-coefficient / zero-multiplication detection)
# ============================================================================
# A variable can be syntactically present in an expression but contribute nothing
# to its value — e.g. x * 0, or an AffExpr term with coefficient 0.0.
# These functions detect such cases for improved diagnostics.

"""Check if an expression is trivially zero regardless of variable values (e.g. `x * 0`)."""
_is_trivially_zero(::VariableRef) = false
_is_trivially_zero(x::Number) = iszero(x)
_is_trivially_zero(::Zero) = true
_is_trivially_zero(expr::AffExpr) = iszero(expr.constant) && all(iszero, values(expr.terms))
function _is_trivially_zero(expr::QuadExpr)
    _is_trivially_zero(expr.aff) && all(iszero, values(expr.terms))
end
function _is_trivially_zero(expr::NonlinearExpr)
    if expr.head === :*
        any(_is_trivially_zero, expr.args)
    elseif expr.head === :+ || expr.head === :-
        all(_is_trivially_zero, expr.args)
    elseif expr.head === :^ && length(expr.args) == 2
        _is_trivially_zero(expr.args[1]) && expr.args[2] isa Number && expr.args[2] > 0
    else
        false
    end
end

"""Like `_has_variables` but ignores variables in trivially-zero subtrees or with zero coefficients."""
_has_effective_variables(::Number) = false
_has_effective_variables(::Zero) = false
_has_effective_variables(::VariableRef) = true
_has_effective_variables(expr::AffExpr) = any(!iszero(c) for (_, c) in expr.terms)
function _has_effective_variables(expr::QuadExpr)
    any(!iszero(c) for (_, c) in expr.terms) ||
    any(!iszero(c) for (_, c) in expr.aff.terms)
end
function _has_effective_variables(expr::NonlinearExpr)
    _is_trivially_zero(expr) && return false
    any(_has_effective_variables(arg) for arg in expr.args)
end

"""Like `collect_variables!` but skips variables in trivially-zero subtrees or with zero coefficients."""
_collect_effective_variables!(vars::Set{VariableRef}, ::Union{Number, Zero}) = vars
function _collect_effective_variables!(vars::Set{VariableRef}, var::VariableRef)
    push!(vars, var)
    vars
end
function _collect_effective_variables!(vars::Set{VariableRef}, expr::AffExpr)
    for (var, coef) in expr.terms
        iszero(coef) || push!(vars, var)
    end
    vars
end
function _collect_effective_variables!(vars::Set{VariableRef}, expr::QuadExpr)
    for (var, coef) in expr.aff.terms
        iszero(coef) || push!(vars, var)
    end
    for (pair, coef) in expr.terms
        if !iszero(coef)
            push!(vars, pair.a)
            push!(vars, pair.b)
        end
    end
    vars
end
function _collect_effective_variables!(vars::Set{VariableRef}, expr::NonlinearExpr)
    _is_trivially_zero(expr) && return vars
    for arg in expr.args
        _collect_effective_variables!(vars, arg)
    end
    vars
end

# ============================================================================
# Diagnostics
# ============================================================================

struct TrivialEquation
    index::Int
    endogenous::VariableRef
    residual::VariableRef
    constant_value::Float64
end

struct OrphanVariable
    endogenous::VariableRef
end

"""
    diagnose(block::Block, data::ModelDictionary)

Analyze a block for structural issues that would cause solver failures.

Substitutes exogenous values from `data` into each equation and checks for:
- **Trivial equations**: equations where no endogenous variable effectively contributes
  after substitution. This includes both fully constant expressions and expressions where
  all variable terms have zero coefficients (e.g. `x * 0`). These are silently dropped
  by some solvers (e.g. GAMS), breaking squareness.
- **Orphan variables**: endogenous variables that don't effectively appear in any
  non-trivial equation after substitution, leaving them undetermined. A variable is
  considered absent if it only appears with zero coefficients or inside trivially-zero
  subtrees (e.g. multiplied by zero).

Returns `(trivial, orphans)` — a `Vector{TrivialEquation}` and a `Vector{OrphanVariable}`.

# Example
```julia
trivial, orphans = diagnose(block, data)
for t in trivial
    println("Equation \$(t.index) for \$(name(t.endogenous)) is trivial (value=\$(t.constant_value))")
end
for o in orphans
    println("Orphan: \$(name(o.endogenous))")
end
```
"""
function diagnose(block::Block, data::ModelDictionary)
    for res in block.residuals
        if !(res ∈ data) || data[res] === nothing
            data[res] = 0.0
        end
    end

    endo_set = block._endogenous_set
    var_map = Dict{VariableRef, VariableRef}()
    for endo_var in block.endogenous
        var_map[endo_var] = endo_var  # identity map — we only need to detect variable presence
    end

    trivial = TrivialEquation[]
    vars_in_equations = Set{VariableRef}()

    for (i, eq) in enumerate(block.equations)
        new_func = transform_expr(eq.func, var_map, data, endo_set)
        if !_has_effective_variables(new_func)
            push!(trivial, TrivialEquation(i, block.endogenous[i], block.residuals[i], _constant_value(new_func)))
        else
            _collect_effective_variables!(vars_in_equations, new_func)
        end
    end

    orphans = OrphanVariable[
        OrphanVariable(v) for v in block.endogenous if v ∉ vars_in_equations
    ]

    return trivial, orphans
end

function _format_diagnostic_error(trivial::Vector{TrivialEquation}, orphans::Vector{OrphanVariable})
    lines = String[]
    if !isempty(trivial)
        push!(lines, "$(length(trivial)) trivial equation(s) (no endogenous variables effectively present after substituting exogenous data):")
        for t in trivial
            rhs = t.constant_value
            if isnan(rhs)
                push!(lines, "  Eq $(t.index): $(name(t.endogenous)) — effectively trivial (constant value could not be determined)")
            else
                status = abs(rhs) < 1e-12 ? "0 = 0 (redundant)" : "$(rhs) = 0 (infeasible!)"
                push!(lines, "  Eq $(t.index): $(name(t.endogenous)) — $status")
            end
        end
    end
    if !isempty(orphans)
        push!(lines, "$(length(orphans)) orphan variable(s) (not effectively present in any non-trivial equation):")
        for o in orphans
            push!(lines, "  $(name(o.endogenous))")
        end
    end
    join(lines, "\n")
end

# ============================================================================
# _build_model (internal)
# ============================================================================

# Create a new model with the same optimizer (including all solver-specific attributes) as src
function _copy_model_config(src)
    Model(() -> deepcopy(unsafe_backend(src)))
end

# Internal function to build a solve model from a block
function _build_model(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing,
    replace_nothing::Union{Nothing, Number} = nothing,
    skip_diagnostics::Bool = false
)
    for res in block.residuals
        if !(res ∈ data) || data[res] === nothing
            data[res] = 0.0
        end
    end

    solve_model = _copy_model_config(block.model)
    endo_set = block._endogenous_set

    var_map = sizehint!(Dict{VariableRef, VariableRef}(), length(block.endogenous))
    for endo_var in block.endogenous
        new_var = @variable(solve_model)
        var_map[endo_var] = new_var

        if has_lower_bound(endo_var)
            set_lower_bound(new_var, lower_bound(endo_var))
        end
        if has_upper_bound(endo_var)
            set_upper_bound(new_var, upper_bound(endo_var))
        end

        start_val = nothing
        if start_values !== nothing
            try
                start_val = start_values[endo_var]
            catch
            end
        end
        if start_val === nothing
            try
                start_val = data[endo_var]
            catch
            end
        end
        if start_val === nothing && replace_nothing !== nothing
            start_val = replace_nothing
        end
        if start_val !== nothing && !isnan(start_val)
            set_start_value(new_var, start_val)
        end
    end

    trivial = skip_diagnostics ? nothing : TrivialEquation[]
    endos_used = skip_diagnostics ? nothing : Set{VariableRef}()
    reverse_map = skip_diagnostics ? nothing : Dict{VariableRef, VariableRef}(v => k for (k, v) in var_map)

    for (i, eq) in enumerate(block.equations)
        new_func = transform_expr(eq.func, var_map, data, endo_set)
        @constraint(solve_model, new_func in eq.set)
        if !skip_diagnostics
            if !_has_effective_variables(new_func)
                push!(trivial, TrivialEquation(i, block.endogenous[i], block.residuals[i], _constant_value(new_func)))
            else
                solve_vars = Set{VariableRef}()
                _collect_effective_variables!(solve_vars, new_func)
                for sv in solve_vars
                    orig = get(reverse_map, sv, nothing)
                    orig !== nothing && push!(endos_used, orig)
                end
            end
        end
    end

    if !skip_diagnostics
        orphans = OrphanVariable[OrphanVariable(v) for v in block.endogenous if v ∉ endos_used]
        if !isempty(trivial) || !isempty(orphans)
            throw(NonSquareError("Model is not effectively square after substituting exogenous values.\n" *
                  _format_diagnostic_error(trivial, orphans)))
        end
    end

    return solve_model, var_map
end

# ============================================================================
# Square-system model construction
# ============================================================================

"""
    square_model(optimizer; options...) -> Model
    square_model(; gamsdir, working_dir=mktempdir(), solver="CONOPT4", options...) -> Model

Construct a JuMP `Model` configured as a square nonlinear system: `FEASIBILITY_SENSE`
(no objective), ready for `solve`/`solve!`.

Pass a positional `optimizer` exactly as in JuMP's `Model(optimizer)` — an optimizer
constructor (e.g. `Ipopt.Optimizer`, `CONOPT.Optimizer`), a factory closure, or an
`optimizer_with_attributes(...)` object.

Alternatively, pass the `gamsdir` keyword to solve the system as a GAMS constrained
nonlinear system (CNS), e.g. `square_model(; gamsdir = "C:/GAMS/53")`. This requires the
optional `GAMS` package — run `using GAMS` first. The GAMS workspace is built so its
listing (`moi.lst`) lands in `working_dir`, which lets `solve!`/`annotate_lst!` locate and
annotate it afterwards.

Extra keyword arguments are applied as optimizer attributes, e.g.
`square_model(Ipopt.Optimizer; tol = 1e-10)` or `square_model(; gamsdir = "C:/GAMS/53", lmmxsf = 1)`.

# GAMS keyword arguments
- `gamsdir`: Path to the GAMS system directory (the folder containing `gams.exe`).
- `working_dir`: Directory for GAMS scratch files and the `moi.lst` listing.
- `solver`: GAMS CNS solver name (default `"CONOPT4"`).
"""
function square_model(optimizer=nothing; gamsdir=nothing, working_dir=nothing, solver="CONOPT4", options...)
    if gamsdir !== nothing
        optimizer === nothing || error("`square_model` takes either a positional `optimizer` or the `gamsdir` keyword, not both.")
        ext = Base.get_extension(@__MODULE__, :SquareModelsGAMSExt)
        ext === nothing && error("`gamsdir` requires the GAMS package — run `using GAMS` first.")
        optimizer = ext._gams_optimizer(; system_dir=gamsdir, working_dir=(working_dir === nothing ? mktempdir() : working_dir), solver)
    elseif optimizer === nothing
        error("`square_model` requires a positional `optimizer` (e.g. `Ipopt.Optimizer`) or the `gamsdir` keyword.")
    end
    model = Model(optimizer)
    set_objective_sense(model, FEASIBILITY_SENSE)
    for (key, value) in options
        set_optimizer_attribute(model, string(key), value)
    end
    return model
end

# ============================================================================
# GAMS listing annotation
# ============================================================================

"""Return the path to the `.lst` listing file written by `model`'s GAMS optimizer,
or `nothing` if `model` is not solved through GAMS (or no listing exists yet)."""
function _gams_lst_path(model)
    ext = Base.get_extension(@__MODULE__, :SquareModelsGAMSExt)
    ext === nothing ? nothing : ext._gams_lst_path(model)
end

"""
    annotate_lst!(block::Block, path; out_path=path)

Rewrite a GAMS `.lst`/`.gms` file in place, replacing GAMS.jl's generated `x<i>`/`eq<i>`
symbols with the JuMP names of `block`'s endogenous variables.

GAMS.jl numbers variables (`x<i>`) and equations (`eq<i>`) by their 1-based add-order, which
matches the order in which `solve`/`solve!` adds `block.endogenous`/`block.equations` to the
intermediate model (equation `i` pins `endogenous[i]`). This makes the otherwise opaque GAMS
listing readable for debugging.

By default `out_path == path`, so the file is overwritten in place; pass `out_path` to write
elsewhere. The file is rewritten with `\\n` line endings. Returns `out_path`.
"""
function annotate_lst!(block::Block, path::AbstractString; out_path::AbstractString=path)
    names = name.(block.endogenous)
    rx = r"\b(x|eq)(\d+)\b"
    function rename(m::AbstractString)
        i = parse(Int, m[(startswith(m, "eq") ? 3 : 2):end])
        1 <= i <= length(names) ? names[i] : m
    end
    # Heuristic, not a guarantee: `\bx\d+\b` also matches tokens like "x86" in the GAMS
    # banner ("WEX-WEI ... x86 64bit"), so skip that line. Out-of-range indices elsewhere
    # fall through unchanged, but a coincidental in-range `x<i>` would be mis-renamed.
    lines = [occursin("WEX-WEI", l) ? l : replace(l, rx => rename) for l in readlines(path)]
    write(out_path, join(lines, "\n") * "\n")
    return out_path
end

# ============================================================================
# solve
# ============================================================================

"""
    solve(block::Block, data::ModelDictionary; start_values=nothing, replace_nothing=nothing, skip_diagnostics=false)

Build, optimize, and extract solution in one step.

Uses the optimizer from the block's model. Creates an intermediate solve model with only
endogenous variables, optimizes it, and returns a new ModelDictionary with the solution.

Before solving, runs diagnostics to detect trivial equations and orphan variables
(set `skip_diagnostics=true` to disable if performance is a concern).

Optimizer attributes (silent mode, time limit) are copied from the block's model to the
intermediate solve model. Use `set_silent(model)` or `set_time_limit_sec(model, seconds)`
on the original model to configure solver behavior.

# Arguments
- `block::Block`: Block defined on a model with an optimizer set
- `data::ModelDictionary`: Data dictionary with values for all variables
- `start_values::Union{Nothing, ModelDictionary}`: Optional starting values (overrides `data`)
- `replace_nothing::Union{Nothing, Number}`: If provided, replace `nothing` values in start
  values with this number. If not provided, `nothing` values will cause errors.
- `skip_diagnostics::Bool`: Skip pre-solve diagnostic checks (default `false`)

# Returns
A new `ModelDictionary` containing the solution values for endogenous variables,
with exogenous values copied from `data`.

# Example
```julia
using Ipopt
model = square_model(Ipopt.Optimizer)
set_silent(model)  # Suppress solver output
@variables model begin
    x
    y
end
data = ModelDictionary(model)
data[x] = 1.0
data[y] = 2.0

block = @block model begin
    x, x == 10
    y, y == 20
end

solution = solve(block, data)
solution[x]  # 10.0
solution[y]  # 20.0
```
"""
function solve(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing,
    replace_nothing::Union{Nothing, Number} = nothing,
    skip_diagnostics::Bool = false
)
    result = copy(data)
    solve!(block, result; start_values, replace_nothing, skip_diagnostics)
    return result
end

"""
    solve!(block::Block, data::ModelDictionary; start_values=nothing, replace_nothing=nothing, skip_diagnostics=false)

Build, optimize, and update data in-place.

Like `solve`, but mutates `data` instead of returning a new ModelDictionary.

Before solving, runs diagnostics to detect trivial equations and orphan variables
(set `skip_diagnostics=true` to disable if performance is a concern).

Optimizer attributes (silent mode, time limit) are copied from the block's model to the
intermediate solve model. Use `set_silent(model)` or `set_time_limit_sec(model, seconds)`
on the original model to configure solver behavior.

# Arguments
- `block::Block`: Block defined on a model with an optimizer set
- `data::ModelDictionary`: Data dictionary to update with solution values
- `start_values::Union{Nothing, ModelDictionary}`: Optional starting values (overrides `data`)
- `replace_nothing::Union{Nothing, Number}`: If provided, replace `nothing` values in start
  values with this number. If not provided, `nothing` values will cause errors.
- `skip_diagnostics::Bool`: Skip pre-solve diagnostic checks (default `false`)

# Returns
The mutated `data` ModelDictionary.

# Example
```julia
solve!(block, data)  # data is updated in-place
```
"""
function solve!(
    block::Block,
    data::ModelDictionary;
    start_values::Union{Nothing, ModelDictionary} = nothing,
    replace_nothing::Union{Nothing, Number} = nothing,
    skip_diagnostics::Bool = false
)
    model, var_map = _build_model(block, data; start_values, replace_nothing, skip_diagnostics)
    optimize!(model)

    lst = _gams_lst_path(model)
    lst === nothing || annotate_lst!(block, lst)

    assert_is_solved_and_feasible(model)

    for (original_var, solve_var) in var_map
        data[original_var] = value(solve_var)
    end
    return data
end
