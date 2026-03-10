# tagged_variables.jl - Custom @variables macro with tags and descriptions
#
# Syntax (using :: for tags, like Julia's Holy trait pattern):
#
#   # Apply tags to individual variables:
#   @variables container begin
#       var_name[indices] :: (tag1, tag2), "Description"
#       var_name[indices] :: tag1
#       var_name[indices], "Description"
#       var_name[indices]
#   end
#
#   # Apply tags to ALL variables in the block:
#   @variables container :: tag begin
#       var_name[indices], "Description"
#       ...
#   end
#
#   # Combine block-level and variable-level tags (they accumulate):
#   @variables container :: BlockTag begin
#       var_name[indices] :: VarTag, "Description"  # Has both BlockTag and VarTag
#   end
#
# Tags are trait-like markers for variable categorization.
# This replaces JuMP's @variables macro.

# ==============================================================================
# Tag Definition (Holy Trait Pattern)
# ==============================================================================
"""
    Tag

A trait-like tag for categorizing variables, following Julia's Holy trait pattern.

Tags are used to mark variables with properties like "growth adjusted" or
"inflation adjusted", enabling trait-based dispatch and grouping.

# Example
```julia
# Define tags as trait markers
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)
const FlatForecast = Tag(:flat_forecast)

# Use with :: syntax (like type annotations)
@variables db begin
    vGDP[t] :: (GrowthAdjusted, InflationAdjusted), "Nominal GDP"
    pGDP[t] :: InflationAdjusted, "GDP deflator"
    qGDP[t] :: GrowthAdjusted, "Real GDP"
end
```
"""
struct Tag
    name::Symbol
end

Base.show(io::IO, t::Tag) = print(io, "Tag(:", t.name, ")")

# ==============================================================================
# Variable Metadata
# ==============================================================================
"""
    VariableMetadata

Stores metadata for a variable: its tags and description.
"""
struct VariableMetadata
    tags::Set{Tag}
    description::String
end

VariableMetadata() = VariableMetadata(Set{Tag}(), "")
VariableMetadata(tags) = VariableMetadata(Set{Tag}(tags), "")
VariableMetadata(tags, desc::AbstractString) = VariableMetadata(Set{Tag}(tags), String(desc))

# Global registry: variable name (Symbol) => VariableMetadata
const _variable_metadata = Dict{Symbol, VariableMetadata}()

"""
    description(var) → String

Get the description of a variable. For indexed variables like `X[2020]`,
returns the description of the base variable `X`.
"""
description(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata()).description
description(var::Symbol) = get(_variable_metadata, var, VariableMetadata()).description

"""
    tags(var) → Set{Tag}

Get all tags associated with a variable. For indexed variables like `X[2020]`,
returns the tags of the base variable `X`.
"""
tags(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata()).tags
tags(var::Symbol) = get(_variable_metadata, var, VariableMetadata()).tags

"""
    has_tag(var, tag::Tag) → Bool

Check if a variable has a specific tag.
"""
has_tag(var::AbstractVariableRef, tag::Tag) = tag ∈ tags(var)
has_tag(var::Symbol, tag::Tag) = tag ∈ tags(var)

"""
    tagged(tag::Tag) → Vector{Symbol}

Get all variable base names that have a specific tag.
"""
tagged(tag::Tag) = [k for (k, m) in _variable_metadata if tag ∈ m.tags]

"""
    metadata(var) → VariableMetadata

Get full metadata for a variable. For indexed variables like `X[2020]`,
returns the metadata of the base variable `X`.
"""
metadata(var::AbstractVariableRef) = get(_variable_metadata, Symbol(base_name(var)), VariableMetadata())
metadata(var::Symbol) = get(_variable_metadata, var, VariableMetadata())

# ==============================================================================
# Parsing Helpers
# ==============================================================================

"""
Parse a variable declaration line.

Handles these forms:
- `var[idx]` → (var_def, [], "")
- `var[idx] :: tag` → (var_def, [tag], "")
- `var[idx] :: (tag1, tag2)` → (var_def, [tag1, tag2], "")
- `var[idx] :: tag, "desc"` → (var_def, [tag], "desc")
- `var[idx], "desc"` → (var_def, [], "desc")

Returns: (var_definition, tag_exprs, description)
"""
function _parse_var_line(expr)
    # Handle tuple form: (var_stuff, "description")
    if isexpr(expr, :tuple)
        if length(expr.args) >= 2 && expr.args[end] isa String
            desc = expr.args[end]
            inner = expr.args[1]
            var_def, tag_exprs, _ = _parse_var_line(inner)
            return (var_def, tag_exprs, desc)
        elseif length(expr.args) == 1
            return _parse_var_line(expr.args[1])
        end
    end

    # Simple variable: just a symbol or ref
    if expr isa Symbol || isexpr(expr, :ref)
        return (expr, [], "")
    end

    # Type annotation expression: var[t] :: tag or var[t] :: (tag1, tag2)
    if isexpr(expr, :(::))
        var_def = expr.args[1]
        tag_part = expr.args[2]
        # Handle single tag or tuple of tags
        if isexpr(tag_part, :tuple)
            tag_exprs = collect(tag_part.args)
        else
            tag_exprs = [tag_part]
        end
        return (var_def, tag_exprs, "")
    end

    # Fallback: treat whole thing as var definition
    return (expr, [], "")
end

"""
Parse block-level tags from `container :: tag` or `container :: (tag1, tag2)`.

Returns: (container_expr, block_tag_exprs)
"""
function _parse_block_tags(expr)
    if isexpr(expr, :(::))
        container = expr.args[1]
        tag_part = expr.args[2]
        if isexpr(tag_part, :tuple)
            return (container, collect(tag_part.args))
        else
            return (container, [tag_part])
        end
    end
    return (expr, [])
end

# ==============================================================================
# Index set extraction for SparseZeroArray auto-wrapping
# ==============================================================================

"""Extract the set expression (RHS) from an index specification in a variable definition."""
_index_set(spec::Symbol) = spec
function _index_set(spec::Expr)
    if spec.head in (:(=), :kw)
        spec.args[2]
    elseif spec.head == :call && spec.args[1] in (:∈, :in)
        spec.args[3]
    else
        spec
    end
end

"""
Extract index set expressions from a :ref variable definition.
Returns a vector of expressions that evaluate to the index sets at runtime.
Skips :parameters (semicolon conditions).
"""
function _extract_index_sets(var_def::Expr)
    isexpr(var_def, :ref) || return []
    sets = []
    for spec in var_def.args[2:end]
        isexpr(spec, :parameters) && continue
        push!(sets, _index_set(spec))
    end
    return sets
end
_extract_index_sets(::Symbol) = []

# ==============================================================================
# @variables Macro
# ==============================================================================
"""
    @variables container begin ... end
    @variables container :: tag begin ... end

Create JuMP variables with optional tags and descriptions.

This is SquareModels' replacement for JuMP's `@variables` macro, adding support
for variable metadata (tags and descriptions) using Julia's `::` syntax,
following the Holy trait pattern.

# Syntax

Tags can be applied at two levels:
- **Block-level**: `@variables container :: tag begin ... end` applies tag(s) to ALL variables
- **Variable-level**: `var :: tag` applies tag(s) to that variable only

When both are used, tags accumulate (union).

Each variable line can have:
- Variable definition (required): `var` or `var[indices]`
- Tags (optional): `:: tag` or `:: (tag1, tag2)` after variable
- Description (optional): `, "description"` at end

```julia
# Define tags as trait markers
const GrowthAdjusted = Tag(:growth_adjusted)
const InflationAdjusted = Tag(:inflation_adjusted)

# Block-level tag: all variables get GrowthAdjusted
@variables db :: GrowthAdjusted begin
    vGDP[t] :: InflationAdjusted, "Nominal GDP"  # Has both tags
    qGDP[t], "Real GDP"                          # Has GrowthAdjusted only
end

# Variable-level tags only
@variables db begin
    pGDP[t] :: InflationAdjusted, "GDP deflator"
    σ, "Substitution elasticity"
end
```

# Access metadata
```julia
description(:vGDP)  # "Nominal GDP"
tags(:vGDP)         # Set([GrowthAdjusted, InflationAdjusted])
has_tag(:vGDP, GrowthAdjusted)  # true
tagged(GrowthAdjusted)  # [:vGDP, :qGDP, ...]
```

Note: If you need JuMP's original `@variables` macro, use `JuMP.@variables` explicitly.
"""
macro variables(container_expr, block)
    # Parse block-level tags from container expression
    container, block_tag_exprs = _parse_block_tags(container_expr)

    # Validate block structure
    if !isexpr(block, :block)
        error("@variables requires a begin...end block")
    end

    code = Expr(:block)
    model_expr = :($container isa ModelDictionary ? $container.model : $container)
    var_names = Symbol[]

    for line in block.args
        # Skip line numbers
        if line isa LineNumberNode
            push!(code.args, line)  # Preserve for error messages
            continue
        end

        # Parse the line
        var_def, var_tag_exprs, desc = _parse_var_line(line)
        var_name = _get_name(var_def)
        push!(var_names, var_name)

        # Generate @variable call
        push!(code.args, :(JuMP.@variable($model_expr, $var_def)))

        # Auto-wrap SparseAxisArray in SparseZeroArray with domain sets (if enabled)
        index_sets = _extract_index_sets(var_def)
        if !isempty(index_sets)
            sets_expr = Expr(:tuple, [:(Set($s)) for s in index_sets]...)
            push!(code.args, :(
                if SquareModels._use_sparse_zero_array[]
                    $var_name = $var_name isa JuMP.Containers.SparseAxisArray ?
                        SquareModels.SparseZeroArray($var_name, $sets_expr) : $var_name
                end
            ))
        end

        # Combine block-level and variable-level tags
        all_tag_exprs = vcat(block_tag_exprs, var_tag_exprs)

        # Register metadata
        if !isempty(all_tag_exprs) || !isempty(desc)
            tags_tuple = Expr(:tuple, all_tag_exprs...)
            push!(code.args, :(
                SquareModels._variable_metadata[$(QuoteNode(var_name))] =
                    SquareModels.VariableMetadata([$tags_tuple...], $desc)
            ))
        else
            push!(code.args, :(
                SquareModels._variable_metadata[$(QuoteNode(var_name))] =
                    SquareModels.VariableMetadata()
            ))
        end
    end

    # Return named tuple of variables (like JuMP.@variables)
    if length(var_names) == 1
        push!(code.args, var_names[1])
    else
        named_tuple = Expr(:tuple, [Expr(:(=), n, n) for n in var_names]...)
        push!(code.args, named_tuple)
    end

    return esc(code)
end
