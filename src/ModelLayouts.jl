# Shared variable metadata. A layout belongs to the model, while datasets own
# their values. Dictionaries are replaced on refresh so datasets can retain an
# old variable-to-slot snapshot while moving values into the refreshed layout.
using Dictionaries: Indices

mutable struct ModelLayout{M<:AbstractModel,V<:AbstractVariableRef}
    model::M
    names::Vector{String}
    variables::Vector{V}
    name_to_slot::Dict{String,Int}
    id_to_slot::Dict{MOI.VariableIndex,Int}
    name_indices::Union{Nothing,Indices{String}}
    selection_cache::IdDict{Any,Any}
    revision::UInt
    n_variables::Int
    growth_stamp::Union{Nothing,Int}
end

const _MODEL_LAYOUT_KEY = :SquareModels_model_layout

# Standard JuMP models own their layout in `ext`, avoiding a global cache whose
# values would keep its weakly keyed models alive through VariableRefs. Custom
# models without extension storage use weak values as well as weak keys.
const _fallback_model_layouts = WeakKeyDict{AbstractModel,WeakRef}()

function ModelLayout(model::AbstractModel)
    # Infer the concrete reference type from the first snapshot, including for
    # empty generic JuMP models, without requiring a second model traversal.
    variables = all_variables(model)
    layout = ModelLayout(model, String[], variables, Dict{String,Int}(),
        Dict{MOI.VariableIndex,Int}(), nothing, IdDict{Any,Any}(), UInt(0), -1, nothing)
    return _refresh_model_layout!(layout, variables)
end

function _refresh_model_layout!(layout::ModelLayout, variables=all_variables(layout.model))
    names = Vector{String}(undef, length(variables))
    name_to_slot = Dict{String,Int}()
    id_to_slot = Dict{MOI.VariableIndex,Int}()
    sizehint!(name_to_slot, length(variables))
    sizehint!(id_to_slot, length(variables))
    n_unnamed = 0
    for (slot, variable) in enumerate(variables)
        var_name = name(variable)
        names[slot] = var_name
        id_to_slot[JuMP.index(variable)] = slot
        # JuMP excludes unnamed variables from name lookup. Zero marks an
        # ambiguous name, preserving JuMP's duplicate-name error on lookup.
        if !isempty(var_name)
            name_to_slot[var_name] = haskey(name_to_slot, var_name) ? 0 : slot
        else
            n_unnamed += 1
        end
    end
    layout.names = names
    layout.variables = variables
    layout.name_to_slot = name_to_slot
    layout.id_to_slot = id_to_slot
    # A ModelDictionary requires every name (including anonymous names) to be
    # unique. The registry itself still supports ambiguous and unnamed models.
    unique_names = length(name_to_slot) + n_unnamed == length(names) && n_unnamed <= 1
    layout.name_indices = unique_names ? Indices(names) : nothing
    empty!(layout.selection_cache)
    layout.n_variables = length(variables)
    layout.growth_stamp = _model_growth_stamp(layout.model)
    layout.revision += UInt(1)
    return layout
end

function _check_model_layout(layout::ModelLayout)
    if hasproperty(layout.model, :ext)
        get(getproperty(layout.model, :ext), _MODEL_LAYOUT_KEY, nothing) === layout ||
            throw(ArgumentError(
                "This model layout is no longer current. Create a new ModelDictionary after empty!(model).",
            ))
    end
    return nothing
end

# Isolated adapter for JuMP's standard MOI cache. MOI.NumberOfVariables counts
# live entries in VariablesContainer.set_mask, which scans the entire model.
# Its storage length instead changes in O(1) whenever a variable is added,
# including a deletion followed by an addition. This relies only on the known
# cache/container implementations below; other backends use the public count.
# Deletion alone and renaming require refresh_model_layout! explicitly.
_model_growth_stamp(::AbstractModel) = nothing
function _model_growth_stamp(model::JuMP.GenericModel)
    backend = JuMP.backend(model)
    backend isa MOI.Utilities.CachingOptimizer || return nothing
    cache = backend.model_cache
    cache isa MOI.Utilities.UniversalFallback && (cache = cache.model)
    cache isa MOI.Utilities.AbstractModel || return nothing
    hasproperty(cache, :variables) || return nothing
    variables = cache.variables
    variables isa MOI.Utilities.VariablesContainer || return nothing
    hasproperty(variables, :set_mask) || return nothing
    mask = variables.set_mask
    mask isa Vector{UInt16} || return nothing
    return length(mask)
end

function _ensure_model_layout!(layout::ModelLayout)
    _check_model_layout(layout)
    growth_stamp = _model_growth_stamp(layout.model)
    changed = growth_stamp === nothing ?
        JuMP.num_variables(layout.model) != layout.n_variables :
        growth_stamp != layout.growth_stamp
    changed && _refresh_model_layout!(layout)
    return layout
end

function _model_layout(model::AbstractModel; refresh::Bool=false)
    if hasproperty(model, :ext)
        storage = getproperty(model, :ext)
        layout = get(storage, _MODEL_LAYOUT_KEY, nothing)
        # Extensions copied without JuMP.copy_extension_data must never reuse
        # variable references belonging to another model.
        if layout === nothing || (layout isa ModelLayout && layout.model !== model)
            layout = ModelLayout(model)
            storage[_MODEL_LAYOUT_KEY] = layout
            return layout
        end
        refresh && return _refresh_model_layout!(layout::ModelLayout)
        return _ensure_model_layout!(layout::ModelLayout)
    end
    reference = get(_fallback_model_layouts, model, nothing)
    layout = reference === nothing ? nothing : reference.value
    if layout === nothing
        layout = ModelLayout(model)
        _fallback_model_layouts[model] = WeakRef(layout)
    elseif refresh
        return _refresh_model_layout!(layout)
    end
    return _ensure_model_layout!(layout)
end

"""
    refresh_model_layout!(model)

Rebuild SquareModels' shared variable names and storage-slot index for `model`.
Call this after deleting or renaming variables, or changing registered variable
containers. Standard cached JuMP models detect additions automatically using an
O(1) growth stamp; querying the number of live variables would scan the model.
Other backends fall back to checking the live variable count. Missing-name
lookups in a synchronized standard model do not scan its variables.

Refreshing invalidates prepared selections and lets datasets synchronize their
values by variable identity when next accessed. Finish a group of model changes
before refreshing, rather than refreshing after every individual variable.
After `empty!(model)`, create new datasets and selections: JuMP may reuse old
variable identities, so data from the emptied model cannot be synchronized.
"""
function refresh_model_layout!(model::AbstractModel)
    return _model_layout(model; refresh=true)
end

function _layout_slot(layout::ModelLayout, var_name::AbstractString)
    slot = get(layout.name_to_slot, String(var_name), -1)
    slot == 0 && error("Multiple variables have the name $var_name.")
    slot == -1 && throw(KeyError(var_name))
    return slot
end

function _layout_slot(layout::ModelLayout, variable::AbstractVariableRef)
    JuMP.owner_model(variable) === layout.model ||
        throw(ArgumentError("Variable belongs to a different model."))
    return get(layout.id_to_slot, JuMP.index(variable)) do
        throw(KeyError(variable))
    end
end

# Copying a JuMP model must not carry references to variables in the original.
JuMP.copy_extension_data(::ModelLayout, new_model::AbstractModel, ::AbstractModel) =
    ModelLayout(new_model)
