# Model layouts own immutable snapshots of names and variable identities. Every
# dataset owns its value vector; no dataset inserts into shared dictionary keys.

function _layout_indices(layout)
    layout.name_indices === nothing && throw(ArgumentError(
        "ModelDictionary requires unique variable names. Name anonymous or duplicate variables before creating a dataset.",
    ))
    return layout.name_indices::Indices{String}
end

# Unlike JuMP.variable_by_name, a dataset can address the empty string when
# exactly one anonymous variable exists. The named-variable map excludes it.
function _storage_name_slot(layout, key::AbstractString)
    isempty(key) || return _layout_slot(layout, key)
    found, token = gettoken(_layout_indices(layout), String(key))
    found || throw(KeyError(key))
    # For Dictionaries.Indices the second token component is the value slot.
    return last(token)
end

function _numeric_storage_type(::Type{V}) where {V}
    # Imported and heterogeneous dictionaries may declare Any even when all
    # their values are numeric. Conversion below validates those values.
    V === Any && return Number
    types = filter(!=(Nothing), Base.uniontypes(V))
    isempty(types) && return Float64
    T = Union{types...}
    T <: Number || throw(ArgumentError("ModelDictionary values must be numbers or nothing."))
    return T
end

function _new_model_dictionary(model, dictionary::Dictionary{String,Union{Nothing,T}},
        layout, full, variables::Vector{V}, positions=nothing) where {T,V<:AbstractVariableRef}
    return ModelDictionary{T,typeof(model),typeof(layout),V}(
        model, dictionary, layout, layout.revision,
        full, variables, positions,
    )
end

"""
    ModelDictionary{T}(model[, values])

Create a dataset with numeric type `T` and `nothing` for unset cells. The default
`ModelDictionary(model)` uses `Float64`; use `ModelDictionary{Number}(model)` to
retain heterogeneous numeric types, or e.g. `ModelDictionary{Bool}` for masks.
Datasets for the same model share name and variable metadata, but own their values.
"""
function ModelDictionary{T}(model::AbstractModel) where {T<:Number}
    layout = _model_layout(model)
    indices = _layout_indices(layout)
    values = Vector{Union{Nothing,T}}(nothing, length(layout.variables))
    dictionary = Dictionary(indices, values)
    return _new_model_dictionary(model, dictionary, layout, true, layout.variables)
end

function ModelDictionary{T}(model::AbstractModel, values::Union{Number,AbstractVector}) where {T<:Number}
    d = ModelDictionary{T}(model)
    _set_initial_values!(d, values)
    return d
end

function _set_initial_values!(d, values::AbstractVector)
    length(values) == length(d.dictionary) || throw(DimensionMismatch("One value per model variable is required."))
    copyto!(d.dictionary.values, values)
    return d
end
_set_initial_values!(d, value::Number) = (fill!(d.dictionary.values, value); d)

# Import selected or reordered dictionary entries. Copy keys so external
# structural mutation cannot corrupt the dataset.
function ModelDictionary(model::AbstractModel, dictionary::Dictionary)
    layout = _model_layout(model)
    names = String.(collect(keys(dictionary)))
    slots = [_storage_name_slot(layout, key) for key in names]
    T = _numeric_storage_type(eltype(dictionary))
    values = Vector{Union{Nothing,T}}(collect(dictionary))
    full = length(slots) == length(layout.variables) && all(i -> slots[i] == i, eachindex(slots))
    variables = full ? layout.variables : layout.variables[slots]
    indices = full ? _layout_indices(layout) : Indices(names)
    positions = full ? nothing : Dict(JuMP.index(v) => i for (i, v) in enumerate(variables))
    return _new_model_dictionary(model, Dictionary(indices, values), layout, full,
        variables, positions)
end

function _derived_dictionary(d::ModelDictionary, values::AbstractVector)
    T = _numeric_storage_type(eltype(values))
    storage = convert(Vector{Union{Nothing,T}}, values)
    return _new_model_dictionary(d.model, Dictionary(keys(d.dictionary), storage),
        d._layout, d._full, d._variables, d._variable_positions)
end

function _subset_dictionary(d::ModelDictionary{T}, positions) where {T}
    variables = d._variables[positions]
    names = [d._layout.names[d._layout.id_to_slot[JuMP.index(v)]] for v in variables]
    dictionary = Dictionary(Indices(names), d.dictionary.values[positions])
    lookup = Dict(JuMP.index(v) => i for (i, v) in enumerate(variables))
    return _new_model_dictionary(d.model, dictionary, d._layout, false, variables, lookup)
end

function _ensure_data_layout!(d::ModelDictionary{T}; expand::Bool=false) where {T}
    layout = _ensure_model_layout!(d._layout)
    full = d._full || expand
    d._revision == layout.revision && full == d._full && return d
    _layout_indices(layout)
    if full
        values = Vector{Union{Nothing,T}}(nothing, length(layout.variables))
        for (i, variable) in enumerate(d._variables)
            slot = get(layout.id_to_slot, JuMP.index(variable), 0)
            slot == 0 || (values[slot] = d.dictionary.values[i])
        end
        dictionary = Dictionary(_layout_indices(layout), values)
        variables = layout.variables
        positions = nothing
    else
        retained = findall(v -> haskey(layout.id_to_slot, JuMP.index(v)), d._variables)
        variables = d._variables[retained]
        names = [layout.names[layout.id_to_slot[JuMP.index(v)]] for v in variables]
        dictionary = Dictionary(Indices(names), d.dictionary.values[retained])
        positions = Dict(JuMP.index(v) => i for (i, v) in enumerate(variables))
    end
    setfield!(d, :dictionary, dictionary)
    setfield!(d, :_variables, variables)
    setfield!(d, :_variable_positions, positions)
    setfield!(d, :_full, full)
    setfield!(d, :_revision, layout.revision)
    return d
end

# Membership and metadata inspection describe the dataset's current contents;
# indexing or explicit synchronization discovers additions. Adopt shared layout
# updates made by other operations.
function _ensure_data_snapshot!(d::ModelDictionary)
    _check_model_layout(d._layout)
    d._revision == d._layout.revision || _ensure_data_layout!(d)
    return d
end

function _variable_position(d::ModelDictionary, variable::AbstractVariableRef)
    slot = _layout_slot(d._layout, variable)
    d._full && return slot
    return get(d._variable_positions, JuMP.index(variable)) do
        throw(KeyError(variable))
    end
end

function _assert_aligned(a::ModelDictionary, b::ModelDictionary)
    ka, kb = keys(a.dictionary), keys(b.dictionary)
    ka === kb && return nothing
    length(ka) == length(kb) && all(isequal(x, y) for (x, y) in zip(ka, kb)) && return nothing
    throw(DimensionMismatch("ModelDictionary operations require the same variable names in the same order."))
end

# A checked array prevents a previously obtained Window from silently writing to
# detached storage after a dataset synchronizes or its model is refreshed.
struct _ModelValues{T,D} <: AbstractVector{T}
    dataset::D
    revision::UInt
    storage::Vector{T}
end
_ModelValues(d::ModelDictionary) = _ModelValues(d, d._revision, d.dictionary.values)
Base.size(a::_ModelValues) = size(a.storage)
Base.IndexStyle(::Type{<:_ModelValues}) = IndexLinear()
Base.parent(a::_ModelValues) = a.storage
Base.dataids(a::_ModelValues) = Base.dataids(a.storage)
function _check_values(a::_ModelValues)
    _check_model_layout(a.dataset._layout)
    a.revision == a.dataset._revision == a.dataset._layout.revision ||
        throw(ArgumentError("This window is stale after a model-layout refresh. Obtain a new window."))
    a.storage === a.dataset.dictionary.values ||
        throw(ArgumentError("This window is stale after dataset synchronization. Obtain a new window."))
    return nothing
end
@inline function Base.getindex(a::_ModelValues, i::Int)
    _check_values(a)
    @boundscheck checkbounds(a.storage, i)
    return @inbounds a.storage[i]
end
@inline function Base.setindex!(a::_ModelValues, value, i::Int)
    _check_values(a)
    @boundscheck checkbounds(a.storage, i)
    @inbounds a.storage[i] = value
    return value
end

# Bulk operations validate a checked Window once, then run Julia's array loops
# on its actual storage. Scalar Window access continues to check on each access.
_window_storage(w) = _uncheck_model_array(w.data_view)

# Preserve checked public views, including saved lazy broadcasts. Unwrap their
# storage only immediately before a bulk operation after validating its inputs.
_uncheck_model_array(a) = a
function _uncheck_model_array(a::_ModelValues)
    _check_values(a)
    return a.storage
end
function _uncheck_model_array(a::SubArray)
    raw = _uncheck_model_array(parent(a))
    raw === parent(a) && return a
    return @inbounds view(raw, parentindices(a)...)
end
function _uncheck_model_array(a::Base.ReshapedArray)
    raw = _uncheck_model_array(parent(a))
    raw === parent(a) && return a
    return reshape(raw, size(a))
end

"""
    ModelSelection

A prepared snapshot of container coordinates and model storage slots. Create it
with [`prepare_selection`](@ref) and use `d[selection]` with datasets for the same
model. Refreshing the model layout invalidates selections; changing data values
does not. Treat the selection's coordinate and slot arrays as read-only.
"""
struct ModelSelection{L,S,I}
    layout::L
    revision::UInt
    slots::S
    indices::I
    varname::Union{Nothing,AbstractString}
end

_selection_slot(layout, v::AbstractVariableRef) = _layout_slot(layout, v)
_selection_slot(layout, v) = _storage_name_slot(layout, string(v))

function _compact_slots(slots::Vector{Int})
    isempty(slots) && return 1:0
    start = first(slots)
    all(i -> slots[i] == start + i - 1, eachindex(slots)) && return start:last(slots)
    return slots
end

function _selection_varname(layout, container)
    isempty(container) && return nothing
    first(container) isa AbstractVariableRef || return nothing
    return first(split(layout.names[_layout_slot(layout, first(container))], "["; limit=2))
end

"""
    prepare_selection(d::ModelDictionary, container)

Resolve a variable container once. Reuse `d[selection]` across datasets for the
same model without repeating name lookup or rebuilding labelled coordinates.
Selections snapshot mutable containers: prepare a new selection after changing
the container. A model-layout refresh invalidates existing selections.

Registered model containers are cached automatically by `d[container]`. Treat
their contents as fixed between calls to [`refresh_model_layout!`](@ref).
Unregistered arrays and temporary slices are never retained in that cache.
"""
function prepare_selection(d::ModelDictionary, container::AbstractArray)
    _ensure_data_layout!(d)
    return _prepare_selection(d._layout, container, _selection_varname(d._layout, container))
end

function _prepare_selection(layout, container, varname)
    slots = Vector{Int}(undef, length(container))
    for (i, variable) in enumerate(container)
        slots[i] = _selection_slot(layout, variable)
    end
    indices = create_window(nothing, container, varname).indices
    return ModelSelection(layout, layout.revision, _compact_slots(slots), indices, varname)
end

function _container_selection(d, container, varname=nothing)
    _ensure_data_layout!(d)
    layout = d._layout
    # Identity lookup also avoids asking JuMP for the first variable's name on
    # repeated access. Only registered model containers are inserted below.
    cached = get(layout.selection_cache, container, nothing)
    cached === nothing || return cached
    varname === nothing && (varname = _selection_varname(layout, container))
    selection = _prepare_selection(layout, container, varname)
    if varname !== nothing
        symbol = Symbol(varname)
        if haskey(d.model, symbol) && d.model[symbol] === container
            layout.selection_cache[container] = selection
        end
    end
    return selection
end

function Base.getindex(d::ModelDictionary, selection::ModelSelection)
    _ensure_data_layout!(d)
    selection.layout === d._layout || throw(ArgumentError("Selection belongs to a different model."))
    selection.revision == d._layout.revision || throw(ArgumentError(
        "Selection is stale after a model-layout refresh. Prepare a new selection.",
    ))
    slots = if d._full
        selection.slots
    else
        [_variable_position(d, selection.layout.variables[i]) for i in selection.slots]
    end
    data_view = @inbounds view(_ModelValues(d), slots)
    return Window(data_view, selection.indices, selection.varname)
end

function Base.setindex!(d::ModelDictionary, value, selection::ModelSelection)
    _set_window!(d[selection], value)
    return value
end
