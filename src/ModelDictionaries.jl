# ModelDictionaries - Variable-to-value mappings for JuMP models
# Integrated into SquareModels

using Dictionaries
using Parquet2
using DataFrames
using CSV
import GAMS

"""
    ModelDictionary

A dictionary mapping variable names to numeric values, with special support for JuMP variables.

`ModelDictionary` provides convenient syntax for storing and retrieving values
associated with JuMP variables. It supports indexing by variable references,
strings, symbols, or dot notation, and integrates with JuMP's `fix` and `set_start_value`
functions.

# Fields
- `model::AbstractModel`: The JuMP model whose variables are tracked
- `dictionary::Dictionary{String, Union{Nothing, Number}}`: Storage for variable values

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)

# Set values using different access methods
d[x] = 1.0
d["y"] = [1, 2, 3]    # String access
d.y = [1, 2, 3]       # Dot notation

# Get values
d[x]       # 1.0
d.y[1]     # 1
```

See also: [`fix`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
struct ModelDictionary
	model::AbstractModel
	dictionary::Dictionary{String, Union{Nothing, Number}}
	_synced_n_vars::Base.RefValue{Int}
	ModelDictionary(model, dictionary) = new(model, dictionary, Ref(0))
end
@forward ModelDictionary.dictionary (
	Base.keys,
	Base.values,
	Base.isassigned,
	Base.length,
	Base.iterate,
	Base.filter,
	Base.haskey,
	Base.get,
)

function Base.show(io::IO, ::MIME"text/plain", md::ModelDictionary)
	n = length(md)
	print(io, "ModelDictionary with ", n, " entries")
	n == 0 && return
	println(io, ":")
	# Show first few and last few entries, similar to Vector display
	max_show = get(io, :limit, false) ? 10 : n
	half = max_show ÷ 2
	ks = collect(keys(md.dictionary))
	vs = collect(values(md.dictionary))
	key_width = maximum(length ∘ string, ks; init=1)
	for i in eachindex(ks)
		if n > max_show && i == half + 1
			println(io, "  ⋮")
			continue
		elseif n > max_show && half < i < n - half + 1
			continue
		end
		print(io, "  ", lpad(ks[i], key_width), " => ", vs[i])
		i < n && println(io)
	end
end

Base.show(io::IO, md::ModelDictionary) = show(io, MIME"text/plain"(), md)

"""
    ModelDictionary(model::AbstractModel)

Create a dictionary mapping all variables in `model` to values (initially `nothing`).

Supports convenient syntax for getting/setting values using variable references,
symbols, or dot notation.

# Arguments
- `model::AbstractModel`: The JuMP model whose variables to track

# Returns
A `ModelDictionary` with all model variables initialized to `nothing`.

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]  # Dot notation

fix(d)  # Fix all variables to their values in d
```

See also: [`fix`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
function ModelDictionary(m)
	md = ModelDictionary(m, Dictionary{String, Union{Nothing, Number}}())
	add_missing_model_variables!(md)
	return md
end

"""
    ModelDictionary(model::AbstractModel, values::Union{Number, AbstractVector})

Create a dictionary with all variables set to the provided values.

# Arguments
- `model::AbstractModel`: The JuMP model whose variables to track
- `values`: A single number (applied to all) or vector of values

# Returns
A `ModelDictionary` with variables initialized to the given values.
"""
function ModelDictionary(m::AbstractModel, values::Union{Number, AbstractVector})
	d = ModelDictionary(m)
	setindex!.(Ref(d), values, all_variables(m))
	return d
end

Base.copy(md::ModelDictionary) = ModelDictionary(md.model, copy(md.dictionary))

"""
    add_missing_model_variables!(md::ModelDictionary)

Add any JuMP model variables that are not yet in the dictionary.

This is useful after defining new blocks (which create residual variables)
to ensure the dictionary includes all model variables.

New variables are initialized to `nothing`.
"""
function add_missing_model_variables!(md::ModelDictionary)
	n = JuMP.num_variables(md.model)
	n == md._synced_n_vars[] && return
	for v in all_variables(md.model)
		k = name(v)
		if k ∉ keys(md.dictionary)
			insert!(md.dictionary, k, nothing)
		end
	end
	md._synced_n_vars[] = n
end

function Base.setindex!(d::ModelDictionary, value, index::String)
	index ∈ keys(d.dictionary) || add_missing_model_variables!(d)
	sym = Symbol(index)
	index ∉ keys(d.dictionary) && haskey(d.model, sym) && return setindex!(d, value, d.model[sym])
	return setindex!(d.dictionary, value, index)
end
Base.setindex!(d::ModelDictionary, value, index::AbstractVariableRef) = setindex!(d, value, name(index))
Base.setindex!(d::ModelDictionary, value, index::Symbol) = setindex!(d, value, String(index))
Base.setindex!(d::ModelDictionary, value, index::AbstractArray) = setindex!.(Ref(d), value, index)

function Base.getindex(d::ModelDictionary, index::String)
	index ∈ keys(d.dictionary) || add_missing_model_variables!(d)
	index ∈ keys(d.dictionary) && return getindex(d.dictionary, index)
	sym = Symbol(index)
	haskey(d.model, sym) && return getindex(d, d.model[sym])
	return d.dictionary[index] # IndexError
end
Base.getindex(d::ModelDictionary, index::AbstractVariableRef) = getindex(d, name(index))
Base.getindex(d::ModelDictionary, index::Symbol) = getindex(d, String(index))
function Base.getindex(d::ModelDictionary, container::AbstractArray{<:AbstractString}, varname::Union{Nothing, AbstractString}=nothing)
	add_missing_model_variables!(d)
	idx = indexin(String[container...], [keys(d.dictionary)...])
	data_view = @view(d.dictionary.values[idx])
	return create_window(data_view, container, varname)
end
Base.getindex(d::ModelDictionary, s::SparseZeroArray) = getindex(d, s.data)
Base.setindex!(d::ModelDictionary, value, s::SparseZeroArray) = setindex!(d, value, s.data)

function Base.getindex(d::ModelDictionary, container::AbstractArray{<:AbstractVariableRef})
	varname = isempty(container) ? nothing : split(name(first(container)), "[")[1]
	getindex(d, name.(container), varname)
end
Base.getindex(d::ModelDictionary, container::AbstractArray) = getindex(d, string.(container), nothing)

# Filtering with a boolean ModelDictionary (e.g., d[d .> 0])
function Base.getindex(d::ModelDictionary, mask::ModelDictionary)
	ks = collect(keys(d.dictionary))
	vs = collect(values(d.dictionary))
	mask_vs = collect(values(mask.dictionary))
	selected = mask_vs .== true
	ModelDictionary(d.model, Dictionary(ks[selected], vs[selected]))
end


"""
    Window{T, S}

A view into a subset of a `ModelDictionary`, indexed like a JuMP variable container.

`Window` provides array-like access to a slice of a `ModelDictionary` that corresponds
to an indexed JuMP variable (e.g., `y[1:3]`). It allows reading and writing values
using the same indices as the original variable.

This is an internal type typically created automatically when indexing a
`ModelDictionary` with a variable container.

# Fields
- `data_view::T`: View into the underlying dictionary values
- `indices::S`: Index mapping matching the variable container's axes

# Examples
```julia
model = Model()
@variable(model, y[1:3])

d = ModelDictionary(model)
d.y = [10, 20, 30]

w = d[y]     # Returns a Window
w[1]         # 10
w[2] = 25    # Modify through the window
d[y[2]]      # 25
```
"""
struct Window{T, S}
	data_view::T
	indices::S
	varname::Union{Nothing, AbstractString}
end
function create_window(data_view, container, varname::Union{Nothing, AbstractString}=nothing)
	indices = (_->0).(container)
	for (i, idx) in enumerate(eachindex(indices))
		indices[idx] = i
	end
	Window(data_view, indices, varname)
end

function Base.getproperty(w::Window, name::Symbol)
	name == :shaped_view && return reshape(w.data_view, size(w.indices))
	return getfield(w, name)
end

@forward Window.indices (
	Base.length,
	Base.size,
	Base.axes,
	Base.ndims,
	Base.keys,
	Base.lastindex,
)
@forward Window.shaped_view (
	Base.iterate,
	Base.collect,
)

_key_to_tuple(k::JuMP.Containers.DenseAxisArrayKey) = k.I
_key_to_tuple(k::CartesianIndex) = Tuple(k)
_key_to_tuple(k::Tuple) = k
_key_to_tuple(k) = (k,)

function Base.show(io::IO, ::MIME"text/plain", w::Window)
	n = length(w)
	ax = axes(w.indices)
	# Header with variable name if available
	varname_str = w.varname === nothing ? "" : string(w.varname)
	if length(ax) == 1
		print(io, n, "-element Window")
	else
		print(io, join(length.(ax), "×"), " Window")
	end
	n == 0 && return
	println(io, ":")
	# Get keys - for DenseAxisArray this gives the actual index values
	ks = collect(keys(w.indices))
	max_show = get(io, :limit, false) ? 10 : n
	half = max_show ÷ 2
	for (i, k) in enumerate(ks)
		if n > max_show && i == half + 1
			println(io, " ⋮")
			continue
		elseif n > max_show && half < i < n - half + 1
			continue
		end
		# Format key with variable name: varname[k1, k2, ...] or varname[k]
		k_tuple = _key_to_tuple(k)
		idx_str = "[" * join(k_tuple, ", ") * "]"
		key_str = varname_str * idx_str
		print(io, " ", key_str, " => ", w.data_view[w.indices[k_tuple...]])
		i < n && println(io)
	end
end
Base.show(io::IO, w::Window) = show(io, MIME"text/plain"(), w)

Base.getindex(w::Window, index::AbstractArray) = length(index) == 1 ? getindex(w, index[]) : getindex.(Ref(w), index)
function Base.getindex(w::Window, indices...)
	idx = w.indices[indices...]
	idx isa Integer && return w.data_view[idx]
	map(i -> w.data_view[i], Array(idx))
end

Base.setindex!(w::Window, value, index::AbstractArray) = setindex!.(Ref(w), value, index)
Base.setindex!(w::Window, value, indices...) = setindex!.(Ref(w.data_view), value, w.indices[indices...])

# Additional array methods for Window
Base.vec(w::Window) = vec(collect(w))

# Broadcasting support for Window - use shaped_view as the broadcastable representation
Base.broadcastable(w::Window) = w.shaped_view

# For broadcast assignment (w .= x), materialize into the underlying data
function Base.materialize!(w::Window, bc::Base.Broadcast.Broadcasted)
	result = Base.materialize(bc)
	if result isa AbstractArray
		w.data_view .= vec(result)
	else
		w.data_view .= result  # scalar broadcast
	end
	return w
end

Base.in(index::String, d::ModelDictionary) = index ∈ keys(d.dictionary)
Base.in(index::Symbol, d::ModelDictionary) = String(index) ∈ d
Base.in(index::AbstractVariableRef, d::ModelDictionary) = name(index) ∈ d
Base.in(index::AbstractArray, d::ModelDictionary) = all(string.(index) .∈ Ref(d))

function Base.replace!(d::ModelDictionary, old_new::Pair...)
	for (k, v) in zip(keys(d), replace(collect(d), old_new...))
		d[k] = v
	end
	return d
end

function Base.replace(d::ModelDictionary, old_new::Pair...)
	d2 = copy(d)
	return replace!(d2, old_new...)
end

# ----------------------------------------------------------------------------------------------------------------------
# JuMP extensions for ModelDictionary
# ----------------------------------------------------------------------------------------------------------------------
"""
    fix(var::VariableRef, d::ModelDictionary)

Fix a single variable to its value in the dictionary.

# Arguments
- `var::VariableRef`: The variable to fix
- `d::ModelDictionary`: Dictionary containing the target value

# Examples
```julia
d = ModelDictionary(model)
d[x] = 5.0
fix(x, d)  # x is now fixed to 5.0
```

See also: [`ModelDictionary`](@ref), [`set_start_value`](@ref)
"""
JuMP.fix(var::VariableRef, d::ModelDictionary) = fix(var, d[var], force=true)

"""
    fix(variables::AbstractArray, d::ModelDictionary)

Fix a collection of variables to their values in the dictionary.

# Arguments
- `variables::AbstractArray`: Array of variable references
- `d::ModelDictionary`: Dictionary containing the target values
"""
JuMP.fix(variables::AbstractArray, d::ModelDictionary) = fix.(variables, Ref(d))

"""
    fix(model::AbstractModel, d::ModelDictionary)
    fix(d::ModelDictionary)

Fix all variables in a JuMP model to their corresponding values in a ModelDictionary.

Variables with `nothing` values in the dictionary are skipped.

# Arguments
- `model::AbstractModel`: The model whose variables to fix (optional if using `fix(d)`)
- `d::ModelDictionary`: Dictionary containing the target values

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]

fix(d)  # Fix all variables to their values in d
# Equivalent to: fix(model, d)
```

See also: [`ModelDictionary`](@ref), [`set_start_value`](@ref), [`value_dict`](@ref)
"""
function JuMP.fix(model::AbstractModel, d::ModelDictionary)
	for var in all_variables(model)
		v = d[var]
		isnothing(v) && error("Cannot fix variable $(name(var)): no value in dictionary. Set it explicitly (e.g., to 0) before fixing.")
		fix(var, v, force=true)
	end
end
function JuMP.fix(d::ModelDictionary)
	n_model_vars = length(all_variables(d.model))
	if length(d) == n_model_vars
		# Full dictionary: require all variables to have values
		fix(d.model, d)
	else
		# Subset dictionary: only fix variables present in the dictionary
		for (k, v) in pairs(d.dictionary)
			isnothing(v) && error("Cannot fix variable $k: no value in dictionary. Set it explicitly (e.g., to 0) before fixing.")
			fix(variable_by_name(d.model, k), v, force=true)
		end
	end
end

"""
    set_start_value(var::VariableRef, d::ModelDictionary)

Set the starting value of a variable from a ModelDictionary.

# Arguments
- `var::VariableRef`: The variable to set the start value for
- `d::ModelDictionary`: Dictionary containing the start value

See also: [`ModelDictionary`](@ref), [`fix`](@ref)
"""
JuMP.set_start_value(var::VariableRef, values::ModelDictionary) = set_start_value(var, values[var]::Number)

"""
    set_start_value(variables::AbstractArray, d::ModelDictionary)

Set the starting values of a collection of variables from a ModelDictionary.

# Arguments
- `variables::AbstractArray`: Array of variable references
- `d::ModelDictionary`: Dictionary containing the start values
"""
JuMP.set_start_value(variables::AbstractArray, values::ModelDictionary) = set_start_value.(variables, Ref(values))

"""
    set_start_value(model::AbstractModel, d::ModelDictionary)
    set_start_value(d::ModelDictionary)

Set starting values for all variables in a model from a ModelDictionary.

Starting values provide hints to the solver about where to begin the optimization.

# Arguments
- `model::AbstractModel`: The model whose variables to set (optional if using `set_start_value(d)`)
- `d::ModelDictionary`: Dictionary containing the start values

# Examples
```julia
model = Model()
@variable(model, x)
@variable(model, y[1:3])

d = ModelDictionary(model)
d[x] = 1.0
d.y = [1, 2, 3]

set_start_value(d)  # Set start values for all variables
```

See also: [`ModelDictionary`](@ref), [`fix`](@ref), [`value_dict`](@ref)
"""
function JuMP.set_start_value(model::AbstractModel, d::ModelDictionary)
	for var in all_variables(model)
		v = d[var]
		isnothing(v) && error("Cannot set start value for $(name(var)): no value in dictionary. Set it explicitly before calling set_start_value.")
		set_start_value(var, v)
	end
end
function JuMP.set_start_value(d::ModelDictionary)
	n_model_vars = length(all_variables(d.model))
	if length(d) == n_model_vars
		# Full dictionary: require all variables to have values
		set_start_value(d.model, d)
	else
		# Subset dictionary: only set start values for variables present in the dictionary
		for (k, v) in pairs(d.dictionary)
			isnothing(v) && error("Cannot set start value for $k: no value in dictionary. Set it explicitly before calling set_start_value.")
			set_start_value(variable_by_name(d.model, k), v)
		end
	end
end

"""
    value_dict(model::AbstractModel) → ModelDictionary

Extract the solution values of all variables as a ModelDictionary.

Call this after `optimize!(model)` to capture the solution in a dictionary
that can be used for warm-starting, comparing solutions, or fixing variables.

# Arguments
- `model::AbstractModel`: A solved JuMP model

# Returns
A `ModelDictionary` containing the optimal value of each variable.

# Examples
```julia
model = Model(Ipopt.Optimizer)
@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + y == 10)
@objective(model, Max, x + 2y)

optimize!(model)

d = value_dict(model)
d[x]  # Optimal value of x
d[y]  # Optimal value of y

# Use solution as starting point for another solve
set_start_value(d)
```

See also: [`ModelDictionary`](@ref), [`fix`](@ref), [`set_start_value`](@ref)
"""
value_dict(model::AbstractModel) = ModelDictionary(model, value.(all_variables(model)))

# ----------------------------------------------------------------------------------------------------------------------
# Parquet serialization
# ----------------------------------------------------------------------------------------------------------------------
"""
    parse_variable_name(name::String) → (base_name, indices)

Parse a JuMP variable name into its base name and index string.

# Examples
```julia
parse_variable_name("K[2025]")       # ("K", "2025")
parse_variable_name("cᵃ[15,2025]")   # ("cᵃ", "15,2025")
parse_variable_name("σˣ")            # ("σˣ", "")
```
"""
function parse_variable_name(name::String)
	m = match(r"^(.+?)\[(.+)\]$", name)
	isnothing(m) && return (name, "")
	return (m.captures[1], m.captures[2])
end

"""
    unload(path::AbstractString, d::ModelDictionary)

Save a ModelDictionary to a Parquet file.

The dictionary is stored as a table with columns:
- `variable`: The base variable name (e.g., "K", "cᵃ")
- `indices`: The index string (e.g., "2025", "15,2025", "" for scalars)
- `value`: The numeric value

# Arguments
- `path`: File path (typically ending in .parquet)
- `d`: The ModelDictionary to save

# Examples
```julia
d = value_dict(model)
unload("solution.parquet", d)
```

See also: [`load`](@ref), [`ModelDictionary`](@ref)
"""
function unload(path::AbstractString, d::ModelDictionary)
	rows = NamedTuple{(:variable, :indices, :value), Tuple{String, String, Float64}}[]
	for (k, v) in pairs(d.dictionary)
		isnothing(v) && continue
		base, indices = parse_variable_name(k)
		push!(rows, (; variable=base, indices=indices, value=Float64(v)))
	end
	Parquet2.writefile(path, DataFrame(rows))
end

"""
    load(path::AbstractString, model::AbstractModel; renames...) → ModelDictionary
    load(path::AbstractString, model::AbstractModel, renames::Pair...) → ModelDictionary

Load a ModelDictionary from a Parquet, CSV, or GDX file.

Iterates over all variables in the model and looks up their values in the data file.
Variables not found in the file will have `nothing` values.

For Parquet files, supports both the simple format (variable, indices, value) and Gekko's format
(with id, name, dim1, dim2, period, value columns).

CSV files use the simple format only.

For GDX files, reads parameters and uses their values. Multi-dimensional parameters have
their indices joined with commas.

# Arguments
- `path`: Path to the Parquet, CSV, or GDX file
- `model`: The JuMP model to associate with the dictionary
- `renames`: Optional name mappings to load variables from differently-named data. Can be passed as keyword arguments
  or as `Pair` arguments. For simple renames, use `ModelVar="GdxName"`. For slicing (extracting a subset of a
  higher-dimensional GDX symbol), use the syntax `ModelVar="GdxSymbol[fixed1,fixed2,:,...]"` where `:` marks
  positions that correspond to the model variable's indices.

# Returns
A `ModelDictionary` populated with values from the file.
Variables in the model that aren't in the file will have `nothing` values.

# Examples
```julia
d = load("solution.parquet", model)
d = load("data.gdx", model)
set_start_value(d)  # Use loaded values as starting point

# Load with name remapping (similar to GAMS \$LOAD path Y=OtherY;)
d = load("data.parquet", model, Y => "OtherY", X => "DataX")
d = load("data.gdx", model; N_a="nPop", L_a="nLHh")

# Slice a higher-dimensional GDX symbol into a model variable:
# If GDX has vC[commodity,t] and model has C[t], extract vC[:cTot,t] into C[t]
d = load("data.gdx", model;
    C = "vC[:cTot,:]",      # C[t] ← vC[:cTot, t]
    X = "vX[:xTot,:]",      # X[t] ← vX[:xTot, t]
    K = "vK[:iTot,:tot,:]", # K[t] ← vK[:iTot, :tot, t]
)
```

See also: [`unload`](@ref), [`ModelDictionary`](@ref)
"""
function load(path::AbstractString, model::AbstractModel, renames::Pair...; kwargs...)
	rename_dict, slice_dict = _build_rename_and_slice_dicts(renames, kwargs)

	# Dispatch based on file extension
	ext = lowercase(path)
	if endswith(ext, ".gdx")
		return _load_gdx(path, model, rename_dict, slice_dict)
	elseif endswith(ext, ".csv")
		return _load_csv(path, model, rename_dict, slice_dict)
	else
		return _load_parquet(path, model, rename_dict, slice_dict)
	end
end

"""Load from a GDX file using GAMS.jl's read_gdx."""
function _load_gdx(path::AbstractString, model::AbstractModel, rename_dict::Dict{String, String}, slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}})
	gdx = GAMS.read_gdx(path)

	# Build index for O(1) lookup: (variable, indices) => value
	data_index = Dict{Tuple{String, String}, Float64}()

	for sym_name in keys(gdx.symbols)
		sym = gdx.symbols[sym_name]
		df = sym.records
		isempty(df) && continue

		# Get the value column name (differs by symbol type)
		value_col = if hasproperty(df, :value)
			:value
		elseif hasproperty(df, :level)
			:level
		else
			continue
		end

		# Get domain columns (all columns except value/level/marginal/etc.)
		domain_cols = [n for n in names(df) if n ∉ ("value", "level", "marginal", "lower", "upper", "scale")]

		for row in eachrow(df)
			indices_str = join([string(row[col]) for col in domain_cols], ",")
			data_index[(string(sym_name), indices_str)] = row[value_col]
		end
	end

	d = ModelDictionary(model)
	for var in all_variables(model)
		base, indices = _var_to_key(var)

		# Check for slice mapping first
		if haskey(slice_dict, base)
			gdx_symbol, fixed_indices, wildcard_positions = slice_dict[base]
			lookup_key = _build_slice_key(indices, fixed_indices, wildcard_positions)
			key = (gdx_symbol, lookup_key)
		else
			# Use renamed base if specified, otherwise use original
			lookup_base = get(rename_dict, base, base)
			key = (lookup_base, indices)
		end

		if haskey(data_index, key)
			d[var] = data_index[key]
		end
	end
	return d
end

"""Normalize a tabular cell to String, treating missing as empty."""
_tab_str(x) = ismissing(x) ? "" : string(x)

const IndexValue = Union{Symbol, Int}

"""Parse one index component from a comma-joined indices string."""
_parse_index_part(s) = (i = tryparse(Int, s)) === nothing ? Symbol(s) : i

"""Parse comma-joined indices from the simple (variable, indices, value) format."""
function _parse_index_tuple(indices::AbstractString)
	isempty(strip(indices)) && error("Empty indices string")
	return Tuple(_parse_index_part(strip(s)) for s in split(indices, ","))
end

"""Extract rows in simple (variable, indices, value) format."""
function _simple_format_df(df::DataFrame)
	("variable" in names(df) && "indices" in names(df)) ||
		error("Expected columns: (variable, indices, value)")
	return df[.!ismissing.(df.value), [:variable, :indices, :value]]
end

function _read_simple_df(path::AbstractString)
	ext = lowercase(path)
	df = endswith(ext, ".csv") ? CSV.read(path, DataFrame) : DataFrame(Parquet2.Dataset(path))
	return _simple_format_df(df)
end

function _read_simple_keyed(path::AbstractString; variable=nothing)
	df = _read_simple_df(path)
	if variable !== nothing
		df = df[_tab_str.(df.variable) .== string(variable), :]
	end
	return Dict(_parse_index_tuple(_tab_str(row.indices)) => Float64(row.value) for row in eachrow(df))
end

"""
Read index components from a simple `(variable, indices, value)` CSV or Parquet file.

Returns a `Vector{Union{Symbol, Int}}` when every row has one index, otherwise an
`n×d` `Matrix{Union{Symbol, Int}}` with one column per index dimension.
"""
function read_indices(path::AbstractString)
	df = _read_simple_df(path)
	isempty(df) && return IndexValue[]
	parsed = [_parse_index_tuple(_tab_str(row.indices)) for row in eachrow(df)]
	ndims = length(first(parsed))
	for p in parsed
		length(p) == ndims || error("Inconsistent number of indices across rows in $path")
	end
	if ndims == 1
		return IndexValue[only(p) for p in parsed]
	end
	mat = Matrix{IndexValue}(undef, length(parsed), ndims)
	for (i, p) in enumerate(parsed)
		mat[i, :] = collect(p)
	end
	return mat
end

"""Read a simple `(variable, indices, value)` file as a `SparseZeroArray` keyed by parsed indices."""
read_sparse_array(path::AbstractString; variable=nothing) = SparseZeroArray(_read_simple_keyed(path; variable))
read_sparse_array(path::AbstractString, variable) = read_sparse_array(path; variable)

"""Read values from a file aligned to a JuMP variable container's keys.

Returns an array with the same shape and key order as `var`. Missing entries
use `default` (default `nothing`).
"""
function read_variable(path::AbstractString, var; default=nothing, variable=base_name(var))
	data = _read_simple_keyed(path; variable)
	return [get(data, _key_to_tuple(key), default) for key in keys(var)]
end

"""Load from a DataFrame in simple (variable, indices, value) format."""
function _load_simple(df::DataFrame, model::AbstractModel, rename_dict::Dict{String, String}, slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}})
	data_index = Dict{Tuple{String, String}, Float64}()
	for row in eachrow(df)
		key = (_tab_str(row.variable), _tab_str(row.indices))
		data_index[key] = row.value
	end

	d = ModelDictionary(model)
	for var in all_variables(model)
		base, indices = _var_to_key(var)

		# Check for slice mapping first
		if haskey(slice_dict, base)
			src_symbol, fixed_indices, wildcard_positions = slice_dict[base]
			lookup_key = _build_slice_key(indices, fixed_indices, wildcard_positions)
			key = (src_symbol, lookup_key)
		else
			# Use renamed base if specified, otherwise use original
			lookup_base = get(rename_dict, base, base)
			key = (lookup_base, indices)
		end

		if haskey(data_index, key)
			d[var] = data_index[key]
		end
	end
	return d
end

"""Load from a Parquet file."""
function _load_parquet(path::AbstractString, model::AbstractModel, rename_dict::Dict{String, String}, slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}})
	df = DataFrame(Parquet2.Dataset(path))
	data_df = if "variable" in names(df) && "indices" in names(df)
		_simple_format_df(df)
	elseif "name" in names(df) && "id" in names(df)
		_convert_gekko_format(df)
	else
		error("Unknown parquet format. Expected columns: (variable, indices, value) or Gekko format (id, name, dim1, dim2, period, value)")
	end
	return _load_simple(data_df, model, rename_dict, slice_dict)
end

"""Load from a CSV file."""
function _load_csv(path::AbstractString, model::AbstractModel, rename_dict::Dict{String, String}, slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}})
	return _load_simple(_simple_format_df(CSV.read(path, DataFrame)), model, rename_dict, slice_dict)
end


"""
Build rename and slice dictionaries from Pair arguments and keyword arguments.

Values containing brackets (e.g., "vC[:cTot,:]") are parsed as slice specifications.
Simple strings are treated as renames.
"""
function _build_rename_and_slice_dicts(renames::Tuple, kwargs)
	rename_dict = Dict{String, String}()
	slice_dict = Dict{String, Tuple{String, Vector{String}, Vector{Int}}}()

	function process_mapping(model_var::String, spec::String)
		if contains(spec, "[")
			# Slice specification
			gdx_symbol, fixed_indices, wildcard_positions = _parse_slice_spec(spec)
			slice_dict[model_var] = (gdx_symbol, fixed_indices, wildcard_positions)
		else
			# Simple rename
			rename_dict[model_var] = spec
		end
	end

	for (k, v) in renames
		process_mapping(_to_base_name(k), string(v))
	end
	for (k, v) in pairs(kwargs)
		process_mapping(string(k), string(v))
	end

	return rename_dict, slice_dict
end

"""Extract base variable name from various input types. Always returns String."""
_to_base_name(x::Symbol) = string(x)
_to_base_name(x::AbstractString) = String(x)
_to_base_name(x::AbstractVariableRef) = String(first(parse_variable_name(name(x))))
function _to_base_name(x::AbstractArray{<:AbstractVariableRef})
	# For JuMP variable containers, extract base name from first element
	String(first(parse_variable_name(name(first(x)))))
end

"""Convert Gekko parquet format to simple (variable, indices, value) format."""
function _convert_gekko_format(df::DataFrame)
	# Separate metadata rows (have name) from data rows (have value but no name)
	metadata = df[.!ismissing.(df.name), [:id, :name, :dim1, :dim2]]
	data = df[ismissing.(df.name) .& .!isnan.(coalesce.(df.value, NaN)), [:id, :period, :value]]

	# Join metadata to data
	joined = leftjoin(data, metadata, on=:id)

	# Build indices string from dim1, dim2, period
	function build_indices(row)
		parts = String[]
		!ismissing(row.dim1) && push!(parts, string(row.dim1))
		!ismissing(row.dim2) && push!(parts, string(row.dim2))
		!ismissing(row.period) && push!(parts, string(row.period))
		return join(parts, ",")
	end

	result = DataFrame(
		variable = coalesce.(joined.name, ""),
		indices = build_indices.(eachrow(joined)),
		value = Float64.(joined.value)
	)

	# Filter out rows with empty variable names
	return result[result.variable .!= "", :]
end

"""
Convert a JuMP variable to the (variable, indices) key format.
E.g., x → ("x", ""), y[1,2] → ("y", "1,2")
"""
function _var_to_key(var::AbstractVariableRef)
	base, indices = parse_variable_name(name(var))
	return (base, indices)
end

"""
Parse a slice specification string into (gdx_symbol, fixed_indices, wildcard_positions).

The string format is "symbol[idx1,idx2,...]" where:
- Fixed indices (like :cTot or cTot) become part of the lookup key
- Wildcards (:) indicate positions that should be filled from the target variable's indices

# Examples
```julia
_parse_slice_spec("vC[:cTot,:]")   # ("vC", ["cTot"], [2])
_parse_slice_spec("vK[:iTot,:tot,:]")  # ("vK", ["iTot", "tot"], [3])
_parse_slice_spec("nPop")  # ("nPop", [], [])  - simple rename
```
"""
function _parse_slice_spec(spec::AbstractString)
	# Handle simple rename case (no brackets)
	m = match(r"^([^\[]+)\[(.+)\]$", spec)
	isnothing(m) && return (spec, String[], Int[])

	gdx_symbol = m.captures[1]
	indices_str = m.captures[2]

	# Split by comma, respecting that indices might contain colons
	parts = split(indices_str, ",")

	fixed_indices = String[]
	wildcard_positions = Int[]

	for (i, part) in enumerate(parts)
		part = strip(part)
		if part == ":"
			push!(wildcard_positions, i)
		else
			# Strip leading colon if present (e.g., :cTot -> cTot)
			idx = startswith(part, ":") ? part[2:end] : part
			push!(fixed_indices, idx)
		end
	end

	return (gdx_symbol, fixed_indices, wildcard_positions)
end

"""
Build the GDX lookup key for a slice mapping.

Given a target variable's indices and a slice specification, constructs the
indices string that should be looked up in the GDX file.

# Arguments
- `target_indices`: Indices from the target variable (e.g., "2025" or "15,2025")
- `fixed_indices`: Fixed parts of the slice (e.g., ["cTot"])
- `wildcard_positions`: Positions where target indices should be inserted

# Example
For target C[2025] with slice "vC[:cTot,:]":
- target_indices = "2025"
- fixed_indices = ["cTot"]
- wildcard_positions = [2]
- Result: "cTot,2025"
"""
function _build_slice_key(target_indices::AbstractString, fixed_indices::Vector{String}, wildcard_positions::Vector{Int})
	isempty(wildcard_positions) && return join(fixed_indices, ",")

	target_parts = isempty(target_indices) ? String[] : split(target_indices, ",")

	# Total positions = fixed + wildcards
	total_positions = length(fixed_indices) + length(wildcard_positions)
	result = Vector{String}(undef, total_positions)

	fixed_idx = 1
	target_idx = 1

	for pos in 1:total_positions
		if pos in wildcard_positions
			result[pos] = target_idx <= length(target_parts) ? string(target_parts[target_idx]) : ""
			target_idx += 1
		else
			result[pos] = fixed_indices[fixed_idx]
			fixed_idx += 1
		end
	end

	return join(result, ",")
end

# ----------------------------------------------------------------------------------------------------------------------
# Dot access
# ----------------------------------------------------------------------------------------------------------------------
Base.setproperty!(d::ModelDictionary, name::Symbol, value) = setindex!(d, value, String(name))
Base.getproperty(d::ModelDictionary, sym::Symbol) = sym in fieldnames(typeof(d)) ? getfield(d, sym) : d[String(sym)]

# ----------------------------------------------------------------------------------------------------------------------
# Broadcasting
# ----------------------------------------------------------------------------------------------------------------------
struct ModelDictionaryStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:ModelDictionary}) = ModelDictionaryStyle()
Base.BroadcastStyle(::ModelDictionaryStyle, ::Broadcast.DefaultArrayStyle{0}) = ModelDictionaryStyle()
Base.BroadcastStyle(s::ModelDictionaryStyle, ::ModelDictionaryStyle) = s

# ModelDictionary participates directly in broadcasting (not converted via collect)
Base.broadcastable(md::ModelDictionary) = md
Base.axes(md::ModelDictionary) = (Base.OneTo(length(md)),)
Base.getindex(md::ModelDictionary, i::Int) = md.dictionary.values[i]

# Find the first ModelDictionary in broadcast arguments (including nested Broadcasted)
_find_model_dict(md::ModelDictionary) = md
_find_model_dict(bc::Broadcast.Broadcasted) = _find_model_dict(bc.args)
_find_model_dict(::Any) = nothing
function _find_model_dict(args::Tuple)
	for arg in args
		result = _find_model_dict(arg)
		isnothing(result) || return result
	end
	nothing
end

_bc_collect(md::ModelDictionary) = collect(md.dictionary.values)
_bc_collect(x) = x

# Lift a function to propagate nothing (like NaN propagation)
_lift(f) = (args...) -> any(isnothing, args) ? nothing : f(args...)

function Base.copy(bc::Broadcast.Broadcasted{ModelDictionaryStyle})
	md = _find_model_dict(bc.args)
	flat = Broadcast.flatten(bc)
	# Unwrap ModelDictionaries to their values, broadcast scalars normally
	unwrapped = map(_bc_collect, flat.args)
	# Lift the function to handle nothing values
	new_values = broadcast(_lift(flat.f), unwrapped...)
	ModelDictionary(md.model, Dictionary(keys(md.dictionary), new_values))
end

# ==============================================================================
# Comparison utilities
# ==============================================================================
"""
	keys_match(a::ModelDictionary, b::ModelDictionary) -> Bool

Check if two ModelDictionaries have matching structure: same keys with `nothing`
values in the same positions.

# Example
```julia
if keys_match(baseline, scenario)
	diffs = abs.(baseline .- scenario)
	# safe to compare numerically
end
```
"""
function keys_match(a::ModelDictionary, b::ModelDictionary)
	keys(a) == keys(b) || return false
	for k in keys(a)
		xor(isnothing(a[k]), isnothing(b[k])) && return false
	end
	return true
end

"""
	assert_no_diff(a::ModelDictionary, b::ModelDictionary; atol=1e-6, rtol=0.0, msg="")

Assert that two ModelDictionaries have no significant differences.

A difference passes if BOTH conditions are met:
1. `|a - b| <= atol` (absolute tolerance must always be satisfied)
2. Either `|b| <= atol` (reference is small, so relative doesn't apply)
   OR `|a - b| / |b| <= rtol` (relative tolerance is satisfied)

This approach uses absolute tolerance for small values and relative tolerance
for large values, avoiding issues with division by near-zero references.

Errors immediately if keys don't match (different keys or nothing/value mismatch).

# Example
```julia
assert_no_diff(pre_solve, post_solve, atol=1e-6, msg="Zero shock test failed")
assert_no_diff(baseline, scenario, atol=1e-6, rtol=0.01, msg="Differences exceed 1%")
```
"""
function assert_no_diff(a::ModelDictionary, b::ModelDictionary; atol::Real=1e-6, rtol::Real=0.0, msg::String="")
	error_msg = isempty(msg) ? "" : "$msg\n"

	# Check structural match
	if keys(a) != keys(b)
		error("$(error_msg)Cannot compare: dictionaries have different keys")
	end
	mismatches = [k for k in keys(a) if xor(isnothing(a[k]), isnothing(b[k]))]
	if !isempty(mismatches)
		error("$(error_msg)Cannot compare: $(length(mismatches)) keys have nothing/value mismatch: $(first(mismatches, 10))$(length(mismatches) > 10 ? "..." : "")")
	end

	# Check differences using MAKRO-style logic:
	# Pass if: |diff| <= atol AND (|ref| <= atol OR |diff/ref| <= rtol)
	violations = Tuple{String, Float64, Float64, Any, Any}[]  # (key, abs_diff, rel_diff, v1, v2)
	for k in keys(a)
		v1, v2 = a[k], b[k]
		isnothing(v1) && continue
		d = abs(v1 - v2)
		abs_ref = abs(v2)
		# Absolute check
		d <= atol && continue
		# If reference is small, only absolute matters (already failed above)
		if abs_ref <= atol
			push!(violations, (k, d, Inf, v1, v2))
		else
			# Check relative tolerance
			rel_d = d / abs_ref
			if rel_d > rtol
				push!(violations, (k, d, rel_d, v1, v2))
			end
		end
	end
	if !isempty(violations)
		sort!(violations, by=x -> -x[2])
		lines = [isinf(rd) ? "  $k: diff=$d ($v1 vs $v2)" : "  $k: diff=$d ($(round(rd*100, digits=2))%) ($v1 vs $v2)"
		         for (k, d, rd, v1, v2) in violations]
		tol_desc = rtol > 0 ? "atol=$atol, rtol=$rtol" : "atol=$atol"
		error("$(error_msg)$(length(violations)) differences exceed tolerance ($tol_desc):\n" * join(lines, "\n"))
	end
	return true
end
