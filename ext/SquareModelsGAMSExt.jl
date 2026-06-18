module SquareModelsGAMSExt

import GAMS
using JuMP: AbstractModel, Model, all_variables, unsafe_backend
using JuMP: set_objective_sense, set_optimizer_attribute, FEASIBILITY_SENSE
import SquareModels: ModelDictionary, _var_to_key, _build_slice_key

function _gams_cns_model(; system_dir::AbstractString, working_dir::AbstractString=mktempdir(), solver::AbstractString="CONOPT4")
	isdir(system_dir) || error("GAMS system directory not found: $system_dir")
	mkpath(working_dir)
	# GAMS.jl's optimize! swaps the args of GAMSWorkspace(working_dir, system_dir),
	# so build the workspace here (correct order) instead of setting SysDir/WorkDir attributes.
	workspace = GAMS.GAMSWorkspace(working_dir, system_dir)
	model = Model(() -> GAMS.Optimizer(workspace))
	set_objective_sense(model, FEASIBILITY_SENSE)
	set_optimizer_attribute(model, GAMS.ModelType(), "CNS")
	set_optimizer_attribute(model, "CNS", solver)
	set_optimizer_attribute(model, GAMS.Solver(), solver)
	set_optimizer_attribute(model, "lmmxsf", 1)
	return model
end

function _gams_lst_path(model)
	backend = unsafe_backend(model)
	backend isa GAMS.Optimizer || return nothing
	work = backend.gamswork
	work === nothing && return nothing
	path = joinpath(work.working_dir, "moi.lst")
	return isfile(path) ? path : nothing
end

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

end
