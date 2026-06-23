module SquareModelsGDXInterfaceExt

import GDXInterface
import SquareModels
using DataFrames: DataFrame, eachrow, names
using JuMP: AbstractModel, all_variables
using SquareModels: ModelDictionary, _build_slice_key, _var_to_key

function SquareModels._load_gdx(
	path::AbstractString,
	model::AbstractModel,
	rename_dict::Dict{String, String},
	slice_dict::Dict{String, Tuple{String, Vector{String}, Vector{Int}}},
)
	gdx = GDXInterface.read_gdx(String(path), DataFrame)

	data_index = Dict{Tuple{String, String}, Float64}()
	for (_, sym) in gdx
		value_col = sym isa GDXInterface.GDXParameter ? :value :
			sym isa Union{GDXInterface.GDXVariable, GDXInterface.GDXEquation} ? :level : continue
		df = sym.records
		isempty(df) && continue

		domain_cols = names(df)[1:length(sym.domain)]
		for row in eachrow(df)
			indices_str = join((string(row[col]) for col in domain_cols), ",")
			data_index[(sym.name, indices_str)] = row[value_col]
		end
	end

	d = ModelDictionary(model)
	for var in all_variables(model)
		base, indices = _var_to_key(var)
		key = if haskey(slice_dict, base)
			gdx_symbol, fixed_indices, wildcard_positions = slice_dict[base]
			(gdx_symbol, _build_slice_key(indices, fixed_indices, wildcard_positions))
		else
			(get(rename_dict, base, base), indices)
		end

		if haskey(data_index, key)
			d[var] = data_index[key]
		end
	end
	return d
end

end
