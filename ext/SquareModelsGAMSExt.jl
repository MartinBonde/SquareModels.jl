module SquareModelsGAMSExt

import GAMS
using JuMP: unsafe_backend, optimizer_with_attributes

function _gams_optimizer(; system_dir::AbstractString, working_dir::AbstractString=mktempdir(), solver::AbstractString="CONOPT4")
	isdir(system_dir) || error("GAMS system directory not found: $system_dir")
	mkpath(working_dir)
	# GAMS.jl's optimize! swaps the args of GAMSWorkspace(working_dir, system_dir),
	# so build the workspace here (correct order) instead of setting SysDir/WorkDir attributes.
	workspace = GAMS.GAMSWorkspace(working_dir, system_dir)
	return optimizer_with_attributes(
		() -> GAMS.Optimizer(workspace),
		GAMS.ModelType() => "CNS",
		"CNS" => solver,
		GAMS.Solver() => solver,
		"lmmxsf" => 1,
	)
end

function _gams_annotation_paths(model)
	backend = unsafe_backend(model)
	backend isa GAMS.Optimizer || return String[]
	work = backend.gamswork
	work === nothing && return String[]
	return filter(isfile, joinpath.(work.working_dir, ("moi.lst", "moi.gms")))
end

end
