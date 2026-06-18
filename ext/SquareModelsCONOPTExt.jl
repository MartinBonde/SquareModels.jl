module SquareModelsCONOPTExt

import CONOPT
using JuMP: Model, set_objective_sense, set_optimizer_attribute, FEASIBILITY_SENSE

function _conopt_model(; options...)
	model = Model(CONOPT.Optimizer)
	set_objective_sense(model, FEASIBILITY_SENSE)
	set_optimizer_attribute(model, "lmmxsf", 1)
	for (key, value) in options
		set_optimizer_attribute(model, string(key), value)
	end
	return model
end

end
