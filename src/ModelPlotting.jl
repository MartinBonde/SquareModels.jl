# ModelPlotting - Plot ModelDictionary variables with Makie
#
# The implementation lives in `ext/SquareModelsCairoMakieExt.jl` and is only
# available once CairoMakie is loaded (`using CairoMakie`). Until then these are
# empty generic functions; calling them raises a `MethodError`.

module ModelPlotting

function series end
function plot_series! end
function plot_variable end
function save_figure end

export series, plot_series!, plot_variable, save_figure

end
