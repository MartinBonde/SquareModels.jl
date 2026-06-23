module TestModelPlotting

using Test
using JuMP
using SquareModels

@testset "Plotting without Makie" begin
	if Base.get_extension(SquareModels, :SquareModelsMakieExt) === nothing
		err = try
			plotseries([labeled([1.0, 2.0], "demo")])
			nothing
		catch err
			err
		end
		@test err isa ErrorException
		@test occursin("using CairoMakie", sprint(showerror, err))
	end
end

using Makie

@testset "Makie extension plotseries methods" begin
	series = [labeled([1.0, 2.0], "demo")]
	@test plotseries(series) isa Makie.Figure
	@test plotseries(only(series)) isa Makie.Figure
end

SquareModels.ModelPlotting.plotseries(series::Vector{SquareModels.LabeledSeries}; kwargs...) = series

model = Model()
JuMP.@variables model begin
	x
	p[[:hh, :firm], 2020:2021]
	q[[:hh, :firm], 2020:2021]
end

baseline = ModelDictionary(model)
baseline[x] = 3
for h in [:hh, :firm], t in 2020:2021
	baseline[p[h, t]] = h == :hh ? t - 2019 : 10 * (t - 2019)
	baseline[q[h, t]] = h == :hh ? 2 : 3
end

shock = ModelDictionary(model)
shock[x] = 4
for h in [:hh, :firm], t in 2020:2021
	shock[p[h, t]] = 2 * baseline[p[h, t]]
	shock[q[h, t]] = baseline[q[h, t]]
end

@testset "Model expression evaluation" begin
	@test @evalexpr(baseline, x) == 3
	@test @evalexpr(baseline, p[:hh, 2020] * q[:hh, 2020]) == 2
	@test (@evalexpr baseline p[:hh, 2020] * q[:hh, 2020]) == 2
	@test @evalexpr(baseline, p[:firm, :] * q[:firm, :]) == [30, 60]
	@test @evalexpr(baseline, p * q) == [2 4; 30 60]
	@test @prt(baseline, [x, p[:hh, 2021] * q[:hh, 2021]]) == [3, 4]
	fq = 2
	@test @prt(baseline, p[:hh, 2021] * q[:hh, 2021] / fq) == 2.0
	@test @prt(baseline, (p[:hh, 2021] * q[:hh, 2021], p[:firm, 2021] * q[:firm, 2020] / fq)) == (4, 30.0)
	@test isequal(@prt(:p, baseline, p[:hh, :]), [NaN, 100.0])
	@test @prt(:m, (shock, baseline), p[:hh, :]) == [1.0, 2.0]
	@test @prt(:q, (shock, baseline), p[:hh, :]) == [100.0, 100.0]
	@test isequal(@evalexpr([:n, :p], baseline, p[:hh, :]), [[1, 2], [NaN, 100.0]])
	@test (@evalexpr :q (shock, baseline) p[:hh, :]) == [100.0, 100.0]
	series = @plot :q (shock, baseline) p[:hh, :]
	@test length(series) == 1
	@test series[1].label == "p[:hh, :] <q>"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]
	series = @plot :p baseline p
	@test length(series) == 2
	@test series[1].label == "p[hh] <p>"
	@test series[1].x == [2020, 2021]
	@test isequal(series[1].y, [NaN, 100.0])
	@test series[2].label == "p[firm] <p>"
	@test series[2].x == [2020, 2021]
	@test isequal(series[2].y, [NaN, 100.0])
	series = @plot :q (shock, baseline) p
	@test length(series) == 2
	@test series[1].y == [100.0, 100.0]
	@test series[2].y == [100.0, 100.0]

	set_default_source!(baseline)
	@test @prt(p[:hh, :]) == [1, 2]
	@test isequal(@prt(:p, p[:hh, :]), [NaN, 100.0])
	@test isequal((@prt :p p[:hh, :]), [NaN, 100.0])

	set_default_source!(baseline => shock)
	@test @prt(:q, p[:hh, :]) == [100.0, 100.0]

	set_default_operator!(:q)
	@test @prt(p[:hh, :]) == [100.0, 100.0]

	set_default_source!(baseline)
	@test @prt(:q, p[:hh, :]) == [0.0, 0.0]

	set_default_source!(baseline, baseline => shock)
	@test @prt(:q, p[:hh, :]) == [[0.0, 0.0], [100.0, 100.0]]
	series = @plot(:q, p[:hh, :])
	@test length(series) == 2
	@test series[1].y == [0.0, 0.0]
	@test series[2].y == [100.0, 100.0]

	set_default_source!(baseline => baseline, baseline => shock)
	@test @evalexpr(:q, p[:hh, :]) == [[0.0, 0.0], [100.0, 100.0]]

	@test_throws ErrorException set_default_source!([baseline => baseline, baseline => shock])
	@test_throws ErrorException set_default_source!((baseline, shock))

	set_default_source!(baseline)
	set_default_operator!(:n)
	set_default_periods!(2021:2021)
	@test @prt(p[:hh, :]) == [2]
	@test @prt(p[:hh, 2020]) == 1
	@test (@prt 2020:2020 p[:hh, :]) == [1]
	@test (@prt 2021:2021 baseline p[:hh, :]) == [2]
	series = @plot p
	@test length(series) == 2
	@test series[1].x == [2021]
	@test series[1].y == [2.0]
	@test series[2].x == [2021]
	@test series[2].y == [20.0]
	series = @plot 2020:2020 p
	@test series[1].x == [2020]
	@test series[1].y == [1.0]
	reset_print_defaults!()
end

end
