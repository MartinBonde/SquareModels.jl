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

@testset "Plot series in a figure grid" begin
	fig = Makie.Figure()
	series = labeled([1.0, 2.0], "demo")
	@test plotseries(fig[1, 2], series; title="Panel", legend=false) === fig
	ax = Makie.content(fig[1, 2])
	@test ax.title[] == "Panel"
	@test count(p -> p isa Makie.Lines, ax.scene.plots) == 1
	@test plotseries(fig[2, 1][1, 1], [series]; legend=false) === fig
	@test length(fig.content) == 2
end

@testset "plotvar labels for single and multiple lines" begin
	m = Model()
	JuMP.@variable(m, amount[[:north, :south], 2020:2022])
	db = ModelDictionary(m)
	db[amount] .= 1.0
	fig = plotvar(db[amount]; label="Custom", legend=false)
	@test [p.label[] for p in Makie.content(fig[1, 1]).scene.plots] == ["amount[north]", "amount[south]"]
	fig = plotvar(db[amount[:north, :]]; label="Custom", legend=false)
	@test only(Makie.content(fig[1, 1]).scene.plots).label[] == "Custom"
end

@testset "Makie extension legend and finalize hook" begin
	series = [labeled([1.0, 2.0], "demo"), labeled([2.0, 3.0], "demo2")]
	has_legend(fig) = any(c isa Makie.Legend for c in fig.content)

	@test has_legend(plotseries(series))  # default axislegend
	@test !has_legend(plotseries(series; legend=false))
	@test has_legend(plotseries(series; legend=(position=:cb,)))

	calls = []
	set_plot_finalize!((fig, ax, s) -> (push!(calls, length(s)); fig))
	@test !has_legend(plotseries(series))  # hook suppresses default legend
	@test calls == [2]
	@test has_legend(plotseries(series; legend=true))  # explicit legend still applies
	@test !has_legend(plotseries(series; legend=false))
	@test calls == [2] # Explicit legend choices never call the default hook.
	reset_plot_finalize!()
	@test plot_finalize() === nothing

	# A hook can mutate the layout without returning the figure.
	set_plot_finalize!((fig, ax, s) -> nothing)
	try
		@test plotseries(series) isa Makie.Figure
	finally
		reset_plot_finalize!()
	end
end

@testset "default y-axis label depends on operator" begin
	m_series = SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :m)
	fig = plotseries([m_series])
	ax = fig.content[1]
	@test ax.ylabel[] == "Difference from baseline"
	line = only(p for p in ax.scene.plots if p isa Makie.Lines)
	@test line.label[] == "qGDP"

	n_series = SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :n)
	@test plotseries([n_series]).content[1].ylabel[] == "Value"

	@test plotseries([n_series]; ylabel="Custom").content[1].ylabel[] == "Custom"

	mixed = [SquareModels.LabeledSeries([1.0, 2.0], [1.0, 2.0], "qGDP", :n), SquareModels.LabeledSeries([1.0, 2.0], [3.0, 4.0], "qGDP", :p)]
	fig = plotseries(mixed)
	@test fig.content[1].ylabel[] == "Value"
	lines = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test Set(l.label[] for l in lines) == Set(["qGDP <n>", "qGDP <p>"])
end

@testset "Percentage plot axes preserve small responses" begin
	response(values, name="Response") = LabeledSeries(collect(1:length(values)), values, name, :q)
	for values in ([0.0, 0.0], [-1e-12, 1e-12], [-0.02, 0.04], [NaN, 0.02])
		ax = Makie.content(plotseries(response(values); legend=false)[1, 1])
		@test ax.limits[][2] == (-0.05, 0.05)
		line = only(ax.scene.plots)
		@test isequal([p[2] for p in line[1][]], values)
	end
	ax = Makie.content(plotseries([response([0.01, 0.02]), response([-2.0, 1.0], "Other")]; legend=false)[1, 1])
	@test ax.limits[][2] == (-2.3, 2.3)
	labels = Makie.get_ticklabels(ax.ytickformat[], [-0.04, -0.02, 0.0, 0.02, 0.04])
	@test allunique(String.(labels))
	@test String(labels[4]) == "0.02%"
	@test String(labels[5]) == "0.04%"
	@test all(label -> endswith(String(label), "%"), labels)
	for op in (:p, :pch, :dp, :gdif, :mp, :rp, :rdp)
		ax = Makie.content(plotseries(LabeledSeries([1, 2], [0.01, 0.02], "Rate", op); legend=false)[1, 1])
		@test String.(Makie.get_ticklabels(ax.ytickformat[], [0.01, 0.02])) == ["0.01%", "0.02%"]
		@test ax.limits[] == (nothing, nothing)
	end

	fig = plotseries([response([0.0, 0.01]), response([-2.0, 1.0], "Other")]; layout=:trellis, legend=false)
	axes = [ax for ax in fig.content if ax isa Makie.Axis]
	@test [ax.limits[][2] for ax in axes] == [(-0.05, 0.05), (-2.3, 2.3)]
	custom = values -> string.(values)
	ax = Makie.content(plotseries(response([0.0, 0.01]); legend=false,
		axis=(limits=(nothing, (-0.01, 0.02)), ytickformat=custom))[1, 1])
	@test ax.limits[][2] == (-0.01, 0.02)
	@test ax.ytickformat[] === custom
	ax = Makie.content(plotseries(response([0.0, 0.01]); legend=false,
		decorate=(ax, _) -> Makie.ylims!(ax, -1, 1))[1, 1])
	@test ax.limits[][2] == (-1, 1)
	for series in ([response([0.0, 0.01]), labeled([1.0, 2.0], "Level")], [response([NaN, NaN])])
		ax = Makie.content(plotseries(series; legend=false)[1, 1])
		@test ax.limits[] == (nothing, nothing)
	end
	fig = Makie.Figure()
	ax = Makie.Axis(fig[1, 1]; limits=(nothing, (-1, 1)), ytickformat=custom)
	plotseries!(ax, response([0.0, 0.01]))
	@test ax.limits[][2] == (-1, 1)
	@test ax.ytickformat[] === custom
end

@testset "Linked trellis response ranges cover every panel" begin
	response(values, name, op=:q) = LabeledSeries([1, 2], values, name, op)
	panel_axes(fig) = [ax for ax in fig.content if ax isa Makie.Axis]
	yrange(rect) = (minimum(rect)[2], maximum(rect)[2])
	small = response([0.0, 0.01], "Small")
	large = response([-2.0, 1.0], "Large")
	for series in ([small, large], [large, small])
		axes = panel_axes(plotseries(series; layout=:trellis, linky=true, legend=false))
		# Inspect the displayed ranges: ax.limits can retain per-axis settings
		# even when linking has replaced targetlimits and finallimits.
		@test all(ax -> yrange(ax.targetlimits[]) == (-2.3, 2.3), axes)
		@test all(ax -> yrange(ax.finallimits[]) == (-2.3, 2.3), axes)
		@test all(ax -> String.(Makie.get_ticklabels(ax.ytickformat[], [0.01])) == ["0.01%"], axes)
		@test [p[2] for p in only(axes[2].scene.plots)[1][]] == series[2].y
	end
	for options in ((;), (; linky=false))
		axes = panel_axes(plotseries([small, large]; layout=:trellis, legend=false, options...))
		@test [yrange(ax.finallimits[]) for ax in axes] == [(-0.05, 0.05), (-2.3, 2.3)]
	end

	# Explicit settings and callbacks retain precedence over response defaults.
	custom_format = values -> string.(values)
	axes = panel_axes(plotseries([small, large]; layout=:trellis, linky=true, legend=false,
		axis=(limits=(nothing, (-5.0, 5.0)), ytickformat=custom_format)))
	@test all(ax -> yrange(ax.finallimits[]) == (-5.0, 5.0), axes)
	@test all(ax -> ax.ytickformat[] === custom_format, axes)
	axes = panel_axes(plotseries([small, large]; layout=:trellis, linky=true, legend=false,
		decorate=(ax, _) -> Makie.ylims!(ax, -1.0, 1.0)))
	@test all(ax -> yrange(ax.finallimits[]) == (-1.0, 1.0), axes)

	level = response([-20.0, 10.0], "Level", :n)
	for series in ([small, level], [level, small])
		axes = panel_axes(plotseries(series; layout=:trellis, linky=true, legend=false))
		@test all(ax -> ax.limits[][2] === nothing, axes)
		ranges = [yrange(ax.finallimits[]) for ax in axes]
		@test ranges[1] == ranges[2]
		@test all(range -> range[1] <= -20.0 && range[2] >= 10.0, ranges)
	end

	# Symmetric linear defaults must not impose negative bounds on log scales.
	axes = panel_axes(plotseries([response([1.0, 2.0], "Small"), response([10.0, 100.0], "Large")];
		layout=:trellis, linky=true, legend=false, axis=(yscale=log10,)))
	@test all(ax -> ax.limits[][2] === nothing, axes)
	@test all(ax -> (range = yrange(ax.finallimits[]); 0 < range[1] <= 1.0 && range[2] >= 100.0), axes)
end

@testset "alternating_dash for repeated variables" begin
	# Same base label twice (e.g. same variable from two sources): same color, different dash.
	series = [labeled([1.0, 2.0], "qGDP"), SquareModels.LabeledSeries([1.0, 2.0], [2.0, 3.0], "qGDP", :r), labeled([3.0, 4.0], "qC")]
	fig = plotseries(series; legend=false)
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] == plots[2].color[]
	@test plots[1].color[] != plots[3].color[]
	@test plots[1].linestyle[] === nothing        # :solid
	@test plots[2].linestyle[] isa AbstractVector # dashed
	@test plots[3].linestyle[] === nothing

	# Unique labels: no restyling unless forced.
	series = [labeled([1.0, 2.0], "a"), labeled([2.0, 3.0], "b")]
	fig = plotseries(series; legend=false)
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] != plots[2].color[]

	fig = plotseries(series; legend=false, alternating_dash=true)  # forced: consecutive pairs
	plots = [p for p in fig.content[1].scene.plots if p isa Makie.Lines]
	@test plots[1].color[] == plots[2].color[]
	@test plots[2].linestyle[] isa AbstractVector
end

@testset "Plot macro options and cached year axes" begin
	m = Model()
	JuMP.@variable(m, amount[2020:2022])
	db = ModelDictionary(m)
	db[amount] = [100, 110, 121]
	ref = copy(db)
	ref[amount] = [50, 55, 60.5]
	periods = 2020:2022
	op = :q
	options = (legend=false, linewidth=4)
	fig = @plot(op, periods, ref=>db, amount; title="Response", options...)
	ax = Makie.content(fig[1, 1])
	@test ax.title[] == "Response"
	line = only(ax.scene.plots)
	@test line.linewidth[] == 4
	@test [p[1] for p in line[1][]] == collect(periods)
	@test [p[2] for p in line[1][]] == [100, 100, 100]

	cached = LabeledArray([100.0, 110.0, 121.0], (collect(periods),))
	grid = Makie.Figure()
	@test @plot(:p, periods, db, $cached; position=grid[2, 1], options...) === grid
	points = only(Makie.content(grid[2, 1]).scene.plots)[1][]
	@test [p[1] for p in points] == collect(periods)
	@test isnan(points[1][2])
	@test [p[2] for p in points[2:end]] ≈ [10, 10]

	cached_grid = LabeledArray([1.0 2.0 3.0; 4.0 5.0 6.0], ([:a, :b], collect(periods)))
	fig = @plot(:n, periods, db, $cached_grid; options...)
	lines = Makie.content(fig[1, 1]).scene.plots
	@test length(lines) == 2
	@test [p[1] for p in lines[2][1][]] == collect(periods)
	@test [p[2] for p in lines[2][1][]] == [4, 5, 6]
	@test endswith(lines[1].label[], "[a]")
	@test endswith(lines[2].label[], "[b]")

	# A labeled comprehension must retain years after evaluating JuMP expressions.
	fig = @plot(:q, periods, ref=>db,
		LabeledArray([2 * amount[t] for t in periods], (periods,)); options...)
	points = only(Makie.content(fig[1, 1]).scene.plots)[1][]
	@test [p[1] for p in points] == collect(periods)
	@test [p[2] for p in points] == [100, 100, 100]

	set_default_source!(db)
	try
		fig = @plot(amount; title="Default source", options...)
		@test Makie.content(fig[1, 1]).title[] == "Default source"
	finally
		reset_print_defaults!()
	end
	fig = @plot db amount title="Named option" legend=false
	@test Makie.content(fig[1, 1]).title[] == "Named option"
end

@testset "Trellis panels retain expressions, indices, and sources" begin
	m = Model()
	JuMP.@variables m begin
		a[[:north, :south], 2020:2022]
		b[2020:2022]
		c[i=[:north, :south], t=2020:2022; i == :north || t != 2021]
	end
	base = ModelDictionary(m)
	base[a] .= 10.0
	base[b] .= 30.0
	base[c] .= 5.0
	shock = copy(base)
	shock[a] .*= 1.1
	shock[b] .*= 1.2
	set_default_source!(base => shock)
	set_default_periods!(2021:2022)
	set_default_operator!(:an)
	try
		seen = []
		fig = @plot([a, b]; layout=:trellis, columns=2, legend=false,
			panel_titles=["North", "South", "Total"], decorate=(ax, s) -> push!(seen, s))
		axes = [x for x in fig.content if x isa Makie.Axis]
		@test length(axes) == 3
		@test [ax.title[] for ax in axes] == ["North", "South", "Total"]
		@test all(length(s) == 2 for s in seen)
		@test all(s.x == [2021, 2022] for panel in seen for s in panel)
		@test seen[1][1].y ≈ [11, 11]
		@test seen[1][2].y == [10, 10]
		@test seen[3][1].y == [36, 36]
		@test seen[3][2].y == [30, 30]
		@test seen[1][1].panel == seen[1][2].panel
		@test seen[1][1].panel != seen[2][1].panel
		set_default_source!(base, shock)
		set_default_operator!(:n)
		fig = @plot(a; layout=:trellis, legend=false)
		@test length([x for x in fig.content if x isa Makie.Axis]) == 2
		@test all(length(ax.scene.plots) == 2 for ax in fig.content if ax isa Makie.Axis)
		set_default_source!(base)
		fig = @plot(c; layout=:trellis, legend=false)
		axes = [x for x in fig.content if x isa Makie.Axis]
		@test length(axes) == 2
		@test isnan(only(axes[2].scene.plots)[1][][1][2])
		@test_throws AssertionError @plot(a; layout=:trellis, columns=0)
		@test_throws AssertionError @plot(a; layout=:trellis, panel_titles=["Only one"])
	finally
		reset_print_defaults!()
	end
end

@testset "Direct line handles and per-series options" begin
	series = [labeled([1.0, 2.0], "a"), labeled([3.0, 4.0], "b")]
	fig = Makie.Figure()
	ax = Makie.Axis(fig[1, 1])
	lines = plotseries!(ax, series; labels=["First", "Second"],
		styles=[(color=:red,), (linewidth=5,)])
	@test length(lines) == 2
	@test lines[1].label[] == "First"
	@test lines[1].color[] == Makie.to_color(:red)
	@test lines[2].linewidth[] == 5
	@test_throws AssertionError plotseries!(ax, series; labels=["One"])
end

@testset "Adding styled lines preserves existing plots" begin
	for alternating_dash in (nothing, true)
		fig = Makie.Figure()
		ax = Makie.Axis(fig[1, 1])
		existing = Makie.lines!(ax, [1.0, 2.0]; color=:black, linestyle=:dash)
		old_color, old_style = existing.color[], copy(existing.linestyle[])
		series = [labeled([2.0, 3.0], "a"), labeled([3.0, 4.0], "a")]
		lines = plotseries!(ax, series; alternating_dash)
		@test existing.color[] == old_color
		@test existing.linestyle[] == old_style
		@test lines[1].color[] == lines[2].color[]
		@test lines[1].linestyle[] === nothing
		@test lines[2].linestyle[] isa AbstractVector
		styled = plotseries!(ax, series; styles=[(color=:red,), (linestyle=:solid,)])
		@test styled[1].color[] == Makie.to_color(:red)
		@test styled[2].linestyle[] === nothing
		@test lines[2].linestyle[] isa AbstractVector
	end
end

@testset "Reference comparisons retain sparse year labels" begin
	function sparse_source(periods; omitted=nothing, scale=1.0)
		m = Model()
		SquareModels.@variables m begin s[t=periods; t != omitted] end
		db = ModelDictionary(m)
		for t in periods
			t == omitted || (db[s[t]] = scale * (t - 2019))
		end
		return db
	end
	baseline = sparse_source(2020:2022)
	scenario = sparse_source(2021:2023; scale=10.0)
	difference = @evalexpr :m baseline=>scenario s
	@test difference.dims == ([2021, 2022, 2023],)
	@test isequal(collect(difference), [18.0, 27.0, NaN])
	@test isequal(collect(@evalexpr(:q, baseline=>scenario, s)), [900.0, 900.0, NaN])
	growth = collect(@evalexpr(:mp, baseline=>scenario, s))
	@test isnan(growth[1]) && isnan(growth[3])
	@test growth[2] ≈ 0.0 atol=1e-10
	for op in (:m, :q, :mp)
		fig = @plot(op, nothing, baseline=>scenario, s; legend=false)
		points = only(Makie.content(fig[1, 1]).scene.plots)[1][]
		@test [p[1] for p in points] == [2021, 2022, 2023]
		@test isequal([p[2] for p in points], Float32.(collect(@evalexpr(op, nothing, baseline=>scenario, s))))
	end

	# Conversion may make just one side dense. Gaps and missing keys must survive.
	gapped = sparse_source(2020:2022; omitted=2021)
	complete = sparse_source(2020:2022; scale=10.0)
	@test isequal(collect(@evalexpr(:m, gapped=>complete, s)), [9.0, NaN, 27.0])
	result = @evalexpr :q complete=>gapped s
	@test Set(keys(result)) == Set([(2020,), (2022,)])
	@test result[2020] == result[2022] == -90.0
end

@testset "Complete sparse time slices use labelled arithmetic" begin
	m = Model()
	SquareModels.@variables m begin
		s[i=[:a, :b], t=2020:2022; i == :a || t != 2021]
		d[i=[:a, :b], t=2020:2022]
	end
	db = ModelDictionary(m)
	db[s] .= 6.0
	db[d] .= 2.0
	set_default_source!(db)
	set_default_periods!(2020:2022)
	try
		@test collect(@evalexpr(s[:a,:] / d[:a,:])) == [3, 3, 3]
		@test collect(@evalexpr(d[:a,:] - s[:a,:])) == [-4, -4, -4]
		@test collect(@evalexpr(sum(s[i,:] for i in [:a]) / d[:a,:])) == [3, 3, 3]
		weights = db[s[:a,2020:2022]]
		@test collect(@evalexpr(s[:a,:] * $weights / d[:a,:])) == [18, 18, 18]
		fig = @plot(s[:a,:] / d[:a,:]; legend=false)
		@test [p[1] for p in only(Makie.content(fig[1, 1]).scene.plots)[1][]] == [2020, 2021, 2022]
		alias = d[:a,:]
		sparse_alias = s[:a,:]
		set_default_periods!(2021:2022)
		@test collect(@evalexpr(alias)) == [2, 2]
		@test collect(@evalexpr(sparse_alias)) == [6, 6]
		set_default_periods!(2020:2022)
		gapped = @evalexpr(s[:b,:])
		@test length(gapped) == 2
		@test !haskey(gapped, (2021,))
		db[s[:a,:]] .= [6, 12, 18]
		@test collect(@evalexpr(s[:a,:] / d[:a,:])) == [3, 6, 9]
	finally
		reset_print_defaults!()
	end
end

SquareModels.ModelPlotting.plotseries(series::Vector{SquareModels.LabeledSeries}; kwargs...) = series

model = Model()
JuMP.@variables model begin
	x
	p[[:hh, :firm], 2020:2021]
	q[[:hh, :firm], 2020:2021]
	L[[:cognitive, :physical], 2020:2021]
end

baseline = ModelDictionary(model)
baseline[x] = 3
for h in [:hh, :firm], t in 2020:2021
	baseline[p[h, t]] = h == :hh ? t - 2019 : 10 * (t - 2019)
	baseline[q[h, t]] = h == :hh ? 2 : 3
end
labor = [:cognitive, :physical]
l = labor
t = 2020:2021
for l in labor, t in 2020:2021
	baseline[L[l, t]] = l == :cognitive ? t - 2019 : 10 * (t - 2019)
end

shock = ModelDictionary(model)
shock[x] = 4
for h in [:hh, :firm], t in 2020:2021
	shock[p[h, t]] = 2 * baseline[p[h, t]]
	shock[q[h, t]] = baseline[q[h, t]]
end
for l in labor, t in 2020:2021
	shock[L[l, t]] = 2 * baseline[L[l, t]]
end

@testset "Model expression evaluation" begin
	@test @evalexpr(baseline, x) == 3
	@test @evalexpr(baseline, p[:hh, 2020] * q[:hh, 2020]) == 2
	@test (@evalexpr baseline p[:hh, 2020] * q[:hh, 2020]) == 2
	@test Array(@evalexpr(baseline, p[:firm, :] * q[:firm, :])) == [30, 60]
	@test Array(@evalexpr(baseline, p[:firm] * q[:firm])) == [30, 60]
	@test Array(@evalexpr(baseline, p * q)) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, p[:firm, :] / p[:firm, 2020])) == [1.0, 2.0]
	@test Array(@evalexpr(baseline, p[:firm, :] .* q[:firm, :])) == [30, 60]
	@test Array(@evalexpr(baseline, p .* q)) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, (@. p * q))) == [2 4; 30 60]
	@test Array(@evalexpr(baseline, sum([L[l, :] for l in l]))) == [11, 22]
	@test Array(@evalexpr(baseline, sum(L[l, :] for l in l))) == [11, 22]
	@test Array(@evalexpr(baseline, sum(L[l] for l in l))) == [11, 22]
	local_vector = [7, 8]
	@test @evalexpr(baseline, local_vector[1]) == 7
	@test @evalexpr(baseline, [x, p[:hh, 2021] * q[:hh, 2021]]) == [3, 4]
	multi_vector = @evalexpr(:m, baseline=>shock, [p[:hh, :], q[:hh, :]])
	@test multi_vector.names == ["p[:hh, :]", "q[:hh, :]"]
	@test count(==("year"), split(sprint(show, MIME"text/plain"(), multi_vector))) == 1
	mixed_dims_print = sprint(show, MIME"text/plain"(), @evalexpr(:m, baseline=>shock, [p[:hh, :], q]))
	@test count(==("year"), split(mixed_dims_print)) == 1
	@test all(label -> occursin(label, mixed_dims_print), ["p[:hh, :]", "q[hh]", "q[firm]"])
	fq = 2
	@test @evalexpr(baseline, p[:hh, 2021] * q[:hh, 2021] / fq) == 2.0
	@test @evalexpr(baseline, (p[:hh, 2021] * q[:hh, 2021], p[:firm, 2021] * q[:firm, 2020] / fq)) == (4, 30.0)
	multi = @evalexpr(baseline, (p[:hh, :], p[:firm, :]))
	@test multi == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(baseline, p[:firm, :])))
	@test occursin("p[:hh, :]", sprint(show, MIME"text/plain"(), multi))
	@test occursin("p[:firm, :]", sprint(show, MIME"text/plain"(), multi))
	@test Array(@evalexpr(baseline, p)) == [1.0 2.0; 10.0 20.0]
	@test occursin("2020", sprint(show, MIME"text/plain"(), @evalexpr(baseline, p)))
	@test isequal(@evalexpr(:p, baseline, p[:hh, :]), [NaN, 100.0])
	@test @evalexpr(:m, baseline=>shock, p[:hh, :]) == [1.0, 2.0]
	@test @evalexpr(:q, baseline=>shock, p[:hh, :]) == [100.0, 100.0]
	@test isequal(map(Array, @evalexpr([:n, :p], baseline, p[:hh, :])), [[1, 2], [NaN, 100.0]])
	an = @evalexpr(:an, baseline=>shock, p)
	@test all(v -> v isa LabeledArray, an)
	@test occursin("year", sprint(show, MIME"text/plain"(), an[2]))
	@test (@evalexpr :q baseline=>shock p[:hh, :]) == [100.0, 100.0]
	op = :q
	@test @evalexpr(op, baseline=>shock, p[:hh, :]) == [100.0, 100.0]
	pair_print = @evalexpr(baseline=>shock, p[:hh, :])
	@test pair_print == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(shock, p[:hh, :])))
	printed_pair = sprint(show, MIME"text/plain"(), pair_print)
	@test !occursin("baseline:p[:hh, :]", printed_pair)
	@test !occursin("shock:p[:hh, :]", printed_pair)
	@test occursin("baseline", printed_pair)
	@test occursin("shock", printed_pair)
	@test occursin("p[:hh, :]", printed_pair)
	# Narrow width forces the 34-char label to wrap across two rows of 24 chars.
	set_column_label_total_width!(48)
	long_print = sprint(show, MIME"text/plain"(), @evalexpr((baseline=>shock, baseline), p[:hh, :] * q[:hh, :] + p[:hh, :]))
	set_column_label_total_width!(72)
	@test !occursin("p[:hh, :] * q[:hh, :] + p[:hh, :]", long_print)
	@test occursin("p[:hh, :] *", long_print)
	@test occursin("q[:hh, :] +", long_print)
	@test SquareModels._column_label_width(1) == 72
	@test SquareModels._column_label_width(2) > SquareModels._column_label_width(8)
	set_column_label_total_width!(100)
	@test SquareModels._column_label_width(1) == 100
	set_column_label_total_width!(72)
	multi_db = @evalexpr((baseline=>shock, baseline), p[:hh, :])
	@test multi_db.names == ["baseline\np[:hh, :]", "shock\np[:hh, :]"]
	@test multi_db == (Array(@evalexpr(baseline, p[:hh, :])), Array(@evalexpr(shock, p[:hh, :])))
	multi_q = @evalexpr(:q, (baseline=>baseline, baseline=>shock), p[:hh, :])
	@test multi_q.names == ["baseline\np[:hh, :]", "shock\np[:hh, :]"]
	@test multi_q == ([0.0, 0.0], [100.0, 100.0])
	printed_p = sprint(show, MIME"text/plain"(), @evalexpr(baseline, p))
	@test occursin("year", printed_p)
	@test occursin("hh", printed_p)
	@test occursin("firm", printed_p)
	printed_slice = sprint(show, MIME"text/plain"(), @evalexpr(baseline, p[:hh, :]))
	@test occursin("p[:hh, :]", printed_slice)
	long_print = sprint(show, MIME"text/plain"(), @evalexpr(:m, baseline => shock, p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :]))
	@test !occursin("p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :] + q[:hh, :] - p[:hh, :]", long_print)
	@test occursin("p[:hh, :]", long_print)
	@test !occursin("(", long_print)
	@test SquareModels.ModelExpressions._expr_label(:(a + b - c + d)) == "a + b - c + d"
	series = @plot :q baseline=>shock p[:hh, :]
	@test length(series) == 1
	@test series[1].label == "p[:hh, :]"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]
	series = @plot :p baseline p
	@test length(series) == 2
	@test series[1].label == "p[hh]"
	@test series[1].op == :p
	@test series[1].x == [2020, 2021]
	@test isequal(series[1].y, [NaN, 100.0])
	@test series[2].label == "p[firm]"
	@test series[2].x == [2020, 2021]
	@test isequal(series[2].y, [NaN, 100.0])
	series = @plot baseline p .* q
	@test length(series) == 2
	@test series[1].label == "p .* q[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "p .* q[firm]"
	@test series[2].x == [2020, 2021]
	@test series[2].y == [30.0, 60.0]
	series = @plot baseline p * q
	@test length(series) == 2
	@test series[1].label == "p * q[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "p * q[firm]"
	@test series[2].y == [30.0, 60.0]
	series = @plot baseline (@. p * q)
	@test length(series) == 2
	@test series[1].label == "(*).(p, q)[hh]"
	@test series[1].x == [2020, 2021]
	@test series[1].y == [2.0, 4.0]
	@test series[2].label == "(*).(p, q)[firm]"
	@test series[2].y == [30.0, 60.0]
	series = @plot :q baseline=>shock p
	@test length(series) == 2
	@test series[1].y == [100.0, 100.0]
	@test series[2].y == [100.0, 100.0]
	series = @plot :q baseline=>shock sum([L[l, t] for l in l])
	@test length(series) == 1
	@test series[1].label == "sum([L[l, t] for l = l])"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]
	series = @plot :q baseline=>shock sum(L[l, t] for l in l)
	@test length(series) == 1
	@test series[1].label == "sum((L[l, t] for l = l))"
	@test series[1].op == :q
	@test series[1].x == [2020, 2021]
	@test series[1].y == [100.0, 100.0]

	set_default_source!(baseline)
	@test Array(@evalexpr(p[:hh, :])) == [1, 2]
	@test isequal(@evalexpr(:p, p[:hh, :]), [NaN, 100.0])
	@test isequal((@evalexpr :p p[:hh, :]), [NaN, 100.0])

	set_default_source!(baseline => shock)
	@test @evalexpr(:q, p[:hh, :]) == [100.0, 100.0]

	set_default_operator!(:q)
	@test @evalexpr(p[:hh, :]) == [100.0, 100.0]

	set_default_source!(baseline)
	@test @evalexpr(:q, p[:hh, :]) == [0.0, 0.0]

	set_default_source!(baseline, baseline => shock)
	default_multi = @evalexpr(:q, p[:hh, :])
	@test default_multi.names == ["baseline1\np[:hh, :]", "s2\np[:hh, :]"]
	@test default_multi == ([0.0, 0.0], [100.0, 100.0])
	series = @plot(:q, p[:hh, :])
	@test length(series) == 2
	@test series[1].y == [0.0, 0.0]
	@test series[2].y == [100.0, 100.0]

	set_default_source!(baseline => baseline, baseline => shock)
	default_pairs = @evalexpr(:q, p[:hh, :])
	@test default_pairs.names == ["s1\np[:hh, :]", "s2\np[:hh, :]"]
	@test default_pairs == ([0.0, 0.0], [100.0, 100.0])

	@test_throws ErrorException set_default_source!([baseline => baseline, baseline => shock])
	@test_throws ErrorException set_default_source!((baseline, shock))

	set_default_source!(baseline)
	set_default_operator!(:n)
	set_default_periods!(2021:2021)
	set_default_source!(baseline => shock)
	@test @evalexpr(:m, p[:hh, :]) == [2.0]
	series = @plot :m p[:hh, :]
	@test length(series) == 1
	@test series[1].x == [2021]
	@test series[1].y == [2.0]
	series = @plot :q p[:hh, :]
	@test series[1].y == [100.0]
	set_default_source!(baseline)
	@test Array(@evalexpr(p[:hh, :])) == [2]
	@test Array(@evalexpr(p[:hh])) == [2]
	@test @evalexpr(p[:hh, 2020]) == 1
	@test @evalexpr(2020, p[:hh]) == 1
	@test Array(@evalexpr 2020:2020 p[:hh, :]) == [1]
	@test Array(@evalexpr 2020:2020 p[:hh]) == [1]
	@test Array(@evalexpr 2021:2021 baseline p[:hh, :]) == [2]
	@test Array(@evalexpr(sum(L[l, :] for l in l))) == [22]
	@test Array(@evalexpr(sum(L[l] for l in l))) == [22]
	series = @plot p
	@test length(series) == 2
	@test series[1].x == [2021]
	@test series[1].y == [2.0]
	@test series[2].x == [2021]
	@test series[2].y == [20.0]
	series = @plot p[:hh]
	@test only(series).x == [2021]
	@test only(series).y == [2.0]
	series = @plot 2020:2020 p
	@test series[1].x == [2020]
	@test series[1].y == [1.0]
	reset_print_defaults!()
end

@testset "nothing propagates through expressions" begin
	m2 = Model()
	JuMP.@variables m2 begin
		y
		a[[:hh], 2020:2022]
		b[[:hh], 2020:2022]
	end
	db = ModelDictionary(m2)
	for t in 2021:2022
		db[a[:hh, t]] = 1.0
	end
	for t in 2020:2022
		db[b[:hh, t]] = 2.0
	end
	# y and a[:hh, 2020] stay nothing
	@test @evalexpr(db, y) === nothing
	@test @evalexpr(db, y * b[:hh, 2020]) === nothing
	@test isequal(Array(@evalexpr(db, a[:hh, :] * b[:hh, :])), [nothing, 2.0, 2.0])
	@test isequal(Array(@evalexpr(db, b[:hh, :] / a[:hh, :])), [nothing, 2.0, 2.0])
	@test @evalexpr(db, sum(a[:hh, t] for t in 2020:2022)) === nothing
	@test occursin("nothing", sprint(show, MIME"text/plain"(), @evalexpr(db, a[:hh, :] * b[:hh, :])))
	@test isequal(@evalexpr(:p, db, a[:hh, :]), [NaN, NaN, 0.0])
end

end
