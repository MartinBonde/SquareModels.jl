using Documenter
using SquareModels

DocMeta.setdocmeta!(SquareModels, :DocTestSetup, :(using SquareModels); recursive=true)

makedocs(;
    modules=[SquareModels, SquareModels.ModelExpressions, SquareModels.ModelPlotting],
    authors="Martin Kirk Bonde and contributors",
    sitename="SquareModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="master",
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "quickstart.md",
        "Core Concepts" => "concepts.md",
        "Solving" => "solving.md",
        "Plotting and Printing" => "plotting.md",
        "Modular Models" => "modular.md",
        "Optimization" => "optimization.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:none,
    doctest=false,
    warnonly=[:cross_references],
)

if haskey(ENV, "GITHUB_REPOSITORY")
    deploydocs(;
        repo="github.com/$(ENV["GITHUB_REPOSITORY"]).git",
        devbranch="master",
        push_preview=true,
    )
end
