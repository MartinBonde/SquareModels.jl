# Modular Models

Larger models can be split into ordinary Julia modules. A useful pattern is:

- define variables and index sets inside the module that owns them,
- import shared objects explicitly,
- expose `define_equations()` and, when needed, `define_calibration()`,
- assemble the full model with `sum(m.define_equations() for m in submodels)`.

The repository contains a runnable version in `examples/modular_example.jl`.

```julia
module Production
    using SquareModels
    import ..data

    const s = [:agri, :manuf]

    @variables data.model begin
        Y[s], "Output by sector"
        p[s], "Price by sector"
        A[s], "Productivity"
    end

    function define_equations()
        @block data begin
            Y[s = s], Y[s] == A[s] * K[s]
            p[s = [:agri]], p[s] == 1
        end
    end

    function define_calibration()
        block = define_equations()
        @endo_exo_swap! block begin
            A, Y
        end
        block
    end
end
```

Other modules can use variables and sets from `Production` through explicit
imports or qualified names. This keeps ownership clear while still letting blocks
be combined into one square system.

```julia
submodels = [Production, Households]

base_model() = sum(m.define_equations() for m in submodels)
calibration_model() = sum(m.define_calibration() for m in submodels)

baseline = solve(calibration_model(), data; replace_nothing=1.0)
solve!(base_model(), scenario)
```

When a model has time loops or scenario-specific horizons, keep terminal periods
as ordinary variables in the parent module and have submodels import them. The
block construction functions will pick up the current values when called.
