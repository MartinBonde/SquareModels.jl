# Dataset lookup and assignment

`ModelDictionary` separates shared model metadata from dataset values. Datasets
for one JuMP model share its variable names, variable-ID lookup, and registered
container mappings. Each dataset owns a typed value vector.

## Numeric storage

`ModelDictionary(model)` uses `Float64` values, with `nothing` for unset cells.
Assignments convert to that numeric type. In particular, assigning an integer
returns a floating-point value on read, and assigning a rational rounds to
`Float64`. Choose the numeric type explicitly when this matters:

```julia
baseline = ModelDictionary(model)             # Float64 plus nothing
exact = ModelDictionary{Number}(model)         # preserve mixed numeric types
integer_data = ModelDictionary{Int}(model)
mask = baseline .> 0                          # Bool plus nothing
```

Concrete numeric storage avoids boxing ordinary values. `Number` storage supports
mixed numeric types. Copying a dataset copies its values,
while preserving shared metadata. Dictionary arithmetic validates ordered names
before combining buffers and allocates only its output. In-place arithmetic
writes directly into the destination:

```julia
scenario .= 1.05 .* baseline
baseline[x] .= a .* b .+ c
```

Overlapping window operands are snapshotted when required to preserve assignment
semantics. Sparse arithmetic retains keyed-container semantics; its specialized
broadcast fallback may allocate an intermediate sparse result.

## Reusing container mappings

Variable-reference access uses model-local variable IDs. String access uses a
hash lookup. Neither access reconstructs a table over all model variables.
After synchronization, resolving an arbitrary container costs work proportional
to the number of requested cells.

Registered model containers are prepared and cached on first access. Treat their
contents as fixed until `refresh_model_layout!(model)`. Temporary arrays and
slices are not retained in the automatic cache. Prepare a selection explicitly
when reusing a slice or applying it to multiple datasets:

```julia
selection = prepare_selection(baseline, x[2020:2030])
baseline[selection] .= 1.0
scenario[selection] .= 2.0
```

A selection captures storage slots and labelled coordinates, not data values.
Contiguous slots use a range. Full datasets reuse the mapping without a new
positions array; filtered datasets translate the selection into their own slots.
Obtaining a full-dataset window is constant work after preparation. Writing its
cells is necessarily proportional to the number of cells written.

Selections are snapshots of container contents. Recreate a selection after
changing a temporary container. Its public fields and the backing dictionary's
keys are read-only implementation metadata: do not insert, delete, or reorder
keys through `d.dictionary` or `keys(d)`.

## Model changes

Variable additions are detected on dataset and name lookup. For standard cached
JuMP models, an isolated adapter checks the allocated variable-slot count in
constant time. Calling `num_variables` on these backends would itself scan every
variable. Other backends use their variable-count query, whose cost depends on
the backend. The shared layout is rebuilt once for a changed model, then each
accessed dataset moves its values by variable identity. Batch model construction
before creating data or repeatedly accessing it.

Membership and metadata inspection describe the dataset's current snapshot.
Indexing or `add_missing_model_variables!` discovers additions; inspecting keys
alone does not populate new variables. An already refreshed shared layout is
adopted on the next dataset access.

`fix(d)` and `set_start_value(d)` also synchronize first. Full datasets require
values for newly added variables; filtered datasets apply only their selected
variables, including the single anonymous variable a dataset permits.

The O(1) addition check for standard cached JuMP models depends on MOI's internal
`VariablesContainer.set_mask` storage length. This adapter is isolated and guarded
by cache and container types; unrecognized storage falls back to the public
variable count. Its addition and deletion/addition tests must pass when upgrading
JuMP or MOI. If a future storage implementation reuses slots without changing
these types, call `refresh_model_layout!` explicitly after additions as well.

Call `refresh_model_layout!(model)` after deleting or renaming variables or
changing a registered container. Prepared selections and saved windows become stale and throw an
error when reused. Obtain a new window or prepare a new selection after refresh.
Values for surviving variables are retained, including after a rename.

After `empty!(model)`, create new datasets. Old datasets and selections are
rejected because the rebuilt model may reuse variable IDs for different objects.
Dataset mutation and layout refresh are not designed for concurrent unsynchronized
use.

Filtered dictionaries remain independent subsets on read, including after new
model variables are added. `add_missing_model_variables!(subset)` explicitly
expands one to cover the current full model; missing values become `nothing`.

## Repeated loading

CSV, Parquet, and simple-format loading cache parsed model-variable keys once per
model layout, so repeated loads do not reparse every variable name. This cache is
shared across datasets and cleared on refresh. It trades additional metadata
memory for faster repeated loading; file reads and source-index construction are
performed for each load.

## Keyed assignment

Assign `KeyedData`, `SparseAxisArray`, or `SparseZeroArray` sources directly:

```julia
model = Model()
@variable(model, x[i = [:a, :b], t = 2025:2026])
baseline = ModelDictionary(model)
scenario = ModelDictionary(model)
source = KeyedData(Dict((:a, 2025) => 10.0, (:b, 2026) => 20.0))
baseline[x] = source
source.data[(:a, 2025)] = 12.0
scenario[x] = source
```

Assignment matches coordinates, preserves stored zeros, and writes `nothing`
for absent source cells. Source coordinates outside the target are ignored.
Source and target must have the same number of axes. Each call reads the source's
current values and keys, so deleting a source cell clears its target on assignment.

Registered containers reuse their cached mappings. For a repeated temporary
slice, use `selection = prepare_selection(baseline, slice)` and assign through
`baseline[selection] = source` or `scenario[selection] = source`. Assignment
work is proportional to the number of selected cells.
