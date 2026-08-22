# Auxein

Auxein is a small unsupervised cognitive engine built from centered kernels,
EMA learning and finite material growth. It has no matrices, labels, target
loss, top-k selection or persistent graph.

The current mathematical/material canon is **v0.5.0** and lives in
[`spec/auxein.md`](spec/auxein.md).

Its core rule is:

> recurrent unknown becomes local knowledge; recognised knowledge is weighted
> by geometric concern and can become higher context; explicit adjacent
> contexts can become private predictive knowledge.

## Structure

There are only two public modes:

```text
geometry
predictive = geometry + private adjacent succession + future readout
```

The persistent structure is:

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ predictive-private T(E)=E⊕E        [predictive only]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

Every cognitive object is a centered kernel `(W, C, V)`: support/mass, vector
center and scalar dispersion.

## Presentations

The canonical public presentation is a finite non-empty list of weighted
kernels:

```python
presentation = [
    [0.25, [1.0, 0.0], 0.0],
    [0.50, [0.0, 1.0], 0.2],
    [0.25, [0.0, 0.0], 0.0],
]
```

with positive weights and total mass in `(0, 1]`. Exact `(C,V)` duplicates are
coalesced.

A non-empty list of vectors remains supported as sugar and denotes a uniform
point-kernel presentation:

```python
[[1.0, 0.0], [0.0, 1.0]]
# == [[0.5, [1.0, 0.0], 0.0], [0.5, [0.0, 1.0], 0.0]]
```

A zero-center kernel carries causal mass but no cognitive direction. It cannot
concern a `CELL`, feed `Σ`, create a seed or enter a vertical context.

## Recognition and vertical growth

For an atom `X_s=(r_s,c_s,v_s)`, each distinct recognised center `C` receives
knowledge mass from the canonical `CONCERN` gain:

```text
g_C(X_s) = ||c_s||² - ||c_s-C||²

ω_sC = r_s g_C / Σ_D g_D
```

This weighting is independent of `CELL` support and of learning
responsibilities. `ALLOCATE` governs learning; concern gain governs current
knowledge.

The recognised point kernels are coalesced into `K_L` and merged into the
single vertical context `(W,C,V)`. A context rises only when `V>0` and `C!=0`.

## Readout

A recognised layer emits its recognised point kernels with their knowledge
weights, completed to total mass one by the zero remainder when necessary.
Persistent `CELL` variance is not emitted.

`geometry`:

```text
{
  "present": [
    [ [weight, center, 0], ... ],   # one presentation per responding layer
    ...
  ]
}
```

`predictive`:

```text
{
  "present": [...],
  "future": [
    [ [weight, predicted_center, 0], [remainder, 0, 0] ],
    ...
  ]
}
```

Each layer presentation owns its own mass universe; layer weights are never
flattened or renormalised together. Distinct future candidates remain distinct
presentations. Predictions are never ranked, fed back or chained.

There is deliberately no public sequence readout. Temporal `CELL`s are private
relations used only by predictive projection.

## Explicit sequences

Causality belongs to an explicit sequence boundary, not to the order in which
API calls happen.

`step(P)` is safe by default: it processes **one atomic sequence**. Successive
calls to `step()` can never learn a temporal relation between one another.

```python
net.step([[1.0]])
net.step([[10.0]])      # no implicit 1 -> 10 relation
```

Use `sequence()` for a real causal sequence:

```python
reports = net.sequence([
    [[1.0]],
    [[10.0]],
])
```

For streaming/multi-call execution, the same contract is available explicitly:

```python
net.begin_sequence()
try:
    r0 = net.sequence_step([[1.0]])
    r1 = net.sequence_step([[10.0]])
finally:
    net.end_sequence()
```

All `previous` registers are cleared at sequence open and close. A singleton
may use previously learned temporal knowledge to predict, but cannot learn an
incoming or outgoing transition.

`begin_sequence(resume=True)` is the explicit opt-in for continuing a
mid-sequence `previous` register restored from persistent state. Loading a state
never creates causal continuity by itself.

## Direct Auxein → Auxein composition

Only the upstream `present` family is automatically composable. Each member is
fed downstream, in layer-depth order, as an independent atomic sequence:

```python
upstream_report = upstream.step(input_presentation)
reports = downstream.consume(upstream_report["readout"]["present"])
```

This preserves geometry and prevents false temporal links between simultaneous
layer outputs. An empty family still establishes a causal boundary. `future` is
never auto-reinjected.

## Predictive mode

For adjacent contexts inside one explicit sequence:

```text
H(t-1) = (W-, C-, V-)
H(t)   = (W+, C+, V+)

Xᵀ = (W-W+, C- ⊕ C+, V- + V+)
```

The private temporal population applies the same centered-kernel learning laws
in dimension `2D`.

Prediction adds no learned state. From current context center `C` and a frozen
temporal `CELL` center `S⊕T`, the source is concerned by canonical point
`CONCERN`:

```text
||C-S||² < ||C||²
and
||C-S||² < ||S||²
```

Each distinct successor `T` becomes its own future presentation. Its local
mass is `W * γ`, where `γ = 1 - ||C-S||² / ||C||²` is the relative source
`CONCERN` gain; the rest is zero mass direction. Distinct futures are never
normalised against one another. If several relations project to the same exact
target, only their maximal `γ` survives. Temporal support and variance have no
predictive authority.

## Reference implementation

`auxein.py` is the stdlib-only Python semantic reference.

```python
from auxein import Auxein

net = Auxein(
    dimension=2,
    memory=20,
    eta=1.0,
    scalar="f64",
    mode="predictive",
    budget=100,
)

report = net.step([[1.0, 0.0]])
print(report["readout"])
```

## Persistent state

The canonical state contains:

- `format_version=5`, `dimension`, `scalar`, `memory`, `eta`, `mode`, `steps_seen`;
- ordered layers;
- geometric `cells` and private `sigma`;
- in `predictive`, private `temporal_cells`, `temporal_sigma`, and optional
  `previous` per layer.

Budget is environmental and is not serialised. `previous` is causal state, not
learned knowledge; forced material contraction invalidates all previous
registers.

```python
state = net.export_state()
restored = Auxein.from_state(state, budget=100)
```

## Tests

```bash
python test.py
```

The v0.5 regression suite covers weighted boundary parsing, gain-weighted
vertical context, support-independence, zero remainder, vertical silence,
material packing, explicit sequence boundaries, atomic isolation, predictive
relative-gain weighting, independent branching, same-target max envelopes, zero
targets, no recursive prediction, state round-trip and direct composition.

## Laboratory

```bash
python lab.py
```

The deterministic lab regenerates `results.json` from `experiments/*.json`.
A phase sets `"causal": true` only when its emitted presentations belong to one
explicit sequence. The built-in semantic battery additionally checks gain
weighting, atomic boundary isolation, explicit sequence learning, singleton
prediction, predictive relative-gain authority and output composability. It exits non-zero if a semantic check
fails.

## Benchmark

```bash
python benchmark.py --mode geometry --scenario singleton --dimension 8
python benchmark.py --mode geometry --scenario weighted-partial --dimension 8
python benchmark.py --mode geometry --scenario pair-context --dimension 8
python benchmark.py --mode geometry --scenario dense --dimension 8 --cells 512
python benchmark.py --mode predictive --scenario predictive-stable --dimension 8
python benchmark.py --mode predictive --scenario predictive-sequence --dimension 8
```

`predictive-sequence` benchmarks presentations inside one explicitly open causal
sequence; the other scenarios benchmark atomic presentations.

## Files

```text
auxein.py          reference engine v0.5.0
test.py            canonical regression suite
benchmark.py       stdlib-only benchmark harness
lab.py             deterministic experiment runner
worlds.py          external toy-world generators
experiments/       v0.5 laboratory scenarios
results.json       last laboratory run
spec/auxein.md     mathematical/material canon v0.5.0
```
