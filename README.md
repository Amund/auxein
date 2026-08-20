# Auxein

Auxein is a small unsupervised cognitive engine built from centered kernels,
EMA learning and finite material growth. It has no matrices, labels, target
loss, top-k selection or persistent graph.

The current mathematical/material canon is **v0.3.0** and lives in
[`spec/auxein.md`](spec/auxein.md).

Its core rule is:

> recurrent unknown becomes local knowledge; knowledge recognised together
> becomes higher context; adjacent recognised contexts can become temporal
> knowledge.

## Structure

Auxein has only three architectural levels:

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ temporal space T(E)=E⊕E        [temporal mode]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

The temporal compartment is structurally associated with a layer but belongs
to the `NETWORK`: the layer never reads it. Geometric and temporal `CELL`s do
not concern, allocate, learn or compete across spaces. They share only the
material economy and the external readout of a step.

Every cognitive object is a centered kernel:

```text
(W, C, V)
```

where `W` is support, `C` is a vector center and `V` is scalar dispersion.
External vectors enter as point kernels `(r, x, 0)`.

For geometry:

```text
presentation of kernels
    ↓
CELL concern / allocation
    ├─ unknown → Σ → recurrence → local CELL
    └─ recognised values → one contextual kernel → next LAYER
```

For temporal mode, after the complete geometric phase of a step, the `NETWORK`
compares each recognised layer context with that layer's context from the
immediately preceding external step:

```text
H(t-1) = (W-, C-, V-)
H(t)   = (W+, C+, V+)

        ↓

Xᵀ = (W- W+, C- ⊕ C+, V- + V+)

        ↓

temporal CELL concern / allocation
    └─ unknown → Σᵀ → recurrence → temporal CELL
```

There is no history window and no `T(T(E))`: canonical time is exactly
`step-1 → step`.

## Modes

The constructor exposes one causal mode:

```text
geometry   default; geometric cognition only
 temporal  geometry + adjacent temporal cognition
```

`mode` is immutable and serialised because it changes the causal state machine.
`predictive` is not a v0.3.0 mode.

```python
from auxein import Auxein

net = Auxein(
    dimension=2,
    memory=20,
    eta=1.0,
    scalar="f64",
    mode="temporal",
    budget=100,
)
```

## Reference implementation

`auxein.py` is the stdlib-only Python semantic reference. It favours a direct
translation of the canon over production-specific optimisation.

```python
net.step([[1.0, 0.0]])
net.step([[1.0, 0.0]])
report = net.step([[1.0, 0.0]])
print(report["readout"])
```

An external presentation is always a non-empty list of vectors. Multiple
vectors in one call are one **logical simultaneous context**, not an execution
batch. Splitting them across calls changes the causal observation.

### Readout

In `geometry` mode the readout is a flat list of exact geometric
recognitions.

```text
[
  [universe, local_input, recognised],
  ...
]
```

In `temporal` mode the complete step context is typed rather than flattened:

```text
{
  "concepts": [
    [universe, local_input, recognised],
    ...
  ],
  "sequences": [
    [
      universe,
      [previous_input, current_input],
      [previous_recognised, current_recognised]
    ],
    ...
  ]
}
```

The two lists merely coexist at the same external causal boundary. No pointer,
CELL id, layer id or persistent concept↔sequence link is created.

### State

The persistent state always contains:

- `format_version=3`, `dimension`, `scalar`, `memory`, `eta`, `mode`, `steps_seen`;
- ordered layers;
- each layer's geometric `cells` and private `sigma` kernels.

In `temporal` mode each layer additionally serialises:

- `temporal_cells` and `temporal_sigma` kernels in dimension `2D`;
- `previous`, the optional recognised context from the immediately preceding
  external step.

`previous` is causal state, not learned knowledge. It advances even at
`eta=0`, and is cleared by a forced material contraction so a sequence cannot
cross a knowledge-destruction boundary.

Budget and universe label belong to the execution/interface environment and are
not serialised.

```python
state = net.export_state()
restored = Auxein.from_state(state, budget=100)
```

## Tests

```bash
python test.py
```

The regression suite covers both modes, including:

- centered-kernel merge and total variance;
- horizontal recurrence;
- conservative multi-winner allocation;
- context mass conservation and vertical silence rules;
- exact geometry-mode regression behaviour;
- strict adjacent temporal order;
- temporal recurrence through `Σᵀ`;
- chain breaking when a step has no recognised context;
- `eta=0` freezing learned state while `previous` still advances;
- temporal state round-trip;
- one global geometric+temporal growth transaction;
- forced contraction invalidating temporal causal registers;
- `0→0` temporal silence;
- f32 persistence and invariances;
- finite f64 geometric extremes and positive-support underflow;
- persistent-boundary closure when f32 projection changes a seed's concern status.

## Experimental lab

The deterministic laboratory accepts `model.mode` in experiment JSON. Experiment
files that omit it default to `geometry`; temporal experiments opt in explicitly.

```bash
python lab.py
```

This regenerates `results.json` from all experiment specifications. Results now
separate geometric and temporal observations:

```text
concept_readout_items
sequence_readout_items
recognised_atoms / unknown_atoms
temporal_recognised_atoms / temporal_unknown_atoms
```

The experiment set includes geometric recursion plus temporal adjacent
recurrence and an explicit gap that breaks the previous-context chain. Worlds
remain external generators only; diagnostic truth never enters Auxein.

A dedicated oracle experiment (`15_temporal_oracle_decode.json`) makes the
v0.3 behaviour human-checkable without adding labels to the engine. The toy
world names two vectors `A` and `B` only on the laboratory side, trains Auxein
from an empty temporal network on `A -> B -> 0`, freezes learning, then probes
with nearby vectors never seen during training. The oracle decodes the returned
geometry after the step and checks that:

- the learned geometric CELLs decode as `A` and `B`;
- the learned temporal CELL decodes as `A -> B`;
- unseen nearby probes are recognised as `A` then `B`;
- the second probe also recognises the learned `A -> B` sequence;
- after a zero-context gap, the reverse probe `B -> A` still recognises both
  concepts but does not recognise `A -> B`.

`results.json` records expected vs decoded readout, squared decoder distances,
and a final-state oracle check. Oracle expectations are compared only after
`network.step()` returns; they are never visible to Auxein. `lab.py` exits
non-zero if any oracle check fails.

A second oracle experiment (`16_temporal_distractor_isolation.json`) learns two
independent directed pairs in the same network and probes crossed/reversed
adjacencies. It checks that the four concepts remain individually recognisable
while only the recurrent directed sequences survive the external decoder.

## Benchmark

`benchmark.py` measures complete causal presentations. The mode can be selected
explicitly so geometry and temporal overhead can be compared on the same
scenario.

```bash
python benchmark.py --mode geometry --scenario singleton --dimension 8
python benchmark.py --mode temporal --scenario singleton --dimension 8
python benchmark.py --mode temporal --scenario temporal-stable --dimension 8
python benchmark.py --mode geometry --scenario pair-context --dimension 8
python benchmark.py --mode geometry --scenario sparse --dimension 8 --cells 512
python benchmark.py --mode geometry --scenario dense --dimension 8 --cells 512
```

`temporal-stable` preloads a known `A→A` temporal `CELL` and measures the full
geometric + temporal recognition path after warmup.

The Python implementation is the semantic reference; production performance is
expected from specialised implementations preserving the same causal
transition.

## Material economy

For persistent scalar size `p` (`4` for f32, `8` for f64):

```text
geometric kernel U_H = (D + 2) p
temporal kernel  U_T = (2D + 2) p
```

The v0.3.0 network header costs `34 + 2p` logical units. A geometry-mode layer
costs `16` units. A temporal-mode layer reserves `33 + U_H` units, including a
fixed slot for its optional previous-context kernel; observing a context can
therefore never cause unbudgeted persistent growth.

New geometric seeds, temporal seeds and an optional frontier layer are committed
in **one global growth transaction**. Before that transaction, every seed request
is projected into the persistent scalar format, zero/covered projected seeds are
discarded, and exact projected clones are coalesced in their own space. Material
affordability is therefore computed from the **net persistent state** that would
actually be committed; an f32 rounding step cannot create a `Σ` kernel that is
already covered by a `CELL`.

A solvable state never destroys knowledge to finance new growth. If contraction
is already mandatory, private `Σ`/`Σᵀ` work is discarded first, then geometric
and temporal `CELL`s share one exact value ordering:

```text
K = ||C||² / (||C||² + V)
```

Equal `K` values live or die together regardless of space.

The ergonomic `budget` argument remains expressed in geometric-kernel units;
all internal affordability decisions use exact integer `budget_units`.

## Files

```text
auxein.py          reference engine v0.3.0
spec/auxein.md     mathematical/material canon v0.3.0
test.py            normative regression tests
benchmark.py       geometry/temporal benchmark harness
lab.py             deterministic experimental runner
worlds.py          external synthetic worlds
experiments/       current-canon experiment specifications
results.json       regenerated experiment output
```

## License

See [`LICENSE`](LICENSE).
