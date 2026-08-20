# Auxein

Auxein is a small unsupervised cognitive engine built from centered kernels,
EMA learning and finite material growth. It has no matrices, labels, target
loss, top-k selection or persistent graph.

The current mathematical/material canon is **v0.4.0** and lives in
[`spec/auxein.md`](spec/auxein.md).

Its core rule is:

> recurrent unknown becomes local knowledge; knowledge recognised together
> becomes higher context; adjacent recognised contexts can become temporal
> knowledge; temporal knowledge can expose known immediate successors.

## Structure

Auxein has only three architectural levels:

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ temporal space T(E)=E⊕E        [temporal / predictive]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

The temporal compartment is structurally associated with a layer but belongs
to the `NETWORK`: the layer never reads it. Geometric and temporal `CELL`s do
not concern, allocate, learn or compete across spaces. They share only the
material economy and the external readout of a step.

Every cognitive object is a centered kernel `(W, C, V)`: support, vector center
and scalar dispersion. External vectors enter as point kernels `(r, x, 0)`.

For geometry:

```text
presentation of kernels
    ↓
CELL concern / allocation
    ├─ unknown → Σ → recurrence → local CELL
    └─ recognised values → one contextual kernel → next LAYER
```

For temporal and predictive modes, after the complete geometric phase of a
step, the `NETWORK` compares each recognised layer context with that layer's
context from the immediately preceding external step:

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

Predictive mode adds no learned state. Before temporal learning for the current
step, it reads the snapshot of already existing temporal `CELL`s. For a current
recognised context center `C` and a temporal center `C- ⊕ C+`, the source is
concerned exactly when the canonical point/point concern holds:

```text
||C - C-||² < ||C||²
and
||C - C-||² < ||C-||²
```

If so, `C+` is emitted as a known possible immediate successor. Temporal
support and variance do not participate: `Vᵀ = V- + V+` does not contain a
recoverable source variance, so predictive mode does not invent one. Multiple
known successors are all emitted; predictions are never ranked, fed back or
chained.

## Modes

There are exactly three cumulative causal modes:

```text
geometry     geometric cognition only
 temporal    geometry + adjacent temporal cognition
 predictive  geometry + temporal + immediate predictive readout
```

Equivalently:

```text
geometry ⊂ temporal ⊂ predictive
```

`mode` is immutable and serialised because it changes the causal state machine.
There is no independent predictive flag: predictive operation always includes
temporal cognition.

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

In `geometry` mode the readout is a flat list:

```text
[
  [universe, local_input, recognised],
  ...
]
```

In `temporal` mode it is typed:

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

In `predictive` mode a third independent list is added:

```text
{
  "concepts": [...],
  "sequences": [...],
  "predictions": [
    [universe, current_context, recognised_source, predicted_successor],
    ...
  ]
}
```

`current_context` is the recognised geometric context of that layer;
`recognised_source` and `predicted_successor` are respectively the left and
right center projections of the temporal `CELL` that was concerned. Exact
duplicate items are coalesced. No CELL id, layer id, pointer or persistent
concept↔sequence↔prediction relation is created.

A source center exactly at zero is predictively silent because the canonical
point concern cannot hold. A successor center exactly at zero is still an
explicit valid prediction and is distinct from no prediction.

### State

The persistent state always contains:

- `format_version=4`, `dimension`, `scalar`, `memory`, `eta`, `mode`, `steps_seen`;
- ordered layers;
- each layer's geometric `cells` and private `sigma` kernels.

In `temporal` **and** `predictive` modes each layer additionally serialises:

- `temporal_cells` and `temporal_sigma` kernels in dimension `2D`;
- `previous`, the optional recognised context from the immediately preceding
  external step.

Predictive mode adds no persistent learned field beyond temporal mode. For the
same knowledge, the two modes therefore have the same material footprint;
only the immutable mode tag differs.

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

The regression suite covers the three modes, including:

- centered-kernel merge, total variance and recurrence;
- conservative multi-winner allocation and context mass conservation;
- vertical silence and exact geometry-mode regression behaviour;
- strict adjacent temporal order, recurrence through `Σᵀ` and gap breaking;
- `eta=0` freezing learned state while `previous` still advances;
- one global geometric+temporal growth transaction and forced contraction;
- f32 persistence, finite f64 extremes and positive-support underflow;
- persistent-boundary closure after scalar projection;
- predictive center projection independent of temporal support/variance;
- multiple successors without selection;
- zero-source silence and explicit zero successor;
- next-step-only predictive authority for newly promoted temporal `CELL`s;
- identical persistent trajectories/economy for temporal and predictive modes;
- predictive state round-trip.

## Experimental lab

The deterministic laboratory accepts all three modes in experiment JSON:

```bash
python lab.py
```

It regenerates `results.json`. Diagnostics count concept, sequence and
prediction readout items separately. External world/oracle truth is never
passed into Auxein.

The temporal oracle experiments learn and decode directed concepts/sequences,
including distractors and reversed/crossed probes. The predictive oracle
`17_predictive_oracle_decode.json` goes one step further:

1. the external lab names two vectors `A` and `B`;
2. Auxein learns `A → B` from an initially empty predictive network;
3. learning is frozen;
4. an unseen nearby `A'` is presented;
5. **before `B'` is observed**, the returned predictive readout decodes to `B`;
6. the following `B'` can then recognise the already learned `A → B` sequence.

The oracle compares expectations only after `network.step()` returns and exits
non-zero on any semantic mismatch.

## Benchmark

`benchmark.py` measures complete causal presentations:

```bash
python benchmark.py --mode geometry --scenario singleton --dimension 8
python benchmark.py --mode temporal --scenario singleton --dimension 8
python benchmark.py --mode predictive --scenario singleton --dimension 8
python benchmark.py --mode temporal --scenario temporal-stable --dimension 8
python benchmark.py --mode predictive --scenario predictive-stable --dimension 8
python benchmark.py --mode geometry --scenario pair-context --dimension 8
python benchmark.py --mode geometry --scenario sparse --dimension 8 --cells 512
python benchmark.py --mode geometry --scenario dense --dimension 8 --cells 512
```

`temporal-stable` preloads a known `A→A` temporal `CELL` and measures geometry +
temporal recognition. `predictive-stable` uses the same persistent knowledge in
predictive mode and additionally exercises the projection/readout path.

The Python implementation is the semantic reference; production performance is
expected from specialised implementations preserving the same causal
transition.

## Material economy

For persistent scalar size `p` (`4` for f32, `8` for f64):

```text
geometric kernel U_H = (D + 2) p
temporal kernel  U_T = (2D + 2) p
network header   U_N = 34 + 2p
geometry layer   U_L = 16
temporal/predictive layer U_L = 33 + U_H
```

Predictive projections and readout are ephemeral and have no persistent cost.

New geometric seeds, temporal seeds and an optional frontier layer are committed
in **one global growth transaction**. Every seed request is first projected into
the persistent scalar format, zero/covered projected seeds are discarded, and
exact projected clones are coalesced in their own space. Affordability is
computed from the net persistent state that would actually be committed.

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
auxein.py          reference engine v0.4.0
spec/auxein.md     mathematical/material canon v0.4.0
test.py            normative regression tests
benchmark.py       geometry/temporal/predictive benchmark harness
lab.py             deterministic experimental runner
worlds.py          external synthetic worlds
experiments/       current-canon experiment specifications
results.json       regenerated experiment output
```

## License

See [`LICENSE`](LICENSE).
