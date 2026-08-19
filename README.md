# Auxein

Auxein is a small unsupervised cognitive engine built from centered kernels,
EMA learning and finite material growth.  It has no matrices, labels, target
loss, top-k selection or persistent graph.

The current mathematical/material canon is **v0.2.0** and lives in
[`spec/auxein.md`](spec/auxein.md).

Its core rule is:

> recurrent unknown becomes local knowledge; knowledge recognised together
> becomes higher context.

## Structure

Auxein has three persistent levels of state:

```text
NETWORK
  └─ ordered LAYERs
       ├─ CELL kernels
       └─ private Σ kernels
```

Every geometric object is a centered kernel:

```text
(W, C, V)
```

where `W` is support, `C` is a vector center and `V` is scalar dispersion.
External vectors enter as point kernels `(r, x, 0)`.  Internal layers receive
at most one contextual kernel from the preceding layer for each logical
presentation.

For a layer:

```text
presentation of kernels
    ↓
CELL concern / allocation
    ├─ unknown → Σ → recurrence → local CELL
    └─ recognised values → one contextual kernel → next LAYER
```

A single recognised value is not a relation and does not create vertical
context.  A context with exactly zero center has no canonical vector direction
and remains silent.

## Reference implementation

`auxein.py` is the stdlib-only Python reference implementation.  It favours a
small, direct translation of the canon over production-specific optimisation.

```python
from auxein import Auxein

net = Auxein(
    dimension=2,
    memory=20,
    eta=1.0,
    scalar="f64",
    budget=100,
)

net.step([[1.0, 0.0]])
net.step([[1.0, 0.0]])       # recurrent unknown can promote locally
report = net.step([[1.0, 0.0]])
print(report["readout"])
```

An external presentation is always a non-empty list of vectors.  Multiple
vectors in one call are one **logical simultaneous context**, not an execution
batch.  Splitting them across calls changes the causal observation.

### State

The persistent state contains only:

- `dimension`, `scalar`, `memory`, `eta`, `steps_seen`;
- ordered layers;
- each layer's `cells` and private `sigma` kernels.

Budget and universe label belong to the execution/interface environment and are
not serialised as learned knowledge.

```python
state = net.export_state()
restored = Auxein.from_state(state, budget=100)
```

## Tests

```bash
python test.py
```

The regression suite focuses on the normative boundaries of v0.2.0, including:

- centered-kernel merge and total variance;
- horizontal recurrence;
- internal `(r,C,V)` concern;
- conservative multi-winner allocation;
- context mass conservation;
- independence of context geometry from learning responsibility;
- singleton vertical silence;
- exact zero-center contextual silence;
- simultaneous-vs-sequential context semantics;
- no artificial deep cascade from a constant input;
- growth transactions and forced solvency;
- strict state round-trip and f32 projection.

## Experimental lab

The laboratory is deliberately built around contextual recursion.  The
`experiments/` directory contains only experiments for the current canon.

```bash
python lab.py
```

This regenerates `results.json` from all experiment specifications.

Current experiments cover:

- local singleton recurrence;
- pair-context learning;
- sequential values not forming a context;
- exactly symmetric zero-center context;
- duplicate/coalescence invariance;
- `eta=0` freeze;
- simultaneous recognised and unknown mass;
- noisy local recurrence and noisy context;
- growth budget gating;
- forced contraction;
- f32 contextual recursion.

Worlds are external generators only.  Diagnostic truth never enters Auxein.

## Benchmark

`benchmark.py` is a small stdlib-only harness.  It measures complete causal
presentations, not isolated arithmetic primitives.

```bash
python benchmark.py --scenario singleton --dimension 8
python benchmark.py --scenario pair-context --dimension 8
python benchmark.py --scenario sparse --dimension 8 --cells 512
python benchmark.py --scenario dense --dimension 8 --cells 512
```

The Python implementation is the semantic reference; production performance is
expected from specialised implementations that preserve the same causal
transition.

## Material economy

The budget is an exact integer capacity.  A kernel costs `(D + 2) * p` logical
units (`p=4` for f32, `p=8` for f64).  Promotion from private `Σ` to public
`CELL` reuses the same payload and has zero marginal material cost.

New seeds and an optional frontier layer are committed in one global growth
transaction.  A solvable state never destroys knowledge merely to finance new
growth.  Forced contraction occurs only when the execution budget is already
violated, and knowledge loss is ordered by the intrinsic value

```text
K = ||C||² / (||C||² + V)
```

which is independent of current support.

## Files

```text
auxein.py          reference engine
spec/auxein.md     mathematical/material canon v0.2.0
test.py            normative regression tests
benchmark.py       performance/persistence benchmark harness
lab.py             deterministic experimental runner
worlds.py          external synthetic worlds
experiments/       current-canon experiment specifications
results.json       regenerated experiment output
```

## License

See [`LICENSE`](LICENSE).
