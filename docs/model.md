# The `model/` package

A PyTorch model for playing ARC-AGI-3, built as a package fully independent
from the agent framework. Nothing here is trained yet — this document
covers what's built, why it's shaped this way, and how to use it once you
implement the architecture.

> **Nota:** este documento describe el scaffold **tal como está hoy**.
> `docs/design.md` fija el contrato del proyecto y la arquitectura acordada,
> y supersede parte de lo de acá — en particular `network.py`, que pasa de
> CNN sobre un frame único a encoder de slots + modelo secuencial sobre el
> episodio. Ver §4.3 de ese documento para qué sobrevive y qué se rehace.

## Why a separate package

`agent/my_agent.py` is the one file the competition's dev loop expects you
to edit (`make play-local`, `make submit`). Model code needs a different
lifecycle — it's tested and trained offline, has its own dependencies
(PyTorch), and changes on a totally different rhythm than agent glue code.
Keeping it in `model/`, imported by the agent through a single narrow
interface, means:

- You can build, test, and train the network without a live game running.
- The agent's baseline strategy never breaks while you experiment — see
  [Falling back to the random baseline](#falling-back-to-the-random-baseline).
- Swapping the architecture later never touches `agent/my_agent.py`.

## Layout

Seven files, each one concern:

| File | Concern |
|---|---|
| `model/config.py` | Hyperparameters (`ModelConfig`, `InferenceConfig`, `TrainConfig`) and the tensor containers passed between modules (`Observation`, `PolicyOutput`, `Decision`). No behavior, just shapes. |
| `model/network.py` | The architecture: `GridBackbone`/`CNNBackbone` (grid → features), `PolicyHead`/`ValueHead` (features → logits), and `ARCPolicyNetwork` (composes them + masks illegal actions). |
| `model/adapters.py` | The only file that imports `arcengine`. Translates `FrameData` → `Observation` and `PolicyOutput` → `GameAction`, both directions. |
| `model/inference.py` | Checkpoint save/load, and `ModelPolicy` — the one class `agent/my_agent.py` is meant to import. |
| `model/data.py` | Loading recorded episodes (`.recording.jsonl`) and a generic replay buffer. Pure I/O/data-structure plumbing — fully implemented. |
| `model/train.py` | Loss functions and the training CLI loop (`python -m model.train`). |
| `model/__init__.py` | Re-exports the public surface. |

## Data flow

```
FrameData (frames[], latest_frame)
        │  model/adapters.py : build_observation()
        ▼
Observation(grid: [1,T,64,64], action_mask: [1,7])
        │  model/network.py : ARCPolicyNetwork.forward()
        ▼
backbone(grid) → features [1, embed_dim]
policy_head(features) → action_logits, x_logits, y_logits
        │  (illegal actions masked to -inf here)
        ▼
PolicyOutput
        │  model/adapters.py : decode_action()
        ▼
GameAction (x, y filled in for ACTION6)
```

`model/inference.py`'s `ModelPolicy.act()` runs the whole pipeline in one
call, with the same signature as `Agent.choose_action`.

Grid geometry is fixed by the game engine, not a design choice:
`arcengine.camera.Camera.MAX_DIMENSION` is 64, so every frame is a 64×64
grid of color indices 0–15. Of the 8 `GameAction` values, the model only
ever chooses among ACTION1–ACTION7 (7 options); `RESET` is handled by the
agent, not the policy — see `model/config.py`'s constants and
`model/adapters.py`'s `POLICY_ACTIONS` for the exact mapping.

## What's implemented vs. what's a placeholder

Deliberately split in two, per your instruction to build the scaffolding
without writing the model's actual intelligence:

**Implemented and tested:**
- `ARCPolicyNetwork.forward`'s composition + illegal-action masking (a hard
  game rule, not a learned behavior).
- Both directions of `model/adapters.py` (frame ↔ tensor, tensor ↔ action).
- Checkpoint save/load (`model/inference.py`), including the
  self-describing format (see [Checkpoints](#checkpoints) below).
- `ModelPolicy` (agent-facing wrapper) including the `try_from_checkpoint`
  graceful-fallback path.
- All of `model/data.py` (episode loading, replay buffer).

**Placeholder (`NotImplementedError`) — this is where your work goes:**
- `CNNBackbone.forward` in `model/network.py` — the actual conv stack.
- `PolicyHead.forward` / `ValueHead.forward` in `model/network.py` — the
  projection layers.
- `policy_loss` / `value_loss` in `model/train.py` — the training
  objective.
- The step loop inside `model/train.py`'s `main()` — currently a `TODO`
  comment showing the intended shape, followed by a `raise
  NotImplementedError`.

Only one backbone (`CNNBackbone`) exists, on purpose: ARC grids are small
(64×64, 16 colors) and spatially local, which is what convolutions are
good at, and there was no second working architecture yet to justify a
selectable registry. `GridBackbone` is still an abstract base class, so
adding an alternative later is one new subclass plus a one-line change in
`ARCPolicyNetwork.__init__` — not a rewrite.

## Checkpoints

`save_checkpoint(path, model, config, step=...)` writes a dict containing
the `state_dict`, the full `ModelConfig` (as plain fields), a step counter,
and a format version. Because the config travels with the weights, a
checkpoint is self-describing — `ModelPolicy.from_checkpoint(path)`
reconstructs the exact architecture it was saved with; you never have to
remember what hyperparameters a given `.pt` file used.

```python
from model import ModelConfig, ARCPolicyNetwork, save_checkpoint

net = ARCPolicyNetwork(ModelConfig())
save_checkpoint("checkpoints/step_1000.pt", net, ModelConfig(), step=1000)
```

Checkpoints are gitignored (`checkpoints/`, `*.pt`, `*.pth` in
`.gitignore`) — they're build artifacts, not source.

## Falling back to the random baseline

`agent/my_agent.py` reads `MODEL_CHECKPOINT` (class attribute, defaults to
the `ARC_MODEL_CHECKPOINT` env var) in `__init__`. If it's unset, or the
checkpoint can't be found or fails to load, `ModelPolicy.try_from_checkpoint`
returns `None` and the agent transparently falls back to the original
random-action strategy — nothing breaks while the model doesn't exist yet
or is mid-experiment. This is why it's always safe to leave
`MODEL_CHECKPOINT` pointed at a path even before you've trained anything.

## Usage

### Playing locally with a trained checkpoint

```bash
ARC_MODEL_CHECKPOINT=checkpoints/step_50000.pt make play-local GAME=ls20
```

### Training

Once `model/network.py`'s `forward` methods and `model/train.py`'s losses
are implemented:

```bash
make train TRAIN_ARGS="--steps 50000 --lr 3e-4"
```

This calls `python -m model.train`; see `parse_args()` in
`model/train.py` for the full flag list. Checkpoints land in
`checkpoints/step_<N>.pt` by default (`--checkpoint-dir` to change it).

Recorded episodes for offline training come from running an agent with
`record=True` (see `agents.recorder.Recorder`), which produces
`.recording.jsonl` files; `model/data.py`'s `RecordingDataset` reads a
directory of them back into `FrameData` sequences.

### Submitting to Kaggle with a trained model

`scripts/build_notebook.py` bundles all of `model/` into the generated
notebook alongside `agent/my_agent.py` (previously only the agent file was
spliced in — `my_agent.py`'s `from model.inference import ModelPolicy`
wouldn't have resolved on Kaggle otherwise). `make submit` picks this up
automatically; no extra steps needed as long as `MODEL_CHECKPOINT` points
somewhere accessible at runtime (e.g. a path inside a Kaggle dataset you've
attached to the notebook).

### Quick sanity check without a real game

```python
import torch
from model import ModelConfig, ARCPolicyNetwork
from model.config import Observation

net = ARCPolicyNetwork(ModelConfig())
obs = Observation(
    grid=torch.zeros(1, 1, 64, 64, dtype=torch.long),
    action_mask=torch.ones(1, 7, dtype=torch.bool),
)
net(obs)  # raises NotImplementedError until CNNBackbone.forward is filled in
```
