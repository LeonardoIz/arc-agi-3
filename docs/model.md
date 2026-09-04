# The `model/` package

A PyTorch model for playing ARC-AGI-3, built as a package fully independent
from the agent framework. The architecture and a single-episode training
loop are implemented and verified end-to-end — see
[What's implemented](#whats-implemented-vs-what-a-placeholder) for exactly
what that covers and what's still Fase 3 scope. This document covers what's
built and how to use it.

> **Nota:** `docs/design.md` fija el contrato del proyecto y la bitácora de
> decisiones de arquitectura (por qué slots, por qué transformer causal,
> etc.) — ver su §4. Este documento describe el paquete tal como está
> implementado hoy, en sincro con ese diseño.

## Why a separate package

`agent/my_agent.py` is the one file the competition's dev loop expects you
to edit (`make play-local`, `make submit`). Model code needs a different
lifecycle — it's tested and trained offline, has its own dependencies
(PyTorch), and changes on a totally different rhythm than agent glue code.
Keeping it in `model/`, imported by the agent through a single narrow
interface (`model.inference.Policy`), means:

- You can build, test, and train the network without a live game running.
- The agent's baseline strategy never breaks while you experiment — see
  [Falling back to the random baseline](#falling-back-to-the-random-baseline).
- Swapping the architecture later never touches `agent/my_agent.py`.

## Layout

Seven files, each one concern:

| File | Concern |
|---|---|
| `model/config.py` | Hyperparameters (`ModelConfig`, `InferenceConfig`, `TrainConfig`) and the tensor containers passed between modules (`Observation`, `PolicyOutput`, `Decision`). No behavior, just shapes. |
| `model/network.py` | The architecture: `SlotAttention`/`FrameEncoder` (frame → learned object slots), a causal transformer with RoPE over the episode, and `ARCPolicyNetwork` composing them + masking illegal actions. See its module docstring for the full rationale. |
| `model/adapters.py` | The only file that imports `arcengine`. Translates `FrameData` → `Observation` and `PolicyOutput` → `GameAction`, both directions. |
| `model/inference.py` | Checkpoint save/load, the `Policy` interface, and `ModelPolicy` — the class `agent/my_agent.py` is meant to import. |
| `model/data.py` | Loading recorded episodes (`.recording.jsonl`) and a generic replay buffer. Pure I/O/data-structure plumbing — fully implemented. |
| `model/train.py` | Loss functions and the training CLI loop (`python -m model.train`). The objective is decided (behavior cloning / Algorithm Distillation) — the loop itself isn't built yet. |
| `model/__init__.py` | Re-exports the public surface. |

## Data flow

```
FrameData history (frames[], latest_frame) — the WHOLE episode so far
        │  model/adapters.py : build_observation()
        ▼
Observation(grid: [1,T,64,64], action_mask: [1,T,7],
            prev_actions: [1,T], level_ups: [1,T], padding_mask: [1,T])
        │  model/network.py : ARCPolicyNetwork.forward()
        ▼
per frame (batched over T):
  color embed → conv stack → Slot Attention → K object slots → mean-pool
        → fused with prev-action embedding + level-up embedding → 1 token
all T tokens → causal transformer (RoPE) → per-step hidden state
        │
action_head / x_head / y_head → per-step logits
        │  (illegal actions masked to -inf here, every step)
        ▼
PolicyOutput  (predictions at EVERY step; online play reads only the last)
        │  model/adapters.py : decode_action()
        ▼
GameAction (x, y filled in for ACTION6)
```

`model/inference.py`'s `ModelPolicy.act()` runs the whole pipeline in one
call, with the same signature as `Agent.choose_action`. Each call rebuilds
the `Observation` from the full frame history and recomputes the whole
forward pass — no KV-cache yet (see `docs/design.md` §4.4's closing note).

Grid geometry is fixed by the game engine, not a design choice:
`arcengine.camera.Camera.MAX_DIMENSION` is 64, so every frame is a 64×64
grid of color indices 0–15. Of the 8 `GameAction` values, the model only
ever chooses among ACTION1–ACTION7 (7 options); `RESET` is handled by the
agent, not the policy — see `model/config.py`'s constants and
`model/adapters.py`'s `POLICY_ACTIONS` for the exact mapping.

## Why this shape

Driven directly by the project contract (`docs/design.md` §3):

- **Slot Attention, not a plain CNN feature vector** (C10): the model has
  to learn its own object decomposition rather than receive one from
  hand-coded segmentation. `FrameEncoder` decomposes each frame into
  `config.num_slots` slots and mean-pools them (permutation-invariant,
  since Slot Attention doesn't guarantee stable slot ordering across
  separate forward calls).
- **A causal transformer over the whole episode, not a fixed window**
  (C4): "memory lives in the model" is satisfied by feeding the model the
  full history and letting self-attention decide what matters, rather than
  agent code maintaining hand-written state between calls. RoPE is used
  instead of learned absolute positions because episode lengths vary and
  aren't capped in advance.
- **No value head**: training is behavior cloning against expert
  trajectories (Algorithm Distillation, §5.2), not actor-critic, so
  nothing consumes a value estimate — and C9 (no search at inference) means
  nothing ever would.
- **Illegal-action masking in `ARCPolicyNetwork.forward`, not learned**:
  this is a hard game rule (`FrameData.available_actions`), not a decision,
  so it's coded directly rather than left for training to discover.

## What's implemented vs. what a placeholder

**Implemented and tested** (forward pass, backward pass/gradients, causal
masking verified with a same-seed diff test, padding-mask correctness, an
integration run of a real agent against a live game using a checkpoint with
random weights, and a real single-episode training run — loss 1.50 → 0.0001
in 500 steps, 100% of the memorized trajectory's actions reproduced in
deterministic mode; see `docs/design.md` §4.3/§7):
- All of `model/network.py`: `SlotAttention`, `FrameEncoder`,
  `CausalSelfAttention`/`RotaryEmbedding`, `TransformerBlock`,
  `ARCPolicyNetwork` (composition + illegal-action masking).
- All of `model/adapters.py`: online (`build_observation`), offline
  (`build_training_example`), and decoding (`decode_action`) — including
  building `prev_actions`/`level_ups` from
  `FrameData.action_input`/`levels_completed`, and dropping any training
  target the game declared illegal at that step.
- Checkpoint save/load (`model/inference.py`), including the
  self-describing format (see [Checkpoints](#checkpoints) below).
- `ModelPolicy` (agent-facing wrapper) including `try_from_checkpoint`'s
  graceful-fallback path and `reset()` (currently a no-op — the model
  carries no state of its own between calls yet; see its docstring).
- All of `model/data.py` (episode loading, replay buffer).
- `model/train.py`: `policy_loss` and the step loop — trains on one
  recorded episode per step, no cross-episode batching yet.

**Not built yet — Fase 3 scope (training at scale), not this package's
current gap:**
- Batching multiple episodes per step with padding (`Observation` and
  `TrainingExample` already carry a `padding_mask`/`valid` for this; the
  loop in `model/train.py` just doesn't use it yet since there's normally
  only one recording to sample from).
- Generating/collecting the actual training distribution — recorded
  episodes today come from `scripts/play_local.py --record`, one at a time;
  Fase 1's procedural generator is what will make this a real dataset.

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

Or with the scoring evaluator (see `scripts/evaluate.py`):

```bash
ARC_MODEL_CHECKPOINT=checkpoints/step_50000.pt make evaluate GAME=ls20
```

### Training

First record at least one episode (see [below](#recording-episodes)), then:

```bash
make train TRAIN_ARGS="--steps 2000 --lr 1e-3"
```

This calls `python -m model.train`; see `parse_args()` in `model/train.py`
for the full flag list (`--recordings-dir`, `--log-every`, etc.).
Checkpoints land in `checkpoints/step_<N>.pt` by default (`--checkpoint-dir`
to change it). With one recording available, this trains on that single
episode every step — the Fase 2 sanity check (overfitting), not Fase 3
(training at scale on a real distribution of games, which needs Fase 1's
procedural generator and cross-episode batching — see
[What's implemented](#whats-implemented-vs-what-a-placeholder)).

#### Recording episodes

```bash
.venv/bin/python scripts/play_local.py --game ls20 --max-steps 150 --record
```

Sets `RECORDINGS_DIR` and passes `record=True` to the agent
(`agents.recorder.Recorder`), producing a `recordings/*.recording.jsonl`
file; `model/data.py`'s `RecordingDataset` reads a directory of them back
into `FrameData` sequences, and `model/adapters.py`'s
`build_training_example` turns one into training tensors.

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

config = ModelConfig()
net = ARCPolicyNetwork(config)

B, T = 1, 3  # batch of 1, a 3-step episode so far
obs = Observation(
    grid=torch.zeros(B, T, 64, 64, dtype=torch.long),
    action_mask=torch.ones(B, T, config.num_simple_actions, dtype=torch.bool),
    prev_actions=torch.full((B, T), config.num_simple_actions, dtype=torch.long),  # "start" token
    level_ups=torch.zeros(B, T, dtype=torch.long),
    padding_mask=torch.ones(B, T, dtype=torch.bool),
)
out = net(obs)
print(out.action_logits.shape)  # [1, 3, 7] — a prediction at every step
```
