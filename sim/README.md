# Sim harness — LIBERO baseline

Agent orchestrator for the LIBERO simulator. An LLM picks tools from a heterogeneous toolbox per step; each tool wraps a deterministic primitive (perception, motion, grasp) or a sub-policy (pi0.5 VLA).

| File | Purpose |
|---|---|
| `agent_sim.py` | Harness loop, tool dispatch, LLM interface (LiteLLM), pi0.5 client. |
| `spatial_tools.py` | Geometric primitives — Cartesian move, contact-aware descent, bbox → world unprojection. |

## Tools exposed to the LLM

`Perceive`, `Grasp`, `Place`, `Release`, `GoHome`, `VLARollout`, `VerifyCandidate`.

Perception is Gemini Robotics ER (numbered bbox overlays with world-coord estimates from calibrated camera + depth ray-cast). Grasps default to top-down with PCA-yaw fallback. Placement is contact-aware with a `container` vs `flat` distinction. `VLARollout` calls pi0.5 (via websocket, verbatim subgoal string).

## Sensors the LLM sees

- **Camera images**: `agentview` (third-person RGB), `wrist` RGB.
- **Robot state**: `robot0_gripper_qpos`, `ee_force`, EE pose via forward kinematics.
- **Detections**: Gemini ER's numbered `#N` bboxes + world-coord estimates. The LLM sees `#N` labels only, not object names.
- **`task_done`** boolean is surfaced for early-stopping (standard LIBERO convention).

Object ground-truth positions and contact state are **not** exposed. Camera intrinsics + extrinsics are read once for `bbox → world` unprojection; those are calibration data with real-robot equivalents. `HARNESS_PERCEPTION` defaults to `gemini_er` and any other value raises at first `Perceive`.

## Dependencies

Install each upstream first; this folder is glue on top.

- **openpi** — <https://github.com/Physical-Intelligence/openpi> — serves `pi05_libero` on `ws://<host>:<port>`.
- **LIBERO** — <https://github.com/Lifelong-Robot-Learning/LIBERO> — the underlying simulator.
- **LIBERO-PRO** — <https://github.com/Zxy-MLlab/LIBERO-PRO> — the extended task suite used for the reported numbers.
- **Gemini Robotics ER** — perception model called via LiteLLM; requires `GEMINI_API_KEY` with preview access to `gemini/gemini-robotics-er-1.6-preview`.
- **LiteLLM** — Python package; agent LLM is model-agnostic (validated with Gemini 3.5 Flash, Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5, Claude Fable 5, GPT-5-mini, Gemini 3.1 Pro).

## Launching the pi0.5 policy server

The harness talks to `pi05_libero` via WebSocket. Start the server first (from your openpi checkout):

```bash
cd /path/to/openpi
CUDA_VISIBLE_DEVICES=0 \
./.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/path/to/pi05_libero_base/59999
```

Boot takes ~60–90 sec (JAX checkpoint restore + norm stats). Verify with `ss -tnl | grep 8000` and tail the log for `INFO:websockets.server:server listening on 0.0.0.0:8000`.

**Non-obvious flags:**

- `--port` MUST come **before** the `policy:checkpoint` subcommand — otherwise it is applied to the wrong subparser and rejected.
- `--policy.dir` points at the checkpoint step directory (e.g. `.../59999`), NOT `.../59999/params`. The loader appends `/params` itself.
- Use `./.venv/bin/python` directly rather than `uv run` — the latter triggers a rebuild and arg-reordering issues.

**Blackwell / B300 GPUs** (compute_cap 10.3+): the JAX XLA build only knows up to sm_100. It falls back to sm_101 and crashes on the fused Triton GEMM (`tcgen05`) at first inference. Add this env var to disable the Triton GEMM path:

```bash
XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' ./.venv/bin/python scripts/serve_policy.py ...
```

Symptom without the flag: server boots and listens, then dies with `ptxas fatal ... 'tcgen05.mma' not supported on .target 'sm_101'` at the first client rollout. Consumer / A100 / H100 GPUs do not need this flag.

## Run the harness

```bash
export ANTHROPIC_API_KEY=...            # or GEMINI_API_KEY / OPENAI_API_KEY
export GEMINI_API_KEY=...               # required for ER perception
# HARNESS_PERCEPTION defaults to gemini_er — override only for baseline experiments

python agent_sim.py \
  --mode harness --suite libero_goal_task --task 0 --episode 2 \
  --policy-host localhost --policy-port 8000 \
  --model gemini/gemini-3.5-flash \
  --max-llm-steps 35 --max-steps-per-rollout 500 \
  --out-dir results/gemflash_libero_goal_task_t0
```

Optional env: `MAX_ENV_STEPS`, `LIBERO_HORIZON`, `PHASE0_MAX_STEPS`, `EXTRA_PYTHONPATH` (for shared cluster installs).

Raw-VLA baseline (no LLM):

```bash
python agent_sim.py --mode raw --suite libero_goal_task --task 0 --episode 2 \
  --policy-host localhost --policy-port 8000
```
