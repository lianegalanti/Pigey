# Addressing the Orchestration Gap in Generalist Robots via Physical Agency

A frozen-skill robot system driven by a closed-loop VLM orchestrator — reasoning-heavy
robotic manipulation with no new policy training.

**🌐 Project page:** https://lianegalanti.github.io/Pigey/

**📄 Paper:** [arXiv:2607.21725](https://arxiv.org/abs/2607.21725)

---

*Liane Galanti · Dhruv Shah · Tri Dao*

---

## Code

Two implementations of the agent, sharing the same design — an LLM picks tools from a heterogeneous toolbox per step:

| Folder | Target | Files |
|---|---|---|
| [`real/`](real/) | Real Franka FR3 + π0.5-DROID stack | `agent.ts` (orchestrator), `agent-system.md` (system prompt) |
| [`sim/`](sim/) | LIBERO simulator | `agent_sim.py` (orchestrator + tools), `spatial_tools.py`, `README.md` |

### `real/` — Real-robot agent

The two files (`agent.ts` + `agent-system.md`) are the novel orchestrator. They call into:

- **TAMP server** (`http://ROBOT_HOST:7777`) — wraps [TiPToP](https://github.com/tiptop-robot/tiptop) (perception + cuRobo planning + grasp prediction).
- **DROID server** (`tcp://ROBOT_HOST:9876`) — wraps [DROID](https://github.com/droid-dataset/droid) for VLA rollout + camera + robot state.
- **π0.5 policy server** (`ws://ROBOT_HOST:8000`) — [openpi](https://github.com/Physical-Intelligence/openpi) serving the `pi05_droid` checkpoint.

Hardware: Franka Research 3, Robotiq 2F-85 gripper, 1× ZED 2i (third-person), 1× ZED-Mini (wrist).

Run:

```bash
curl -fsSL https://bun.sh/install | bash

export ANTHROPIC_API_KEY=...
export TAMP_URL=http://<workstation>:7777
export DROID_HOST=<workstation>
export PI0_URL=ws://<workstation>:8000
bun agent.ts "pick up the cup and put it in the basket"
```

Optional env vars: `AGENT_MAX_STEPS`, `AGENT_FAIL_LIMIT`, `AGENT_LLM_TIMEOUT_MS`, `AGENT_SYSTEM`, `AGENT_SKIP_WRIST_REC`.

### `sim/` — LIBERO sim agent

See [`sim/README.md`](sim/README.md) for the full setup, including the pi0.5 policy server launch and the Blackwell/B300 GPU workaround. Short version:

```bash
export ANTHROPIC_API_KEY=...     # or GEMINI_API_KEY / OPENAI_API_KEY
export GEMINI_API_KEY=...        # required for Gemini Robotics ER perception

python agent_sim.py \
  --mode harness --suite libero_goal_task --task 0 --episode 2 \
  --policy-host localhost --policy-port 8000 \
  --model gemini/gemini-3.5-flash \
  --max-llm-steps 35 --max-steps-per-rollout 500 \
  --out-dir results/
```

Perception is Gemini Robotics ER. The agent sees camera images, robot state (`gripper_qpos`, `ee_force`, EE pose via forward kinematics), and Gemini ER's numbered `#N` bbox detections. Object ground-truth positions and contact state are not surfaced.

### Dependencies

All public. Install each upstream first; this repo is the glue on top.

- **openpi** (π0.5 policy server) — <https://github.com/Physical-Intelligence/openpi>
- **TiPToP** (TAMP: perception + cuRobo + grasp prediction) — <https://github.com/tiptop-robot/tiptop>
- **LIBERO** (sim benchmark) — <https://github.com/Lifelong-Robot-Learning/LIBERO>
- **LIBERO-PRO** (extended task suite used for the reported numbers) — <https://github.com/Zxy-MLlab/LIBERO-PRO>
- **DROID** (real-robot camera + Franka wrapper) — <https://github.com/droid-dataset/droid>

Transitive (installed by TiPToP or via HF on first use): M2T2 (grasp), FoundationStereo (depth), SAM2 (segmentation), cuRobo (planning).

Model checkpoints: `pi05_libero` for `sim/` (see openpi docs); `pi05_droid` for `real/` at `gs://openpi-assets/checkpoints/pi05_droid`.

LLM: model-agnostic (LiteLLM). Validated with Gemini 3.5 Flash, Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5, Claude Fable 5, Gemini 3.1 Pro, GPT-5-mini.

Gemini Robotics ER (perception in the sim harness): called via LiteLLM as `gemini/gemini-robotics-er-1.6-preview`; requires `GEMINI_API_KEY`.
