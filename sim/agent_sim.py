"""
Cortex Toolbox harness — LLM as orchestrator over a heterogeneous primitive set.

Architecture:
    LLM (brain) ──tools──> harness ──> LIBERO env (in-process)
                                  ├──> pi0.5 policy (WebSocket, language-conditioned VLA)
                                  └──> TipTop bridge (geometric grasp, collision-aware motion, TAMP solver)

The LLM picks the right primitive per step from a heterogeneous toolbox:
    Core (open-loop):     MoveTo, Grasp, Release
    Perception:           Perceive, LocateBox, DetectObjects
    Motor — neural:       VLARollout (pi0.5 — dexterous, language-conditioned, can close on air)
    Motor — geometric:    GraspGeometric (deterministic top-down grasp, no language, robust on small objects)
    Motion planning:      PlanReach (collision-aware Cartesian motion, vs open-loop MoveTo)
    Task planning:        PlanFullTask (full TAMP solve via TipTop, last-resort escalation)
    Control flow:         Done

Each tool has different strengths and failure modes — the LLM is responsible for routing.
The thesis: capability is latent in the toolbox; the LLM unlocks it by picking the right tool per step.

Usage:
    # With harness (LLM orchestration over full toolbox):
    ANTHROPIC_API_KEY=... python agent_sim.py --mode harness --suite libero_object_swap --task 0 \
        --model claude-sonnet-4-6 --policy-host localhost --policy-port 8000

    # Raw VLA baseline (no LLM):
    python agent_sim.py --mode raw --suite libero_object_swap --task 0 \
        --policy-host localhost --policy-port 8000
"""
from __future__ import annotations

# Removing OWL-ViT broke an import-order side-effect that SAM relied on.
try:
    import transformers.models.owlv2  # noqa: F401
except Exception:
    pass

import argparse
import base64
import collections
import dataclasses
import io
import json
import math
import os
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np

# Late imports below for libero/openpi/anthropic so --help works without them.


# ============================================================================
# Constants (match OpenPI's LIBERO main.py)
# ============================================================================
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# Higher render resolution gives the LLM better visual detail for bboxes;
# resize_with_pad in the policy obs builder still produces 224x224 for pi0.5.
LIBERO_ENV_RESOLUTION = int(os.environ.get("LIBERO_ENV_RES", "512"))
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10  # wait for objects to settle after env reset
DEFAULT_REPLAN_STEPS = 5

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "libero_100": 400,
}


# ============================================================================
# Argument parsing
# ============================================================================
@dataclasses.dataclass
class Args:
    mode: str  # "raw" or "harness"
    suite: str
    task: int
    episode: int
    seed: int
    policy_host: str
    policy_port: int
    model: str
    max_steps_per_rollout: int
    max_llm_steps: int
    replan_steps: int
    out_dir: str
    no_video: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "harness"], default="harness")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-steps-per-rollout", type=int, default=60,
                   help="Max env steps per VLARollout call (harness mode)")
    p.add_argument("--max-llm-steps", type=int, default=10,
                   help="Max LLM tool-loop iterations per task (harness mode)")
    p.add_argument("--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--no-video", action="store_true")
    return Args(**vars(p.parse_args()))


# ============================================================================
# LIBERO runner (in-process env + pi0.5 client)
# ============================================================================
class LiberoRunner:
    def __init__(self, suite_name: str, task_id: int, seed: int,
                 policy_host: str, policy_port: int, replan_steps: int):
        # LIBERO-PRO uses `from libero import benchmark`; standard libero uses `from libero.libero import benchmark`.
        try:
            from libero import benchmark, get_libero_path
            from libero.envs import OffScreenRenderEnv
        except ImportError:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        # Blackwell-compat: pi0.5's first call needs ~3-5min cuDNN autotune.
        # Disable websocket keepalive pings so the client doesn't time out during JIT.
        try:
            import websockets.sync.client as _ws_sync
            if not getattr(_ws_sync, "_cortex_no_keepalive_patched", False):
                _ws_sync_orig_connect = _ws_sync.connect
                def _cortex_no_keepalive_connect(*args, **kwargs):
                    kwargs['ping_interval'] = None
                    kwargs['ping_timeout'] = None
                    kwargs.setdefault('open_timeout', 60)
                    kwargs.setdefault('close_timeout', 60)
                    kwargs.setdefault('max_size', 2**28)
                    return _ws_sync_orig_connect(*args, **kwargs)
                _ws_sync.connect = _cortex_no_keepalive_connect
                _ws_sync._cortex_no_keepalive_patched = True
        except Exception as _e_ws:
            print(f"[warn] could not patch websockets keepalive: {_e_ws}", flush=True)
        from openpi_client import websocket_client_policy as wsp

        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed
        self.replan_steps = replan_steps

        np.random.seed(seed)
        bd = benchmark.get_benchmark_dict()
        if suite_name not in bd:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(bd.keys())}")
        self.task_suite = bd[suite_name]()
        if task_id >= self.task_suite.n_tasks:
            raise ValueError(f"task_id {task_id} >= n_tasks {self.task_suite.n_tasks}")

        self.task = self.task_suite.get_task(task_id)
        self.initial_states = self.task_suite.get_task_init_states(task_id)

        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        # Pick the instruction string we feed to the agent + pi0.5:
        #  - On _task perturbation suites (libero_*_task), the BDDL :language is the
        #    TRUE goal and differs semantically from the filename (e.g. filename says
        #    "alphabet_soup", BDDL says "cream cheese"; or "turn ON" -> "turn OFF").
        #    We MUST use BDDL because the env checks against the BDDL goal predicate.
        #  - On _swap (and other non-task) suites, the BDDL :language is only
        #    STYLISTICALLY different from the filename (e.g. "Pick the X" vs
        #    "pick up the X") and targets the same object. pi0.5 was trained on the
        #    filename-derived phrasing, so feeding BDDL costs us familiarity for no
        #    semantic gain. Stay with self.task.language (filename-derived) in this case.
        if "_task" in suite_name:
            try:
                _bddl_src = pathlib.Path(bddl_file).read_text()
                import re as _re
                _m = _re.search(r"\(:language\s+([^)]*)\)", _bddl_src)
                self.task_description = _m.group(1).strip() if _m else self.task.language
            except Exception:
                self.task_description = self.task.language
        else:
            self.task_description = self.task.language
        env_args = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
            # Extend default horizon (~1000) so the harness has room for multi-step recovery.
            # CaP-X's own config uses max_steps=8000 for this reason.
            "horizon": int(os.environ.get("LIBERO_HORIZON", "2000")),
            # Depth required for image-based LocateBox (spatial path).
            "camera_depths": True,
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(seed)

        self.client = wsp.WebsocketClientPolicy(host=policy_host, port=policy_port)

        self.obs = None
        self.done = False
        self.success = False
        self.t = 0
        self.action_plan: collections.deque = collections.deque()
        self.replay_frames: list[np.ndarray] = []  # for video out

    def snapshot_initial_scene(self):
        """No-op. Initial-scene OWL-ViT snapshot is unused with gemini_er primary
        perception. Caller relies on per-step Perceive (gemini_er) instead."""
        self._initial_scene = {}

    def reset(self, episode_idx: int = 0):
        self.episode_idx = episode_idx  # remember for reset_current()
        self.env.reset()
        self.obs = self.env.set_init_state(self.initial_states[episode_idx])
        self.t = 0
        self.done = False
        self.success = False
        self.action_plan.clear()
        self.replay_frames.clear()
        # Wait for objects to settle
        for _ in range(NUM_STEPS_WAIT):
            self.obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
            self.t += 1
        # Take an initial-scene snapshot via OWL-ViT (vision-based, not sim peeking).
        # ChangedObjects can later diff current scene against this baseline.
        self.snapshot_initial_scene()

    def repose(self, target_noun: str) -> dict:
        """No-op. repose() used the OWL-ViT-based initial-scene snapshot.
        With gemini_er primary perception, this internal helper is disabled."""
        return {"ok": False, "note": "repose disabled (OWL-ViT removed)"}


    def right_object(self, target_noun: str, max_env_steps: int = 500):
        """Convenience tool: VLARollout with a 'stand X upright' prompt. Use this
        when a previous VLARollout knocked an object on its side (lying bottle,
        tipped cup, etc.) and you want to right it before retrying the actual
        task. No special sim manipulation — just pi0.5 with a specific prompt."""
        return self.vla_rollout(f"stand the {target_noun} upright on the table", max_env_steps)

    def rush_retry(self, refined_subgoal: str, max_repose: int = 3,
                   max_env_steps_per_call: int = 500) -> dict:
        """Single LLM step that compresses the v6m-style recovery pattern into
        ONE tool call: detect displaced objects -> restore them in place via
        Repose -> retry the task with the refined VLA subgoal.

        Sequence inside the harness (no LLM steps consumed for each sub-action):
          1) Run ChangedObjects to find which objects moved >3cm since trial
             start. Sort by displacement distance (largest first).
          2) For each of the top-`max_repose` displaced objects, call Repose()
             to put it back at its initial position. Skip on Repose failure.
          3) Run VLARollout(refined_subgoal). Standard early-stop on success.
          4) Return a structured result summarizing the cycle.

        This is the legitimate equivalent of the v6m "ResetEnv + retry" pattern:
        each cycle gives pi0.5 a clean(-ish) shot, but the restoration is done
        via real pick-and-place (no sim teleportation)."""
        if self.done:
            return {"ok": False, "note": "task already done", "steps": 0}

        out: dict = {"reposed": [], "skipped": [], "vla": None, "steps": 0}

        # Step 1: identify displaced objects
        try:
            cr = self.changed_objects()
        except Exception as e:
            return {"ok": False, "note": f"ChangedObjects failed: {e}", "steps": 0}
        changed = sorted(cr.get("changed", []), key=lambda o: -o.get("distance_m", 0))

        # Step 2: repose top-N displaced objects
        for obj in changed[:max_repose]:
            if self.done:
                break
            name = obj.get("name", "")
            # Strip _main / _can / _bottle suffixes to get the noun the LLM uses
            tgt = name.replace("_main", "").replace("_", " ").strip()
            try:
                rr = self.repose(tgt)
            except Exception as e:
                out["skipped"].append({"name": name, "reason": f"exc: {e}"})
                continue
            out["steps"] += rr.get("steps", 0)
            if rr.get("ok"):
                out["reposed"].append({"name": name, "moved_to": rr.get("moved_to")})
                if self.done:
                    out["ok"] = True
                    return out
            else:
                out["skipped"].append({"name": name, "reason": rr.get("note", "?")})

        # Step 3: retry VLA with refined subgoal
        if self.done:
            out["ok"] = True
            return out
        try:
            vr = self.vla_rollout(refined_subgoal, max_env_steps_per_call)
        except Exception as e:
            out["ok"] = False
            out["note"] = f"VLA failed: {e}"
            return out
        out["vla"] = vr
        out["steps"] += vr.get("steps", 0)
        out["ok"] = True
        return out

    def put_aside(self):
        """Move whatever is in the gripper to a fixed 'stash' pose on the side of
        the workspace and release it. Standard recovery primitive — every real
        robot has a 'stash' area for unwanted items.

        Procedure: lift +10cm, fly to a safe-stash xy, descend ~5cm, release.
        Stash xy is hardcoded to a corner of the workspace that's unlikely to
        interfere with any LIBERO scene (NOT a LIBERO-specific calibration — just
        a fixed workspace 'park' position the same way a real arm has one)."""
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import move_ee_to, get_current_ee_pos
        if self.done:
            return {"steps": 0, "note": "task already done"}
        # 1) Lift first to clear the scene.
        cur = get_current_ee_pos(self.env)
        lift_target = cur + np.array([0.0, 0.0, 0.12], dtype=float)
        r1 = move_ee_to(self.env, lift_target, max_steps=60, pos_tol=0.03, gripper_open=False)
        # 2) Fly to stash xy at a moderate height. The stash is at the front-left
        # of the workspace, well clear of any LIBERO objects.
        stash_xyz = np.array([0.30, -0.30, 1.05], dtype=float)
        r2 = move_ee_to(self.env, stash_xyz, max_steps=200, pos_tol=0.05, gripper_open=False)
        # 3) Open the gripper to drop whatever's held.
        for _ in range(20):
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.t += 1
            if done_flag:
                self.done = True; self.success = True; break
        # 4) Lift back up so we're not still hovering over the stashed item.
        cur2 = get_current_ee_pos(self.env)
        r3 = move_ee_to(self.env, cur2 + np.array([0.0, 0.0, 0.08], dtype=float),
                        max_steps=60, pos_tol=0.03, gripper_open=True)
        # Refresh obs.
        self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
        self.t += 1
        if done_flag:
            self.done = True; self.success = True
        steps = (r1.get("steps",0) + r2.get("steps",0) + 20 + r3.get("steps",0) + 1)
        return {"steps": steps, "reached": r2.get("reached", False),
                "final_err": r2.get("final_err", -1.0)}

    def changed_objects(self, threshold=0.03):
        """No-op (depends on OWL-ViT initial-scene snapshot which we no longer take)."""
        return {"changed": [], "note": "changed_objects disabled"}


    def go_home(self):
        """Return the arm to the trial's home pose. Opens the gripper first so
        anything held is released, then drives the EE to the home Cartesian
        position. Object positions are NOT restored — this is just the standard
        "go home" primitive every real robot has."""
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import move_ee_to, get_current_ee_pos
        if self.done:
            return {"steps": 0, "note": "task already done"}
        # 1) Open gripper so anything held is released in place.
        for _ in range(15):
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.t += 1
            if done_flag:
                self.done = True; self.success = True
                return {"steps": self.t, "done": True}
        # 2) Drive EE to canonical home Cartesian (LIBERO's typical reset pose).
        home_xyz = np.array([-0.208, 0.000, 1.173], dtype=float)
        result = move_ee_to(self.env, home_xyz, max_steps=200, pos_tol=0.03, gripper_open=True)
        self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
        self.t += result.get("steps", 0) + 1
        if done_flag:
            self.done = True; self.success = True
        return {"steps": result.get("steps", 0), "reached": result.get("reached", False),
                "final_err": result.get("final_err", -1.0)}

    def _build_policy_obs(self, prompt: str) -> dict:
        from openpi_client import image_tools
        img = np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1, ::-1])
        # Optional: when self._focus_bbox is set (via vla_focused), draw a thick
        # colored rectangle on the agentview before pi0.5 sees it. This is the
        # "tell pi0.5 which object to grab by visually marking it" trick — fully
        # honest (just image preprocessing) and bypasses semantic disambiguation.
        fb = getattr(self, "_focus_bbox", None)
        if fb is not None:
            try:
                x1, y1, x2, y2 = fb
                H, W = img.shape[:2]
                x1c = max(0, min(int(x1), W - 1)); y1c = max(0, min(int(y1), H - 1))
                x2c = max(x1c + 1, min(int(x2), W)); y2c = max(y1c + 1, min(int(y2), H))
                # 4-pixel thick red rectangle (R=255, G=0, B=0)
                t = 4
                img[y1c:y1c + t, x1c:x2c] = [255, 0, 0]
                img[max(0, y2c - t):y2c, x1c:x2c] = [255, 0, 0]
                img[y1c:y2c, x1c:x1c + t] = [255, 0, 0]
                img[y1c:y2c, max(0, x2c - t):x2c] = [255, 0, 0]
            except Exception:
                pass
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
        return {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": np.concatenate((
                self.obs["robot0_eef_pos"],
                _quat2axisangle(self.obs["robot0_eef_quat"]),
                self.obs["robot0_gripper_qpos"],
            )),
            "prompt": prompt,
        }

    def vla_focused(self, x1: int, y1: int, x2: int, y2: int,
                    subgoal: str, max_env_steps: int) -> dict:
        """Run VLARollout but draw a red bounding box around the (x1,y1,x2,y2)
        region of the agentview image throughout the rollout. The visual marker
        tells pi0.5 exactly which object to interact with, bypassing semantic /
        spatial disambiguation.

        Use AFTER the LLM has identified the correct target via VerifyCandidate.
        The bbox is in agentview pixel coordinates (256x256, top-left origin),
        in the SAME orientation the LLM saw in Perceive's image."""
        self._focus_bbox = (int(x1), int(y1), int(x2), int(y2))
        try:
            result = self.vla_rollout(subgoal, max_env_steps)
        finally:
            self._focus_bbox = None
        return result

    def vla_rollout(self, prompt: str, max_env_steps: int) -> dict:
        """Run pi0.5 for up to max_env_steps with the given prompt. Returns status."""
        steps_run = 0
        if self.done:
            return {"steps": 0, "done": True, "success": self.success, "note": "task already done"}
        for _ in range(max_env_steps):
            if not self.action_plan:
                self.action_plan.clear()
                element = self._build_policy_obs(prompt)
                try:
                    action_chunk = self.client.infer(element)["actions"]
                except Exception as e:
                    return {"steps": steps_run, "done": False, "success": False, "error": f"policy infer failed: {e}"}
                self.action_plan.extend(action_chunk[: self.replan_steps])
            action = self.action_plan.popleft()
            try:
                self.obs, reward, done_flag, info = self.env.step(action.tolist() if hasattr(action, "tolist") else list(action))
            except Exception as e:
                return {"steps": steps_run, "done": False, "success": False, "error": f"env step failed: {e}"}
            self.replay_frames.append(np.ascontiguousarray(self.obs["agentview_image"]))
            steps_run += 1
            self.t += 1
            if done_flag:
                self.done = True
                self.success = True
                break
        return {"steps": steps_run, "done": self.done, "success": self.success}

    def perceive_pngs(self) -> tuple[bytes, bytes]:
        """Return (agentview_png, wrist_png) bytes for the current obs.
        Use [::-1, ::-1] (180 deg) to match OpenPI's policy preprocessing AND
        the convention assumed by spatial_tools.unproject_bbox_to_world's
        llm_view_rotated=True.

        In gemini_er perception mode, overlay numbered bboxes on the agentview
        image so the LLM can visually identify each detection (instead of
        relying on text labels).
        """
        from PIL import Image, ImageDraw, ImageFont
        agent = self.obs["agentview_image"]
        agent = np.ascontiguousarray(agent[::-1, ::-1])
        wrist = self.obs["robot0_eye_in_hand_image"]
        wrist = np.ascontiguousarray(wrist[::-1, ::-1])

        # Annotate agentview with numbered bboxes (gemini_er mode only).
        # If state_text() has run during this dispatch, _gemini_er_bboxes_ordered
        # is fresh and we reuse it. Otherwise (e.g. post-VLARollout) run a fresh
        # detection so we annotate the current image, not stale bboxes.
        if os.environ.get("HARNESS_PERCEPTION", "").lower() == "gemini_er":
            if not getattr(self, "_perceive_pngs_skip_detect", False):
                try:
                    world_dict = self._gemini_er_localize()
                    bboxes = getattr(self, "_gemini_er_bboxes", {})
                    self._gemini_er_bboxes_ordered = [bboxes[k] for k in world_dict if k in bboxes]
                except Exception:
                    self._gemini_er_bboxes_ordered = []
        bboxes_for_annotate = getattr(self, "_gemini_er_bboxes_ordered", None)
        if bboxes_for_annotate:
            pil = Image.fromarray(agent).convert("RGB")
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            for i, (x1, y1, x2, y2) in enumerate(bboxes_for_annotate):
                # Cycle through high-contrast colors
                colors = [(0, 255, 0), (255, 165, 0), (0, 255, 255), (255, 0, 255),
                          (255, 255, 0), (0, 128, 255), (255, 128, 128)]
                color = colors[i % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                # Label with index number
                tx, ty = x1, max(0, y1 - 16)
                draw.rectangle([tx, ty, tx + 22, ty + 16], fill=color)
                draw.text((tx + 4, ty), f"#{i}", fill=(0, 0, 0), font=font)
            agent = np.array(pil)

        out = []
        for arr in (agent, wrist):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            out.append(buf.getvalue())
        return out[0], out[1]

    def state_text(self) -> str:
        ee_pos = self.obs["robot0_eef_pos"]
        gripper = self.obs["robot0_gripper_qpos"]
        gripper_open = float(gripper[0]) > 0.03  # heuristic
        base = (f"step={self.t}, EE=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}), "
                f"gripper={'open' if gripper_open else 'closed'} (q={gripper[0]:.3f},{gripper[1]:.3f}), "
                f"task done={self.done}")
        # Append detected object positions from Gemini ER (bbox → world unproject
        # via calibrated camera intrinsics + depth).
        mode = os.environ.get("HARNESS_PERCEPTION", "gemini_er").lower()
        try:
            if mode == "gemini_er":
                world_dict = self._gemini_er_localize()
                bboxes = getattr(self, "_gemini_er_bboxes", {})
                # Number detections #0, #1, ... and emit positions WITHOUT semantic labels.
                # Store an ordered list of bboxes so perceive_pngs can overlay matching
                # #N annotations on the agentview image — the LLM identifies objects by
                # looking at the image, not by reading our labels.
                ordered = []
                world_ordered = []
                parts = []
                for i, (name, xyz) in enumerate(world_dict.items()):
                    bb = bboxes.get(name)
                    if bb:
                        ordered.append(bb)
                        world_ordered.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                        parts.append(f"#{i} world=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) bbox=[{bb[0]},{bb[1]},{bb[2]},{bb[3]}]")
                    else:
                        parts.append(f"#{i} world=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})")
                self._gemini_er_bboxes_ordered = ordered
                self._gemini_er_world_ordered = world_ordered
              
                # are in a normalized space; some LLMs were passing
                # those coords directly to Grasp, which expects 0-{LIBERO_ENV_RES}
                # pixel space. Result: bbox→world raycast pointed off-scene and
                # tasks failed. Reverted so Perceive output matches exactly.
            else:
                raise ValueError(
                    f"HARNESS_PERCEPTION={mode!r} not supported; only 'gemini_er' is available."
                )
            if parts:
                base += "\nobjects: " + "; ".join(parts)
                base += ("\nIMPORTANT: detections are labeled #0, #1, ... and overlaid as colored rectangles on the agentview image. "
                         "Identify your target by LOOKING at the image and pass its integer as target_index to Grasp/Place. "
                         "Use LocateBox ONLY if target is not in this list. NEVER bbox the whole image.")
            base += f"\nperception_mode={mode}"
        except Exception as e:
            base += f"\nperception error: {e}"
        return base

    def _gemini_er_localize(self):
        """Use Gemini multimodal vision to detect ALL visible objects in the
        agentview, with open vocabulary. Returns dict[snake_case_name -> (x,y,z)].
        """
      
        # reuse the previous result. Halves Gemini ER calls (one per Perceive +
        # one per perceive_pngs annotation, but both now share if same self.t).
        if getattr(self, '_gemini_er_cached_at_step', -1) == self.t:
            self._gemini_er_bboxes = getattr(self, '_gemini_er_cached_bboxes', {})
            return getattr(self, '_gemini_er_cached_world', {})
        import base64, sys as _sys, io as _io, re as _re, json as _json
        _extra = os.environ.get("EXTRA_PYTHONPATH", "").strip()
        if _extra and _extra not in _sys.path:
            _sys.path.insert(0, _extra)
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("[gemini_er] no GEMINI_API_KEY — returning empty", flush=True)
            return {}

        agent_arr = self.obs.get("agentview_image")
        depth = self.obs.get("agentview_depth")
        if agent_arr is None or depth is None:
            return {}
        # llm-view orientation
        agent_rgb = np.ascontiguousarray(agent_arr[::-1, ::-1])
        from PIL import Image as _Image
        _buf = _io.BytesIO()
        _Image.fromarray(agent_rgb).save(_buf, format="PNG")
        b64 = base64.b64encode(_buf.getvalue()).decode("ascii")

        prompt_text = (
            "This is a tabletop manipulation scene. Detect every distinct manipulable "
            "object resting on the table/floor.\n\n"
            "Reply as a single JSON array; each entry has:\n"
            "  - 'label': a short PURELY VISUAL descriptor — describe what you SEE, "
            "do NOT guess semantic identity. Use color + shape + size, in snake_case. "
            "Examples of good labels: 'red_can_short', 'red_bottle_tall', "
            "'orange_carton', 'white_box_flat', 'wicker_basket', 'brown_bottle_small'. "
            "If two objects look visually similar, distinguish them with an _a / _b suffix "
            "(e.g., 'red_can_a', 'red_can_b'). Do NOT name them by what they might be "
            "semantically ('tomato_sauce', 'ketchup', etc.) — that is the agent's job.\n"
            "  - 'bbox': 4-int list [ymin, xmin, ymax, xmax] normalized 0-1000.\n\n"
            "Be conservative: skip the table, the robot arm, the background. Only include "
            "actual manipulable objects. Output ONLY the JSON array, no markdown fences."
        )
        try:
            import litellm
            model = os.environ.get("GEMINI_ER_MODEL", "gemini/gemini-robotics-er-1.6-preview")
            resp = litellm.completion(
                model=model,
                api_key=api_key,
                max_tokens=8000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[gemini_er] API call failed: {e}", flush=True)
            return {}

        # Strip markdown fences if Gemini added them despite the instruction
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text)
        detections = None
        # 1) try to parse the whole response as JSON
        try:
            detections = _json.loads(text)
            if not isinstance(detections, list):
                detections = None
        except Exception:
            pass
        # 2) salvage: extract individual {...} blocks with regex if full parse failed
        if detections is None:
            salvaged = []
            for m in _re.finditer(r"\{[^{}]*\}", text):
                try:
                    obj = _json.loads(m.group(0))
                    if isinstance(obj, dict) and ("label" in obj or "name" in obj) and "bbox" in obj:
                        salvaged.append(obj)
                except Exception:
                    continue
            if salvaged:
                print(f"[gemini_er] full JSON parse failed; salvaged {len(salvaged)} detections from partial output", flush=True)
                detections = salvaged
        # 3) if STILL nothing, retry once with stricter prompt
        if detections is None or not detections:
            try:
                retry_prompt = prompt_text + "\n\nIMPORTANT: Output STRICT valid JSON. No comments, no trailing commas, no markdown. Just the JSON array."
                resp2 = litellm.completion(
                    model=model, api_key=api_key, max_tokens=8000,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": retry_prompt},
                    ]}],
                )
                text2 = (resp2.choices[0].message.content or "").strip()
                text2 = _re.sub(r"^```(?:json)?\s*", "", text2)
                text2 = _re.sub(r"\s*```$", "", text2)
                detections = _json.loads(text2)
                if not isinstance(detections, list): detections = []
                print(f"[gemini_er] retry succeeded with {len(detections)} detections", flush=True)
            except Exception as e2:
                print(f"[gemini_er] retry also failed: {e2}; raw1={text[:150]!r}", flush=True)
                detections = []

        from spatial_tools import unproject_bbox_to_world
        H, W = agent_rgb.shape[:2]
      
        # Old filter [-0.05, 0.20] only kept tabletop; rejected spatial/goal suite raycasts
        # whose Z fell elsewhere (bowl rim vs bottom, cabinet shelf, drawer interior, etc).
        # Empirically Flash perception success was 99% on object suites, 0% on spatial/goal.
        # New range covers everything from drawer interiors (Z~-0.2) to high cabinets (Z~1.5).
        Z_MIN, Z_MAX = -0.30, 1.50
        world = {}
        n_rejected_z = 0
        n_rejected_area = 0
        n_rejected_other = 0
        self._gemini_er_bboxes = {}  # label -> (x1,y1,x2,y2) in llm-view pixel coords
        for det in detections:
            try:
                name = str(det.get("label", det.get("name", ""))).strip().lower().replace(" ", "_").replace("-", "_")
                bb = det.get("bbox", det.get("box_2d", []))
                if not name or len(bb) < 4:
                    n_rejected_other += 1
                    continue
                ymin_n, xmin_n, ymax_n, xmax_n = [int(v) for v in bb[:4]]
                x1 = max(0, min(W - 2, int(round(xmin_n / 1000.0 * W))))
                y1 = max(0, min(H - 2, int(round(ymin_n / 1000.0 * H))))
                x2 = max(x1 + 1, min(W,     int(round(xmax_n / 1000.0 * W))))
                y2 = max(y1 + 1, min(H,     int(round(ymax_n / 1000.0 * H))))
                area = (x2 - x1) * (y2 - y1)
                if area < 16 or area > (W * H * 0.95):
                    n_rejected_area += 1
                    continue
                xyz = unproject_bbox_to_world(
                    self.env, (x1, y1, x2, y2),
                    camera="agentview", img_h=LIBERO_ENV_RESOLUTION, img_w=LIBERO_ENV_RESOLUTION,
                    depth_arr=depth, llm_view_rotated=True,
                )
                if not (Z_MIN <= float(xyz[2]) <= Z_MAX):
                    n_rejected_z += 1
                    print(f"[gemini_er DEBUG] rejected '{name}' z={float(xyz[2]):.3f} out of [{Z_MIN},{Z_MAX}] bb={bb} xy=({x1},{y1},{x2},{y2})", flush=True)
                    continue
                base_name = name
                k = 2
                while name in world:
                    name = f"{base_name}_{k}"; k += 1
                world[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                self._gemini_er_bboxes[name] = (x1, y1, x2, y2)
            except Exception:
                continue
        print(f"[gemini_er] detected {len(world)} objects: {list(world.keys())} "
              f"(rejected: z={n_rejected_z} area={n_rejected_area} other={n_rejected_other} "
              f"out of {len(detections)} dets)", flush=True)
      
        self._gemini_er_cached_world = world
        self._gemini_er_cached_bboxes = dict(self._gemini_er_bboxes)
        self._gemini_er_cached_at_step = self.t
        # Stash the raw Gemini response so state_text() can include it for the agent
        # as a fallback view (in case our parsing/raycast lost some detections).
        self._gemini_er_raw_text = text
        return world

    def save_video(self, path: str, fps: int = 10):
        if not self.replay_frames:
            return False
        import imageio
        imageio.mimwrite(path, [np.asarray(f) for f in self.replay_frames], fps=fps)
        return True

    # ---- Spatial tools (image-based, no privileged state at runtime) ----
    def locate_box(self, bbox) -> dict:
        """LLM-supplied bbox in agentview LLM-view image (after [::-1, ::-1] rotation)
        → world (x, y, z).

        Bbox erosion (shrink by 15%) before unprojection. Coarse bboxes
        from Gemini ER often include neighboring objects (e.g., the cookie box
        edge next to a target bowl). Eroding pulls the sampling point toward
        the object's interior, reducing the "bbox includes occluder" failure.

        (SAM centroid path was tried and removed — the centroid coords returned
        by spatial_tools.get_object_point_cloud_and_pose() were in a different
        frame than expected and produced off-table grasp targets.)
        """
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import unproject_bbox_to_world

        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        # Skip erosion for tiny bboxes (<30px) AND scale erosion by
        # the SHORTER dimension. For narrow tall objects (wine bottle ~40x100),
        # scaling by short-dim (40) instead of each-dim (40 AND 100) prevents
        # over-erosion of the narrow axis where the object is just barely wide
        # enough — fixed Flash regression on goal_swap t9.
        if bw < 30 or bh < 30:
            ebbox = (x1, y1, x2, y2)
        else:
            e = int(round(min(bw, bh) * 0.075))
            ebbox = (x1 + e, y1 + e, x2 - e, y2 - e)
            if ebbox[2] - ebbox[0] < 4 or ebbox[3] - ebbox[1] < 4:
                ebbox = (x1, y1, x2, y2)

        depth = self.obs.get("agentview_depth")
        world = unproject_bbox_to_world(
            self.env, ebbox, camera="agentview",
            img_h=LIBERO_ENV_RESOLUTION, img_w=LIBERO_ENV_RESOLUTION,
            depth_arr=depth, llm_view_rotated=True,
        )
        print(f"[locate_box] bbox-eroded unproject -> world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f}) "
              f"(bbox {bbox}->{ebbox})", flush=True)
        return {"world_xyz": [float(v) for v in world]}

    def _dbg_dump(self, label):
        """Save agentview + wrist images and key state to HARNESS_PLACE_DBG_DIR if set.
        Used to step through locate_and_place phases and SEE what happens."""
        import os
        dbg = os.environ.get("HARNESS_PLACE_DBG_DIR")
        if not dbg:
            return
        try:
            os.makedirs(dbg, exist_ok=True)
            if not hasattr(self, "_dbg_idx"):
                self._dbg_idx = 0
            self._dbg_idx += 1
            idx = self._dbg_idx
            from PIL import Image
            import numpy as _np_dbg
            agent = self.obs.get('agentview_image')
            wrist = self.obs.get('robot0_eye_in_hand_image')
            if agent is not None:
                Image.fromarray(_np_dbg.asarray(agent)).save(f"{dbg}/dbg_{idx:02d}_{label}_agent.png")
            if wrist is not None:
                Image.fromarray(_np_dbg.asarray(wrist)).save(f"{dbg}/dbg_{idx:02d}_{label}_wrist.png")
            # State dump
            from spatial_tools import get_current_ee_pos as _ee_dbg
            ee = _ee_dbg(self.env)
            gq = self.obs.get('robot0_gripper_qpos', None)
            with open(f"{dbg}/dbg_{idx:02d}_{label}.txt", 'w') as f:
                f.write(f"label: {label}\n")
                f.write(f"step (env t): {getattr(self, 't', 'n/a')}\n")
                f.write(f"EE: ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f})\n")
                if gq is not None:
                    f.write(f"gripper_qpos: {list(gq)}\n")
                # Object ground-truth positions intentionally not read; the release
                # is non-privileged. Debug dumps only record EE + gripper state.
        except Exception as e:
            print(f"[_dbg_dump fail at {label}]: {e}")

    def locate_and_place(self, bbox) -> dict:
        """Contact-aware placement with v4 patches (low hover + small descent +
        gripper-closed descent + actual-descent tracking + OSC early-stop +
        lateral-clear before lift). Set HARNESS_PLACE_DBG_DIR to dump per-substep
        images for offline inspection."""
        if self.done:
            return {"steps": 0, "released": False, "done": True}
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))

        loc = self.locate_box(bbox)
        wx, wy, wz = loc["world_xyz"]
        total_steps = 0

      
        # Both were premature optimizations; Place byte-identical to now.
        # Only legitimate non-privileged change kept: F/T sensor in locate_and_grasp.

        # Hover LOW above destination. The Franka gripper body is wider than most
        # LIBERO containers (baskets, bowls, drawers), so the OSC controller fights
        # any descent into them and drifts the gripper UP. Starting closer means
        # the object falls a short distance from release instead of bouncing out
        # from 10cm above.
        self._dbg_dump("00_enter_holding")
        # GEOMETRIC CENTER via plane-intersection.
        # unproject_bbox_to_world uses mj_ray which returns the first solid
        # surface hit -> for an angled agentview, that's the basket NEAR rim,
        # not the interior center. Same problem affects any depth-based fix
        # (depth-max sees PAST the open top to floor beyond the basket).
        # Fix: take the bbox CENTER pixel, build its camera ray, and intersect
        # the ray analytically with the Z=wz plane (rim height). The (x,y) of
        # that intersection IS the basket's geometric center at rim height.
        try:
            import sys as _sys_pp
            _sys_pp.path.insert(0, os.path.dirname(__file__))
            import numpy as _np_pp
            import robosuite.utils.camera_utils as _CU_pp
            sim_pp = self.env.sim
            img_w_pp = LIBERO_ENV_RESOLUTION; img_h_pp = LIBERO_ENV_RESOLUTION
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            # LLM-view -> native: X-flip only (matches unproject_bbox_to_world)
            x1n = (img_w_pp - 1) - bx2; x2n = (img_w_pp - 1) - bx1
            y1n = by1; y2n = by2
            cx_native = (x1n + x2n) / 2.0
            cy_native = (y1n + y2n) / 2.0
            K_pp = _CU_pp.get_camera_intrinsic_matrix(sim_pp, "agentview", img_h_pp, img_w_pp)
            fx_pp, fy_pp = K_pp[0,0], K_pp[1,1]
            cx_pp, cy_pp = K_pp[0,2], K_pp[1,2]
            cam_id_pp = sim_pp.model.camera_name2id("agentview")
            cam_pos_pp = _np_pp.array(sim_pp.data.cam_xpos[cam_id_pp], dtype=float)
            cam_mat_pp = _np_pp.array(sim_pp.data.cam_xmat[cam_id_pp], dtype=float).reshape(3, 3)
            xn_pp = (cx_native - cx_pp) / fx_pp
            yn_pp = (cy_native - cy_pp) / fy_pp
            ray_cam_pp = _np_pp.array([xn_pp, -yn_pp, -1.0])
            ray_world_pp = cam_mat_pp @ ray_cam_pp
            if abs(ray_world_pp[2]) > 1e-4:
                t_pp = (wz - cam_pos_pp[2]) / ray_world_pp[2]
                if t_pp > 0:
                    hit = cam_pos_pp + t_pp * ray_world_pp
                    ix, iy = float(hit[0]), float(hit[1])
                    # Sanity guard: geometric center should be within 15cm of the
                    # rim point (same container, just a different point on it).
                    if abs(ix - wx) < 0.15 and abs(iy - wy) < 0.15:
                        print(f"[locate_and_place] geom-center: rim=({wx:.3f},{wy:.3f},{wz:.3f}) -> center=({ix:.3f},{iy:.3f},{wz:.3f})", flush=True)
                        wx, wy = ix, iy
                    else:
                        print(f"[locate_and_place] geom-center OUT-OF-RANGE rim=({wx:.3f},{wy:.3f}) center=({ix:.3f},{iy:.3f}) - keeping rim", flush=True)
        except Exception as _e_pp:
            print(f"[locate_and_place] geom-center failed: {_e_pp}", flush=True)
        # Hover above the (now-corrected) destination XY
        hover_clearance = 0.10
        r1 = self.plan_reach(wx, wy, wz + hover_clearance, collision_aware=False)
        total_steps += r1.get("steps", 0)
        self._dbg_dump("01_after_hover")
        if self.done:
            return {"steps": total_steps, "released": False, "done": True, "world_xyz": [wx, wy, wz]}

        # (gripper_geoms init removed)

        # (contact-signature helper removed)
            return out

      
        # The Franka gripper is wider than typical LIBERO containers — descending
        # into them causes gripper-rim collisions and OSC drift, which is worse
        # than the hover-release. Flat surface placements use XY+F/T below.
        use_v19m_path = getattr(self, "_place_use_v19m", True)
        descended = 0.0
        if not use_v19m_path:
            print("[locate_and_place] container placement -> v5 hover-release (skipping XY+F/T)", flush=True)
            descent_log = [{"approach": "no_descent_container_placement"}]
        else:
            descent_log = [{"approach": "xy_correct_then_ft_descent"}]
            # XY correction with grasp-offset compensation
            try:
                import sys as _sys_xy
                _sys_xy.path.insert(0, os.path.dirname(__file__))
                from spatial_tools import move_ee_to as _move_xy, get_current_ee_pos as _ee_xy
                import numpy as _np_xy
                cur_xy = _ee_xy(self.env)
                grasp_off = getattr(self, "_last_grasp_offset_xy", (0.0, 0.0))
                ee_target_x = wx - grasp_off[0]
                ee_target_y = wy - grasp_off[1]
                xy_target = _np_xy.array([ee_target_x, ee_target_y, float(cur_xy[2])], dtype=float)
                r_xy = _move_xy(self.env, xy_target, max_steps=20, pos_tol=0.005, gripper_open=False)
                total_steps += r_xy.get("steps", 0)
                ee_after_xy = _ee_xy(self.env)
                print(f"[locate_and_place] XY correction (grasp_off=({grasp_off[0]:+.3f},{grasp_off[1]:+.3f})): "
                      f"EE ({cur_xy[0]:.3f},{cur_xy[1]:.3f}) -> ({ee_after_xy[0]:.3f},{ee_after_xy[1]:.3f}) "
                      f"target ({ee_target_x:.3f},{ee_target_y:.3f}) so obj_center -> dest ({wx:.3f},{wy:.3f}) "
                      f"({r_xy.get('steps',0)} steps)", flush=True)
            except Exception as _e_xy:
                print(f"[locate_and_place] XY correction error: {_e_xy}", flush=True)
          
            try:
                import sys as _sys_dd
                _sys_dd.path.insert(0, os.path.dirname(__file__))
                from spatial_tools import move_ee_to as _move_dd, get_current_ee_pos as _ee_dd
                import numpy as _np_dd
                _robot_dd = None
                try:
                    _robot_dd = self.env.env.robots[0]
                except Exception:
                    pass
                if _robot_dd is not None:
                    ee_force_bias_dd = _np_dd.asarray(_robot_dd.ee_force).copy()
                    F_PLACE_THRESHOLD = 1.5
                    MAX_DESCENT_DD = 0.10
                    STEP_DD = 0.01
                    contact_hit = False
                    for _di in range(int(MAX_DESCENT_DD / STEP_DD)):
                        cur_ee = _ee_dd(self.env)
                        target = _np_dd.array([cur_ee[0], cur_ee[1], cur_ee[2] - STEP_DD], dtype=float)
                        r_dd = _move_dd(self.env, target, max_steps=8, pos_tol=0.005, gripper_open=False)
                        total_steps += r_dd.get("steps", 0)
                        descended += STEP_DD
                        try:
                            delta_F = float(_np_dd.linalg.norm(_np_dd.asarray(_robot_dd.ee_force) - ee_force_bias_dd))
                        except Exception:
                            delta_F = 0.0
                        descent_log.append({"step_m": round(descended, 3), "delta_F_N": round(delta_F, 2)})
                        if delta_F > F_PLACE_THRESHOLD:
                            contact_hit = True
                            print(f"[locate_and_place] contact at descent={descended:.3f}m delta_F={delta_F:.2f}N", flush=True)
                            break
                    if not contact_hit:
                        print(f"[locate_and_place] no contact in {descended:.3f}m descent", flush=True)
                else:
                    print("[locate_and_place] no robot handle, skipping F/T descent", flush=True)
            except Exception as _e_dd:
                print(f"[locate_and_place] descent error: {_e_dd}", flush=True)

        # COMBINED release (playground-confirmed 2026-06-21).
        # Each iter does TWO things:
        #   (1) env.step([0,..,0, -1.0]) — explicitly advances sim with the
        #       gripper-open command. This is what actually opens the gripper.
        #       It also drifts the EE (per-step ~1.4mm based on S1 playground).
        #   (2) move_ee_to(snapshot, max_steps=4, pos_tol=0.003, gripper_open=True)
        #       — actively pulls EE back to snapshot, counters the drift.
        # Together: gripper opens (env.step ticks accumulate -1.0) AND EE stays
        # at snapshot (correction each cycle). Playground S6 showed +2mm net
        # drift with full release, vs v6's 0 drift but NO actual release.
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import move_ee_to as _move_ee_to_release
        from spatial_tools import get_current_ee_pos as _get_ee_release
        release_target = _get_ee_release(self.env).copy()
        for _release_i in range(25):
            # (1) env.step with gripper-open — opens gripper, drifts EE
            self.obs, _, _done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.t += 1
            total_steps += 1
            if _done_flag:
                self.done = True; self.success = True; break
            # (2) pull EE back to snapshot — counters drift, gripper stays open
            r_hold = _move_ee_to_release(self.env, release_target, max_steps=4,
                                          pos_tol=0.003, gripper_open=True)
            self.t += r_hold.get("steps", 0)
            total_steps += r_hold.get("steps", 0)
            if self.done:
                break
        self._dbg_dump("03_after_release")

        if self.done:
            return {"steps": total_steps, "released": True, "done": True,
                    "world_xyz": [wx, wy, wz], "descended_m": round(descended, 3),
                    "descent_log": descent_log}

        # WAIT for physics to settle after release. Without this, the lift
        # immediately after release knocks things around while the object is
        # still mid-fall. 40 sim steps of idle (gripper open, zero arm motion)
        # lets the object fall + settle into the destination before any motion.
        for _ in range(40):
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.t += 1
            total_steps += 1
            if done_flag:
                self.done = True; self.success = True; break
        self._dbg_dump("04_after_wait")

        if self.done:
            return {"steps": total_steps, "released": True, "done": True,
                    "world_xyz": [wx, wy, wz], "descended_m": round(descended, 3),
                    "descent_log": descent_log}

        # Lift vertically. With the higher hover (wz+0.10), finger tips were
        # ~5cm above the rim — the lift is in free space and doesn't catch.
        r_lift = self.move_to(0.0, 0.0, 0.10)
        total_steps += r_lift.get("steps", 0)
        self._dbg_dump("05_after_lift")

      
        # Flat surfaces (plate / stove / rack) have strict ~3cm xy tolerance so
        # precision matters and a re-grasp+correct cycle pays off. Containers
        # (basket / drawer / bowl / bin) have a wide cavity — a 3-4cm off-center
        # release is still cleanly "in" the container, so verify-and-nudge would
        # both be unnecessary AND risk re-grasp collisions with the rim. The
        # LLM-supplied surface_type (already used to route XY+F/T vs hover-release)
        # also gates this step.
        if not self.done and getattr(self, "_place_use_v19m", True):
            try:
                # Fresh Gemini ER snapshot of the post-place scene
                self._gemini_er_cached_at_step = -1
                world_dict = self._gemini_er_localize()
                bboxes = getattr(self, "_gemini_er_bboxes", {})
                best_name, best_xyz = None, None
                for nm, xyz in world_dict.items():
                    d = ((float(xyz[0]) - wx) ** 2 + (float(xyz[1]) - wy) ** 2) ** 0.5
                    if d > 0.05:
                        continue
                    if best_xyz is None or float(xyz[2]) > float(best_xyz[2]):
                        best_name, best_xyz = nm, xyz
                if best_name is not None and best_xyz is not None:
                    bx_w, by_w, bz_w = float(best_xyz[0]), float(best_xyz[1]), float(best_xyz[2])
                    xy_err = ((bx_w - wx) ** 2 + (by_w - wy) ** 2) ** 0.5
                    print(f"[verify] placed object='{best_name}' world=({bx_w:.3f},{by_w:.3f},{bz_w:.3f}) "
                          f"dest=({wx:.3f},{wy:.3f}) xy_err={xy_err:.3f}m", flush=True)
                    if xy_err > 0.025 and best_name in bboxes and not self.done:
                        print(f"[nudge] xy_err={xy_err:.3f}m > 2.5cm — re-grasping bowl and moving to dest", flush=True)
                        grasp_result = self.locate_and_grasp(bboxes[best_name])
                        total_steps += grasp_result.get("steps", 0)
                        if grasp_result.get("held") and not self.done:
                            cur_ee_n = _get_ee_release(self.env)
                            nudge_target = np.array([wx, wy, float(cur_ee_n[2])], dtype=float)
                            r_nudge = _move_ee_to_release(self.env, nudge_target,
                                                           max_steps=40, pos_tol=0.005,
                                                           gripper_open=False)
                            total_steps += r_nudge.get("steps", 0)
                            # Release
                            for _ri in range(25):
                                self.obs, _, _df, _ = self.env.step([0.0]*6 + [-1.0])
                                self.t += 1
                                total_steps += 1
                                if _df:
                                    self.done = True; self.success = True; break
                            print(f"[nudge] re-grasp+move+release complete; done={self.done}", flush=True)
                        else:
                            print(f"[nudge] re-grasp failed (held={grasp_result.get('held')})", flush=True)
                else:
                    print(f"[verify] no detection within 5cm of dest ({wx:.3f},{wy:.3f}); skipping nudge", flush=True)
            except Exception as _e_vn:
                print(f"[verify-nudge error] {_e_vn}", flush=True)

        return {"steps": total_steps, "released": True, "done": self.done,
                "world_xyz": [wx, wy, wz], "descended_m": round(descended, 3),
                "descent_log": descent_log}

    def _measure_grasp_offset_wrist(self):
        """measure (held_object_xy - EE_xy) via wrist depth — find the
        smallest-depth (closest-to-camera) cluster anywhere in the wrist view,
        not just at image center. Bowls have hollow interiors so their RIM is
        the closest surface (not the center, which sees through to the table).
        Solid objects (cans, boxes) show their top surface as the closest cluster.
        Either way, the centroid of the closest-depth pixels is the held object's
        top-surface center. Pure numpy + depth; no SAM, no Gemini ER.
        """
        self._last_grasp_offset_xy = (0.0, 0.0)
        steps_used = 0
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(__file__))
            from spatial_tools import move_ee_to, get_current_ee_pos, unproject_bbox_to_world
            cur_ee_before = get_current_ee_pos(self.env)
            lift_target = cur_ee_before + np.array([0.0, 0.0, 0.08], dtype=float)
            r_lift = move_ee_to(self.env, lift_target, max_steps=30, pos_tol=0.01, gripper_open=False)
            steps_used += r_lift.get("steps", 0)
            # Refresh self.obs so the wrist depth matches the post-lift pose
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [1.0])
            self.t += 1
            steps_used += 1
            if done_flag:
                self.done = True; self.success = True
                return steps_used
            wrist_depth = self.obs.get("robot0_eye_in_hand_depth")
            if wrist_depth is None:
                print("[harness] no wrist depth; offset=(0,0)", flush=True)
                return steps_used
            wd = np.asarray(wrist_depth).squeeze().astype(float)
            H, W = wd.shape[:2]
            # Filter invalid depths (zeros / NaN)
            valid_mask = (wd > 1e-4) & np.isfinite(wd)
            if not valid_mask.any():
                print("[harness] no valid depth pixels; offset=(0,0)", flush=True)
                return steps_used
            # Find the cluster of pixels CLOSEST to the wrist camera (smallest
            # depth values among the valid ones). For a bowl, this is the rim;
            # for a can/box, the top face; either way, this is the held object's
            # top surface, NOT the table (which is farther / larger depth).
            valid_depths = wd[valid_mask]
            close_thr = float(np.percentile(valid_depths, 5))  # closest 5%
            close_mask = valid_mask & (wd <= close_thr)
            ys, xs = np.where(close_mask)
            if len(xs) < 30:
                print(f"[harness] insufficient close pixels (count={len(xs)}); offset=(0,0)", flush=True)
                return steps_used
            cx_pix = float(np.mean(xs))
            cy_pix = float(np.mean(ys))
            # Build a tiny bbox around the centroid for unprojection
            tiny_bbox = (int(cx_pix) - 5, int(cy_pix) - 5, int(cx_pix) + 5, int(cy_pix) + 5)
            try:
                world = unproject_bbox_to_world(
                    self.env, tiny_bbox, camera="robot0_eye_in_hand",
                    img_h=H, img_w=W,
                    depth_arr=wrist_depth, llm_view_rotated=False,
                )
            except Exception as _e_u:
                print(f"[harness] wrist unproject failed: {_e_u}; offset=(0,0)", flush=True)
                return steps_used
            bx, by, bz = float(world[0]), float(world[1]), float(world[2])
            cur_ee_after = get_current_ee_pos(self.env)
            # Elevation sanity: closest cluster should be near gripper height,
            # not at table level.
            if bz < float(cur_ee_after[2]) - 0.15:
                print(f"[harness] closest cluster at bz={bz:.3f} below ee_z-0.15={cur_ee_after[2]-0.15:.3f}; offset=(0,0)", flush=True)
                return steps_used
            ox = bx - cur_ee_after[0]
            oy = by - cur_ee_after[1]
            # Sanity clip: in-gripper offset should not exceed gripper half-width (~3cm).
            if abs(ox) > 0.03 or abs(oy) > 0.03:
                print(f"[harness] |offset|>3cm ({ox:+.3f},{oy:+.3f}); clipped to (0,0)", flush=True)
                return steps_used
            self._last_grasp_offset_xy = (ox, oy)
            print(f"[harness] grasp_offset_xy = ({ox:+.3f}, {oy:+.3f}) "
                  f"(closest-cluster pix=({cx_pix:.0f},{cy_pix:.0f}) world=({bx:.3f},{by:.3f},{bz:.3f}), "
                  f"EE=({cur_ee_after[0]:.3f},{cur_ee_after[1]:.3f},{cur_ee_after[2]:.3f}))", flush=True)
            return steps_used
        except Exception as _e_gxy:
            print(f"[harness] offset measurement failed: {_e_gxy}", flush=True)
            return steps_used

    def locate_and_grasp(self, bbox) -> dict:
        """Combined LocateBox + contact-aware GraspGeometric in a single harness
        operation. The LLM passes a bbox (from Perceive's detection list); the
        harness internally unprojects to world (x, y, z) and grasps.

        SPIRAL SEARCH: on NOTHING HELD, retry at small xy offsets to handle the
        typical 2-3cm error in bbox->world projection for small/cylindrical objects.

        Returns: {steps, held, world_xyz, descended_m, descent_log}."""
        if self.done:
            return {"steps": 0, "held": False, "done": True, "note": "task already done"}
        loc = self.locate_box(bbox)
        wx, wy, wz = loc["world_xyz"]
        # First attempt at the projected world coord
        g = self.grasp_geometric(wx, wy, wz)
        g["world_xyz"] = [wx, wy, wz]
        if g.get("held") or self.done:
            g["search_offsets_tried"] = [(0.0, 0.0)]
            if g.get("held") and not self.done:
                g["steps"] = g.get("steps", 0) + self._measure_grasp_offset_wrist()
            return g
        # NOTHING HELD on first try — spiral search ±2.5cm
        search = [(0.025, 0.0), (-0.025, 0.0), (0.0, 0.025), (0.0, -0.025)]
        tried = [(0.0, 0.0)]
        total_steps = g.get("steps", 0)
        for dx, dy in search:
            if self.done: break
            # Open gripper before retry (was closed on empty air)
            for _ in range(15):
                self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
                if done_flag:
                    self.done = True; self.success = True; break
            if self.done: break
            g2 = self.grasp_geometric(wx + dx, wy + dy, wz)
            total_steps += g2.get("steps", 0)
            tried.append((dx, dy))
            if g2.get("held"):
                g2["world_xyz"] = [wx + dx, wy + dy, wz]
                g2["search_offsets_tried"] = tried
                if not self.done:
                    total_steps += self._measure_grasp_offset_wrist()
                g2["steps"] = total_steps
                return g2
      
        # more attempt at the original (wx, wy) but descending 2cm past first
        # F/T contact (catches "first contact was a glancing brush above the
        # actual rim" cases where the gripper closed on air). Pure sensor-driven;
        # no privileged sim data.
        if not self.done:
            for _ in range(15):
                self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
                if done_flag:
                    self.done = True; self.success = True; break
            if not self.done:
                g_deep = self.grasp_geometric(wx, wy, wz, descend_past_contact=0.02)
                total_steps += g_deep.get("steps", 0)
                if g_deep.get("held"):
                    g_deep["world_xyz"] = [wx, wy, wz]
                    g_deep["search_offsets_tried"] = tried + [("deep_retry", 0.02)]
                    if not self.done:
                        total_steps += self._measure_grasp_offset_wrist()
                    g_deep["steps"] = total_steps
                    return g_deep

        # All offsets + deep retry failed
        g["world_xyz"] = [wx, wy, wz]
        g["search_offsets_tried"] = tried
        g["steps"] = total_steps
        return g

    def move_to(self, dx: float, dy: float, dz: float, max_steps: int = 200) -> dict:
        """Drive EE by Cartesian relative offset using OSC_POSE deltas (no VLA)."""
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import move_ee_to, get_current_ee_pos
        if self.done:
            return {"steps": 0, "reached": False, "note": "task already done"}
        cur = get_current_ee_pos(self.env)
        target = cur + np.array([dx, dy, dz], dtype=float)
        # Hold gripper at whatever it currently is (open if currently open, else closed)
        gripper_open = float(self.obs.get("robot0_gripper_qpos", [0.04])[0]) > 0.03
        result = move_ee_to(self.env, target, max_steps=max_steps, pos_tol=0.02,
                            gripper_open=gripper_open)
        # We need to refresh self.obs because move_ee_to stepped the env directly
        # without going through our normal vla_rollout path. Capture a fresh obs.
        self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0 if gripper_open else 1.0])
        self.t += result["steps"] + 1
        if done_flag:
            self.done = True
            self.success = True
        return result

    def do_grasp(self, n_steps: int = 25) -> dict:
        """Close gripper for n_steps (no VLA)."""
        if self.done:
            return {"steps": 0, "note": "task already done"}
        for _ in range(n_steps):
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [1.0])
            self.replay_frames.append(np.ascontiguousarray(self.obs["agentview_image"]))
            self.t += 1
            if done_flag:
                self.done = True
                self.success = True
                break
        return {"steps": n_steps, "done": self.done}

    def do_release(self, n_steps: int = 25) -> dict:
        """Open gripper for n_steps (no VLA)."""
        if self.done:
            return {"steps": 0, "note": "task already done"}
        for _ in range(n_steps):
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.replay_frames.append(np.ascontiguousarray(self.obs["agentview_image"]))
            self.t += 1
            if done_flag:
                self.done = True
                self.success = True
                break
        return {"steps": n_steps, "done": self.done}

    # ===== Vision-based verification (used by VLARollout post-hoc detection) =====

    def _vlm_yesno(self, image_png_bytes, prompt_text):
        """Ask Gemini Flash Lite (via LiteLLM) a YES/NO/UNSURE question about an image.
        Gemini Flash Lite has generous rate limits, supports vision, fast + cheap."""
        import base64, sys as _sys
        _extra = os.environ.get("EXTRA_PYTHONPATH", "").strip()
        if _extra and _extra not in _sys.path:
            _sys.path.insert(0, _extra)
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return "UNSURE"
        try:
            import litellm
            b64 = base64.b64encode(image_png_bytes).decode("ascii")
            resp = litellm.completion(
                model="gemini/gemini-3.1-flash-lite",
                api_key=api_key,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt_text +
                         " Reply with exactly one word: YES, NO, or UNSURE."},
                    ],
                }],
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            if text.startswith("YES"):
                return "YES"
            if text.startswith("NO"):
                return "NO"
            return "UNSURE"
        except Exception:
            return "UNSURE"


    def gemini_locate(self, target_description: str) -> dict:
        """Ask Gemini multimodal to locate an object in the current agentview by
        pixel bounding box. Bypasses OWL-ViT entirely. Useful when OWL-ViT's
        fixed-threshold detection fails or when the target requires spatial
        reasoning (e.g., 'the bowl NOT between the plate and the ramekin') that
        OWL-ViT can't parse but a multimodal LLM can.

        Returns: {'bbox': (x1,y1,x2,y2)} on success, {'bbox': None, 'note': ...} on failure.
        """
        import base64, sys as _sys, io as _io, re as _re
        _extra = os.environ.get("EXTRA_PYTHONPATH", "").strip()
        if _extra and _extra not in _sys.path:
            _sys.path.insert(0, _extra)
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return {"bbox": None, "note": "no GEMINI_API_KEY"}
        try:
            agent_arr = self.obs.get("agentview_image")
            if agent_arr is None:
                return {"bbox": None, "note": "no agentview obs"}
            agent_rgb = np.ascontiguousarray(agent_arr[::-1, ::-1])
            from PIL import Image as _Image
            _buf = _io.BytesIO()
            _Image.fromarray(agent_rgb).save(_buf, format="PNG")
            png_bytes = _buf.getvalue()
            b64 = base64.b64encode(png_bytes).decode("ascii")
            import litellm
            # Use Gemini's NATIVE bbox format: [ymin, xmin, ymax, xmax] normalized 0-1000.
            # This is Google's documented convention for Gemini multimodal grounding.
            prompt_text = (
                f"Detect: {target_description}.\n\n"
                f"Reply with a single line in this EXACT format:\n"
                f"[ymin, xmin, ymax, xmax]\n\n"
                f"All four values are integers normalized to 0-1000 (NOT pixels). "
                f"That is, ymin is the top edge of the bounding box (0=top of image, "
                f"1000=bottom); xmin is the left edge (0=left, 1000=right). "
                f"Output ONLY the bracketed list, no other text. If the object is not "
                f"visible in the image, reply with exactly: NONE")
            resp = litellm.completion(
                model="gemini/gemini-3.1-flash-lite",
                api_key=api_key,
                max_tokens=40,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }],
            )
            text = (resp.choices[0].message.content or "").strip()
            if "NONE" in text.upper() and "[" not in text:
                return {"bbox": None, "note": "Gemini reports not visible"}
            # Extract 4 ints from response (Gemini's normalized 0-1000)
            ints = [int(m) for m in _re.findall(r"\d+", text)]
            if len(ints) < 4:
                return {"bbox": None, "note": f"could not parse bbox from: {text[:80]!r}"}
            # Gemini's format is [ymin, xmin, ymax, xmax] normalized 0-1000
            ymin_n, xmin_n, ymax_n, xmax_n = ints[:4]
            H, W = agent_rgb.shape[:2]
            x1 = int(round(xmin_n / 1000.0 * W))
            y1 = int(round(ymin_n / 1000.0 * H))
            x2 = int(round(xmax_n / 1000.0 * W))
            y2 = int(round(ymax_n / 1000.0 * H))
            # Clamp + sanity check
            x1 = max(0, min(x1, W - 2)); y1 = max(0, min(y1, H - 2))
            x2 = max(x1 + 1, min(x2, W)); y2 = max(y1 + 1, min(y2, H))
            # Reject if bbox is degenerately small or the whole image
            area = (x2 - x1) * (y2 - y1)
            if area < 16 or area > (W * H * 0.95):
                return {"bbox": None, "note": f"degenerate bbox after rescale: [{x1},{y1},{x2},{y2}], raw={text[:80]!r}"}
            return {"bbox": (x1, y1, x2, y2), "raw": text}
        except Exception as e:
            return {"bbox": None, "note": f"gemini_locate err: {e}"}

    def verify_candidate(self, x1: int, y1: int, x2: int, y2: int, target_noun: str) -> str:
        """Crop the agentview at the bbox and ask the VLM whether the cropped
        region is the target_noun. Returns 'YES' / 'NO' / 'UNSURE'.

        Use this BEFORE grasping when DetectObjects returns multiple candidates
        for a confusable target (e.g., 3 red bottles when asked for 'ketchup').
        Verify each candidate, then geometric-grasp the verified one."""
        try:
            agent_arr = self.obs.get("agentview_image")
            if agent_arr is None:
                return "UNSURE"
            # rotate 180 like perceive
            agent_rgb = np.ascontiguousarray(agent_arr[::-1, ::-1])
            H, W = agent_rgb.shape[:2]
            # clamp bbox to image bounds
            x1c = max(0, min(int(x1), W - 1))
            y1c = max(0, min(int(y1), H - 1))
            x2c = max(x1c + 1, min(int(x2), W))
            y2c = max(y1c + 1, min(int(y2), H))
            # crop with small context padding (8 pixels)
            pad = 8
            cx1 = max(0, x1c - pad); cy1 = max(0, y1c - pad)
            cx2 = min(W, x2c + pad); cy2 = min(H, y2c + pad)
            crop = agent_rgb[cy1:cy2, cx1:cx2]
            from PIL import Image as _Image
            import io as _io
            _buf = _io.BytesIO()
            _Image.fromarray(crop).save(_buf, format="PNG")
            return self._vlm_yesno(
                _buf.getvalue(),
                f"This is a cropped region from a robot agentview camera. "
                f"Is the object shown clearly a '{target_noun}'? Ignore other "
                f"objects on the edges — focus on the main object in the center.")
        except Exception:
            return "UNSURE"



    def smart_grasp(self, bbox: tuple) -> dict:
        """Oriented grasp using a SAM-mask point cloud + PCA. Computes the
        object's principal axis from its point cloud, yaws the gripper to
        align the grasp axis with the object's SHORT axis, then descends and
        closes. Better than top-down GraspGeometric for thin, elongated, or
        lying objects (e.g., tipped bottles).

        bbox: (x1, y1, x2, y2) in agentview pixels (LLM-view orientation).
        Returns dict with steps, held, pose_info (debugging).
        """
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import get_object_point_cloud_and_pose, move_ee_to, get_current_ee_pos

        if self.done:
            return {"steps": 0, "held": False, "done": True, "note": "already done"}

        # Get RGB + depth from current obs
        depth = self.obs.get("agentview_depth")
        rgb = self.obs.get("agentview_image")
        if depth is None or rgb is None:
            return {"steps": 0, "held": False, "note": "missing rgb/depth"}

        pose = get_object_point_cloud_and_pose(
            self.env, bbox, camera="agentview",
            img_h=LIBERO_ENV_RESOLUTION, img_w=LIBERO_ENV_RESOLUTION,
            depth_arr=depth, rgb_arr=np.asarray(rgb), llm_view_rotated=True,
        )
        if pose is None:
            return {"steps": 0, "held": False, "note": "PCA pose failed (no SAM mask or empty point cloud)"}

        cx, cy, _cz = pose["centroid"]
        top_z = pose["top_z"]
        yaw_rad = pose["yaw_rad"]
        width = pose["width"]

        total_steps = 0

        # 1) Hover 12cm above object top with gripper open
        target_hover = np.array([cx, cy, top_z + 0.12], dtype=float)
        r1 = move_ee_to(self.env, target_hover, max_steps=200, pos_tol=0.025, gripper_open=True)
        total_steps += r1.get("steps", 0)
        if self.done:
            return {"steps": total_steps, "held": False, "done": True, "pose": pose}

        # 2) Yaw the gripper to align with short axis. OSC_POSE accepts
        # [dx, dy, dz, drx, dry, drz, gripper] with rotation deltas scaled by
        # 0.5 rad per unit. We split the yaw into small chunks of 0.4 rad each
        # and step until target yaw is reached.
        # Note: This is the EE-frame z-rotation. For LIBERO's franka with the
        # gripper pointing down, EE-z aligns with world-z, so drz rotates yaw.
        remaining = float(yaw_rad)
        # Wrap to [-pi/2, +pi/2] — the gripper is symmetric so 180° rotations are equivalent.
        while remaining > np.pi / 2: remaining -= np.pi
        while remaining < -np.pi / 2: remaining += np.pi
        # Send rotation in chunks
        chunk = 0.30  # rad per OSC step
        n_chunks = max(1, int(np.ceil(abs(remaining) / chunk)))
        per_chunk = remaining / n_chunks
        for _ in range(n_chunks):
            drz_norm = float(np.clip(per_chunk / 0.5, -1.0, 1.0))  # OSC rot scale = 0.5 rad/unit
            action = [0.0, 0.0, 0.0, 0.0, 0.0, drz_norm, -1.0]  # gripper open
            for _step in range(8):  # 8 sim steps per chunk for the rotation to actuate
                self.obs, _, done_flag, _ = self.env.step(action)
                self.t += 1
                total_steps += 1
                if done_flag:
                    self.done = True; self.success = True
                    return {"steps": total_steps, "held": False, "done": True, "pose": pose}

        # 3) Descend straight down. Hover is at top_z + 0.12. Target descent
        # depth: TCP needs to reach (top_z - max(0.01, width/2)) so the fingers
        # straddle/wrap the object. Compute descent = hover_z - target_z.
        target_z = top_z - max(0.01, width * 0.5)
        descend_dz = -((top_z + 0.12) - target_z)  # negative = downward
        descend_dz = max(descend_dz, -0.20)  # cap descent at 20cm
        cur_ee = get_current_ee_pos(self.env)
        descend_target = cur_ee + np.array([0.0, 0.0, descend_dz], dtype=float)
        r3 = move_ee_to(self.env, descend_target, max_steps=120, pos_tol=0.02, gripper_open=True)
        total_steps += r3.get("steps", 0)
        if self.done:
            return {"steps": total_steps, "held": False, "done": True, "pose": pose}

        # 4) Close gripper
        r4 = self.do_grasp(n_steps=25)
        total_steps += r4.get("steps", 0)
        gripper = self.obs.get("robot0_gripper_qpos", [0.0, 0.0])
        finger_gap = abs(float(gripper[0]) - float(gripper[1])) if len(gripper) >= 2 else 0.0
        held = finger_gap > 0.005

      
        if held and not self.done:
            total_steps += self._measure_grasp_offset_wrist()
        else:
            self._last_grasp_offset_xy = (0.0, 0.0)

        # 5) Lift +12 cm (CONDITIONAL: only if held to preserve scene)
        if held:
            cur_ee = get_current_ee_pos(self.env)
            lift_target = cur_ee + np.array([0.0, 0.0, 0.12], dtype=float)
            r5 = move_ee_to(self.env, lift_target, max_steps=80, pos_tol=0.03, gripper_open=False)
            total_steps += r5.get("steps", 0)
        else:
            # Release + small clear
            for _ in range(15):
                self.obs, _, _, _ = self.env.step([0.0]*6 + [-1.0])
            cur_ee = get_current_ee_pos(self.env)
            r5 = move_ee_to(self.env, cur_ee + np.array([0.0, 0.0, 0.05]),
                              max_steps=40, pos_tol=0.02, gripper_open=True)
            total_steps += r5.get("steps", 0)

        return {"steps": total_steps, "held": held, "done": self.done,
                "centroid": pose["centroid"], "yaw_rad": pose["yaw_rad"],
                "width": pose["width"], "top_z": pose["top_z"]}

    def grasp_geometric(self, tx: float, ty: float, tz: float,
                        descend_past_contact: float = 0.0) -> dict:
        """Contact-aware top-down grasp at world (tx, ty, tz):
            1. PlanReach to (tx, ty, tz + 0.10) — hover.
            2. Descend incrementally (1.5cm/step) until the gripper fingers
               touch SOMETHING — OR a safety maximum is reached (25cm).
            3. If descend_past_contact > 0, descend that many additional
               meters past the F/T contact point — for "deeper retry" on objects
               where the first contact was a glancing brush above the rim.
            4. Close gripper.
            5. Lift +0.10.
        This is robust to wrong z estimates: if LocateBox returned a z that's
        5-30cm off (common for elevated targets where depth is unreliable), we
        still find the target by touch. Mirrors how real-robot grasping works.

        Returns dict with total steps + held (bool, inferred from gripper qpos)."""
        if self.done:
            return {"steps": 0, "note": "task already done", "held": False, "done": True}
        total_steps = 0

        # 1. Hover over target
        r1 = self.plan_reach(tx, ty, tz + 0.10, collision_aware=False)
        total_steps += r1.get("steps", 0)
        if self.done: return {"steps": total_steps, "held": False, "done": True}

        # 2. Contact-aware descent. Identify gripper-finger geom IDs (Franka
        # robosuite naming). Then descend in 1.5cm steps, breaking as soon as
        # any of those geoms touches something.
        sim = self.env.sim
        # (gripper_geoms init removed)

      
        # (the sensor data API — equivalent to a real wrist-mounted F/T sensor).
        # We do NOT read sim.data.ncon / sim.data.contact (which would give us
        # privileged contact-pair identity).
        # Pattern follows robosuite's own Wipe task (wipe.py:720, 768):
        #   bias = robot.ee_force when in free space  (gripper weight + gravity)
        #   contact = |robot.ee_force - bias| > threshold
        try:
            _robot = self.env.env.robots[0]
        except Exception:
            _robot = None
        def _ee_force_mag():
            if _robot is None: return 0.0
            try:
                return float(np.linalg.norm(np.asarray(_robot.ee_force)))
            except Exception:
                return 0.0
      
        # brushing an adjacent object (e.g. plate next to the target bowl) tripped
        # the sensor BEFORE the gripper reached the object's rim, stopping descent
        # 3cm too short and closing on air. 1.5N requires real contact force
        # buildup — matches the descent depth ncon would have produced in.
        FORCE_THRESHOLD = 1.5  # Newtons

        # Snapshot baseline F/T in free space (gripper weight + gravity reads here).
        ee_force_bias = (np.asarray(_robot.ee_force).copy() if _robot is not None
                         else np.zeros(3, dtype=float))
        step_size = 0.015     # 1.5 cm per increment
        max_descent = 0.25    # safety: never descend more than 25 cm total
        descended = 0.0
        descent_log = []
        new_contact = False
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import move_ee_to as _move_ee_to_descent
        from spatial_tools import get_current_ee_pos as _get_ee_descent
        while descended < max_descent and not new_contact and not self.done:
            cur_ee_pos = _get_ee_descent(self.env)
            tgt = cur_ee_pos + np.array([0.0, 0.0, -step_size], dtype=float)
            r = _move_ee_to_descent(self.env, tgt, max_steps=12, pos_tol=0.005,
                                     gripper_open=True)
            self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
            self.t += r.get("steps", 0) + 1
            if done_flag:
                self.done = True; self.success = True
            total_steps += r.get("steps", 0) + 1
            descended += step_size
            # F/T-based contact detection: delta in wrist force from baseline.
            # Robosuite Wipe pattern. No sim.data.ncon access.
            if _robot is not None:
                try:
                    delta_F = float(np.linalg.norm(np.asarray(_robot.ee_force) - ee_force_bias))
                except Exception:
                    delta_F = 0.0
            else:
                delta_F = 0.0
            # Require minimum 0.10m descent before allowing contact-stop
            # (small/cylindrical objects: gripper finger may brush early).
            if descended >= 0.10:
                new_contact = (delta_F > FORCE_THRESHOLD)
            descent_log.append({"descended_m": round(descended, 3),
                                "delta_F_N": round(delta_F, 2)})
            if r.get("steps", 0) < 3 and descended > 0.05:
                # EE motion stalled AFTER descending at least 5cm — likely hit
                # something solid (object top, table, etc). Treat as contact.
                break

        if self.done: return {"steps": total_steps, "held": False, "done": True,
                               "descent_log": descent_log}

      
        # First contact often happens 1-2cm ABOVE the actual rim (glancing brush
        # from finger pad on adjacent geometry). Going an additional 1-2cm down
        # lets the fingers wrap around the object instead of closing on air.
        if descend_past_contact > 0.0 and not self.done:
            extra_descended = 0.0
            extra_step = 0.01  # 1cm per step
            while extra_descended < descend_past_contact and not self.done:
                cur_ee_pos = _get_ee_descent(self.env)
                tgt = cur_ee_pos + np.array([0.0, 0.0, -extra_step], dtype=float)
                r_extra = _move_ee_to_descent(self.env, tgt, max_steps=10,
                                               pos_tol=0.005, gripper_open=True)
                self.obs, _, done_flag, _ = self.env.step([0.0]*6 + [-1.0])
                self.t += r_extra.get("steps", 0) + 1
                if done_flag:
                    self.done = True; self.success = True
                total_steps += r_extra.get("steps", 0) + 1
                extra_descended += extra_step
                descended += extra_step
            print(f"[grasp_geometric] descended +{extra_descended:.3f}m past contact (total descent {descended:.3f}m)", flush=True)

        # 3. Close gripper
        r3 = self.do_grasp(n_steps=25)
        total_steps += r3.get("steps", 0)

        # Inspect gripper qpos to decide "held" — both fingers should be > ~0.005 apart for an object
        gripper = self.obs.get("robot0_gripper_qpos", [0.0, 0.0])
        finger_gap = abs(float(gripper[0]) - float(gripper[1])) if len(gripper) >= 2 else 0.0
        held = finger_gap > 0.005

        if self.done: return {"steps": total_steps, "held": held, "done": True,
                               "descent_log": descent_log, "descended_m": round(descended, 3)}

        # 4. Lift (CONDITIONAL: only if actually holding something to avoid
        # disturbing the scene on failed grasps; preserves scene for VLA fallback recovery)
        if held:
            r4 = self.move_to(0.0, 0.0, 0.10)
            total_steps += r4.get("steps", 0)
        else:
            # Release fingers + small upward retreat to clear (no disturbing lift)
            for _ in range(15):
                self.obs, _, _, _ = self.env.step([0.0]*6 + [-1.0])
            small_lift = self.move_to(0.0, 0.0, 0.05)  # gentle 5cm clear
            total_steps += small_lift.get("steps", 0)

        return {"steps": total_steps, "held": held, "done": self.done,
                "descended_m": round(descended, 3),
                "descent_log": descent_log}

    def plan_reach(self, tx: float, ty: float, tz: float,
                   collision_aware: bool = False, max_steps: int = 200) -> dict:
        """Drive EE to ABSOLUTE world target (tx, ty, tz). Computes delta internally
        and delegates to move_to. The collision_aware flag is recorded for the LLM's
        information; the underlying motion is open-loop OSC (CuRobo collision-aware
        planning is wired through tiptop_bridge — see PlanFullTask)."""
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from spatial_tools import get_current_ee_pos
        if self.done:
            return {"steps": 0, "reached": True, "note": "task already done"}
        cur = get_current_ee_pos(self.env)
        dx = float(tx) - float(cur[0])
        dy = float(ty) - float(cur[1])
        dz = float(tz) - float(cur[2])
        if collision_aware:
            # Future: hand off to TipTop's CuRobo motion planner via bridge.
            # For now, fall through to open-loop OSC and tag the result.
            r = self.move_to(dx, dy, dz, max_steps=max_steps)
            r["collision_aware"] = False  # not actually wired yet
            r["note"] = "collision_aware=True requested but CuRobo bridge not wired — fell back to open-loop OSC"
            return r
        return self.move_to(dx, dy, dz, max_steps=max_steps)

    def plan_full_task(self, task: str) -> dict:
        """Hand the full task to TipTop's cuTAMP planner (via WebSocket bridge), receive a
        plan, and execute it via existing motion primitives.

        For now this is a guarded stub — returns {available: False} unless
        TIPTOP_WS_URL is set in the environment AND the bridge module imports cleanly.
        When wired, the bridge collects (rgb, depth, intrinsics, world_from_cam, q_init),
        sends them over WS, gets a plan, and executes step-by-step."""
        ws_url = os.environ.get("TIPTOP_WS_URL", "").strip()
        if not ws_url:
            return {
                "available": False,
                "reason": "TIPTOP_WS_URL not set in environment. Wire the TipTop bridge first.",
            }
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(__file__))
            from tiptop_bridge import call_tiptop_planner  # noqa: F401
        except Exception as e:
            return {
                "available": False,
                "reason": f"tiptop_bridge import failed: {e}",
            }
        try:
            from tiptop_bridge import call_tiptop_planner
            plan_response = call_tiptop_planner(
                ws_url=ws_url,
                rgb=self.obs["agentview_image"],
                depth=self.obs.get("agentview_depth"),
                env=self.env,
                task=task,
            )
            if not plan_response.get("success"):
                return {
                    "available": True,
                    "done": False,
                    "exec_steps": 0,
                    "plan_steps": 0,
                    "note": f"planner returned failure: {plan_response.get('error', 'unknown')}",
                }
            from tiptop_bridge import execute_plan
            exec_result = execute_plan(self, plan_response["plan"])
            return {
                "available": True,
                "done": self.done,
                "exec_steps": exec_result.get("steps", 0),
                "plan_steps": len(plan_response["plan"]),
            }
        except Exception as e:
            return {
                "available": True,
                "done": False,
                "exec_steps": 0,
                "plan_steps": 0,
                "note": f"plan_full_task errored: {e}",
            }


def _quat2axisangle(quat):
    """From OpenPI's libero main.py."""
    q = np.array(quat, dtype=np.float64)
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


# ============================================================================
# Anthropic client + tool loop
# ============================================================================
TOOLS = [
    {
        "name": "Perceive",
        "description": "Look at the scene. Returns the agentview camera image, the wrist camera image, and current robot state (end-effector position, gripper aperture).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "VLARollout",
        "description": (
            "Execute a physical sub-goal using the pi0.5 visual policy. Give a clear "
            "natural-language sub-goal and pi0.5 will run a closed-loop visual policy "
            "to accomplish it. Examples: 'pick up the black bowl', 'place the bowl on "
            "the plate', 'open the middle drawer'. Break complex tasks into sub-goals "
            "and call VLARollout once per sub-goal. After the rollout, you get a new "
            "image so you can verify the result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subgoal": {
                    "type": "string",
                    "description": "Natural-language sub-goal for the VLA policy (e.g. 'pick up the bowl')."
                },
            },
            "required": ["subgoal"],
        },
    },
    {
        "name": "Release",
        "description": "Open the gripper for ~25 steps (no VLA). Use after MoveTo placed the held object over the destination (e.g. above the basket).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
  
    # (the harness loop exits when runner.success=True). Exposing Done lets the
    # LLM spam it — especially Pro, which called Done up to 53× per trial,
    # wasting LLM-step budget. No tool = no spam possible. Completion is
    # detected automatically from the env's task_done flag.
    {
        "name": "GoHome",
        "description": (
            "Return the arm to its trial home pose (standard 'go home' primitive every real "
            "robot has). Opens the gripper first (releasing whatever was held, IN PLACE — so it "
            "drops wherever the EE currently is), then drives the EE back to the canonical "
            "starting Cartesian position with the gripper open. Does NOT restore object positions. "
            "If you're holding a wrong object, prefer PutAside (drops at a safe corner) over GoHome "
            "(drops in place, possibly in the middle of the workspace)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ===== Heterogeneous-toolbox extensions =====
    {
        "name": "Place",
        "description": (
            "Place the held object onto/into the destination identified by target_index "
            "(one of the #N rectangles shown on the annotated agentview image). You MUST "
            "also specify surface_type: 'flat' for placement ON TOP of a continuous flat "
            "surface (plate / stove / rack / cabinet top), 'container' for release ABOVE "
            "an interior cavity (basket / drawer / bowl / bin). The harness picks the "
            "right placement strategy automatically. Falls back to bbox-input (x1,y1,x2,y2) "
            "ONLY for destinations not in the detection list (use LocateBox first instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_index": {"type": "integer", "description": "Destination object's #N index from the annotated Perceive image."},
                "surface_type": {"type": "string", "enum": ["flat", "container"], "description": "'flat' = place ON TOP of a continuous surface; 'container' = release ABOVE an interior cavity. Decide by LOOKING at the image: is the destination an opening you'd drop something INTO, or a surface you'd rest something ON?"},
                "description": {"type": "string", "description": "Short label (logging only)."},
                "x1": {"type": "integer", "description": "(fallback) bbox if target_index unavailable."},
                "y1": {"type": "integer", "description": "(fallback)"},
                "x2": {"type": "integer", "description": "(fallback)"},
                "y2": {"type": "integer", "description": "(fallback)"},
            },
            "required": ["surface_type"],
        },
    },
    {
        "name": "Grasp",
        "description": (
            "Grasp the object identified by target_index (one of the #N rectangles shown "
            "on the annotated agentview image). The harness internally resolves index -> "
            "bbox -> world via SAM-masked depth, hovers above, and descends with contact-"
            "aware stop. First tries top-down (yaw=0); on NOTHING HELD, auto-retries with "
            "PCA-aligned gripper orientation. Falls back to bbox-input (x1,y1,x2,y2) ONLY "
            "when the target is not in the detection list (call LocateBox first instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_index": {"type": "integer", "description": "Target object's #N index from the annotated Perceive image."},
                "description": {"type": "string", "description": "Short description (logging only)."},
                "x1": {"type": "integer", "description": "(fallback) bbox if target_index unavailable."},
                "y1": {"type": "integer", "description": "(fallback)"},
                "x2": {"type": "integer", "description": "(fallback)"},
                "y2": {"type": "integer", "description": "(fallback)"},
            },
            "required": [],
        },
    },
    {
        "name": "VerifyCandidate",
        "description": (
            "Crop the agentview at a bbox and ask a VLM whether the crop is actually the named "
            "target. Returns YES/NO/UNSURE. Use when Perceive returns multiple visually-similar "
            "candidates (e.g. two `red_can` detections) and you need to disambiguate BEFORE grasping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x1": {"type": "integer"}, "y1": {"type": "integer"},
                "x2": {"type": "integer"}, "y2": {"type": "integer"},
                "target": {"type": "string", "description": "Noun to verify (e.g. 'ketchup bottle')."},
            },
            "required": ["x1", "y1", "x2", "y2", "target"],
        },
    },
]


V1F_PROMPT = """You control a Franka Emika robot arm in the LIBERO simulator. You see two cameras: agentview (third-person) and wrist.

## HOW YOU IDENTIFY OBJECTS

Every Perceive returns the agentview image with detected objects highlighted as colored rectangles labelled `#0`, `#1`, `#2`, ... The text list shows each detection's `world=(x,y,z)` and `bbox=[x1,y1,x2,y2]` — **no semantic name**. **YOUR JOB**: look at the annotated image, decide which `#N` rectangle is your target, and pass that integer as `target_index` to Grasp/Place.

You do NOT need to output bbox pixel coordinates. The harness resolves `target_index -> bbox -> world` for you. Only fall back to bbox-input if your target is genuinely not in the detected list — in that case call `LocateBox(description)` first to add it.

## Tools (8)

- **Perceive()** — captures the annotated agentview + wrist images + robot state + a numbered detection list. Mandatory first action of every trial.
- **VerifyCandidate(x1,y1,x2,y2,target)** — VLM crops the bbox and confirms it's the target. YES/NO/UNSURE. Use only when multiple `#N` rectangles look like plausible candidates and you can't tell from the image alone.
- **Grasp(target_index, description)** — contact-aware grasp on detection `#target_index`. Top-down first, PCA fallback on miss.
- **Place(target_index, surface_type, description)** — contact-aware placement on detection `#target_index`. You MUST specify `surface_type`:
    - `surface_type="flat"` if the destination is a continuous surface where the held object will rest ON TOP (e.g. plate, stove top, cabinet top, rack, table). Harness does fine XY correction + force-monitored descent.
    - `surface_type="container"` if the destination has an interior cavity you'd drop the object INTO from above (e.g. basket, drawer, bowl, bin). Harness hovers above and releases — no descent (gripper would collide with the rim).
    Decide by LOOKING at the image: does the destination present an opening above an interior, or a flat surface? When unsure, prefer `"flat"` (descent is more precise; on a true container it would just contact the rim and release).
- **VLARollout(subgoal)** — run pi0.5 on the subgoal verbatim. After a VLARollout, task_done may be False even if pi0.5 thought it succeeded — call `Perceive` to inspect.
- **GoHome()** — reset arm to home pose. Object positions unchanged. ALWAYS call before retrying a grasp or after a VLA call disturbs the arm.
- **Release()** — open the gripper unconditionally.
- (Task completion is auto-detected by the simulator's predicate; no explicit Done tool — the harness exits the loop as soon as the task succeeds.)

## FIRST-ACTION RULE
The very first tool call of every trial is `Perceive`. NEVER call a VLA tool before you have at least one observation — unverified bare VLA can knock objects off the table irrecoverably.

## PI0.5 (VLA) KNOWN LIMITATIONS — READ FIRST

`VLARollout` invoke pi0.5, a learned VLA policy. It has documented failure modes. **Calling pi0.5 on a task it can't handle does NOT just fail to solve — it ACTIVELY DISTURBS the scene** (grabs the wrong object, lifts it high, drops it mid-air, scatters the layout). This makes structured recovery much harder. **Avoid VLA when the task has these features:**

1. **Negation**: any "NOT between", "NOT next to", "NOT on", "NOT in" — pi0.5 drops the "not" and picks the wrong instance.
2. **Above-table placement**: "on top of cabinet", "on the rack", "top of the drawer" — pi0.5 can't reach above table.
3. **Push verbs**: "push the X to Y" — pi0.5 doesn't slide objects (~0.5% historical success).
4. **Spatial disambiguation among similar objects**: task identifies the target by spatial relation to ANOTHER object — "the bowl ON THE cookie box", "the bowl NEXT TO the ramekin", "the bowl BETWEEN X and Y". Pi0.5 doesn't reliably ground these to the correct instance, AND its failures scramble the very spatial info needed for structured recovery.

For ANY task matching the above, **NEVER call `VLARollout`** — not as primary action, not as fallback, not as recovery. Use the structured GEOMETRIC PIPELINE only.

For tasks WITHOUT these features (clean pick-place by unique object name, articulated open/close, turn on/off): pi0.5 may work. Note however that a silent phase 0 already ran before you were invoked — if you're in this context, that attempt may have already happened.

## CONTEXT: SCENE STATE

Before you were invoked, the harness may have attempted a silent pi0.5 rollout (phase 0). The "PRE-VLA" reference image, if shown, is the canonical initial state; the "CURRENT" image is the scene after phase 0. The scene may be disturbed. Practical implications:
- Always begin with `Perceive` to see where things actually are.
- If the arm is in an awkward pose, call `GoHome` first.
- If the gripper appears closed on nothing, you may still proceed; geometric tools handle this.
- **If phase 0 ran and failed**, the task is by definition one pi0.5 already couldn't solve. **Do NOT call VLARollout again** — same model + same task = same failure + more disturbance. Use GEOMETRIC PIPELINE only.

## DECISION RULE (after Perceive)

Look at the task string and the known-limitations list above:

### Branch A — Pi0.5-incompatible task → GEOMETRIC PIPELINE ONLY
Task has negation / above-table / push / spatial-disambiguation → use structured pipeline, never call VLA tools.

### Branch B — Articulated verb (open, close, turn on/off, lift the lid)
Geometric primitives cannot model articulated motion. If phase 0 didn't already solve it, try `VLARollout(task verbatim)` ONCE. If it fails, the task may simply be unsolvable.

### Branch C — Plain pick/put/place (no limitations triggered)
Use GEOMETRIC PIPELINE. VLA fallback (`VLARollout`) is OK after a failed `Grasp` only on these clean tasks — never on Branch A. (Caveat: if phase 0 already ran and failed, skip the VLA fallback.)

## GEOMETRIC PIPELINE (Branches B & C)

This sequence handles ALL pick-and-place. Each step is needed:

1. **Look at the annotated Perceive image.** Each colored numbered rectangle `#N` is a detected object. Visually examine the pixels inside each rectangle and decide which `#N` is your target. The task tells you what to grasp (e.g. "tomato sauce", "milk", "basket"); your job is to find which numbered rectangle in the picture is that object — based on what it looks like, not on your prior beliefs about real-world products. The LIBERO assets are simplified renderings; the actual texture may look different from real-world objects.
2. (Optional) If two `#N` rectangles look like plausible candidates, call `VerifyCandidate(bbox, target)` on each to disambiguate.
3. `Grasp(target_index=N, description="...")` — N is the integer label of the chosen rectangle. Top-down first; on NOTHING HELD auto-retries with PCA-aligned yaw.
4. After a successful grasp (HELD), call `Perceive` again, identify the destination's `#N`, then `Place(target_index=N, surface_type="flat"|"container", description="...")`. Choose `surface_type` by looking at the destination in the image (see Place tool description).

## VERIFYING AFTER VLA

The simulator's task_done flag may be False even if pi0.5 *thought* it succeeded. After ANY `VLARollout`, call `Perceive` to inspect the post-VLA scene. If you need to confirm specific facts ("is X in the gripper?" "is Y on top of Z?") use `VerifyCandidate` over the relevant bbox.

## FALLBACK FLOW (after geometric fails)

If the GEOMETRIC PIPELINE has failed twice on the same target (object did not move, Perceive confirms):
1. `Release` + `GoHome` to reset arm state.
2. Re-Perceive — maybe the bbox needs refinement; try a slightly adjusted bbox.
3. **Branch A task (pi0.5-incompatible)**: do NOT call VLA as fallback. Accept that the task may be unsolvable from this disturbed state; call `Done` if no progress.
4. **Branch B/C task (pi0.5-compatible)**: only if phase 0 did NOT run, you may try `VLARollout(task verbatim)` ONCE as a last-resort. If phase 0 already ran, skip this — same failure expected.

Cap: at most TWO retries on the same target. Do not loop forever.

## PI0.5 STRING FIDELITY (CRITICAL)

`VLARollout(subgoal)` is passed verbatim to pi0.5. Pi0.5 was trained on the EXACT LIBERO task strings; any extra wording pushes the embedding off-distribution.

GOOD: `pick up the salad dressing and place it in the basket` (verbatim), `pick up the salad dressing` (strict shortening).
BAD: visual descriptors (dark label / black cap), positional descriptors (in the center), state words (unopened).

## Task
The LIBERO goal you must accomplish: \"{TASK}\"
"""

def get_system_prompt(task_description, max_steps_per_rollout):
    return V1F_PROMPT.replace("{ROLL}", str(max_steps_per_rollout)).replace("{TASK}", task_description)




# harness: retry-with-backoff for transient LLM API errors.
# Catches 429 (rate limit), 5xx (server errors), timeouts, and connection errors.
# Each retry does NOT count as an LLM step in the agent loop.
import random as _random_v32

_V32_RETRYABLE_PATTERNS = [
    "429", "rate limit", "Rate limit", "Too many requests",
    "503", "Service unavailable", "ServiceUnavail", "overloaded", "Overloaded",
    "502", "Bad Gateway", "504", "Gateway Timeout",
    "500", "Internal Server Error",
    "timeout", "Timeout", "Read timed out", "Connection",
    "connection reset", "connection aborted", "RemoteProtocolError",
    "credit_limit", "Credit limit",  # if credits get topped up mid-run, retry will succeed
]

def _v32_is_retryable(exc):
    s = str(exc)
    return any(p.lower() in s.lower() for p in _V32_RETRYABLE_PATTERNS)

def call_with_retry(call_fn, *args, **kwargs):
    """Wrap an LLM API call with INDEFINITE retry on transient errors.

    Retries forever for 429 / 5xx / timeout / network / credit_limit errors.
    These are recoverable: rate limits reset, services come back, credits
    get topped up. We keep waiting and trying until the call succeeds.

    Only fails fast (re-raises) on non-retryable errors (400 / 401 / 403
    invalid request / bad auth). Backoff capped at 90s between attempts."""
    import time
    start = time.time()
    attempt = 0
    while True:
        try:
            if attempt > 0:
                elapsed = time.time() - start
                print(f"[retry] attempt {attempt + 1} (total wait so far: {elapsed:.0f}s)", flush=True)
            return call_fn(*args, **kwargs)
        except Exception as e:
            if not _v32_is_retryable(e):
                print(f"[retry] non-retryable error, failing fast: {str(e)[:200]}", flush=True)
                raise
            attempt += 1
            # Backoff: 5s, 10s, 20s, 40s, 80s, capped at 90s + jitter
            wait = min(5 * (2 ** min(attempt - 1, 5)), 90) + _random_v32.uniform(0, 5)
            elapsed = time.time() - start
            print(f"[retry] transient error ({str(e)[:160]}) — waiting {wait:.1f}s before retry (elapsed: {elapsed:.0f}s)", flush=True)
            time.sleep(wait)

def call_claude(api_key: str, model: str, messages: list, system: str, tools: list) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    # Large-context models get a generous
    # output budget so multi-step reasoning + tool_use blocks don't get truncated.
    # Anthropic models work fine at 1024.
    _max_tok = 8192 if ('fable' in model.lower() or 'mythos' in model.lower()) else 1024
    resp = client.messages.create(
        model=model,
        max_tokens=_max_tok,
        system=system,
        messages=messages,
        tools=tools,
    )
    # Convert SDK object → dict for uniform handling
    content = []
    for block in resp.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return {"content": content, "usage": usage, "stop_reason": resp.stop_reason}


def call_gpt(api_key: str, model: str, messages: list, system: str, tools: list,
             base_url: str = None) -> dict:
    """OpenAI Chat Completions equivalent of call_claude.

    Internal harness messages are Anthropic-shaped. We convert to OpenAI shape
    on the way in and back to Anthropic-shaped content blocks on the way out.

    If base_url is provided, route to an OpenAI-compatible endpoint (Together AI for Kimi, etc).
    """
    import json as _json
    import openai
    if base_url:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = openai.OpenAI(api_key=api_key)

    oai_tools = [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        },
    } for t in tools]

    oai_msgs = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        cont = msg["content"]
        if isinstance(cont, str):
            oai_msgs.append({"role": role, "content": cont})
            continue
        if role == "user":
            tool_msgs = []
            user_parts = []
            for blk in cont:
                btype = blk.get("type")
                if btype == "tool_result":
                    tr = blk.get("content")
                    text_chunks = []
                    image_chunks = []
                    if isinstance(tr, list):
                        for sub in tr:
                            if sub.get("type") == "text":
                                text_chunks.append(sub.get("text", ""))
                            elif sub.get("type") == "image":
                                s = sub.get("source", {})
                                d = s.get("data", "")
                                mt = s.get("media_type", "image/png")
                                image_chunks.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mt};base64,{d}"},
                                })
                    else:
                        text_chunks.append(str(tr))
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": blk["tool_use_id"],
                        "content": "\n".join(text_chunks) if text_chunks else "(ok)",
                    })
                    if image_chunks:
                        user_parts.extend(image_chunks)
                        user_parts.append({"type": "text", "text": f"(images from tool_use_id={blk['tool_use_id']})"})
                elif btype == "text":
                    user_parts.append({"type": "text", "text": blk.get("text", "")})
                elif btype == "image":
                    s = blk.get("source", {})
                    d = s.get("data", "")
                    mt = s.get("media_type", "image/png")
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,{d}"},
                    })
            oai_msgs.extend(tool_msgs)
            if user_parts:
                oai_msgs.append({"role": "user", "content": user_parts})
        elif role == "assistant":
            text_parts = []
            tool_calls_oai = []
            for blk in cont:
                if blk.get("type") == "text":
                    if blk.get("text"):
                        text_parts.append(blk["text"])
                elif blk.get("type") == "tool_use":
                    tool_calls_oai.append({
                        "id": blk["id"],
                        "type": "function",
                        "function": {
                            "name": blk["name"],
                            "arguments": _json.dumps(blk.get("input", {})),
                        },
                    })
            asst = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
            if tool_calls_oai:
                asst["tool_calls"] = tool_calls_oai
            oai_msgs.append(asst)

    kwargs = dict(
        model=model,
        messages=oai_msgs,
        tools=oai_tools if oai_tools else None,
        tool_choice="auto" if oai_tools else None,
    )
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        # Reasoning models: bigger budget for hidden CoT.
        kwargs["max_completion_tokens"] = 8192
        # gpt-5.5 and newer reject reasoning_effort + tools on chat/completions;
        # only set it for older gpt-5.x and o-series families that accept it.
        if model.startswith(("gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5-codex", "o1", "o3", "o4")) or model == "gpt-5":
            kwargs["reasoning_effort"] = "low"
            kwargs["verbosity"] = "low"
    else:
        kwargs["max_tokens"] = 1024
    resp = client.chat.completions.create(**{k: v for k, v in kwargs.items() if v is not None})

    msg = resp.choices[0].message
    content_blocks = []
    if msg.content:
        content_blocks.append({"type": "text", "text": msg.content})
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            try:
                inp = _json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                inp = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": inp,
            })
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    stop = "tool_use" if any(b.get("type") == "tool_use" for b in content_blocks) else "end_turn"
    return {"content": content_blocks, "usage": usage, "stop_reason": stop}


def call_gpt_responses(api_key: str, model: str, messages: list, system: str, tools: list) -> dict:
    """OpenAI Responses API equivalent of call_gpt, for reasoning models that
    reject reasoning_effort + tools on chat/completions (gpt-5.4-mini, gpt-5.5+).

    Converts Anthropic-shaped internal messages to Responses API input items.
    """
    import json as _json
    import openai
    client = openai.OpenAI(api_key=api_key)

    # Tools: Responses format = {type, name, description, parameters} (no nested function)
    oai_tools = [{
        "type": "function",
        "name": t["name"],
        "description": t.get("description", ""),
        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
    } for t in tools]

    # Map internal Anthropic-shaped messages -> Responses API input items list
    items = []
    for msg in messages:
        role = msg["role"]
        cont = msg["content"]
        if isinstance(cont, str):
            # plain text user/assistant message
            if role == "user":
                items.append({"role": "user", "content": [{"type": "input_text", "text": cont}]})
            elif role == "assistant":
                items.append({"role": "assistant", "content": [{"type": "output_text", "text": cont}]})
            continue
        # cont is a list of blocks
        if role == "user":
            user_parts = []
            for blk in cont:
                bt = blk.get("type")
                if bt == "tool_result":
                    tr = blk.get("content")
                    text_chunks = []
                    image_parts_after = []
                    if isinstance(tr, list):
                        for sub in tr:
                            if sub.get("type") == "text":
                                text_chunks.append(sub.get("text", ""))
                            elif sub.get("type") == "image":
                                s = sub.get("source", {})
                                d = s.get("data", "")
                                mt = s.get("media_type", "image/png")
                                image_parts_after.append({"type": "input_image",
                                                          "image_url": f"data:{mt};base64,{d}"})
                    else:
                        text_chunks.append(str(tr))
                    items.append({
                        "type": "function_call_output",
                        "call_id": blk["tool_use_id"],
                        "output": "\n".join(text_chunks) if text_chunks else "(ok)",
                    })
                    if image_parts_after:
                        items.append({"role": "user", "content": image_parts_after + [
                            {"type": "input_text", "text": f"(images from tool_use_id={blk['tool_use_id']})"}]})
                elif bt == "text":
                    user_parts.append({"type": "input_text", "text": blk.get("text", "")})
                elif bt == "image":
                    s = blk.get("source", {})
                    d = s.get("data", "")
                    mt = s.get("media_type", "image/png")
                    user_parts.append({"type": "input_image", "image_url": f"data:{mt};base64,{d}"})
            if user_parts:
                items.append({"role": "user", "content": user_parts})
        elif role == "assistant":
            asst_text_parts = []
            asst_calls = []
            for blk in cont:
                if blk.get("type") == "text" and blk.get("text"):
                    asst_text_parts.append({"type": "output_text", "text": blk["text"]})
                elif blk.get("type") == "tool_use":
                    asst_calls.append({
                        "type": "function_call",
                        "call_id": blk["id"],
                        "name": blk["name"],
                        "arguments": _json.dumps(blk.get("input", {})),
                    })
                elif blk.get("type") == "reasoning_block":
                    rb_item = {"type": "reasoning", "summary": blk.get("summary", []) or []}
                    if blk.get("id"): rb_item["id"] = blk["id"]
                    if blk.get("encrypted_content"): rb_item["encrypted_content"] = blk["encrypted_content"]
                    asst_calls.append(rb_item)
            if asst_text_parts:
                items.append({"role": "assistant", "content": asst_text_parts})
            for call_item in asst_calls:
                items.append(call_item)

    kwargs = dict(
        model=model,
        input=items,
        instructions=system,  # system prompt goes here in Responses API
        tools=oai_tools if oai_tools else None,
        tool_choice="auto" if oai_tools else None,
        max_output_tokens=8192,
        store=False,
        include=["reasoning.encrypted_content"],
        reasoning={"effort": "medium"},
    )
    resp = client.responses.create(**{k: v for k, v in kwargs.items() if v is not None})

    # Map response.output back to Anthropic-shaped content blocks
    content_blocks = []
    for out_item in (resp.output or []):
        itype = getattr(out_item, "type", None)
        if itype == "message":
            for cblk in (out_item.content or []):
                ctype = getattr(cblk, "type", None)
                if ctype == "output_text" and getattr(cblk, "text", None):
                    content_blocks.append({"type": "text", "text": cblk.text})
        elif itype == "function_call":
            try:
                inp = _json.loads(out_item.arguments) if out_item.arguments else {}
            except Exception:
                inp = {}
            content_blocks.append({
                "type": "tool_use",
                "id": out_item.call_id,
                "name": out_item.name,
                "input": inp,
            })
        elif itype == "reasoning":
            content_blocks.append({
                "type": "reasoning_block",
                "id": getattr(out_item, "id", None),
                "encrypted_content": getattr(out_item, "encrypted_content", None),
                "summary": getattr(out_item, "summary", []) or [],
            })

    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "input_tokens", 0),
        "output_tokens": getattr(u, "output_tokens", 0),
    }
    stop = "tool_use" if any(b.get("type") == "tool_use" for b in content_blocks) else "end_turn"
    return {"content": content_blocks, "usage": usage, "stop_reason": stop}


def call_gemini(api_key: str, model: str, messages: list, system: str, tools: list) -> dict:
    """Gemini call via LiteLLM. LiteLLM uses Google's native genai API under the hood,
    which handles Gemini 3.x thought_signatures automatically (Google's OpenAI-compat
    endpoint requires thought_signature on every assistant tool_call but the openai
    Python SDK drops the field on the way in, causing 400 errors on turn 2+).

    litellm is on a non-standard PYTHONPATH set via EXTRA_PYTHONPATH (e.g. shared conda
    quota is full). We prepend that path here, AFTER transformers/huggingface_hub
    have already been imported at harness startup, so the scratch huggingface_hub
    (1.8.x) does not displace the conda huggingface_hub (0.35.x) that transformers needs.
    """
    import sys as _sys
    _scratch = os.environ.get("EXTRA_PYTHONPATH", "").strip()
    if _scratch not in _sys.path:
        _sys.path.insert(0, _scratch)
    import litellm  # noqa: F401
    litellm_model = model if model.startswith("gemini/") else f"gemini/{model}"
    return _litellm_call(litellm_model, api_key, messages, system, tools)


def _litellm_call(litellm_model, api_key, messages, system, tools):
    """LiteLLM dispatch. Builds OpenAI-shaped messages and calls litellm.completion(),
    which auto-handles Gemini 3.x thought_signatures."""
    import json as _json
    import litellm
    oai_tools = [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        },
    } for t in tools]

    oai_msgs = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        cont = msg["content"]
        if isinstance(cont, str):
            oai_msgs.append({"role": role, "content": cont})
            continue
        if role == "user":
            content_parts = []
            for blk in cont:
                if isinstance(blk, dict):
                    if blk.get("type") == "text":
                        content_parts.append({"type": "text", "text": blk["text"]})
                    elif blk.get("type") == "image":
                        src_ = blk.get("source", {})
                        b64 = src_.get("data", "")
                        mt = src_.get("media_type", "image/png")
                        content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}})
                    elif blk.get("type") == "tool_result":
                        tu_id = blk.get("tool_use_id", "")
                        inner = blk.get("content", [])
                        text = ""
                        if isinstance(inner, list):
                            for ib in inner:
                                if isinstance(ib, dict) and ib.get("type") == "text":
                                    text += ib.get("text", "")
                        else:
                            text = str(inner)
                        oai_msgs.append({"role": "tool", "tool_call_id": tu_id, "content": text})
            if content_parts:
                oai_msgs.append({"role": "user", "content": content_parts})
        elif role == "assistant":
            text_parts = []
            tool_calls = []
            for blk in cont:
                if isinstance(blk, dict):
                    if blk.get("type") == "text":
                        text_parts.append(blk.get("text", ""))
                    elif blk.get("type") == "tool_use":
                        tc = {
                            "id": blk.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": blk.get("name", ""),
                                "arguments": _json.dumps(blk.get("input", {})),
                            },
                        }
                        sig = blk.get("_thought_signature")
                        if sig is not None:
                            tc["thought_signature"] = sig
                        tool_calls.append(tc)
            assistant_msg = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "".join(text_parts)
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if "content" not in assistant_msg and not tool_calls:
                assistant_msg["content"] = ""
            oai_msgs.append(assistant_msg)

  
    # uses thousands of thought-tokens per call which causes over-deliberation
    # in structured agent loops (hallucinated bbox strategies, VerifyCandidate
    # spam, or Done spam). 512 tokens lets the LLM briefly identify the
    # target and pick the tool, but not invent multi-step strategies.
  
    # parameter (Anthropic-style; LiteLLM translates to Gemini's thinking_budget).
    _thinking_kw = {}
    if "pro" in litellm_model.lower() and "customtools" not in litellm_model.lower():
        _thinking_kw = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        print(f"[harness] {litellm_model}: thinking budget=2048 (brief-reasoning)", flush=True)
    # Force a tool call every turn for Gemini Pro. Pro emits text-
    # only responses ("thinking aloud") that waste LLM-step budget. Flash
    # benefits from the default tool_choice=auto behavior (its brief between-
    # action reasoning is useful). Applying tool_choice=required to Flash
    # caused a large regression vs the baseline.
    _tc_kw = {}
    if oai_tools and ("pro" in litellm_model.lower() and "customtools" not in litellm_model.lower()):
        _tc_kw = {"tool_choice": "required"}
    resp = litellm.completion(
        model=litellm_model,
        messages=oai_msgs,
        tools=oai_tools if oai_tools else None,
        max_tokens=4096,
        api_key=api_key,
        **_thinking_kw,
        **_tc_kw,
    )
    msg = resp.choices[0].message
    content_blocks = []
    if msg.content:
        content_blocks.append({"type": "text", "text": msg.content})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args_parsed = _json.loads(tc.function.arguments)
            except Exception:
                args_parsed = {}
            block = {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": args_parsed,
            }
            sig = getattr(tc, "thought_signature", None)
            if sig is None:
                try:
                    sig = tc.model_dump().get("thought_signature")
                except Exception:
                    pass
            if sig is not None:
                block["_thought_signature"] = sig
            content_blocks.append(block)
    stop_reason = "tool_use" if (msg.tool_calls and len(msg.tool_calls) > 0) else "end_turn"
    return {
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        },
    }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _img_block(png_bytes: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(png_bytes)}}


def _v19i_normalize_bbox_input(input_: dict) -> dict:
    """detect Gemini-style 0-1000 normalized bbox coords and scale to
    LIBERO_ENV_RESOLUTION pixel space. Gemini Flash 3.5 (and other Google
    models) natively output bboxes in 0-1000 normalized space even when shown
    a 512x512 image. If we see any coord > LIBERO_ENV_RESOLUTION we treat the
    whole bbox as 0-1000-normalized and rescale.

    Mutates a copy of input_ — original is unchanged.
    """
    if not any(k in input_ for k in ("x1", "y1", "x2", "y2")):
        return input_
    try:
        x1 = int(input_.get("x1", 0)); y1 = int(input_.get("y1", 0))
        x2 = int(input_.get("x2", 0)); y2 = int(input_.get("y2", 0))
    except Exception:
        return input_
    if max(x1, y1, x2, y2) > LIBERO_ENV_RESOLUTION:
        scale = LIBERO_ENV_RESOLUTION / 1000.0
        out = dict(input_)
        out["x1"] = int(round(x1 * scale))
        out["y1"] = int(round(y1 * scale))
        out["x2"] = int(round(x2 * scale))
        out["y2"] = int(round(y2 * scale))
        print(f"[bbox rescale] [{x1},{y1},{x2},{y2}] (0-1000) -> "
              f"[{out['x1']},{out['y1']},{out['x2']},{out['y2']}] (0-{LIBERO_ENV_RESOLUTION})", flush=True)
        return out
    return input_


def exec_tool(name: str, input_: dict, runner: LiberoRunner, max_steps_per_rollout: int) -> dict:
    """Execute a single tool call. Returns dict with 'text', optional 'agent_png', 'wrist_png', 'done'."""
  
    input_ = _v19i_normalize_bbox_input(input_)
  
    # action that actually changes the scene. Perceive/VerifyCandidate alone are
    # NOT corrective — they're just observation. Done after observation-only is
    # still Done-spam.
    _CORRECTIVE_ACTIONS = ("Grasp", "Place", "LocateAndGrasp", "LocateAndPlace",
                           "MoveTo", "GoHome", "VLARollout", "Release",
                           "GraspGeometric", "PlanReach", "SmartGrasp")
    if name in _CORRECTIVE_ACTIONS:
        runner._wasted_done_count = 0
    if name == "Perceive":
        # state_text() runs the gemini_er detection and populates
        # runner._gemini_er_bboxes_ordered. Tell perceive_pngs to reuse those
        # bboxes (skip a second detection call) so we make one Gemini call.
        text = runner.state_text()
        runner._perceive_pngs_skip_detect = True
        try:
            agent_png, wrist_png = runner.perceive_pngs()
        finally:
            runner._perceive_pngs_skip_detect = False
        return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": False}

    if name == "VLARollout":
        subgoal = str(input_.get("subgoal", ""))
        if not subgoal:
            return {"text": "VLARollout failed: missing subgoal", "done": False}
        t0 = time.time()
        result = runner.vla_rollout(subgoal, max_steps_per_rollout)
        elapsed = time.time() - t0
        agent_png, wrist_png = runner.perceive_pngs()
        if "error" in result:
            text = f"VLARollout '{subgoal}' errored after {result['steps']} steps: {result['error']}"
        else:
            tag = " TASK SUCCESS!" if result["success"] else ""
            text = (f"VLARollout '{subgoal}' ran for {result['steps']} steps in {elapsed:.1f}s.{tag} "
                    f"{runner.state_text()}")
      
        # responsible for verifying VLA outcomes by calling Perceive (Gemini ER)
        # and VerifyCandidate over relevant bboxes. This keeps perception in
        # one place (Gemini ER).
        return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": result.get("success", False)}

    if name == "LocateBox":
        desc = str(input_.get("description", ""))
        x1 = int(input_.get("x1", 0)); y1 = int(input_.get("y1", 0))
        x2 = int(input_.get("x2", 0)); y2 = int(input_.get("y2", 0))
        if x1 >= x2 or y1 >= y2:
            return {"text": f"LocateBox '{desc}' failed: invalid bbox [{x1},{y1},{x2},{y2}] (need x1<x2, y1<y2)."}
        try:
            r = runner.locate_box((x1, y1, x2, y2))
            w = r["world_xyz"]
            cur_ee = runner.obs["robot0_eef_pos"]
            return {
                "text": (f"LocateBox '{desc}' bbox=[{x1},{y1},{x2},{y2}] -> "
                         f"world=({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}). "
                         f"Current EE=({cur_ee[0]:.3f}, {cur_ee[1]:.3f}, {cur_ee[2]:.3f}). "
                         f"To hover 10cm above target: MoveTo "
                         f"dx={(w[0]-cur_ee[0]):.3f}, dy={(w[1]-cur_ee[1]):.3f}, dz={(w[2]-cur_ee[2]+0.10):.3f}"),
                "done": False,
            }
        except Exception as e:
            return {"text": f"LocateBox '{desc}' errored: {e}", "done": False}

    if name == "MoveTo":
        dx = float(input_.get("dx", 0.0))
        dy = float(input_.get("dy", 0.0))
        dz = float(input_.get("dz", 0.0))
        try:
            r = runner.move_to(dx, dy, dz)
            agent_png, wrist_png = runner.perceive_pngs()
            tag = " TASK SUCCESS!" if r.get("done") else ""
            note = "" if r.get("reached", False) else f" (didn't fully reach within tol; final_err={r.get('final_err', '?'):.3f})" if isinstance(r.get('final_err'), (int, float)) else ""
            return {
                "text": f"MoveTo (dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}) ran for {r['steps']} steps.{note}{tag} {runner.state_text()}",
                "agent_png": agent_png, "wrist_png": wrist_png,
                "done": runner.success,
            }
        except Exception as e:
            return {"text": f"MoveTo errored: {e}", "done": False}

    # Legacy "Grasp" (close-gripper-at-current-pose) dispatcher removed in.
    # The new merged Grasp = locate-and-grasp w/ PCA fallback is later in this function.

    if name == "Release":
      
        # Guard against Release with an already-open gripper.
        try:
            _gq = runner.obs.get("robot0_gripper_qpos", [0.04, -0.04])
            if float(_gq[0]) > 0.03:
                return {"text": (
                    "REJECTED: Release is a no-op (gripper already open, q="
                    + f"{float(_gq[0]):.3f}" + "). Nothing held to release. Use a different tool."
                ), "done": False, "llm_done": False}
        except Exception:
            pass
        try:
            r = runner.do_release()
            agent_png, wrist_png = runner.perceive_pngs()
            tag = " TASK SUCCESS!" if r.get("done") else ""
            return {"text": f"Release ran for {r['steps']} steps.{tag} {runner.state_text()}",
                    "agent_png": agent_png, "wrist_png": wrist_png, "done": runner.success}
        except Exception as e:
            return {"text": f"Release errored: {e}", "done": False}

    if name == "Done":
        # Server-side check: reject premature Done if simulator hasn't registered success.
        if not runner.success:
          
            # pattern (calling Done 15-53× per trial, wasting LLM budget). After
            # 2 consecutive rejections, force the agent to take a corrective
            # action before allowing another Done.
            prior = int(getattr(runner, "_wasted_done_count", 0)) + 1
            runner._wasted_done_count = prior
            if prior >= 2:
                runner._done_blocked_until_action = True
                return {"text": ("HARD REJECT (Done blocked): you called Done {N} times this trial "
                                 "without the task actually completing. The harness now requires you to "
                                 "(1) call Perceive to see CURRENT state, (2) call a CORRECTIVE action "
                                 "(Grasp, Place, MoveTo, GoHome, VLARollout — anything that changes the "
                                 "scene), and only THEN may you try Done again. Stop guessing — the "
                                 "predicate is checked by physics, not by your reasoning. If the bowl "
                                 "looks placed but the predicate says false, the placement is OFF by "
                                 "a few cm — re-grasp and try Place again.").replace("{N}", str(prior)),
                        "done": False, "llm_done": False}
            return {"text": ("REJECTED: you called Done but the simulator's task_done flag is False. "
                             "Task is NOT actually complete. Re-Perceive to see current state, "
                             "diagnose what's wrong (wrong object grasped? not yet in basket? gripper still closed?), "
                             "and continue with corrective actions. Do NOT call Done until task done=True."),
                    "done": False, "llm_done": False}
        return {"text": "Task marked done by harness.", "done": True, "llm_done": True}

    if name == "GoHome":
        try:
            r = runner.go_home()
        except Exception as e:
            return {"text": f"GoHome failed: {e}", "done": False}
        agent_png, wrist_png = runner.perceive_pngs()
        return {
            "text": (f"GoHome: arm returned to home pose (gripper open). "
                     f"Object positions are unchanged from before this call. {runner.state_text()}"),
            "agent_png": agent_png,
            "wrist_png": wrist_png,
            "done": False,
        }

    # SmartGrasp dispatcher removed in (merged into Grasp as the PCA fallback).

    if name == "Repose":
        tgt = str(input_.get("target", "")).strip()
        if not tgt:
            return {"text": "Repose failed: missing 'target' parameter", "done": False}
        try:
            r = runner.repose(tgt)
        except Exception as e:
            return {"text": f"Repose errored: {e}", "done": False}
        agent_png, wrist_png = runner.perceive_pngs()
        if r.get("ok"):
            mt = r.get("moved_to", (0,0,0))
            text = (f"Repose('{tgt}'): SUCCESS — moved {tgt} back to its initial position "
                    f"({mt[0]:.3f}, {mt[1]:.3f}, {mt[2]:.3f}). {r.get('steps', 0)} env steps. "
                    f"{runner.state_text()}")
        else:
            text = f"Repose('{tgt}') failed: {r.get('note', 'unknown')}. {runner.state_text()}"
        return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": r.get("ok") and runner.done}

    if name == "RushRetry":
        refined = str(input_.get("refined_subgoal", "")).strip()
        if not refined:
            return {"text": "RushRetry failed: missing 'refined_subgoal' parameter", "done": False}
        t0 = time.time()
        try:
            r = runner.rush_retry(refined, max_repose=3,
                                  max_env_steps_per_call=max_steps_per_rollout)
        except Exception as e:
            return {"text": f"RushRetry errored: {e}", "done": False}
        elapsed = time.time() - t0
        agent_png, wrist_png = runner.perceive_pngs()
        reposed_names = [o.get("name", "?") for o in r.get("reposed", [])]
        skipped = r.get("skipped", [])
        vla_res = r.get("vla") or {}
        vla_success = bool(vla_res.get("success"))
        vla_steps = int(vla_res.get("steps", 0))
        tag = " TASK SUCCESS!" if (runner.done or vla_success) else ""
        text = (f"RushRetry: reposed {len(reposed_names)} displaced object(s): "
                f"{reposed_names if reposed_names else '(none)'}. "
                f"VLA('{refined}') ran for {vla_steps} env steps.{tag} "
                f"Total RushRetry took {elapsed:.1f}s, {r.get('steps', 0)} env steps. "
                f"{runner.state_text()}")
        if skipped:
            text += "\nSkipped: " + "; ".join(f"{s.get('name','?')}({s.get('reason','?')})" for s in skipped[:3])
        return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png,
                "done": runner.done or vla_success}

    if name == "RightObject":
        target = str(input_.get("target", "")).strip()
        if not target:
            return {"text": "RightObject failed: missing 'target' parameter", "done": False}
        try:
            t0 = time.time()
            r = runner.right_object(target, max_steps_per_rollout)
            elapsed = time.time() - t0
        except Exception as e:
            return {"text": f"RightObject failed: {e}", "done": False}
        agent_png, wrist_png = runner.perceive_pngs()
        tag = " TASK SUCCESS!" if r.get("success") else ""
        text = (f"RightObject('{target}'): VLARollout('stand the {target} upright on the table') "
                f"ran for {r.get('steps', 0)} steps in {elapsed:.1f}s.{tag} {runner.state_text()}")
        return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": r.get("success", False)}

    if name == "PutAside":
        try:
            r = runner.put_aside()
        except Exception as e:
            return {"text": f"PutAside failed: {e}", "done": False}
        agent_png, wrist_png = runner.perceive_pngs()
        return {
            "text": (f"PutAside: dropped held item at stash corner, returned EE there with gripper "
                     f"open. ran for {r.get('steps', 0)} env steps. {runner.state_text()}"),
            "agent_png": agent_png,
            "wrist_png": wrist_png,
            "done": False,
        }

    if name == "ChangedObjects":
        try:
            r = runner.changed_objects()
        except Exception as e:
            return {"text": f"ChangedObjects failed: {e}", "done": False}
        if not r.get("changed"):
            text = "ChangedObjects: no objects moved >3cm vs initial scene."
        else:
            lines = [f"ChangedObjects: {r['n_changed']} of {r['n_baseline']} tracked objects moved"]
            for o in r["changed"]:
                lines.append(f"  {o['name']}: initial={o['initial_xyz']} -> current={o['current_xyz']} "
                             f"(moved {o['distance_m']}m)")
            text = "\n".join(lines)
        return {"text": text, "done": False}

    # ===== Heterogeneous-toolbox extensions =====
    if name == "DetectObjects":
        query = str(input_.get("query", "")).strip()
        max_results = int(input_.get("max_results", 5))
        if not query:
            return {"text": "DetectObjects failed: missing query", "done": False}
        try:
            r = runner.detect_objects(query, max_results=max_results)
            if not r["detections"]:
                return {"text": f"DetectObjects '{query}': no detections above threshold. Try a simpler descriptor.", "done": False}
            lines = [f"DetectObjects '{query}' found {len(r['detections'])} detection(s):"]
            for i, det in enumerate(r["detections"]):
                b = det["bbox"]; w = det["world_xyz"]
                lines.append(f"  [{i}] bbox=[{b[0]},{b[1]},{b[2]},{b[3]}] score={det['score']:.2f} -> world=({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")
            cur_ee = runner.obs["robot0_eef_pos"]
            lines.append(f"Current EE=({cur_ee[0]:.3f}, {cur_ee[1]:.3f}, {cur_ee[2]:.3f}). Use any world (x,y,z) above as target for GraspGeometric / PlanReach.")
            return {"text": "\n".join(lines), "done": False}
        except Exception as e:
            return {"text": f"DetectObjects '{query}' errored: {e}", "done": False}


    if name == "VerifyCandidate":
        try:
            x1 = int(input_.get("x1", 0)); y1 = int(input_.get("y1", 0))
            x2 = int(input_.get("x2", 0)); y2 = int(input_.get("y2", 0))
            target = str(input_.get("target", "")).strip()
            if not target or x1 >= x2 or y1 >= y2:
                return {"text": f"VerifyCandidate failed: bad inputs (bbox=[{x1},{y1},{x2},{y2}], target='{target}')", "done": False}
            verdict = runner.verify_candidate(x1, y1, x2, y2, target)
            return {"text": f"VerifyCandidate bbox=[{x1},{y1},{x2},{y2}] target='{target}' -> {verdict}", "done": False}
        except Exception as e:
            return {"text": f"VerifyCandidate errored: {e}", "done": False}

    if name == "Place" or name == "LocateAndPlace":  # legacy name accepted
        try:
            desc = str(input_.get("description", "")).strip() or "destination"
          
            target_index = input_.get("target_index", None)
            if target_index is not None:
                try:
                    idx = int(target_index)
                    bbox_list = getattr(runner, "_gemini_er_bboxes_ordered", [])
                    if 0 <= idx < len(bbox_list):
                        x1, y1, x2, y2 = bbox_list[idx]
                        print(f"[Place] target_index={idx} -> bbox=[{x1},{y1},{x2},{y2}]", flush=True)
                    else:
                        return {"text": f"Place failed: target_index={idx} out of range (have {len(bbox_list)} detections); re-Perceive and pick a valid #N", "done": False}
                except Exception as _e:
                    return {"text": f"Place failed parsing target_index: {_e}", "done": False}
            else:
                x1 = int(input_.get("x1", 0)); y1 = int(input_.get("y1", 0))
                x2 = int(input_.get("x2", 0)); y2 = int(input_.get("y2", 0))
                if x1 >= x2 or y1 >= y2:
                    return {"text": f"Place failed: need target_index OR a valid bbox; got bbox=[{x1},{y1},{x2},{y2}]", "done": False}
              
                # Pre-unproject the bbox to world; reject if outside the table footprint.
                # Catches cases where the LLM invents off-table coords (e.g. Pro's
                # "put salad dressing aside" hallucination at world (-0.53, -0.18)).
                try:
                    _wsloc = runner.locate_box((x1, y1, x2, y2))
                    _wx, _wy, _wz = _wsloc["world_xyz"]
                    if abs(_wx) > 0.4 or abs(_wy) > 0.4 or _wz < 0.5 or _wz > 1.5:
                        return {"text": (f"Place rejected: bbox [{x1},{y1},{x2},{y2}] unprojects to off-table "
                                          f"world=({_wx:.3f},{_wy:.3f},{_wz:.3f}). Use target_index from a "
                                          f"detected #N in Perceive output, or call LocateBox first to register "
                                          f"a destination that's actually on the workspace."),
                                "done": False}
                except Exception as _e_ws:
                    print(f"[workspace check] failed: {_e_ws}", flush=True)
          
            surface_type = str(input_.get("surface_type", "flat")).lower().strip()
            if surface_type not in ("flat", "container"):
                surface_type = "flat"
            runner._place_use_v19m = (surface_type == "flat")
            print(f"[Place] desc='{desc}' surface_type='{surface_type}' -> use_v19m_xy_ft={runner._place_use_v19m}", flush=True)
            t0 = time.time()
            r = runner.locate_and_place((x1, y1, x2, y2))
            elapsed = time.time() - t0
            agent_png, wrist_png = runner.perceive_pngs()
            wx, wy, wz = r.get("world_xyz", [0, 0, 0])
            tag = " TASK SUCCESS!" if runner.done else " (released)"
            descended = r.get("descended_m", "?")
            text = (f"LocateAndPlace '{desc}' bbox=[{x1},{y1},{x2},{y2}] -> "
                    f"world=({wx:.3f}, {wy:.3f}, {wz:.3f}). Contact-aware descent "
                    f"of {descended}m, then released gripper.{tag} {r.get('steps', 0)} env steps, "
                    f"{elapsed:.1f}s. {runner.state_text()}")
            return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": runner.done}
        except Exception as e:
            return {"text": f"LocateAndPlace errored: {e}", "done": False}

  
    if name == "Grasp" or name == "LocateAndGrasp":  # legacy name accepted for safety
        try:
            desc = str(input_.get("description", "")).strip() or "target"
          
            target_index = input_.get("target_index", None)
            if target_index is not None:
                try:
                    idx = int(target_index)
                    bbox_list = getattr(runner, "_gemini_er_bboxes_ordered", [])
                    if 0 <= idx < len(bbox_list):
                        x1, y1, x2, y2 = bbox_list[idx]
                        print(f"[Grasp] target_index={idx} -> bbox=[{x1},{y1},{x2},{y2}]", flush=True)
                    else:
                        return {"text": f"Grasp failed: target_index={idx} out of range (have {len(bbox_list)} detections); re-Perceive and pick a valid #N", "done": False}
                except Exception as _e:
                    return {"text": f"Grasp failed parsing target_index: {_e}", "done": False}
            else:
                x1 = int(input_.get("x1", 0)); y1 = int(input_.get("y1", 0))
                x2 = int(input_.get("x2", 0)); y2 = int(input_.get("y2", 0))
                if x1 >= x2 or y1 >= y2:
                    return {"text": f"Grasp failed: need target_index OR a valid bbox; got bbox=[{x1},{y1},{x2},{y2}]", "done": False}
              
                try:
                    _wsloc = runner.locate_box((x1, y1, x2, y2))
                    _wx, _wy, _wz = _wsloc["world_xyz"]
                    if abs(_wx) > 0.4 or abs(_wy) > 0.4 or _wz < 0.5 or _wz > 1.5:
                        return {"text": (f"Grasp rejected: bbox [{x1},{y1},{x2},{y2}] unprojects to off-table "
                                          f"world=({_wx:.3f},{_wy:.3f},{_wz:.3f}). Use target_index from a "
                                          f"detected #N in Perceive output, or call LocateBox first."),
                                "done": False}
                except Exception as _e_ws:
                    print(f"[workspace check] failed: {_e_ws}", flush=True)
            t0 = time.time()
            r = runner.locate_and_grasp((x1, y1, x2, y2))
            elapsed = time.time() - t0
            wx, wy, wz = r.get("world_xyz", [0, 0, 0])
            held = r.get("held", False)
            descended = r.get("descended_m", "?")
            steps_top = r.get("steps", 0)
            # PCA-aligned fallback if top-down didn't grab anything.
          
            # flow: agent did GoHome between LocateAndGrasp and SmartGrasp). Without
            # this, the gripper occludes the wrist camera and SAM returns empty mask.
            fallback_text = ""
            if not held and not runner.done:
                try:
                    runner.go_home()
                except Exception:
                    pass
                try:
                    r2 = runner.smart_grasp((x1, y1, x2, y2))
                    steps_pca = r2.get("steps", 0)
                    if r2.get("note"):
                        fallback_text = (f" Top-down NOTHING HELD; PCA-fallback failed: "
                                         f"{r2['note']} ({steps_pca} env steps).")
                    else:
                        held = r2.get("held", False)
                        yaw_deg = r2.get("yaw_rad", 0.0) * 180.0 / 3.141592653589793
                        fallback_text = (f" Top-down NOTHING HELD; PCA-fallback ran "
                                         f"{steps_pca} steps yaw={yaw_deg:.1f}deg "
                                         f"-> {'HELD' if held else 'NOTHING HELD'}.")
                except Exception as _e_pca:
                    fallback_text = f" PCA-fallback errored: {_e_pca}."
            elapsed_total = time.time() - t0
            agent_png, wrist_png = runner.perceive_pngs()
            tag = " TASK SUCCESS!" if runner.done else (" HELD" if held else " NOTHING HELD")
            text = (f"Grasp '{desc}' bbox=[{x1},{y1},{x2},{y2}] -> "
                    f"world=({wx:.3f}, {wy:.3f}, {wz:.3f}). Top-down descended {descended}m "
                    f"and closed gripper.{tag}{fallback_text} {elapsed_total:.1f}s total. "
                    f"{runner.state_text()}")
            return {"text": text, "agent_png": agent_png, "wrist_png": wrist_png, "done": runner.done}
        except Exception as e:
            return {"text": f"Grasp errored: {e}", "done": False}

    # VLAFocused dispatcher removed in (tool was eliminated from the agent's toolset).

    if name == "GraspGeometric":
        try:
            tx = float(input_.get("target_x", 0.0))
            ty = float(input_.get("target_y", 0.0))
            tz = float(input_.get("target_z", 0.0))
            r = runner.grasp_geometric(tx, ty, tz)
            agent_png, wrist_png = runner.perceive_pngs()
            tag = " TASK SUCCESS!" if r.get("done") else ""
            held = "HELD" if r.get("held", False) else "NOTHING HELD (closed on air)"
            return {
                "text": f"GraspGeometric -> ({tx:.3f}, {ty:.3f}, {tz:.3f}) done in {r['steps']} steps. {held}.{tag} {runner.state_text()}",
                "agent_png": agent_png, "wrist_png": wrist_png,
                "done": runner.success,
            }
        except Exception as e:
            return {"text": f"GraspGeometric errored: {e}", "done": False}

    if name == "PlanReach":
        try:
            tx = float(input_.get("target_x", 0.0))
            ty = float(input_.get("target_y", 0.0))
            tz = float(input_.get("target_z", 0.0))
            collision_aware = bool(input_.get("collision_aware", False))
            r = runner.plan_reach(tx, ty, tz, collision_aware=collision_aware)
            agent_png, wrist_png = runner.perceive_pngs()
            tag = " TASK SUCCESS!" if r.get("done") else ""
            mode = "collision-aware" if collision_aware else "open-loop"
            note = "" if r.get("reached", True) else f" (didn't fully reach; final_err={r.get('final_err', '?'):.3f}m)" if isinstance(r.get('final_err'), (int, float)) else ""
            return {
                "text": f"PlanReach ({mode}) -> ({tx:.3f}, {ty:.3f}, {tz:.3f}) done in {r['steps']} steps.{note}{tag} {runner.state_text()}",
                "agent_png": agent_png, "wrist_png": wrist_png,
                "done": runner.success,
            }
        except Exception as e:
            return {"text": f"PlanReach errored: {e}", "done": False}

    if name == "PlanFullTask":
        task = str(input_.get("task", "")).strip()
        if not task:
            return {"text": "PlanFullTask failed: missing task description", "done": False}
        try:
            r = runner.plan_full_task(task)
            agent_png, wrist_png = runner.perceive_pngs()
            if not r.get("available", False):
                return {"text": f"PlanFullTask UNAVAILABLE: {r.get('reason', 'planner not configured')}. Fall back to step-by-step orchestration.",
                        "agent_png": agent_png, "wrist_png": wrist_png, "done": False}
            tag = " TASK SUCCESS!" if r.get("done") else ""
            return {
                "text": f"PlanFullTask '{task}' returned plan with {r.get('plan_steps', '?')} steps; executed in {r.get('exec_steps', 0)} env steps.{tag} {runner.state_text()}",
                "agent_png": agent_png, "wrist_png": wrist_png,
                "done": runner.success,
            }
        except Exception as e:
            return {"text": f"PlanFullTask errored: {e}", "done": False}

    return {"text": f"Unknown tool: {name}", "done": False}


# ============================================================================
# Logging
# ============================================================================
class TrialLog:
    def __init__(self, out_dir: str, mode: str, model: str, suite: str, task_id: int,
                 episode: int, task_description: str):
        ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
        slug = f"{suite}_t{task_id:02d}_e{episode:02d}"
        model_dir = (
            "raw" if mode == "raw"
            else ("sonnet" if "sonnet" in model.lower()
                  else ("haiku" if "haiku" in model.lower()
                        else ("opus" if "opus" in model.lower()
                              else ("gpt5" if model.lower().startswith(("gpt-5", "gpt5"))
                                    else ("gpt" if "gpt" in model.lower() else "other")))))
        )
        self.dir = pathlib.Path(out_dir) / model_dir / slug / ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "mode": mode,
            "model": model,
            "suite": suite,
            "task_id": task_id,
            "episode": episode,
            "task_description": task_description,
            "started": ts,
            "transcript": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        self._img_n = 0

    def save_image(self, png_bytes: bytes, label: str) -> str:
        self._img_n += 1
        fname = f"img_{self._img_n:03d}_{label}.png"
        (self.dir / fname).write_bytes(png_bytes)
        return fname

    def append(self, role: str, content: str, **extra):
        self.data["transcript"].append({"role": role, "content": content[:4000], "ts": time.time(), **extra})

    def finish(self, success: bool, env_steps: int, llm_steps: int, duration_s: float):
        self.data["success"] = success
        self.data["env_steps"] = env_steps
        self.data["llm_steps"] = llm_steps
        self.data["duration_s"] = round(duration_s, 1)
        (self.dir / "trial.json").write_text(json.dumps(self.data, indent=2, default=str))
        return str(self.dir / "trial.json")


# ============================================================================
# Modes
# ============================================================================
def run_raw(args: Args, runner: LiberoRunner, log: TrialLog) -> bool:
    """Pi0.5 alone with the original task description, run until done or max_steps."""
    max_steps = SUITE_MAX_STEPS.get(args.suite, 400)
    print(f"[raw] task='{runner.task_description}' max_env_steps={max_steps}", flush=True)
    log.append("config", f"raw mode, max_env_steps={max_steps}, prompt='{runner.task_description}'")
    t0 = time.time()
    result = runner.vla_rollout(runner.task_description, max_steps)
    elapsed = time.time() - t0
    err_note = f" ERROR: {result['error']}" if "error" in result else ""
    print(f"[raw] done in {result['steps']} steps, success={result['success']}, elapsed={elapsed:.1f}s.{err_note}", flush=True)
    log.append("result", f"steps={result['steps']} success={result['success']}{err_note}")
    log.finish(result["success"], result["steps"], 0, elapsed)
    return result["success"]


def run_harness(args: Args, runner: LiberoRunner, log: TrialLog) -> bool:
    """LLM tool-loop orchestrating pi0.5."""
    model_lc = args.model.lower()
    is_together = "kimi" in model_lc or model_lc.startswith(("moonshotai/", "meta-llama/", "qwen/", "deepseek/"))
    is_gemini = (not is_together) and model_lc.startswith("gemini")
    is_openai = (not is_together) and (not is_gemini) and model_lc.startswith(("gpt", "o1", "o3", "o4", "openai"))
    if is_together:
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            print("TOGETHER_API_KEY not set", file=sys.stderr)
            sys.exit(1)
    elif is_gemini:
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("GEMINI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
    elif is_openai:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("OPENAI_API_KEY not set", file=sys.stderr)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
    if not api_key:
        sys.exit(1)

    # ========================================================================
    # Phase 0: silent raw-VLA attempt (no LLM, no tokens)
    # If pi0.5 can solve the task with the literal task_description, we save
    # all LLM cost. On failure, the env is NOT reset — the LLM agent inherits
    # the disturbed scene and must recover from it. This preserves any partial
    # progress pi0.5 made (e.g., grasp succeeded but place missed) and avoids
    # wasting the phase 0 compute budget.
    # Gated on CORTEX_NO_PHASE0 — set to "1" to skip phase 0 entirely.
    # ========================================================================
    phase0_attempted = False
    pre_vla_agent_png = None
    pre_vla_wrist_png = None
    # Linguistic gate: skip phase 0 when the task string contains markers
    # that predict (a) pi0.5 will fail AND (b) failure will catastrophically
    # disturb the layout the LLM agent needs to reason about. This trades a
    # few easy phase 0 wins for a canonical-state recovery on the hard tasks.
    def _vla_hostile(task_desc: str) -> tuple[bool, str]:
        """Generic linguistic gate. No LIBERO-specific object names. Skip phase 0 when:
        1. Negation (semantic inversion)
        2. Above-table destinations (pi0.5 can't reach)
        3. Push verbs (pi0.5 historically ~0.5%)
        4. Spatial preposition inside the PICK clause — i.e. target is identified
           by a spatial relation to another object. Pi0.5's failures scatter the
           spatial layout the LLM agent needs for recovery.
        """
        t = task_desc.lower().strip()
        # 1. Negation
        for m in ["not between", "not next to", "not in", "not on"]:
            if m in t: return True, f"negation: {m}"
        # 2. Above-table destinations — generic prepositional patterns,
        # no LIBERO-specific furniture nouns hardcoded.
        for m in ["on top of the", "top of the", "on the rack", "on the shelf"]:
            if m in t: return True, f"above-table: {m}"
        # 3. Push verbs
        if t.startswith("push "): return True, "push"
        # 4. Spatial preposition in the pick clause.
        # Split on " and " — first clause is usually the "pick X" part, second
        # is the "place at Y" part. We only look at the pick clause so that
        # destination prepositions ("place on the stove") don't trigger.
        clauses = t.split(" and ", 1)
        for clause in clauses:
            if clause.startswith("pick") or " pick " in clause:
                for prep in [" between ", " next to ", " behind ",
                             " in front of ", " on top of ", " on the "]:
                    if prep in clause:
                        return True, f"spatial-pick: {prep.strip()}"
                break
        return False, ""
    _vla_skip, _vla_skip_reason = _vla_hostile(runner.task_description)
    if os.environ.get("CORTEX_NO_PHASE0", "0") == "1":
        print("\n=== Phase 0: SKIPPED (CORTEX_NO_PHASE0=1) ===", flush=True)
        log.append("phase0_skipped", "CORTEX_NO_PHASE0=1")
    elif _vla_skip:
        print(f"\n=== Phase 0: SKIPPED (linguistic gate — {_vla_skip_reason}) ===", flush=True)
        log.append("phase0_skipped", f"linguistic_gate: {_vla_skip_reason}")
    else:
        # Snapshot the canonical scene BEFORE phase 0 disturbs it. If phase 0
        # fails, we hand both images to the agent so it can visually compare
        # initial vs disturbed state and identify the target by its original
        # spatial position.
        try:
            pre_vla_agent_png, pre_vla_wrist_png = runner.perceive_pngs()
        except Exception as _e_pre:
            print(f"[phase0] pre-snapshot failed: {_e_pre}", flush=True)
        phase0_attempted = True
        phase0_budget = int(os.environ.get("PHASE0_MAX_STEPS", str(args.max_steps_per_rollout)))
        print(f"\n=== Phase 0: silent raw-VLA attempt (budget={phase0_budget} env steps) ===", flush=True)
        log.append("phase0_start", f"task={runner.task_description}; budget={phase0_budget}")
        p0_t0 = time.time()
        p0_result = runner.vla_rollout(runner.task_description, phase0_budget)
        p0_elapsed = time.time() - p0_t0
        p0_steps = p0_result.get("steps", 0)
        p0_success = bool(runner.success)
        log.append("phase0_done", f"steps={p0_steps} success={p0_success} elapsed={p0_elapsed:.1f}s")
        if p0_success:
            print(f"[phase0] raw VLA SUCCEEDED in {p0_steps} env steps ({p0_elapsed:.1f}s). No LLM needed.", flush=True)
            log.finish(True, p0_steps, 0, p0_elapsed)
            return True
        # NO RESET — agent inherits the disturbed scene and recovers from it,
        # but it ALSO sees the pre-VLA snapshot so it can locate the target
        # by its original position.
        print(f"[phase0] raw VLA failed after {p0_steps} env steps ({p0_elapsed:.1f}s). Handing disturbed scene + pre-VLA reference to LLM agent.", flush=True)
        log.append("phase0_handed_to_agent", "scene NOT reset; agent has pre-VLA + current images")
        if pre_vla_agent_png:
            log.save_image(pre_vla_agent_png, "pre_vla_agent")
        if pre_vla_wrist_png:
            log.save_image(pre_vla_wrist_png, "pre_vla_wrist")

    # ========================================================================
    # Phase 1: LLM agent loop with full heterogeneous toolbox
    # ========================================================================
    system_prompt = get_system_prompt(runner.task_description, args.max_steps_per_rollout)

    # Initial observation (post-Phase-0 state — agent sees what VLA left behind)
    agent_png, wrist_png = runner.perceive_pngs()
    log.save_image(agent_png, "initial_agent")
    log.save_image(wrist_png, "initial_wrist")

    if phase0_attempted:
        recovery_note = ("\nNOTE: A silent pi0.5 rollout just attempted this task and did NOT succeed. "
                         "Two pairs of images are shown:\n"
                         " (1) PRE-VLA (canonical initial state) — agentview + wrist, taken before the VLA attempt.\n"
                         " (2) CURRENT (disturbed) — agentview + wrist as the scene looks NOW.\n"
                         "Use the PRE-VLA images to identify the target object by its original position "
                         "(e.g. 'the bowl that was on the cookie box in PRE-VLA'). Then locate it in the "
                         "CURRENT scene (it may have moved) and recover. GoHome if the arm is in an awkward pose. "
                         "Do NOT retry VLA first — it just failed.\n")
    else:
        recovery_note = ""
    initial_user = []
    if phase0_attempted and pre_vla_agent_png is not None:
        initial_user.append({"type": "text", "text": "PRE-VLA snapshot (canonical initial state) — agentview, then wrist:"})
        initial_user.append(_img_block(pre_vla_agent_png))
        initial_user.append(_img_block(pre_vla_wrist_png))
        initial_user.append({"type": "text", "text": "CURRENT scene (post-VLA, disturbed) — agentview, then wrist:"})
    initial_user.extend([
        _img_block(agent_png),
        _img_block(wrist_png),
        {"type": "text", "text": (
            f"Task: {runner.task_description}\n"
            f"{runner.state_text()}\n"
            f"{recovery_note}"
            f"Use tools to complete the task.")},
    ])
    messages = [{"role": "user", "content": initial_user}]
    log.append("system", system_prompt)
    log.append("user", f"task={runner.task_description}; initial state recorded")

    t0 = time.time()
    success = False
    llm_done = False
    step = 0
    total_iterations = 0
    wasted_done_steps = 0  # v30: rejected-Done steps not charged to budget
    while step < args.max_llm_steps and total_iterations < args.max_llm_steps * 2:
        total_iterations += 1
        print(f"\n--- LLM step {step + 1}/{args.max_llm_steps} ---", flush=True)
        try:
            if is_together:
                api_result = call_with_retry(call_gpt, api_key, args.model, messages, system_prompt, TOOLS,
                                      base_url="https://api.together.xyz/v1")
            elif is_gemini:
                api_result = call_with_retry(call_gemini, api_key, args.model, messages, system_prompt, TOOLS)
            elif is_openai:
                # Route gpt-5.4-mini and gpt-5.5-mini to Responses API (medium reasoning effort).
                # gpt-5.5+ chat/completions rejects reasoning_effort + tools. Responses API
                # is the path that sets reasoning.effort=medium for these models.
                if args.model.startswith(("gpt-5.4-mini", "gpt-5.5-mini")):
                    api_result = call_with_retry(call_gpt_responses, api_key, args.model, messages, system_prompt, TOOLS)
                else:
                    api_result = call_with_retry(call_gpt, api_key, args.model, messages, system_prompt, TOOLS)
            else:
                api_result = call_with_retry(call_claude, api_key, args.model, messages, system_prompt, TOOLS)
        except Exception as e:
            print(f"[error] LLM call failed: {e}", flush=True)
            log.append("error", f"LLM call failed: {e}")
            break
        log.data["usage"]["input_tokens"] += api_result["usage"]["input_tokens"]
        log.data["usage"]["output_tokens"] += api_result["usage"]["output_tokens"]

        text_block = ""
        tool_calls = []
        for b in api_result["content"]:
            if b["type"] == "text":
                text_block = b["text"]
            elif b["type"] == "tool_use":
                tool_calls.append(b)

        if text_block:
            print(f"[claude] {text_block[:300]}", flush=True)
        log.append("assistant", text_block, tool_calls=[t["name"] for t in tool_calls])

        if not tool_calls:
            messages.append({"role": "assistant", "content": api_result["content"]})
            messages.append({"role": "user", "content": "Please use a tool to proceed."})
            step += 1
            continue

        messages.append({"role": "assistant", "content": api_result["content"]})

        tool_result_blocks = []
        rejected_done_count = 0
        non_done_tool_called = False
        for tc in tool_calls:
            print(f"  [tool] {tc['name']} {json.dumps(tc.get('input', {}))[:200]}", flush=True)
            log.append("tool_call", json.dumps(tc.get("input", {})), tool=tc["name"])

            result = exec_tool(tc["name"], tc.get("input", {}), runner, args.max_steps_per_rollout)

            # harness fix: track rejected Done calls for budget refund
            if tc["name"] == "Done" and not result.get("done", False) and "REJECTED" in result.get("text", ""):
                rejected_done_count += 1
            else:
                non_done_tool_called = True

            img_files = []
            if "agent_png" in result:
                img_files.append(log.save_image(result["agent_png"], f"step{step + 1}_{tc['name']}_agent"))
            if "wrist_png" in result:
                img_files.append(log.save_image(result["wrist_png"], f"step{step + 1}_{tc['name']}_wrist"))
            print(f"    └─ {result['text'][:300]}{' ' + ','.join(img_files) if img_files else ''}", flush=True)
            log.append("tool_result", result["text"], tool=tc["name"], images=img_files)

            tr_content = []
            if "agent_png" in result:
                tr_content.append(_img_block(result["agent_png"]))
            if "wrist_png" in result:
                tr_content.append(_img_block(result["wrist_png"]))
            tr_content.append({"type": "text", "text": result["text"]})
            tool_result_blocks.append({"type": "tool_result", "tool_use_id": tc["id"], "content": tr_content})

            if result.get("done"):
                if result.get("llm_done"):
                    llm_done = True
                if runner.success:
                    success = True
                    break

        messages.append({"role": "user", "content": tool_result_blocks})

        if runner.success or llm_done:
            break

      
        # returns "executing action in terminated episode", every subsequent
        # tool call errors. Continuing the LLM loop is pure waste.
        if any("terminated episode" in (blk.get("content")[-1].get("text", "") if isinstance(blk.get("content"), list) else "")
               for blk in tool_result_blocks):
            print("[harness] episode terminated — exiting LLM loop early", flush=True)
            log.append("episode_terminated", "exited LLM loop early due to terminated-episode error")
            break

        # harness fix: if every tool call this iteration was a rejected Done,
        # the LLM made no real progress — refund the step.
        if rejected_done_count > 0 and not non_done_tool_called:
            wasted_done_steps += 1
        else:
            step += 1

    elapsed = time.time() - t0
    # report effective LLM steps (excluding rejected-Done waste).
    effective_steps = step if (runner.success or llm_done) else step
    print(f"\n[harness] done — success={runner.success} (llm_done={llm_done}), llm_steps={effective_steps}, wasted_done={wasted_done_steps}, total_iterations={total_iterations}, env_steps={runner.t}, elapsed={elapsed:.1f}s", flush=True)
    log.finish(runner.success, runner.t, effective_steps, elapsed)
    return runner.success


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    args = parse_args()
    print(f"=== cortex-capx ===", flush=True)
    print(f"mode={args.mode}, suite={args.suite}, task={args.task}, episode={args.episode}", flush=True)
    print(f"policy={args.policy_host}:{args.policy_port}, model={args.model}", flush=True)

    runner = LiberoRunner(args.suite, args.task, args.seed,
                          args.policy_host, args.policy_port, args.replan_steps)
    print(f"task='{runner.task_description}'", flush=True)
    runner.reset(episode_idx=args.episode)
    print(f"env reset; initial {runner.state_text()}", flush=True)

    log = TrialLog(args.out_dir, args.mode, args.model, args.suite, args.task,
                   args.episode, runner.task_description)
    print(f"results dir: {log.dir}", flush=True)

    try:
        if args.mode == "raw":
            ok = run_raw(args, runner, log)
        else:
            ok = run_harness(args, runner, log)
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        log.append("error", f"fatal: {e}\n{traceback.format_exc()}")
        log.finish(False, runner.t, 0, 0)
        return 1

    if not args.no_video and runner.replay_frames:
        try:
            video_path = log.dir / "rollout.mp4"
            runner.save_video(str(video_path), fps=10)
            print(f"video: {video_path}", flush=True)
        except Exception as e:
            print(f"video save failed: {e}", flush=True)

    print(f"\nresult: success={ok}, env_steps={runner.t}, dir={log.dir}", flush=True)
    return 0 if ok else 0  # don't fail the process on task failure


if __name__ == "__main__":
    sys.exit(main())