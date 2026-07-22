/**
 * Hybrid TAMP + VLA Agent — text-only orchestrator (no image context).
 *
 *   Claude (haiku, by default) -> tools -> tamp_server (TAMP via TiPToP) + droid_server (VLA via pi0.5)
 *
 * Tools: Pick(label), Place(label, target), MoveTo(label), VLARollout(subgoal), Release, Done
 *
 * Design notes:
 *   - Claude never sees raw images; perception happens inside Pick/Place via Gemini-ER. So we keep
 *     ZERO images in the LLM context — only text scene-memory + tool results.
 *   - tamp_server (HTTP :7777) subprocesses tiptop-run for each Pick/Place (reliable, ~45s/call).
 *   - droid_server (TCP :9876) bridges to pi0.5 WebSocket for VLARollout.
 *   - Trial dirs from each tiptop-run subprocess are copied into the run dir for offline review.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'fs'
import { join } from 'path'
import { Socket } from 'net'
import { createInterface } from 'readline'
import { spawn } from 'child_process'

// ---- Config ----
const API_KEY = process.env.ANTHROPIC_API_KEY ?? ''
const MODEL = process.env.CLAUDE_MODEL ?? 'claude-opus-4-7'
const MAX_STEPS = parseInt(process.env.AGENT_MAX_STEPS ?? '15', 10)
const CONSECUTIVE_FAIL_LIMIT = parseInt(process.env.AGENT_FAIL_LIMIT ?? '5', 10)
const LLM_TIMEOUT_MS = parseInt(process.env.AGENT_LLM_TIMEOUT_MS ?? '60000', 10)

const ARGS = parseArgs(process.argv.slice(2))
const TAMP_URL  = ARGS.tamp ?? process.env.TAMP_URL  ?? 'http://ROBOT_HOST:7777'
const DROID_HOST = ARGS['droid-host'] ?? process.env.DROID_HOST ?? 'ROBOT_HOST'
const DROID_PORT = parseInt(ARGS['droid-port'] ?? process.env.DROID_PORT ?? '9876', 10)
const PI0_URL   = ARGS.pi0  ?? process.env.PI0_URL  ?? 'ws://ROBOT_HOST:8000'
const TASK = ARGS._[0] ?? 'pick up the cup and put it in the basket'
const SYSTEM_PROMPT_PATH = process.env.AGENT_SYSTEM ?? join(import.meta.dir, 'agent-system.md')

function parseArgs(argv: string[]) {
  const out: any = { _: [] as string[] }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) { out[a.slice(2)] = argv[++i] } else { out._.push(a) }
  }
  return out as Record<string, string> & { _: string[] }
}

// ---- Output dir + ANSI ----
const RUN_DIR = join('results', 'agent', slug(TASK), new Date().toISOString().replace(/[:.]/g, '-'))
mkdirSync(RUN_DIR, { recursive: true })
// reasoning.txt accumulates one block per turn for easy grep-ability — the same text is
// also embedded in messages.json (the canonical record), but this file is plain-text.
const REASONING_PATH = join(RUN_DIR, 'reasoning.txt')
const B='\x1b[1m', R='\x1b[0m', G='\x1b[32m', Y='\x1b[33m', C='\x1b[36m', RE='\x1b[31m', D='\x1b[2m'
function slug(s: string) { return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60).replace(/^-|-$/g, '') }

// ---- Scene memory (text-only) ----
type SceneMemory = {
  currently_holding: string | null
  objects_seen: { label: string; last_seen_turn: number }[]
  actions_taken: { turn: number; tool: string; args: any; result: string }[]
  in_container: Record<string, string>   // label → container it was placed into (cleared on next successful Pick)
  latest_known_objects: string[]         // last seen known_objects list from any tool result
  latest_objects_detail: Record<string, { centroid: number[]; extents: number[] }>  // label → 3D centroid + extents (meters, robot frame)
  consecutive_pick_fails: number         // count of failed Picks since last successful non-Perceive action; resets on success or non-Pick action
}
const memory: SceneMemory = { currently_holding: null, objects_seen: [], actions_taken: [], in_container: {}, latest_known_objects: [], latest_objects_detail: {}, consecutive_pick_fails: 0 }

// ---- Structured trial result (written at end of run for paper analysis) ----
type ToolCall = {
  turn: number
  tool: string
  args: any
  success: boolean | null  // null for Perceive/Done (no success flag), bool otherwise
  fail_reason: string | null
  elapsed_s: number | null
  is_grasped?: boolean
  gripper_width_m?: number
}
type FinalStatus = 'done' | 'aborted_consecutive_fails' | 'max_steps' | 'llm_error' | 'no_tool_call'
type TrialResult = {
  task: string
  model: string
  started_at: string
  ended_at: string | null
  duration_s: number | null
  n_turns: number
  final_status: FinalStatus | null
  done_reason: string | null
  tool_calls: ToolCall[]
  consecutive_fails_at_abort: number | null
  tool_usage_histogram: Record<string, number>
}
const tStart = Date.now()
const trialResult: TrialResult = {
  task: '',
  model: '',
  started_at: new Date(tStart).toISOString(),
  ended_at: null,
  duration_s: null,
  n_turns: 0,
  final_status: null,
  done_reason: null,
  tool_calls: [],
  consecutive_fails_at_abort: null,
  tool_usage_histogram: {},
}

function updateMemory(turn: number, tool: string, args: any, result: string, detected_labels?: string[]) {
  memory.actions_taken.push({ turn, tool, args, result })
  if (detected_labels) {
    memory.latest_known_objects = [...detected_labels]
    for (const l of detected_labels) {
      const i = memory.objects_seen.findIndex(o => o.label === l)
      if (i >= 0) memory.objects_seen[i].last_seen_turn = turn
      else memory.objects_seen.push({ label: l, last_seen_turn: turn })
    }
  }
  if (tool === 'Pick' && result.startsWith('success')) {
    memory.currently_holding = args?.label ?? null
    // Picking removes the label from any container it was in.
    if (args?.label) delete memory.in_container[args.label]
    memory.consecutive_pick_fails = 0
  }
  if ((tool === 'Place' || tool === 'DropAbove' || tool === 'Release') && result.startsWith('success')) {
    // Track WHERE the held object went, so the next Pick can refuse if it lands in a container.
    if (tool === 'DropAbove' && memory.currently_holding && args?.target_label) {
      memory.in_container[memory.currently_holding] = args.target_label
    }
    memory.currently_holding = null
    memory.consecutive_pick_fails = 0
  }
  // Pick-fail bookkeeping for the must-Perceive-then-escalate protocol.
  // Only count REAL pick failures (where the robot actually attempted a grasp).
  // Label-resolution failures ("not in centroids") don't count — no grasp was tried,
  // and the error message already gave the agent the corrected label set.
  if (tool === 'Pick' && result.startsWith('fail')) {
    const labelMismatch = /not in centroids|label .* not in/i.test(result)
    if (!labelMismatch) {
      memory.consecutive_pick_fails += 1
    }
  }
  // VLARollout completes a sub-task — reset the Pick-fail chain.
  if (tool === 'VLARollout') {
    memory.consecutive_pick_fails = 0
  }
  // A successful Perceive gives the agent fresh labels — reset the Pick-fail chain so
  // it can retry with the new ground truth. Without this, a label-drift sequence
  // gets stuck: Pick(stale) fails → Perceive → Pick(fresh) still blocked by old count.
  // Perceive's result string is 'perceived' (set in dispatchTool, line ~585).
  if (tool === 'Perceive' && result === 'perceived') {
    memory.consecutive_pick_fails = 0
  }
}

function memoryAsText(): string {
  const held = memory.currently_holding ? `currently holding: ${memory.currently_holding}` : 'currently holding: nothing'
  const objs = memory.objects_seen.map(o => `  - ${o.label} (last seen T${o.last_seen_turn})`).join('\n') || '  (none yet)'
  const recent = memory.actions_taken.slice(-5).map(a => `  T${a.turn} ${a.tool}(${JSON.stringify(a.args)}) -> ${a.result}`).join('\n') || '  (none)'
  // PATCH 2026-06-02: surface the LATEST known_objects on every turn. Gemini-ER
  // re-runs detection on every Perceive (including the verifier's hidden Perceive
  // calls), and labels can drift between runs (`blue_elephant_toy` → `blue_toy_elephant`
  // → `blue_plush_toy`). Without this, the agent carries a stale label from an earlier
  // turn and the next Pick fails with "not in centroids". Always use the labels
  // BELOW for any Pick/DropAbove call.
  const fresh = memory.latest_known_objects.length
    ? memory.latest_known_objects.map(l => `  - ${l}`).join('\n')
    : '  (none yet — call Perceive)'
  return `Scene memory:\n${held}\nLATEST known_objects (use these labels for Pick/DropAbove — they may differ from earlier turns):\n${fresh}\nObjects seen so far:\n${objs}\nRecent actions:\n${recent}`
}

// ---- TAMP HTTP client ----
// Explicit 300s timeout. /perceive runs tiptop detect-only + wrist capture
// (legit 30-60s). With a shorter default fetch timeout the agent was retrying
// mid-call and racing the ZED handle, hanging the whole TAMP server.
async function tampPost(path: string, body: any = {}) {
  const r = await fetch(`${TAMP_URL}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    signal: AbortSignal.timeout(300_000),
  })
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`)
  return r.json() as Promise<any>
}
const callPick = (label: string) => tampPost('/pick', { label })
const callDropAbove = (target_label: string, dx_m: number = 0, dy_m: number = 0,
  abs_x: number | null = null, abs_y: number | null = null, abs_z: number | null = null) =>
  tampPost('/drop_above', { target_label, dx_m, dy_m, abs_x, abs_y, abs_z })
// NOTE: deliberately do NOT pass the user's task to /perceive. Passing it makes
// Gemini-ER prompt-biased — it primes the detector to "find an X" and the detector
// hallucinates one if X isn't actually in the scene (attaches the label to the most
// X-like visible object). The server's default is "identify every object on the
// table", which gives unbiased object names. The agent does the task-to-label
// mapping itself, in its own reasoning.
const callPerceive = () => tampPost('/perceive', {})
const callRelease = () => tampPost('/release')
const callLookAway = (wait_s: number = 30.0) => tampPost('/look_away', { wait_s })
const callLookBack = (verify_only: boolean = false, wait_s: number = 60.0) => tampPost('/look_back', { verify_only, wait_s })

// ---- droid_server TCP client for VLA ----
type DroidResp = Record<string, any> & { ok: boolean; error?: string }
let _droidSock: any = null
async function ensureDroidConnected() {
  if (_droidSock) return _droidSock
  const sock = new Socket()
  const queue: Array<{ res: (r: DroidResp) => void; rej: (e: Error) => void }> = []
  await new Promise<void>((res, rej) => { sock.connect(DROID_PORT, DROID_HOST, () => res()); sock.once('error', rej) })
  const rl = createInterface({ input: sock })
  rl.on('line', (line) => { const p = queue.shift(); if (!p) return; try { p.res(JSON.parse(line)) } catch { p.rej(new Error(`bad json: ${line}`)) } })
  sock.on('close', () => { for (const p of queue) p.rej(new Error('droid_server closed')); queue.length = 0; _droidSock = null })
  _droidSock = {
    send(cmd: any, timeoutMs = 60_000): Promise<DroidResp> {
      return new Promise((res, rej) => {
        queue.push({ res, rej })
        sock.write(JSON.stringify(cmd) + '\n')
        if (timeoutMs > 0) setTimeout(() => rej(new Error(`droid timeout ${timeoutMs}ms`)), timeoutMs)
      })
    },
  }
  // consume banner if any
  await new Promise(r => setTimeout(r, 200))
  return _droidSock
}

async function safeClose() {
  // PATCH 2026-05-30: use release_cams (releases ZEDs for TAMP, keeps RobotEnv alive).
  // The next VLA call hits the persistent-env fast path (~25s) instead of
  // recreating RobotEnv + restarting polymetis (~135s).
  try { await (_droidSock as any)?.send?.({ cmd: 'release_cams' }, 10_000) } catch {}
}

// PATCH 2026-05-31: route pi0.5 rollout through OG path (ServerInterface → zerorpc :4242
// on the NUC). Bamboo shim runs in parallel on :5555/:5559 for tiptop. Both clients share
// the same polymetis backend. OG path gives ~14.2 Hz rollout (vs bamboo's ~11 Hz).
// `reset` ensures RobotEnv + ZED cameras are ready after any prior `release_cams`.
async function callVLA(subgoal: string) {
  try {
    const sock = await ensureDroidConnected()
    // PRE-VLA: reset homes the robot via env.reset → update_joints(reset_joints, blocking=True).
    // Also re-acquires ZEDs if release_cams was called previously.
    await sock.send({ cmd: 'reset', record_video: true }, 90_000)
    const r = await sock.send({ cmd: 'vla_rollout', prompt: subgoal, pi0_url: PI0_URL, max_steps: 300, record: true }, 120_000)
    let video_path: string | undefined
    if ((r as any)?.ok) {
      const save = await sock.send({ cmd: 'save_video', path: '/tmp/agent_vla.mp4', fps: 10 }, 30_000) as any
      if (save?.ok) video_path = save.path
    }
    // POST-VLA: return arm to home so the next tool (tiptop or another VLA) starts predictable.
    // NOTE: gripper state is NOT touched — VLA may be used mid-task while still holding an object,
    // and the LLM can call Release explicitly when it wants the gripper open.
    try { await sock.send({ cmd: 'move_to_start' }, 60_000) } catch (e: any) { console.log(`${Y}post-VLA home failed: ${e?.message}${R}`) }
    // Release ZED cams so the next tiptop Pick/DropAbove can grab them.
    try { await sock.send({ cmd: 'release_cams' }, 10_000) } catch {}
    // IMPORTANT: VLA's "ok=true" only means the 300-step rollout finished — NOT that
    // the task succeeded. There's no programmatic grasp-check for VLA (unlike Pick's
    // is_grasped). Force the LLM to verify via Perceive afterwards by returning
    // success=null (unknown) instead of true.
    const ok = (r as any)?.ok !== false
    return {
      success: null,
      rollout_completed: ok,
      note: 'VLA rollout finished but task success is NOT verified — you MUST call Perceive next to visually check the outcome before proceeding.',
      ...(r as object),
      video_path,
    }
  } catch (e: any) {
    await safeClose()
    return { success: false, error: e?.message ?? String(e) }
  }
}

// ---- Tool registry (Anthropic format) ----
const TOOLS = [
  { name: 'Perceive', description: 'Capture a fresh wrist-cam image and robot state. Use to visually verify scene/grasp.',
    input_schema: { type: 'object', properties: {}, required: [] } },
  { name: 'Pick', description: 'TiPToP pipeline pick on a labeled rigid object. Detect → segment → depth → M2T2 grasp → cuRobo plan → execute → close. Returns {success, executed_trajectory, grasp_attempted, is_grasped, gripper_width_m, fail_reason, elapsed_s} + before/after wrist images. AUTO-perceives the whole scene; after a successful Pick the location of target_label AND every other detected object is cached for DropAbove.',
    input_schema: { type: 'object', properties: { label: { type: 'string', description: 'Natural-language description of the object (e.g. "red cup", "yellow banana", "toy eggplant").' } }, required: ['label'] } },
  { name: 'DropAbove', description: 'Release the currently-held object at a target location and open the gripper. Two modes: (1) RELATIVE — pass target_label (+ optional dx_m, dy_m) and the drop point is reference_centroid_xy + offset at reference_top_z + clearance. (2) ABSOLUTE — pass abs_x, abs_y, abs_z and the drop point is exactly that world coordinate, ignoring target_label. Use ABSOLUTE for memory/restore tasks where you remember an exact position from an earlier Perceive. There is no separate Place tool.',
    input_schema: { type: 'object', properties: {
      target_label: { type: 'string', description: 'Label of the reference object whose centroid the drop is computed from. Ignored if abs_x/abs_y/abs_z are all provided.' },
      dx_m: { type: 'number', description: 'Optional XY offset in +x (forward) direction in meters. Use to drop next to the reference instead of on it. Default 0.' },
      dy_m: { type: 'number', description: 'Optional XY offset in +y (left) direction in meters. Use to drop next to the reference instead of on it. Default 0.' },
      abs_x: { type: 'number', description: 'Absolute world x (meters). Provide together with abs_y AND abs_z to drop at an exact position (skips target_label lookup).' },
      abs_y: { type: 'number', description: 'Absolute world y (meters). See abs_x.' },
      abs_z: { type: 'number', description: 'Absolute world z (meters). See abs_x.' },
    }, required: ['target_label'] } },
  { name: 'VLARollout', description: 'pi0.5 closed-loop sub-goal — fallback for deformable / cable / occluded / non-grasp actions.',
    input_schema: { type: 'object', properties: { subgoal: { type: 'string' } }, required: ['subgoal'] } },
  { name: 'Release', description: 'Open the gripper.',
    input_schema: { type: 'object', properties: {}, required: [] } },
  { name: 'LookAway', description: 'Move the arm so the wrist camera points AWAY from the table (base rotated 180°), then HOLD that pose for ~30 seconds while the user shuffles the scene. Returns ONLY after the wait completes, with side_b64 showing the post-shuffle scene. Use this in memory/restore tasks. Pair with LookBack to return to the capture pose. There is no need to poll — one call gives the user the full shuffle window.',
    input_schema: { type: 'object', properties: { wait_s: { type: 'number', description: 'How long to hold the look-away pose (seconds). Default 30.' } }, required: [] } },
  { name: 'LookBack', description: 'Two-mode tool for memory/restore tasks. (1) verify_only=true: SLEEPS for wait_s seconds (default 60) then returns the side-cam image WITHOUT moving the arm — gives the user a fixed window (e.g. 60s) to shuffle without you polling repeatedly. The wrist stays in the look-away pose throughout. (2) verify_only=false (default): inspects + moves the arm back to capture pose. Typical pattern: call LookAway, then LookBack(verify_only=true) ONCE (which waits 60s and returns a side image), inspect that image; if shuffle is done call LookBack() to actually move arm back; if still shuffling call LookBack(verify_only=true) again for another 60s window.',
    input_schema: { type: 'object', properties: {
      verify_only: { type: 'boolean', description: 'If true, sleep wait_s and return side cam without moving. If false (default), inspect AND move arm back.' },
      wait_s: { type: 'number', description: 'In verify_only mode, how long to sleep before capturing the side image (seconds). Default 60.' },
    }, required: [] } },
  { name: 'Done', description: 'Signal that the task is complete.',
    input_schema: { type: 'object', properties: { reason: { type: 'string' } }, required: [] } },
]

// Keep ONLY the most-recent tool_result's images intact. Strip images from every
// earlier message (replacing them with a tiny text placeholder so the assistant
// still sees that an image was previously there). Scene memory carries the
// state information that the LLM actually needs for prior turns.
function pruneOldImages(messages: any[]) {
  let mostRecentToolResultIdx = -1
  outer: for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role !== 'user' || !Array.isArray(m.content)) continue
    for (const c of m.content) {
      if (c?.type === 'tool_result' && Array.isArray(c.content) && c.content.some((x: any) => x?.type === 'image')) {
        mostRecentToolResultIdx = i
        break outer
      }
    }
  }
  for (let i = 0; i < messages.length; i++) {
    if (i === mostRecentToolResultIdx) continue
    const m = messages[i]
    if (!Array.isArray(m.content)) continue
    for (const c of m.content) {
      if (c?.type === 'tool_result' && Array.isArray(c.content)) {
        for (let k = 0; k < c.content.length; k++) {
          if (c.content[k]?.type === 'image') {
            c.content[k] = { type: 'text', text: '[older wrist image pruned]' }
          }
        }
      }
    }
  }
}

// ---- LLM call ----
async function callClaude(messages: any[], system: string): Promise<any> {
  const r = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: MODEL, max_tokens: 4096, system, messages, tools: TOOLS }),
    signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
  })
  if (!r.ok) throw new Error(`Claude API ${r.status}: ${await r.text()}`)
  return r.json()
}

// ---- Independent task verifier ----
// Runs an LLM call with only the task, action history, and current scene — no
// agent history, no system prompt — so it can't be co-opted by the agent's reasoning.
// Used after every world-modifying action (DropAbove/VLARollout/Release) to gate
// per-subgoal progression; also at Done.
async function runVerifier(
  task: string,
  actions: { turn: number; tool: string; args: any; result: string }[],
  objectsDetail: Record<string, { centroid: number[]; extents: number[] }>,
  sideImageB64?: string,
): Promise<{ ok: boolean; reason: string }> {
  const verifierSystem = `You are an independent task-completion verifier for a robot system. You see only the user's task, the actions the robot has taken, and the current scene (3D object centroids/extents in meters, robot frame, +x forward / +y left / +z up). You DO NOT see the executing agent's reasoning — judge purely from observed evidence.

Be literal and strict:
- "Empty" means the container has no other objects inside its xy-footprint besides the placed item.
- "In X" means the object's xy-centroid is within X's xy-footprint.
- "On X" means the object's xyz-centroid is at or above X's top surface within its xy-footprint.
- "Smallest / largest" must match measured extents.
- If the task has multiple sub-steps separated by "then", "first/second", or sequencing words, check whether the SUBSET completed so far is consistent with progress — don't penalize incomplete work, only penalize incorrect work.
- Do NOT weaken predicates based on what the agent intended. Past assertions ("was empty earlier") don't satisfy current-state predicates.

Respond with EXACTLY one JSON object on a single line: {"ok": true|false, "reason": "≤15 words"}.

ok:true = the actions taken so far are consistent with progress toward task completion.
ok:false = the most recent action produced an incorrect or task-violating state and the agent must recover.`

  const userText = `TASK: ${task}

ACTIONS_TAKEN:
${JSON.stringify(actions.slice(-12), null, 2)}

CURRENT_SCENE_OBJECTS_DETAIL:
${JSON.stringify(objectsDetail, null, 2)}

Verify.`

  const content: any[] = [{ type: 'text', text: userText }]
  if (sideImageB64) {
    const isPng = sideImageB64.startsWith('iVBOR')
    content.push({ type: 'image', source: { type: 'base64', media_type: isPng ? 'image/png' : 'image/jpeg', data: sideImageB64 } })
  }

  try {
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': API_KEY, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({ model: MODEL, max_tokens: 200, system: verifierSystem, messages: [{ role: 'user', content }] }),
      signal: AbortSignal.timeout(60_000),
    })
    if (!r.ok) return { ok: true, reason: `(verifier API ${r.status} — letting through)` }
    const j = await r.json() as any
    const text = j?.content?.find((c: any) => c.type === 'text')?.text ?? ''
    const m = text.match(/\{[\s\S]*?"ok"\s*:\s*(true|false)[\s\S]*?\}/i)
    if (!m) return { ok: true, reason: `(verifier returned no JSON — letting through)` }
    try {
      const parsed = JSON.parse(m[0])
      return { ok: !!parsed.ok, reason: String(parsed.reason ?? '') }
    } catch {
      return { ok: true, reason: `(verifier JSON parse failed — letting through)` }
    }
  } catch (e: any) {
    return { ok: true, reason: `(verifier exception ${e?.message} — letting through)` }
  }
}

// Track async SCP-back fetches so main() can wait for them at the end
const _pendingFetches: Promise<void>[] = []
function fetchTrialDirAsync(remoteDir: string, localDir: string) {
  const p = new Promise<void>((res) => {
    try {
      const proc = spawn('py', ['-3', 'scripts/_fetch_trial_dir.py', remoteDir, localDir], { stdio: 'ignore' })
      proc.on('close', () => res())
      proc.on('error', () => res())
    } catch { res() }
  })
  _pendingFetches.push(p)
}
function fetchRemoteFileAsync(remoteFile: string, localFile: string) {
  const p = new Promise<void>((res) => {
    try {
      const proc = spawn('py', ['-3', 'scripts/_fetch_remote_file.py', remoteFile, localFile], { stdio: 'ignore' })
      proc.on('close', () => res())
      proc.on('error', () => res())
    } catch { res() }
  })
  _pendingFetches.push(p)
}

// ---- Tool dispatch ----
async function dispatchTool(turn: number, name: string, input: any): Promise<{ resultBlock: any; done: boolean }> {
  const { __tool_use_id, ...cleanInput } = input
  console.log(`${Y}[T${turn}] ${B}${name}${R} ${C}${JSON.stringify(cleanInput)}${R}`)
  let out: any
  // PATCH 2026-05-31: Rule 4 enforcement — if the target was just DropAbove'd into a
  // container, Pick can't grab it (rimmed surface defeats tiptop's plane-fit grasp planner).
  // Refuse the Pick and tell the LLM to use VLARollout.
  if (name === 'Pick' && cleanInput.label && memory.in_container[cleanInput.label]) {
    const container = memory.in_container[cleanInput.label]
    const msg = `Rule 4 violation: ${cleanInput.label} is currently INSIDE ${container} (placed there by a prior DropAbove). tiptop's Pick fails on rimmed/in-container objects. Use VLARollout("pick up the ${cleanInput.label} from the ${container} and put it in/on the {destination}") instead.`
    console.log(`${RE}  → blocked: ${msg}${R}`)
    out = { success: false, fail_reason: msg, rule_violated: 'rule_4_in_container' }
  }
  // PATCH 2026-06-01: Pick-recovery protocol.
  // After 2+ Pick fails in the current chain → block Pick, force VLARollout escalation.
  // After 1 Pick fail (no Perceive yet) → block Pick, force a Perceive first.
  // Chain resets on any successful action or on VLARollout.
  else if (
    name === 'Pick' &&
    memory.consecutive_pick_fails >= 2
  ) {
    const msg = `Pick failed ${memory.consecutive_pick_fails} times in this chain. Tiptop can't grasp this object reliably (geometry or shape issue). Escalate to VLARollout("pick up the {target} and put it on/in {destination}") for the full pick-and-place subgoal.`
    console.log(`${RE}  → blocked: must escalate to VLARollout after ${memory.consecutive_pick_fails} Pick fails${R}`)
    out = { success: false, fail_reason: msg, rule_violated: 'must_escalate_to_vla' }
  }
  else if (
    name === 'Pick' &&
    memory.actions_taken.length > 0 &&
    (() => {
      const last = memory.actions_taken[memory.actions_taken.length - 1]
      return last.tool === 'Pick' && last.result.startsWith('fail')
    })()
  ) {
    const msg = `Forced re-Perceive: the previous Pick failed and Gemini's labels can mutate between calls. Call Perceive first to ground with current known_objects, then decide whether to retry the same label or pick a different (now-current) one.`
    console.log(`${RE}  → blocked: must Perceive between consecutive failed Picks${R}`)
    out = { success: false, fail_reason: msg, rule_violated: 'must_perceive_after_fail' }
  }
  // PATCH 2026-06-01: Pre-Done verification guard (two layers).
  // Layer 1: force a fresh Perceive before Done so the LLM grounds against the
  // CURRENT scene. Layer 2: if the task contains a checkable predicate ("empty",
  // "alone in", etc.), verify it against the latest objects_detail and refuse Done
  // if violated. The data check fires regardless of how the agent rationalizes — we
  // count objects inside the destination's footprint and block if the count is
  // inconsistent with the predicate.
  else if (
    name === 'Done' &&
    (() => {
      for (let i = memory.actions_taken.length - 1; i >= 0; i--) {
        const a = memory.actions_taken[i]
        if (a.tool === 'Done') continue
        return a.tool !== 'Perceive'
      }
      return true
    })()
  ) {
    const msg = `Pre-Done verification required: you must call Perceive before Done so you can check the prompt's success condition against the CURRENT scene state. Action results (success: true on DropAbove etc.) only confirm the motion ran — they do not confirm the task's worded condition is satisfied. After Perceive, re-read the task, check the predicate against objects_detail, and recover if the condition is violated.`
    console.log(`${RE}  → blocked: must Perceive immediately before Done${R}`)
    out = { success: false, fail_reason: msg, rule_violated: 'must_perceive_before_done' }
  }
  else {}

  // Layer 2 guard: "empty" predicate enforcement. If the previous guards didn't
  // already block and this is a Done call on a task with "empty"-class predicates,
  // count objects inside the destination's footprint and refuse Done if violated.
  // This fires regardless of how the agent rationalizes — the check is on the data.
  if (
    !out &&
    name === 'Done' &&
    /\b(empty|with nothing|by itself|alone)\b/i.test(TASK) &&
    Object.keys(memory.latest_objects_detail).length > 0
  ) {
    let dest: string | null = null
    for (let i = memory.actions_taken.length - 1; i >= 0; i--) {
      const a = memory.actions_taken[i]
      if ((a.tool === 'DropAbove' || a.tool === 'Place') && a.result.startsWith('success') && a.args?.target_label) {
        dest = a.args.target_label as string
        break
      }
    }
    let destEntry: { label: string; centroid: number[]; extents: number[] } | null = null
    if (dest) {
      const exact = memory.latest_objects_detail[dest]
      if (exact) destEntry = { label: dest, centroid: exact.centroid, extents: exact.extents }
      else {
        const destLow = dest.toLowerCase()
        const keyword = destLow.split('_').filter(p => p.length > 2)[0] ?? destLow
        for (const k of Object.keys(memory.latest_objects_detail)) {
          if (k.toLowerCase().includes(keyword)) {
            const e = memory.latest_objects_detail[k]
            destEntry = { label: k, centroid: e.centroid, extents: e.extents }
            break
          }
        }
      }
    }
    if (destEntry) {
      const [cx, cy] = destEntry.centroid
      const [ex, ey] = destEntry.extents
      const xMin = cx - ex / 2, xMax = cx + ex / 2
      const yMin = cy - ey / 2, yMax = cy + ey / 2
      const inside: string[] = []
      for (const [label, info] of Object.entries(memory.latest_objects_detail)) {
        if (label === destEntry.label) continue
        const [ox, oy] = info.centroid
        if (ox >= xMin && ox <= xMax && oy >= yMin && oy <= yMax) inside.push(label)
      }
      if (inside.length > 1) {
        const msg = `Predicate violated at Done time: task asks for an "empty" destination but ${destEntry.label} currently contains ${inside.length} objects: ${JSON.stringify(inside)}. The destination is NOT empty in the current scene. Do NOT rationalize this as "empty at time of placement" — the predicate refers to the final state. Recover: take the placed object out and put it in a destination that actually satisfies the predicate (or the fallback per the task), then Done.`
        console.log(`${RE}  → blocked: destination not empty (${inside.length} objects inside ${destEntry.label})${R}`)
        out = { success: false, fail_reason: msg, rule_violated: 'predicate_empty_violated' }
      }
    }
  }

  // If no guard blocked, run the tool normally.
  if (!out) {
    try {
      switch (name) {
        case 'Perceive': out = await callPerceive(); break
        case 'Pick':     out = await callPick(cleanInput.label); break
        case 'DropAbove': out = await callDropAbove(cleanInput.target_label, cleanInput.dx_m ?? 0, cleanInput.dy_m ?? 0, cleanInput.abs_x ?? null, cleanInput.abs_y ?? null, cleanInput.abs_z ?? null); break
        case 'VLARollout': out = await callVLA(cleanInput.subgoal); break
        case 'Release':  out = await callRelease(); break
        case 'LookAway': out = await callLookAway(cleanInput.wait_s ?? 30.0); break
        case 'LookBack': out = await callLookBack(cleanInput.verify_only ?? false, cleanInput.wait_s ?? 60.0); break
        case 'Done':     return { resultBlock: null, done: true }
        default: out = { success: false, error: `unknown tool ${name}` }
      }
    } catch (e: any) {
      out = { success: false, error: e?.message ?? String(e) }
    }
  }

  // Programmatic grasp check (#9): tiptop may report success=true even when the
  // gripper closed on air. Trust the bamboo width sensor over the parsed log.
  if (name === 'Pick' && out.success === true && out.is_grasped === false) {
    out.success = false
    out.fail_reason = out.fail_reason ?? `gripper width=${out.gripper_width_m} reports nothing grasped`
  }

  // PATCH 2026-06-01: Per-subgoal verifier. After every world-modifying action
  // (DropAbove / VLARollout / Release), auto-refresh the scene and run an
  // independent LLM verifier against TASK + actions + current scene. If the
  // verifier says the current state is inconsistent with progress, downgrade
  // the action to fail with the verifier's reason — the agent then recovers.
  // Skipped on Done (handled by the Done-time guards) and on Perceive/Pick
  // (Pick is verified by the grasp sensor + visual; not a state placement).
  if (
    (name === 'DropAbove' || name === 'VLARollout' || name === 'Release') &&
    out && out.success !== false  // success === true OR null (VLA returns null)
  ) {
    try {
      // Refresh scene so verifier sees the post-action state.
      const fresh = await callPerceive()
      if (Array.isArray(fresh?.known_objects)) memory.latest_known_objects = fresh.known_objects
      if (fresh?.objects_detail && typeof fresh.objects_detail === 'object') {
        memory.latest_objects_detail = fresh.objects_detail
      }
      const verdict = await runVerifier(
        TASK,
        [...memory.actions_taken, { turn, tool: name, args: cleanInput, result: 'pending-verification' }],
        memory.latest_objects_detail,
        typeof fresh?.side_b64 === 'string' ? fresh.side_b64 : undefined,
      )
      if (!verdict.ok) {
        out.success = false
        out.fail_reason = `Verifier rejected: ${verdict.reason}. The action ran but the current scene does NOT match the task's expected state. Recover before proceeding.`
        out.verifier_rejected = true
        console.log(`${RE}  → verifier rejected: ${verdict.reason}${R}`)
      } else {
        console.log(`${D}  ✓ verifier accepted: ${verdict.reason}${R}`)
      }
    } catch (e: any) {
      console.log(`${D}  (verifier skipped: ${e?.message})${R}`)
    }
  }

  const resultSummary = out.success === false ? `fail: ${out.fail_reason ?? out.error ?? '?'}` :
                        out.success === true  ? `success${out.confidence !== undefined ? ` (confidence=${out.confidence.toFixed(2)})` : ''}` :
                        (out.image_b64 || out.after_b64) ? 'perceived' : 'ok'
  updateMemory(turn, name, cleanInput, resultSummary, Array.isArray(out?.known_objects) ? out.known_objects : undefined)
  if (out?.objects_detail && typeof out.objects_detail === 'object') {
    memory.latest_objects_detail = out.objects_detail
  }
  console.log(`${out.success === false ? RE : G}  → ${resultSummary}${R}`)

  // Append to structured trial_result
  trialResult.tool_calls.push({
    turn, tool: name, args: cleanInput,
    success: out.success === undefined ? null : !!out.success,
    fail_reason: out.fail_reason ?? out.error ?? null,
    elapsed_s: out.elapsed_s ?? null,
    is_grasped: out.is_grasped,
    gripper_width_m: out.gripper_width_m,
  })
  trialResult.tool_usage_histogram[name] = (trialResult.tool_usage_histogram[name] ?? 0) + 1

  // Async SCP-back of trial dir so we have frames + videos + JSON locally for review.
  if (out.trial_dir) {
    const turnDir = join(RUN_DIR, `turn${String(turn).padStart(2, '0')}_${name}`)
    fetchTrialDirAsync(out.trial_dir, turnDir)
  }
  // VLARollout returns a path to a third-person MP4 saved on the lab box; SCP it back.
  if (out.video_path) {
    const localVideo = join(RUN_DIR, `turn${String(turn).padStart(2, '0')}_${name}_video.mp4`)
    fetchRemoteFileAsync(out.video_path, localVideo)
  }

  // Save returned wrist-cam image(s) locally + build image content blocks.
  // The server may return:
  //   - image_b64 (Perceive / Release): single wrist frame
  //   - before_b64 + after_b64 (Pick / Place): perception-pose + post-trajectory
  const imageBlocks: any[] = []
  const addImage = (b64: string | undefined, suffix: string) => {
    if (!b64) return
    // Sniff PNG vs JPEG from base64 prefix: PNG → "iVBOR...", JPEG → "/9j/..."
    const isPng = b64.startsWith('iVBOR')
    const mediaType = isPng ? 'image/png' : 'image/jpeg'
    const ext = isPng ? 'png' : 'jpg'
    try {
      const imgPath = join(RUN_DIR, `turn${String(turn).padStart(2, '0')}_${name}_${suffix}.${ext}`)
      writeFileSync(imgPath, Buffer.from(b64, 'base64'))
    } catch {}
    imageBlocks.push({ type: 'image', source: { type: 'base64', media_type: mediaType, data: b64 } })
  }
  if (out.before_b64 || out.after_b64) {
    addImage(out.before_b64, 'before')
    addImage(out.after_b64, 'after')
  } else if (out.image_b64) {
    addImage(out.image_b64, 'wrist')
  }
  // LookBack(verify_only=true) returns ONLY side_b64 (no wrist). the agent needs to
  // see that side image to detect "shuffle done" in memory/restore tasks.
  // For other tools (Pick, DropAbove), side_b64 is a stale pre-action snapshot
  // and we skip it (see comment below).
  if (name === 'LookBack' && out.side_b64) {
    addImage(out.side_b64, 'side')
  }
  // Third-person side cam: save to disk for paper figures but DO NOT send to the agent.
  // Reason (2026-05-31): adding it caused two failure modes — (a) on Pick/DropAbove
  // the side image is the stale pre-action rgb.png, confusing the LLM; (b) even on
  // Perceive, the agent's visual size-reasoning is less reliable than Gemini-ER's
  // task-conditional output. Side cam is surfaced to the agent ONLY through dedicated
  // tools (PeekSide / LookBack response) for memory-restore tasks.
  if (out.side_b64) {
    try {
      const isPng = out.side_b64.startsWith('iVBOR')
      const ext = isPng ? 'png' : 'jpg'
      const sidePath = join(RUN_DIR, `turn${String(turn).padStart(2, '0')}_${name}_side.${ext}`)
      writeFileSync(sidePath, Buffer.from(out.side_b64, 'base64'))
    } catch {}
  }

  // side2 (paper-figure cam): save locally only, not sent to LLM.
  if (out.side2_b64) {
    try {
      const side2Path = join(RUN_DIR, `turn${String(turn).padStart(2, '0')}_${name}_side2.jpg`)
      writeFileSync(side2Path, Buffer.from(out.side2_b64, 'base64'))
    } catch {}
  }

  // Strip large fields from the textual result; images go in their own blocks
  const clean: any = {
    success: out.success,
    fail_reason: out.fail_reason,
    elapsed_s: out.elapsed_s,
    executed_trajectory: out.executed_trajectory,
    grasp_attempted: out.grasp_attempted,
    gripper_width_m: out.gripper_width_m,
    is_grasped: out.is_grasped,
    ee_pose_xyz: out.ee_pose_xyz,
    known_objects: out.known_objects,
    // Gemini-ER's task-conditional reasoning (Perceive only — server returns []
    // for action calls). gemini_atoms is the goal predicate set Gemini believes
    // solves the task; gemini_movables/surfaces is its Movable/Surface tagging;
    // objects_detail has per-object 3D centroid + bbox extents (in meters).
    gemini_atoms: out.gemini_atoms,
    gemini_movables: out.gemini_movables,
    gemini_surfaces: out.gemini_surfaces,
    objects_detail: out.objects_detail,
  }
  const textBlock = { type: 'text', text: JSON.stringify({ result: clean, memory: memoryAsText() }, null, 2) }
  const content = imageBlocks.length > 0 ? [textBlock, ...imageBlocks] : [textBlock]
  return { resultBlock: { type: 'tool_result', tool_use_id: __tool_use_id, content }, done: false }
}

// ---- Main loop ----
async function main() {
  if (!API_KEY) { console.error(`${RE}Missing ANTHROPIC_API_KEY${R}`); process.exit(1) }
  if (!existsSync(SYSTEM_PROMPT_PATH)) { console.error(`${RE}Missing system prompt at ${SYSTEM_PROMPT_PATH}${R}`); process.exit(1) }
  const system = readFileSync(SYSTEM_PROMPT_PATH, 'utf8')
  trialResult.task = TASK
  trialResult.model = MODEL
  console.log(`${C}TAMP:  ${TAMP_URL}\ndroid: tcp://${DROID_HOST}:${DROID_PORT}\npi0:   ${PI0_URL}\nmodel: ${MODEL}\ntask:  "${TASK}"\ndir:   ${RUN_DIR}${R}\n`)

  // Always start with an open gripper
  try { await callRelease(); console.log(`${D}Gripper pre-opened.${R}`) } catch (e: any) { console.log(`${Y}Pre-open release failed: ${e?.message}${R}`) }

  // Start the LAB-SIDE wrist-cam continuous recorder. Cooperative pause/resume
  // via /tmp/wrist_rec_pause so the wrist ZED can be momentarily released to
  // tiptop's /perceive and /pick subprocesses. Captures continuous wrist video
  // covering the shuffle phase of memory/restore tasks.
  // Set AGENT_SKIP_WRIST_REC=1 to disable (when debugging ZED contention).
  const wristSessionName = RUN_DIR.replace(/[\\/]/g, '_').replace(/[^A-Za-z0-9_-]/g, '')
  let wristRecOk = false
  if (process.env.AGENT_SKIP_WRIST_REC === '1') {
    console.log(`${Y}Wrist recording disabled (AGENT_SKIP_WRIST_REC=1)${R}`)
  } else {
    try {
      const wristStart: any = await tampPost('/start_wrist_recording', { session_name: wristSessionName })
      if (wristStart?.ok) {
        console.log(`${D}Wrist recording started → ${wristStart.mp4_expected}${R}`)
        wristRecOk = true
      } else {
        console.log(`${Y}Wrist recording start failed: ${wristStart?.error ?? '?'}${R}`)
      }
    } catch (e: any) {
      console.log(`${Y}Wrist recording start exception: ${e?.message}${R}`)
    }
  }

  // Start the LOCAL paper-cam recorder (paper cam plugged into this PC's USB).
  // Defensive: kill any orphan local_paper_recorder.py from a prior trial first.
  try {
    const { execSync } = require('child_process')
    execSync(
      `powershell -NoProfile -Command "Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match 'local_paper_recorder.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"`,
      { stdio: 'ignore', timeout: 10000 },
    )
    await new Promise(r => setTimeout(r, 1000))
  } catch {}
  let sessionRecordingProc: ReturnType<typeof spawn> | null = null
  const sessionMp4Path = join(RUN_DIR, 'session.mp4')
  const sessionLogPath = join(RUN_DIR, 'recorder.log')
  try {
    const fsmod = require('fs') as typeof import('fs')
    const logFd = fsmod.openSync(sessionLogPath, 'w')
    sessionRecordingProc = spawn('py', ['-3', 'scripts/local_paper_recorder.py', '--out', sessionMp4Path], { stdio: ['ignore', logFd, logFd] })
    await new Promise(r => setTimeout(r, 3000))
    try {
      const logNow = fsmod.readFileSync(sessionLogPath, 'utf8')
      if (logNow.includes('RECORDING_STARTED')) {
        console.log(`${D}Session recording started locally → ${sessionMp4Path}${R}`)
      } else {
        console.log(`${Y}WARN: recorder did NOT signal start within 3s; log: ${logNow.slice(-300)}${R}`)
      }
    } catch {}
  } catch (e: any) {
    console.log(`${Y}Local session recording failed to start: ${e?.message}${R}`)
    sessionRecordingProc = null
  }

  const messages: any[] = [{ role: 'user', content: `Task: ${TASK}\n\nThe gripper has been pre-opened.\n\nFIRST decide: does the task name a specific target object to grab (e.g. "put the cup in the basket" → target is "cup")?\n- If YES and you already know the exact object: call Pick(target) directly — perception runs inside Pick automatically.\n- If NO — the task is abstract or multi-object (e.g. "childproof the table", "clear the table", "set the table for a vegetarian", "put away the toys") — you do NOT yet know what is on the table. Call Perceive FIRST to get the real object labels, THEN decide what to Pick. Do NOT guess a label from the task text (e.g. do not Pick the "basket" just because the word appears in the task — the basket is usually the destination, not a hazard).\n\nAfter each tool call, decide the next step from the returned status (success/executed_trajectory/fail_reason) and the known_objects list.` }]

  let consecutiveFails = 0
  let lastTurnReached = 0
  for (let turn = 1; turn <= MAX_STEPS; turn++) {
    lastTurnReached = turn
    let resp: any
    try { resp = await callClaude(messages, system) } catch (e: any) {
      console.error(`${RE}LLM error: ${e.message}${R}`)
      trialResult.final_status = 'llm_error'
      break
    }
    const assistantContent = resp.content ?? []
    messages.push({ role: 'assistant', content: assistantContent })

    const textBlock = assistantContent.find((b: any) => b.type === 'text')
    if (textBlock?.text) console.log(`${D}${textBlock.text.slice(0, 400)}${R}`)

    const toolBlocks = assistantContent.filter((b: any) => b.type === 'tool_use')
    // Persist Claude's per-turn reasoning + chosen tool calls to reasoning.txt for easy review.
    try {
      const toolSummary = toolBlocks.map((b: any) => `${b.name}(${JSON.stringify(b.input)})`).join(', ') || '(no tool call)'
      const block = `\n=== Turn ${turn} ===\n${textBlock?.text ?? '(no text)'}\n\n→ tool: ${toolSummary}\n`
      appendFileSync(REASONING_PATH, block)
    } catch {}
    if (toolBlocks.length === 0) {
      console.log(`${Y}(no tool call; ending)${R}`)
      trialResult.final_status = 'no_tool_call'
      break
    }

    const userContent: any[] = []
    let done = false
    let anySuccessThisTurn = false
    let anyFailureThisTurn = false
    for (const tb of toolBlocks) {
      const { resultBlock, done: d } = await dispatchTool(turn, tb.name, { ...tb.input, __tool_use_id: tb.id })
      if (resultBlock) {
        userContent.push(resultBlock)
        const txt = resultBlock.content?.find((c: any) => c.type === 'text')?.text ?? ''
        if (txt.includes('"success": true')) anySuccessThisTurn = true
        else if (txt.includes('"success": false')) anyFailureThisTurn = true
      }
      if (d) {
        done = true
        trialResult.final_status = 'done'
        trialResult.done_reason = tb.input?.reason ?? null
        console.log(`${G}${B}DONE${R} ${tb.input?.reason ?? ''}`)
        break
      }
    }
    if (done) break

    if (anySuccessThisTurn) consecutiveFails = 0
    else if (anyFailureThisTurn) consecutiveFails++
    if (consecutiveFails >= CONSECUTIVE_FAIL_LIMIT) {
      console.error(`${RE}Aborting: ${consecutiveFails} consecutive failed tool calls (limit=${CONSECUTIVE_FAIL_LIMIT})${R}`)
      trialResult.final_status = 'aborted_consecutive_fails'
      trialResult.consecutive_fails_at_abort = consecutiveFails
      break
    }

    messages.push({ role: 'user', content: userContent })
    pruneOldImages(messages)
    writeFileSync(join(RUN_DIR, 'memory.json'), JSON.stringify(memory, null, 2))
  }

  if (trialResult.final_status === null) trialResult.final_status = 'max_steps'
  trialResult.n_turns = lastTurnReached
  trialResult.ended_at = new Date().toISOString()
  trialResult.duration_s = Math.round((Date.now() - tStart) / 1000)

  // Stop the lab-side wrist recorder + SCP the wrist mp4 back. Retry the stop
  // call up to 3 times — the lab can be momentarily unreachable at trial end
  // (verifier perceptions overlap with the stop request), and a single failure
  // would leave the recorder running and the segments un-stitched.
  if (wristRecOk) {
    let wristStop: any = null
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        wristStop = await tampPost('/stop_wrist_recording', {})
        break
      } catch (e: any) {
        console.log(`${Y}Wrist recording stop attempt ${attempt}/3 failed: ${e?.message}${R}`)
        if (attempt < 3) await new Promise(r => setTimeout(r, 5000))
      }
    }
    if (wristStop?.ok && wristStop?.mp4 && wristStop?.mp4_exists) {
      const remoteMp4 = wristStop.mp4 as string
      const localMp4 = join(RUN_DIR, 'wrist.mp4')
      try {
        const { execSync } = require('child_process')
        execSync(`scp REMOTE_USER@ROBOT_HOST:"${remoteMp4}" "${localMp4}"`, { stdio: 'ignore', timeout: 600000 })
        console.log(`${D}Wrist recording saved → ${localMp4} (${wristStop.mp4_size_bytes} bytes)${R}`)
      } catch (e: any) {
        console.log(`${Y}Wrist mp4 SCP failed: ${e?.message}${R}`)
      }
    } else if (wristStop) {
      console.log(`${Y}Wrist recording stop returned no mp4: ${JSON.stringify(wristStop)?.slice(0, 300)}${R}`)
    } else {
      console.log(`${Y}Wrist recording stop failed after 3 attempts — recorder may still be running on lab; SVOs still recoverable.${R}`)
    }
  }

  // Stop the local paper-cam recorder. After stop-file, the recorder runs
  // ffmpeg to transcode raw mp4v -> H.264 (~30-60s for a 2-min clip). Wait up
  // to 180s. Do NOT force-kill — that corrupts the MP4 mid-transcode.
  if (sessionRecordingProc !== null) {
    try {
      const fsmod = require('fs') as typeof import('fs')
      const stopFile = join(RUN_DIR, '.stop_recording')
      fsmod.writeFileSync(stopFile, '')
      await new Promise<void>(res => {
        const t = setTimeout(() => res(), 180000)
        sessionRecordingProc!.on('exit', () => { clearTimeout(t); res() })
      })
      if (existsSync(sessionMp4Path) && fsmod.statSync(sessionMp4Path).size > 0) {
        const sz = fsmod.statSync(sessionMp4Path).size
        console.log(`${D}Session recording stopped → ${sessionMp4Path} (${sz} bytes)${R}`)
      } else {
        console.log(`${Y}Session recording: no MP4 at ${sessionMp4Path} yet (may still be transcoding in background)${R}`)
      }
    } catch (e: any) {
      console.log(`${Y}Session recording stop failed: ${e?.message}${R}`)
    }
  }

  writeFileSync(join(RUN_DIR, 'memory.json'), JSON.stringify(memory, null, 2))
  writeFileSync(join(RUN_DIR, 'messages.json'), JSON.stringify(messages, null, 2))
  writeFileSync(join(RUN_DIR, 'trial_result.json'), JSON.stringify(trialResult, null, 2))
  if (_pendingFetches.length > 0) {
    console.log(`${D}Waiting for ${_pendingFetches.length} trial-dir fetch(es) to finish...${R}`)
    await Promise.all(_pendingFetches)
  }

  // Step 1 (always works): copy each turn's per-action external_cam.mp4 + hand_cam.mp4
  // up to the run dir with flat names (turnNN_<Tool>_action.mp4 / _wrist.mp4).
  // tiptop records these server-side via --enable-recording; they're already SCP'd back.
  try {
    const fs = require('fs') as typeof import('fs')
    const sideFiles = fs.readdirSync(RUN_DIR)
      .filter(f => /^turn\d+_.*_side\.(jpg|png)$/i.test(f))
      .sort()
    let copied = 0
    for (const sideFile of sideFiles) {
      const m = sideFile.match(/^(turn\d+_\w+)_side\./)
      if (!m) continue
      const turnPrefix = m[1]
      const subdir = join(RUN_DIR, turnPrefix)
      if (!fs.existsSync(subdir)) continue
      try {
        const innerDirs = fs.readdirSync(join(subdir, 'eval')).filter(d => /\d/.test(d))
        for (const inner of innerDirs) {
          const ext = join(subdir, 'eval', inner, 'external_cam.mp4')
          const wrist = join(subdir, 'eval', inner, 'hand_cam.mp4')
          if (fs.existsSync(ext)) { fs.copyFileSync(ext, join(RUN_DIR, `${turnPrefix}_action.mp4`)); copied++ }
          if (fs.existsSync(wrist)) { fs.copyFileSync(wrist, join(RUN_DIR, `${turnPrefix}_wrist.mp4`)); copied++ }
        }
      } catch {}
    }
    if (copied > 0) console.log(`${D}Copied ${copied} per-action video file(s) to run dir.${R}`)
  } catch (e: any) {
    console.log(`${D}(per-action mp4 copy skipped: ${e?.message?.split('\n')[0]})${R}`)
  }

  // Step 2 (best-effort): build a combined trial.mp4 by concatenating per-action MP4s
  // and 1s freezes of the still frames. Tries ffmpeg from PATH, then from winget's
  // install location, then gives up gracefully. Requires ffmpeg with libx264.
  try {
    const { execSync } = require('child_process')
    const fs = require('fs') as typeof import('fs')
    const path = require('path') as typeof import('path')
    // Resolve ffmpeg path: PATH first, then winget install dir on Windows.
    let ffmpegBin = 'ffmpeg'
    try { execSync(`${ffmpegBin} -version`, { stdio: 'pipe' }) } catch {
      const wingetGlob = join(
        process.env.LOCALAPPDATA ?? '',
        'Microsoft', 'WinGet', 'Packages',
      )
      try {
        const dirs = fs.readdirSync(wingetGlob).filter(d => d.startsWith('Gyan.FFmpeg'))
        for (const d of dirs) {
          const inner = fs.readdirSync(join(wingetGlob, d)).filter(x => x.startsWith('ffmpeg-'))
          for (const i of inner) {
            const candidate = join(wingetGlob, d, i, 'bin', 'ffmpeg.exe')
            if (fs.existsSync(candidate)) { ffmpegBin = `"${candidate}"`; break }
          }
          if (ffmpegBin !== 'ffmpeg') break
        }
      } catch {}
    }
    const sideFiles = fs.readdirSync(RUN_DIR)
      .filter(f => /^turn\d+_.*_side\.(jpg|png)$/i.test(f))
      .sort()
    const entries: { file: string; isVideo: boolean }[] = []
    for (const sideFile of sideFiles) {
      const m = sideFile.match(/^(turn\d+_\w+)_side\./)
      if (!m) continue
      const turnPrefix = m[1]
      const actionMp4 = join(RUN_DIR, `${turnPrefix}_action.mp4`)
      if (fs.existsSync(actionMp4)) entries.push({ file: actionMp4, isVideo: true })
      else entries.push({ file: join(RUN_DIR, sideFile), isVideo: false })
    }
    if (entries.length > 0) {
      const segDir = join(RUN_DIR, '_seg')
      try { fs.mkdirSync(segDir, { recursive: true }) } catch {}
      const segPaths: string[] = []
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i]
        const seg = join(segDir, `s${String(i).padStart(2, '0')}.mp4`)
        const filt = `scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1`
        // Action videos: play at native speed (the actual robot motion). Stills
        // (Perceive turns): freeze for 1s as a transition between actions.
        const cmd = e.isVideo
          ? `${ffmpegBin} -y -i "${e.file}" -vf "${filt}" -an -c:v libx264 -pix_fmt yuv420p -r 30 "${seg}"`
          : `${ffmpegBin} -y -loop 1 -i "${e.file}" -vf "${filt}" -an -c:v libx264 -pix_fmt yuv420p -r 30 -t 1 "${seg}"`
        execSync(cmd, { stdio: 'pipe' })
        segPaths.push(seg)
      }
      const listInSeg = join(segDir, '_list.txt')
      fs.writeFileSync(listInSeg, segPaths.map(p => `file '${path.basename(p)}'`).join('\n'))
      const outMp4 = join(RUN_DIR, 'trial.mp4')
      execSync(`${ffmpegBin} -y -f concat -safe 0 -i "${listInSeg}" -c copy "${outMp4}"`, { stdio: 'pipe' })
      try { fs.rmSync(segDir, { recursive: true, force: true }) } catch {}
      const vidCount = entries.filter(e => e.isVideo).length
      console.log(`${D}Wrote trial.mp4 from ${entries.length} segments (${vidCount} action videos + ${entries.length - vidCount} stills).${R}`)
    }
  } catch (e: any) {
    console.log(`${D}(trial.mp4 build skipped: ${e?.message?.split('\n')[0] ?? 'unavailable'})${R}`)
  }

  console.log(`\n${C}Saved trajectory to ${RUN_DIR} (final_status=${trialResult.final_status}, turns=${trialResult.n_turns}, duration=${trialResult.duration_s}s)${R}`)
}

main().catch(e => { console.error(`${RE}fatal: ${e?.stack ?? e}${R}`); process.exit(1) })
