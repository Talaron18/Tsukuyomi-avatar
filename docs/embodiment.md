# Embodiment: a body the model knows and controls

Status: design, 2026-10. Stage 1 (tags → events → motion, with voice and a
live loop) works. This document is the plan for what comes after it.

## Why tags are not enough

Today the language model picks symbols from a menu (`[nod]`, `[tilt:left]`,
`[gesture:touch_head]`). `flow/events.py` turns each into a planned
minimum-jerk move to a target taken from the rig's fitted statistics, held,
then released. There is no feedback: nothing compares what is rendered with
what was meant.

A stage-2 model trained to pair words with these events would still only
learn "word ↔ symbol". It would not know that a head exists, where it is,
how far it can turn, what the body looks like right now, or what a raised
right hand is in space. A vision-language model can reason about spatial
geometry; tags throw that ability away.

The goal is to let the model control a body it has a model of: joints and
regions with states, limits and positions, seen from the outside as well
(its own rendered image), the way a robot policy controls an arm. Live2D is
an easy case for this (no contact, no dynamics beyond decorative physics),
and the same body description should carry over to 3D avatars.

## The baseline stays

The tag pipeline is not replaced; it is the baseline every embodiment step
is measured against, and the fallback while those steps are unproven:

- **Stage 1** (works now): API model writes tags; director, events, voice
  and the live loop play them.
- **Stage 2** (the original plan): a small local model maps its hidden
  states to the same event vocabulary, trained on words ↔ events. Much
  easier than E4 and may well reach good quality on its own.

Each embodiment step (E2 onwards) is judged against these with the same
tools: blind A/B on rendered clips (`flow/legacy/abtest.py`), the motion
scorer (`flow/score.py`), and latency in the live loop. A step that does not
beat the baseline does not replace it.

## 1. One body description for every model

A canonical body state, shared by Live2D and 3D, that the model thinks in:

- head yaw / pitch / roll, gaze (x, y), eyelids;
- spine / chest / hips orientation and lean;
- per arm: shoulder, elbow, wrist; per hand: a small set of hand poses;
- face: expression shapes (ARKit's 52 blendshapes or FACS action units as
  the vocabulary; a rig drives the subset it has);
- positions of end effectors (hands, head) in body coordinates, in units of
  body size (e.g. shoulder width), so targets read like "right hand in front
  of the chest, at shoulder height".

Each rig declares which parts it has and their limits. A part the rig lacks
is absent, not faked (the same rule as the director's offered tags).

**3D (VRM).** VRM defines a humanoid skeleton and expression blendshapes, so
the mapping is close to direct; inverse kinematics turns end-effector
targets into joint angles.

**Live2D.** Parameters are artist-defined deformers, not joints, so the
mapping is measured, not declared:

- Set one parameter at a time across its range and record the geometry it
  produces. `l2dprobe` already does this on rendered images (`out/`). The
  Cubism core also computes every drawable's mesh vertices on the CPU, so the
  measurement can use exact geometry instead of pixels: per parameter, how
  each body region's vertices move (a forward model, parameters → region
  positions/orientations).
- Body regions come from drawable / part ids and names (the same hints
  `flow/gestures.py` and `flow/behaviors.py` use), checked against the
  measured motion.
- Inverting the forward model (least squares with limits) gives the solver
  "put the right hand here → set these parameters". Switch-like parameters
  (arm pose swaps) become discrete choices the solver can make; whole-model
  transforms stay held.

## 2. Self-perception

The model gets what it currently looks like: a rendered frame, the current
canonical body state, and its limits. It can then check its own output ("did
the hand go up?") and correct, which closes the loop the event pipeline lacks.
Live2D makes this cheap: the live loop renders at 30 fps in about 10 ms a
frame on the server's CPU.

## 3. Control: a VLM plus an action model

The π0 pattern: a VLM backbone reads the conversation (and optionally its
own image); a separate action model, conditioned on the VLM's hidden states,
outputs short chunks of canonical body-state trajectory, continuous, not
symbolic. `flow/legacy/flowprior.py` is already a flow-matching chunk model;
it failed as an unconditioned idle generator, but as a conditioned action
expert it has the right shape. The idle, the director's tags and the
retargeting below all remain useful as fallbacks and as training signal.

## Training data

Retarget into the canonical description:

- **BEAT2, TalkSHOW**: speech paired with co-speech gesture and face (SMPL-X),
  i.e. words ↔ how the body moves, with real geometry.
- **HumanML3D, Motion-X**: text descriptions paired with full-body motion
  (Motion-X with face and hands).
- **Our rigs' authored clips** (40 Azur Lane rigs, Cubism samples), once
  mapped through the Live2D forward model.

Related work to learn from: MotionGPT (motion as language tokens),
RT-2 / OpenVLA (actions as tokens), π0 (flow-matching action expert).

## Stages

| Step | What | Training |
|---|---|---|
| E1 | Canonical body description; Live2D geometry from mesh vertices; the solver from body targets to parameters for Hiyori, lafei, xuefeng | none |
| E2 | Test without training: the API model gets its body description, limits and its own image, and writes continuous body targets (JSON) instead of tags; E1's solver plays them; compare with tags in the live loop | none |
| E3 | A VRM avatar driven by the same body targets (IK + blendshapes), side by side with Live2D | none |
| E4 | Retarget BEAT2 / HumanML3D into the canonical description; train the action model on a small local VLM's hidden states | GPU needed |

E1 to E3 run on the current CPU server and decide whether E4 is worth renting
a GPU for. If E2 shows an API model can direct its own body from a description
plus a picture, the action model mainly buys latency and smoothness; if not,
training is the route.

E1 also helps the current pipeline: gesture descriptions can say "raises the
right hand to face height" from measurement instead of naming parameters.

## E2 as built (2026-10)

`flow/pose.py`; `python -m flow.director --mode body` (or `--body-targets`)
offline, `python -m flow.live --mode body` live, `scripts/body_vs_tags.sh`
for the comparison.

- **Prompt**: generated from `rigs/<rig>.body.json` alone (byte-stable per
  file): parts, the character's own left and right, landmarks, every offered
  variable with its rest and reachable range (union over the drawn arm
  poses, so Hiyori's hands read "[-2.24, -1.66] or [-0.32, -0.02]"), reach
  per hand per drawn pose in words, and what cannot be done (lafei's left
  hand cannot reach shoulder height; Hiyori's hands cannot stop between the
  hanging and the raised drawing). No parameter ids. The pose now goes in
  each user message; a picture of the current frame too with
  `--body-image` on Anthropic, SiliconFlow VLMs or Ollama (DeepSeek chat is
  text only and gets none).
- **Output**: `[pose {"hand.R.y": "@shoulder.y", "head.tilt": 8}]` inline,
  optional `"dur"` / `"hold"`; plus `[blink]` and `[pause:x]`, and the
  rig's own motions (`[gesture:NAME]`, as tag mode offers them) unless
  `--no-body-gestures`. Face shapes are canonical variables, a blink is the
  autonomic layer's transient. A gesture gives way to a pose that moves
  an arm. Markers time like tags (words, moras, sentence streaming).
- **Prompt guidance** (after the first A/B, below): brief beats (a glance
  away and back, a nod as a quick pair), short holds unless stressing one
  moment, the hand that reaches highest named with how high it goes,
  variety from reply to reply, no held half-shut eyes (they stop the
  blinks).
- **Play**: solved with the prior toward the pose the body is in; only
  what moves is played; minimum-jerk move, hold, release back into the
  idle, over the idle frame in both `runtime` and `stream`; arm variant
  parts swap at the midpoint; a pose moving an arm keeps idle gestures
  from starting during it (`director.merge`), and the idle's head, gaze,
  brow and body moves run on under every pose. A hold is not frozen: it
  carries 0.35 of the idle's motion and settles 0.2 of the way back to it
  (`flow.pose.LIFE`, `SETTLE`). Tag mode is unchanged byte for byte.
- **Feedback**: per target, the forward model's value and the Core's own
  geometry go to `actions.json` / the live log; `--body-check` runs one
  offline correction round (`check/check.json`).

### First A/B (runs/e2) and what changed

Blind A/B on the server, 12 pairs (DeepSeek, Hiyori and lafei, 6 lines
each): a tie overall (body 46%, tags 54%); Hiyori body 2-1 with 3 ties,
lafei tags 3-1 with 2 ties. The body replies were long static holds (hold
2-3 s, 9 of 13 Hiyori markers with an explicit hold; head and gaze frozen
in 58% of the hold frames), face-only on lafei (2 of 12 poses moved a
hand), few quick moves; the tags gave quick nods and looks plus the rig's
own gestures. And on lafei the breath bobbed the whole figure (~3 px std
at 0.25 Hz). Changed for the second round (runs/e2b):

- breath calibrated per rig from the body file (`flow.runtime.
  breath_setting`): Hiyori 1, Natori 0.467, lafei 0.109, xuefeng 0.107 of
  ParamBreath's range, so it moves each body at most 0.0075 hu;
- held poses keep the idle's life (LIFE 0.35, SETTLE 0.2) and the idle's
  head/gaze/brow/body moves run on under poses; default hold 1.5 -> 1 s;
- the prompt guidance above, and gestures offered by default.


## E3 as built (2026-10)

`flow/gltf.py`, `flow/vrmbody.py`, `flow/vrm.py`, `flow/vrmrender.py`,
`flow/web/` (vrm.js, vrm.html, render.html, vendor/: three 0.180.0 and
@pixiv/three-vrm 3.5.5, MIT, sha256 in vendor/README.md). Sample avatars in
`assets/vrm/` (Seed-san, VRM 1.0, VRM Public License 1.0; three-vrm-girl,
VRM 0, VRoid Hub licence; both allow redistribution and modification, see
assets/vrm/README.md). Body files `rigs/seed-san.body.json`,
`rigs/three-vrm-girl.body.json`.

- **Body file** (`python -m flow.vrmbody --vrm PATH --out FILE`):
  tsukuyomi.body/2 with `"kind": "vrm"`, loaded by `flow.body.load` into a
  `VRMBody` with BodyModel's interface, so flow.pose's prompt, solve loop,
  reach report and play run unchanged. Frame as Live2D: origin the chin
  (lowest head-bone vertex in front), 1 hu = chin to top of skull (hair
  left out): 0.246 m on Seed-san; +x the character's left (VRM 0, facing
  -Z, is turned round), z toward the viewer. Parameters are the body's own
  joint angles (head/torso yaw-pitch-tilt split over neck+head and
  spine+chest+upperChest; per arm elevation, plane of elevation, twist,
  elbow bend; shoulders) and face weights in the Live2D standard ranges.
  Joint limits: AAOS active ranges cut to a talking avatar (head yaw +-50,
  pitch +-35, tilt +-30; torso +-25/15/12; elevation 0..170, plane
  -90..130 with the extension held to 60 hanging / 45 level / 30 overhead,
  twist -100..90 measured from the swing out of hanging (swing-twist: an
  Euler twist's zero turned with the plane, and behind the back needed
  -110 of it), elbow 0..145; a wrist on the hand bone, roll -20..160 from
  the palm-down T-pose, flexion -70..80, deviation -30..20).
- **Solver**: declared variables and joints set directly (clamped);
  a hand target is two-bone IK with a pole (elbow bend by bisection on the
  real bone vectors, swivel searched for the elbow nearest out-down-back
  within the limits and the current pose); a target given in x, y only
  keeps the palm's depth, pushed in front of the torso / face and within
  reach. ~3 ms per solve (two hands meeting ~30 ms, a detour ~0.2 s). The
  reverse is forward kinematics (`perceive`).
- **Collisions** (body file revision 2): the trunk, neck and head as one
  signed distance lofted through 0.2 hu slabs of the skinned rest mesh
  (hair left out), narrowed where the resting arms sank in (a loose
  shirt's sides are skinned to the chest); arms as capsules (upper arm past
  the shoulder, forearm, two across the palm), radii from their vertices.
  The swivel search orders limits, then penetration, then the pole. A
  target inside the body moves onto it; one reachable only through the
  body (or, one hand moving, the other arm) moves to the nearest clear
  target; both are MISS with the reason in the report. Two hands moving
  together are solved for the body each and their targets parted, half
  each, until they touch. When the straight joint-angle path (what the
  pose layer plays) crosses the body, the solver gives a via pose that
  flow.pose plays on the way in and before the release, and that pose
  holds still (no settle: drifting each angle toward the rest put the
  hands 0.32-0.43 hu into the hips by the end of a hold behind the back). Old solver vs now
  (deepest penetration, hu; Seed-san / three-vrm-girl): hands together
  0.08 / 0.07 into the chest and 0.085 / 0.09 into each other -> 0.018 /
  0.007; crossed forearms 0.20 / 0.18 through each other -> 0.02; hands
  behind the back (z given) 0.32 / 0.30 into the hips -> 0.01, and 0.41 /
  0.35 on the straight path -> 0.01 by the detour; hand on the other
  shoulder 0.11 -> 0.02. Tolerance 0.02 hu (5 mm).
- **Composite targets** (VRM only): `"hands": "@together"` (palm to palm
  before the chest, fingers up), `"@clasped"` (before the waist),
  `"@behind_back"`; `"hand.R": "@chest"` (or any landmark: the palm flat
  on the body there), `"@hip"`, `"@back"`. Hand positions plus a hand
  orientation the wrist turns to; the VRM prompt lists them. The Live2D
  prompt and parser have none (byte-identical): in `--avatar both` the
  reply is Hiyori's, so "hands together" arrives as two plain x, y targets
  and the VRM's hands meet side by side (reported), never through each
  other; head.pitch markers are converted from ParamAngleY units.
- **Kinematics = three-vrm's normalized rig**: the stream sends each
  bone's local quaternion; Python FK and three-vrm in Chromium agree to
  0.01 mm on both samples (tests/test_vrm.py). VRM 0: x and z of each
  quaternion negated in the page; three-vrm's look-at pitch is + down.
- **Face map** (per avatar, from what it has): VRoid's per-feature morphs
  where present (Fcl_EYE_Joy_L/R for eye smile, Fcl_BRW_Surprised / Angry
  / Fun for brow lift +/- and form +/-, Fcl_MTH_Fun / Sorrow for mouth
  form), else the emotion presets at reduced gain (happy 0.5 eye smile,
  surprised / angry / relaxed for the brows, happy / sad 0.4 for the
  mouth). Limits: VRoid brows are one shape for both sides (each side gives
  half); presets move the whole face. Blinks on blinkLeft/Right, visemes
  a/i/u/e/o on aa/ih/ou/ee/oh, gaze on vrm.lookAt (gaze 1 = the look-at
  range's input maximum: the eyes at the avatar's own extremes, Seed-san
  10 deg, three-vrm-girl 8-12, as Live2D's EyeBall 1 is the drawn eye's
  edge; capped at 15 deg for a bone look-at whose range map says more).
  The camera sits at the eyes' height looking down at the framing: gaze 0
  with the head at rest looks into it (it sat 0.7 hu lower: the rest gaze
  passed 3.7 / 3.2 deg over the viewer). Blush only from a custom expression or a morph
  named for it -- Seed-san has none, so cheek.blush is absent (reported).
- **Idle and events**: VRMRig gives MotionStream an EventRig's surface; the
  roles land on canonical variables in the Cubism standard units the calm
  profile is in (head_x 1 = 30 deg yaw, head_y 1 = 15 deg pitch -- Live2D
  draws its pitch at ~0.48 of its yaw, the mouth's shift at ParamAngleY
  +-30 over its shift at ParamAngleX 30 on Hiyori, Natori, lafei,
  xuefeng; at 30 the fitted idle bent the head 18 deg down at its 2nd
  percentile -- body 10 deg, gaze/brows 1). Arms
  rest relaxed (elevation 8, plane 20, elbow 14); breath = shoulders up +
  upper chest back, calibrated to BREATH_TARGET_HU like the Live2D rigs.
  No gestures for VRM: body poses only. Spring bones (hair) and node
  constraints run in three-vrm's update.
- **Pipeline**: `flow.director --avatar vrm --vrm PATH` plans into
  states.json (+ speech.wav, subs, tokens; `--render` -> render/preview.mp4
  and demo.mp4 via headless Chromium). `flow.live --avatar vrm` streams
  states on /states (SSE, server clock); the page interpolates them with
  the audio on the same playout delay. `--avatar both`: the Live2D rig and
  the VRM side by side from the same sentences (prompt from the Live2D
  body; the VRM resolves landmarks on its own body). No `--body-image` for
  VRM (the server cannot draw it).

## Motion primitives (2026-10)

Precise asks played well in body mode; open ones ("dance for me", "let's
celebrate") came out as a few held poses in a row, nothing on the beat.
`flow/moves.py` adds parameterised continuous motions the model calls by
name next to its poses:

    Let's dance! [move {"name": "dance", "beats": 8, "tempo": 120}] Come on!

- **Primitives**: sway (upper body side to side, head countering; a VRM's
  hips sway too), bounce (down on every beat: knees via the VRM's hips,
  shoulders, a little nod), head_bob, shimmy (shoulders, twice a beat),
  look_around, clap (hands meet on the beat; a VRM's through the
  `@together` composite, palms facing), cheer (fist pump, side L/R/both),
  wave (a raised hand side to side; both: both hands up, swaying), twirl
  (hands rolling around each other), and dance (2-4 beat segments of a
  trunk figure plus an arm figure, seeded, no figure twice in a row,
  mixed over a beat in target space). Aliases: nod_along (head_bob at
  half tempo), fist_pump, arm_wave, shoulder_shake, twirl_hands,
  celebrate. Params: beats 1-32 (8), tempo 60-180 bpm (110), amp 0.1-1
  (0.7), side.
- **Per body**: curves in canonical variables, offsets on the declared
  variables, hand targets solved on keyframes (8 a beat, 16 for a shimmy;
  repeating keyframes once; every other one solved) with the body's own
  solver: VRM collision-aware IK (no detour search: 87% of the time),
  Live2D forward model in one arm drawing with the arm alone. Capability
  per body decides the prompt's menu: Seed-san and three-vrm-girl all;
  Hiyori no cheer (her raised drawing lifts no fist with the arm alone);
  lafei no clap, cheer or wave. A side a body lacks is degraded with a log
  line; a primitive it lacks is skipped.
- **Play**: MoveLayer under the PoseLayer, PCHIP between keyframes (no
  overshoot at a clap's contact), min-jerk ease in and out (one beat,
  0.9 s for a clip that moves an arm), additive over the idle (the idle's
  head, gaze, breath keep going), absolute arms blended from it; a later
  move on the same parameters takes them over; a pose takes its parts and
  gives them back (held without LIFE on a move's joints). On a VRM a trunk
  figure carries a figure's hands with the chest; a free arm hangs,
  analytically (0.3 of the chest's turn, raised off the hips only as far as
  clears them): rigid arms went 0.10-0.15 hu into the hips at torso.tilt 6,
  and IK targets at a hanging palm swapped swivel between keyframes. The
  hips' sway and bob are a render offset of the whole avatar ("hips" in
  the state, only while a move writes it).
- **Checked** (scratchpad renders, every frame, idle on): VRM penetration
  <= 0.024 hu (PEN_TOL 0.02) for every primitive, a pose over a clap, two
  moves in a row, and a 16-beat dance, on Seed-san and three-vrm-girl;
  onset/end steps at the idle's level; Hiyori's dance draws one arm at a
  time, no parameter faster than 2.5 spans a second. Build time per marker
  (it adds to the sentence's first audio in live): a figure 0.02-0.15 s, a
  16-beat dance ~1.1 s on the VRMs, ~0.9 s on Hiyori; the menu at startup
  < 1 s on Hiyori and lafei, ~3.5 s on Natori.
- **Not done**: beats are not synced to the speech (beat 1 is the marker).
- `--no-body-moves`: the prompt and parser of before. Without [move]
  markers every output is the bytes it was.

## Stage 2 on one 3090 Ti (2026-10)

The baseline's Stage 2 on the user's server: one RTX 3090 Ti (24 GB,
Ampere: bf16, no FP8, no FlashAttention-3). Two students of the body-mode
director, from the data engine (`flow/dataset.py`): (a) a local model
LoRA-tuned to write the same marked replies (`flow/train/`), (b) an action
head on a local model's hidden states predicting the directed trajectory
(`flow/actionhead.py`). Nothing GPU-side has run here (no GPU in this
container): the code paths ran on CPU with a tiny random Qwen3 and tiny
random data; the 4-bit, Unsloth and llama.cpp steps have not run at all.

### What is built

- **Moves in the data** (`flow.dataset`): `[move]` markers are built on
  the body as `director.plan` builds them (seeded by the turn's seed and
  the marker's place), a gesture under an arm move gives way, and the
  move tokens stream through the same layers in both trajectories; the
  markers show `played: true`, records carry `moves`, a VRM's trajectory
  gains `hips.x` / `hips.y`. Records from before (trajectory version 1):
  `python -m flow.dataset replay OUT` re-streams them from their replay
  records with no model call (version 2 replay records give back the
  arrays they were made with, bit for bit; tested on seed-san, Hiyori lite
  and the synthetic Live2D rig).
- **One template** (`flow/train/chat.py`): Qwen3 with thinking off as
  ChatML with the empty `<think>\n\n</think>\n\n` on every assistant turn,
  history included (Qwen3's own template strips it from the history, so a
  whole session would not train as one sequence; on a single turn the two
  are the same bytes, which the trainer checks against the real tokenizer
  before it starts). The trainer, `--backend hf`, the merged tokenizer's
  `chat_template` and the Ollama Modelfile's `TEMPLATE` all render the
  same bytes (tests run the Go template the way Ollama calls it). Contents
  are never trimmed: the body prompt ends in a newline and Ollama renders
  it as is. ChatML (Qwen2.5) and Gemma 3 too.
- **Trainer** (`scripts/train_sft.py` = `python -m flow.train.sft`):
  `stats` (token lengths: whole examples, system prompt, user turns,
  replies; the share that fits at each length), `train` (session split by
  hash, labels on replies and their end token only, cut at a reply
  boundary, QLoRA NF4 r32 alpha 32 on every projection, gradient
  checkpointing, bf16, batch 1 x 16 accumulation, paged 8-bit AdamW,
  Unsloth when it imports else transformers + PEFT with the logits only at
  labelled positions, eval loss, resumable checkpoints guarded by a data
  hash, a speed probe), `merge` (fp16, into the unquantized base), `gguf`
  (llama.cpp commands), `modelfile` (TEMPLATE, stops, num_ctx 8192,
  sampling; `--create` runs `ollama create`).
- **Teaching the reachable value** (`flow.dataexport --correct-misses`,
  on by default for `sft`, `stats` and the trainer's dataset-folder
  loading): the student of the first run asked for hand positions that
  lafei's narrow-armed rig cannot reach (17% of its targets missed against
  the teacher's 8%). The solver records, per target, the value it did reach
  (`got`); the export writes that number into the marker in place of the
  missed one (an `"@shoulder.y"` anchor that missed becomes the number;
  reached targets, `dur`, `hold` and the spoken text stay byte for byte).
  A turn whose misses are all corrected no longer counts as a MISS turn
  for the export, so a session is no longer cut there; the `miss` flag and
  the reach rates stay as recorded, and `stats` counts markers and targets
  corrected per avatar and what was skipped (records from before the
  report, markers that do not line up with their pose records, a target
  whose key cannot be found). The reached value is not always a fixed
  point: asked for again, ~95% of corrected markers land (random hand
  targets, 80 per body: Hiyori 78/78 and lafei 76/80 ok, seed-san 47/56,
  where hands that would pass through the body come back shifted again);
  the action export keeps what was played. `--avatar-weight lafei=2`
  repeats an avatar's sessions in the export (also `scripts/train_sft.py`).
- **Streamer manners** (`flow.dataexport --manners`, on by default for
  `sft`, `stats` and the trainer's dataset-folder loading; `--no-manners`
  for the export of before, byte for byte; `--max-poses N`, default 2):
  user feedback on renders of the first student was arms raised "for no
  reason" (worst on lafei, whose arm can only swing out sideways). The
  teacher writes ~2.3 pose markers a reply and about a third of them move
  a hand; the student copied that, where streamers move head, face, gaze
  and a little torso, and the hands rarely and on purpose. The export
  decides per turn whether the user asked for a body action (the topic is
  `pose`, `show` or `invite`, or the line matches the en / ja / zh word
  lists `ASK_WORDS`, with `ASK_NOT` cutting false friends such as 相手,
  举例, "on the other hand"; the decision is counted by `stats`). Not
  asked: pose markers keep head, gaze, brow, eye, mouth and cheek targets,
  torso yaw / pitch / tilt up to 6 degrees, `dur` and `hold`; hand, arm,
  shoulder and bigger torso targets and composites (`"hands":
  "@together"`) go, and a marker with nothing left is removed with its
  extra space; `[move]` other than `head_bob` / `look_around` (and their
  aliases nod, bop, look) and every `[gesture:NAME]` go (a directed
  gesture plays the arms on the Cubism rigs); `[blink]` and `[pause]`
  stay. Asked: the body markers stay. Either way at most `--max-poses`
  pose markers a reply stay (the first; markers with body targets first
  on an asked turn). It runs after `--correct-misses`, on history turns of
  a per-turn export too, and a turn whose spoken text would change (the
  pose parser's reading, before and after) is left alone and counted. It
  changes the training text, so a run needs a new `--out`. No record of
  the real teacher data was at hand when it was written: the word lists
  were checked against the personas file's topic examples and the tests'
  mocked records only; read `stats` on r1 first (below).
- **Serving**: Ollama (the GGUF and the Modelfile) through the existing
  `--backend ollama`, or `--backend hf` (transformers + the adapter,
  `TextIteratorStreamer`) in `flow.director` and `flow.live`.
  `--ollama-ctx N` sets num_ctx per request for a stock model.
- **Eval** (`scripts/eval_sft.sh`, `flow.train.evalsft`): the held-out
  sessions turn by turn through the student, the teacher live and the
  teacher's recorded replies, each keeping its own history; per model the
  parse-warning rate, refusals (the backend's and by text: local backends
  never report one), cut-offs, markers per turn by tag, unplayed and
  not-played markers, reach / MISS from the pose solve, first-text latency
  and tokens/s; `AB=1` renders first turns for a blind A/B page
  (`flow.legacy.abtest`, the `body_vs_tags.sh` layout).
- **Action head** (`scripts/extract_hidden.py` = `python -m
  flow.train.hidden`, `python -m flow.actionhead train | eval | export`):
  module docstrings (motion history, prosody, layer mix, per-avatar
  norm, the eval metrics and the ablation grid: "Action head ablations on
  BEAT2" below). `export` solves a trajectory back to parameters and
  writes move tokens: `tokens.json` for `flow.runtime --tokens-in`
  (Live2D) and `states.json` for `flow.vrmrender`, the directed clip
  through the same path, plus a `render.sh`. Known limit: a trajectory has
  hand positions and no palm orientation, so at a clap's contact the
  VRM's collision model keeps the hands ~0.1 hu wider than the clip that
  had `@together`'s orientation.

### Install

`requirements.txt` is the Stage 1 pipeline; `requirements-train.txt`
this section's pip packages (torch with CUDA first, see its header).

Tested on CPU here (these exact versions; released 2026-09/10):

    pip install "transformers==5.20.2" "peft==0.21.2" "accelerate==1.15.0" \
        "tokenizers==0.23.3" sentencepiece

GPU side, tested range unknown (never run here). PyTorch with CUDA for an
Ampere card from the official index (pick the CUDA build that matches the
driver; torch 2.14.1 is the version the CPU tests used):

    pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu128
    pip install "bitsandbytes>=0.47"          # 4-bit NF4 and paged AdamW
    pip install unsloth                       # optional, faster (it pins its own deps:
                                              # install it in its own venv)

The code also takes transformers 4.56+ (`dtype` vs `torch_dtype` and
`processing_class` vs `tokenizer` are chosen by version; unknown
TrainingArguments are dropped with a log line). TRL is not needed: its
SFTTrainer's jobs here, templating and masking, are `flow.train.chat`'s so
that they match serving exactly. llama.cpp for GGUF:

    git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
    cd ~/llama.cpp && pip install -r requirements.txt && cmake -B build && cmake --build build -j

### Memory (estimates, 24 GB)

| base | 4-bit weights + bf16 embeddings | LoRA r32 (fp32 + grads + 8-bit Adam) | activations at 8k (checkpointed) | logits at labelled positions | total |
|---|---|---|---|---|---|
| Qwen3-8B | ~6.5 GB | ~0.9 GB | ~4 GB | ~2.5 GB (152k vocab) | ~14 GB |
| Qwen3-4B | ~3.5 GB | ~0.6 GB | ~3 GB | ~2.5 GB | ~10 GB |
| gemma-3-12b-it | ~9.5 GB | ~1.3 GB | ~5 GB | ~4 GB (262k vocab) | ~20 GB: `--max-len 6144` if it does not fit |

Without the labelled-position trick (logits everywhere) an 8k sequence's
fp32 logits alone are ~5 GB for Qwen3 and ~8.5 GB for Gemma 3. The
`--max-len auto` default is the 99th percentile of the examples (a
session of 6 turns with a ~2.3k-token prompt is ~4-5k tokens).
Serving: Qwen3-8B Q4_K_M in Ollama is ~5 GB plus the KV cache at num_ctx
8192 (~1.2 GB); `--backend hf` in bf16 ~16.5 GB, `--hf-4bit` ~6.5 GB.
Hidden states: one 4096-wide layer in fp16 is 8 KB a token; 15k replies
of ~150 tokens are ~18 GB (`--proj 1024`: ~4.6 GB).

### Time (estimates)

15k turns as 6-turn sessions are ~2.5k examples of ~4.5k tokens, ~11M
tokens an epoch. QLoRA on an 8B with gradient checkpointing runs at very
roughly 1-1.5k tokens/s with PEFT and 2-3k with Unsloth on this card: ~1-3
h an epoch, the default 2 epochs 2-6 h. The trainer's first steps log the
real tokens/s, peak memory and the epoch time they imply: run
`--max-steps 20` first. Eval: ~1-2 s a turn on Ollama (Q4_K_M, 8B) plus
the teacher's own latency. Extraction: one forward of ~3-5k tokens a turn,
~0.3-0.5 s on the 8B (~2 h for 15k turns); the action head trains in
minutes.

### The sequence on the server

    # 1. data (DeepSeek teacher; resumable; export with session ids)
    SESSIONS=2500 TURNS=6 CONC=8 BUDGET_USD=20 OUT=data/datagen/r1 \
        nohup bash scripts/datagen.sh > datagen.log 2>&1 &
    python -m flow.dataset replay data/datagen/r1      # older records: moves played
    python -m flow.dataexport stats data/datagen/r1

    # 2. lengths and the template check, then a speed probe, then the run
    python scripts/train_sft.py stats --data data/datagen/r1 --base Qwen/Qwen3-8B
    python scripts/train_sft.py train --data data/datagen/r1 --out runs/sft/q8 \
        --base Qwen/Qwen3-8B --max-steps 20          # tokens/s, peak memory
    python scripts/train_sft.py train --data data/datagen/r1 --out runs/sft/q8 \
        --base Qwen/Qwen3-8B --restart               # the full run (resumable)

    # 3. merge, GGUF, Ollama
    python scripts/train_sft.py merge --run runs/sft/q8
    python scripts/train_sft.py gguf --run runs/sft/q8 --llama-cpp ~/llama.cpp
    python scripts/train_sft.py modelfile --run runs/sft/q8 --name tsk-q8 --create

    # 4. eval against the teacher (and a blind A/B with AB=1)
    RUN=runs/sft/q8 STUDENT_MODEL=tsk-q8 bash scripts/eval_sft.sh
    # or through transformers: STUDENT_BACKEND=hf STUDENT_MODEL=Qwen/Qwen3-8B \
    #   HF_ADAPTER=runs/sft/q8/adapter bash scripts/eval_sft.sh

    # 5. live
    python -m flow.live --mode body --model-json <rig>.model3.json \
        --rig rigs/Hiyori.npz --backend ollama --model tsk-q8 --tts auto
    # (or --backend hf --model Qwen/Qwen3-8B --hf-adapter runs/sft/q8/adapter)

    # 6. action head
    python -m flow.dataexport action data/datagen/r1 --out data/datagen/r1/export/action
    python scripts/extract_hidden.py --export data/datagen/r1/export/action \
        --data data/datagen/r1 --model runs/sft/q8/merged --layers -8 \
        --proj 1024 --out runs/hidden/q8
    python -m flow.actionhead train --export data/datagen/r1/export/action \
        --hidden runs/hidden/q8 --out runs/ah/q8 --steps 3000
    python -m flow.actionhead eval --ckpt runs/ah/q8 \
        --export data/datagen/r1/export/action --hidden runs/hidden/q8 \
        --markers runs/sft/q8/eval/student.jsonl --data data/datagen/r1
    python -m flow.actionhead export --ckpt runs/ah/q8 \
        --export data/datagen/r1/export/action --hidden runs/hidden/q8 \
        --data data/datagen/r1 --id <session>/t02 --out runs/ah/q8/side
    bash runs/ah/q8/side/render.sh                  # pred/ and directed/ clips

The action head's markers baseline needs another model's markers (the
student's, from the eval): on teacher text the markers played again are
the target itself.

### Retraining with streamer manners (the GPU machine)

WSL, repo at `the repo`, data `data/datagen/r1` (the first student
was `~/runs/sft/q8`). First what the transform does to the data: the
turns asked / not asked per avatar, pose markers per reply before and
after, the hand share of targets, the moves and gestures dropped, the
replies left with no body marker:

    cd Tsukuyomi
    python -m flow.dataexport stats data/datagen/r1 --json ~/runs/manners-stats.json
    # per turn: "manners"."by" in the json (topic / keyword / topic+keyword / null)
    python -m flow.dataexport stats data/datagen/r1 --no-manners   # the old numbers

Then the run (manners and correct-misses are on by default; a new `--out`,
the data hash differs from `q8`'s), and the rest as in "The sequence on the
server":

    python scripts/train_sft.py train --data data/datagen/r1 \
        --out ~/runs/sft/q8m --base Qwen/Qwen3-8B --avatar-weight lafei=2
    python scripts/train_sft.py merge --run ~/runs/sft/q8m
    python scripts/train_sft.py gguf --run ~/runs/sft/q8m --llama-cpp ~/llama.cpp
    python scripts/train_sft.py modelfile --run ~/runs/sft/q8m --name tsk-q8m --create
    RUN=~/runs/sft/q8m STUDENT_MODEL=tsk-q8m TEACHER=0 bash scripts/eval_sft.sh
    python -m flow.live --mode body --model-json <rig>.model3.json \
        --rig rigs/Hiyori.npz --backend ollama --model tsk-q8m --tts auto

The eval's markers-per-turn table should show fewer pose markers and a
lower hand share than `q8`'s. Its held-out sessions are the same ones as
`q8`'s (the split is by session hash). Words that wrongly count as a
request keep a hand; words that wrongly do not drop one: edit `ASK_WORDS`
/ `ASK_NOT` in `flow/dataexport.py` and run `stats` again before training.

### Expression beats (r2 data)

Real VTubers (Kaf, Shigure Ui) and an AI VTuber (Shizuku), all framed chest
up, move mostly the face: several expression changes a sentence (eyes
closed in a smile at the funny parts, mouth shapes, brows), a small head
tilt or turn with each one, a body that is almost still and the hands out
of the picture. The r1 replies held one expression a reply. Three changes,
none of which touch the r1 files or the flags of a run in progress:

- **Teacher prompt** (`flow.pose.system_prompt(..., prompt="beats")`;
  `flow.dataset gen --teacher-prompt {beats,v1}`, default `beats`; `v1`
  regenerates r1's prompt byte for byte, tested). The "How to move" part
  gains, worded from the variables this body has (`pose.offered`; lafei has
  no `eye.*.smile`, so it is told to close the eyes with `eye.*.open`
  instead, and brows / blush only where they exist):

      Your face does most of the talking, as a streamer's does, and it
      never holds still: every sentence gets at least one expression beat,
      a longer sentence two or three, so that something changes about every
      1.5 to 2 seconds of speech ... Vary them, not the same smile each
      time: the eyes, the brows, mouth shapes, a blush, the gaze, small head
      moves (worded from the variables the body has) ...
      Each beat carries a small head move in the same marker, a few degrees
      (head.tilt, head.yaw, head.pitch between -8 and 8) ... Never write
      that head move as a marker of its own.
      Keep the rest of your body almost still, hands out of the picture: no
      hand, arm or big torso moves unless the user asks you to show or do
      something, or the moment is a big one (a greeting, a celebration), and
      then once in a reply at most.
      (That last paragraph is replaced by the camera, budget and `[frame]`
      lines of "Framing and motion budget" below.)

  The rule becomes "At least one expression beat in every sentence, two or
  three in a longer one" (the 20-session pilot had 0.65 beats a sentence,
  2.3 a reply of about 9 s of speech, against the one to three asked for;
  real streamers change expression every 1-2 s). The definition of a beat
  is unchanged, so the stats count the same thing. The "use
  your whole body" line is dropped and the placement examples are face
  beats with their head angles. The director's own default stays `v1`
  (`--body-prompt`): a student is served with the prompt it was trained
  on, so serve an r2 student with `--body-prompt beats`.
- **Measurement.** `dataexport stats` prints an "expression beats" block
  under the manners one, per avatar and in all: beats per sentence, the face
  and hand share of all pose targets, the share of beats with a head
  angle, pose markers per reply, replies with a beat. A *beat* is a pose
  marker whose targets are face variables (brow, eye, mouth, cheek, gaze)
  and optionally `head.*`, nothing else. With manners on the rates read
  "before manners -> after" (and "beats" in the json under `beats`, with
  `before`). Compare r1 and r2 by running it on both roots.
- **Export.** Manners caps pose markers at `--max-poses` (2), which would
  cut a reply of beats to two. `--max-beats N` caps the beats on their own
  and leaves `--max-poses` to the other markers. Default: 9 when the
  dataset's `avatars.json` carries the beats prompt, else 0, which is
  manners as before, so r1 exports (and the hash of a run such as q8m)
  do not change. The trainer takes the same flag (`--max-beats`). The
  default is 16 (was 12, 9 before): 3-4 sentences with 2 beats each are
  6-8 markers, and the prompt's dense example (below) is 8 beats in 4
  sentences; a long reply written like it (5-6 sentences) reaches 14, which
  12 would have cut at the tail, against the density asked for.

**Idle head drift.** Before, the calm idle held the head still between
fixations (holds of 3 s or more, `events.IDLE`), nothing moved it
continuously, so a quiet stream looked frozen. `flow.stream.HeadDrift` adds
a slow wander: per axis the sum of three sines (periods 4-10 s, seeded
phases), peaking at 2.5 deg yaw, 1.5 pitch and 2 tilt, and a breathing
sway of the body angles (`body_x`, `body_z`: 0.8 and 0.6 deg, periods 4-7
s) where the rig binds them. It is added to the head parameters before
moves and poses are applied, so face-only beats leave it running, and a
pose that targets the head moves from where the drift has the head to its
target and releases back to it (the held target keeps the idle's small
life under it, as for any pose). `flow.live` takes `--head-drift` (default
on), `--no-head-drift` and `--head-drift-scale X` (they sit in the
director's parser, which it shares). The offline director render has it on
by default too (`runtime --head-drift-scale`, which the director passes;
the runtime's own default is 0, so a bare `flow.runtime` run and the
dataset's trajectories and the action head's data are as before:
`MotionStream(head_drift=None)` is off and `dataset.director_args` turns
the director's flags off).

**Motion in bursts** (q8r2 demo review: "every figure moves in bursts, even
a swaying head"). Measured on `Mao_10/demo.mp4`, 34% of the 0.1 s bins had
almost nothing moving, between short spikes. Causes: (1) the offline render
had no head drift (it was only in `flow.live`), so between the calm idle's
fixations (3 s or more) the figure was still; (2) the model's beats are 0.15-0.3 s
eases to a held target, so a head beat is one short spike, then a hold with
nothing moving. Fixed in the playback only (targets and the onsets of the
model's markers are untouched; `--no-head-drift`, `--no-pose-ease` give
the old render):

- the drift above, now in the offline render (and a livelier period range:
  4-10 s, was 5-13 s, because near a sine's turning points all axes were
  slow together);
- `PoseLayer(ease=True)` (`--pose-ease`, default on in the director and
  `flow.live`; the dataset and the exporter keep the beats as written):
  head, torso and body angles take at least `pose.ANGLE_DUR` = 0.35 s to
  arrive and `ANGLE_RELEASE` = 0.7 s to relax; other (face) parameters at
  least `FACE_DUR` = 0.12 s and `RELEASE_MIN` = 0.5 s. All minimum jerk (an
  S-curve). The hold is as long as the marker wrote, from when the longer
  move is done; the held target keeps the `LIFE` and `SETTLE` it had (the
  drift continues underneath).

`python scripts/motion_still.py plan.npz|demo.mp4` prints the share of
0.1 s bins with nearly no motion (mouth, blinks and breath left out of a
plan; a video's differences include them) and the burstiness (p95 / median
of the bin speeds). Target for the offline path: under 10% still. On four 0.2 s head
beats over the calm idle (test rig, 10 s, mean of 8 seeds): still bins
23.0% before, 2.6% after (the idle alone: 38.8% before, 5.4% after);
burstiness 11.0 before, 6.7 after. Measure a real run with its `plan.npz`.

**Dense worked example.** Two pilots of the beats prompt gave 0.69 beats a
sentence, about 0.30 a second of speech, against the 0.5-0.65 a streamer's
face changes (every 1.5-2 s); stronger rule text barely moved it. The
placement examples of the beats prompt now start with one whole reply
(`pose._dense_example`, 4 sentences, about 10 s of speech, 8 beats, a beat
at nearly every clause), so the teacher has a density to copy, not only a
rule: brows up, eye smile, eyes closed in a laugh, gaze away and back,
brows knit with the mouth down, a blush, brows up on a question, a smile
for the last line, each with a small head move in the same marker and no
hand. It is built from `pose.offered(bm)` like the other examples, so a
variable the body lacks is left out; lafei (no `eye.*.smile`) gets a half-
shut squint (`eye.*.open` 0.45) for the smile beats and `eye.*.open` 0 for
the laugh; a VRM gets the same with its own variables. v1 is untouched
(tested by digest per body); the words "expression beats" stay in the
rule text, where `dataexport.BEATS_MARK` finds them. Hiyori's:

    [pose {"brow.L.lift": 0.7, "brow.R.lift": 0.7, "head.pitch": 4, "head.yaw": -4, "dur": 0.2, "hold": 0.4}] Oh, you really did that? [pose {"eye.L.smile": 0.8, "eye.R.smile": 0.8, "mouth.form": 0.9, "head.tilt": 6, ...}] That is the funniest thing, [pose {"eye.L.open": 0, "eye.R.open": 0, "mouth.form": 1, "head.tilt": -5, ...}] and I am not even sorry. [pose {"gaze.x": -0.6, "gaze.y": 0.2, "head.yaw": -6, ...}] Honestly, I was [pose {"gaze.x": 0, "gaze.y": 0, "brow.L.lift": -0.5, "brow.R.lift": -0.5, "mouth.form": -0.4, ...}] a little scared, [pose {"cheek.blush": 0.6, "mouth.form": 0.4, ...}] but it was brave, [pose {"brow.L.lift": 0.5, "brow.R.lift": 0.5, ...}] wasn't it? [pose {"eye.L.smile": 0.8, "eye.R.smile": 0.8, "mouth.form": 0.7, ...}] Now tell me the rest.

**Speech-driven micro-expressions** (`flow.micro`). Between the model's own
beats, a layer in the manner of the head drift adds small, short face
changes from the TTS audio. The audio features: the loudness envelope (RMS
in dB over 40 ms, smoothed over 120 ms) and, from `flow.beat2.prosody`
(YIN), the pitch's rise over its local median in semitones; their sum,
against the sentence's own median, has a stress peak where it tops 0.7 and
is a local maximum over +-0.2 s (at most 0.7 a second of voiced speech).

- **Brow**: a lift of 0.2-0.3 of the brow's range for 0.25-0.4 s at a stress
  peak (a little before it).
- **Smile**: `eye.*.smile` (else `mouth.form` up) at 0.25-0.35 of the range,
  easing in over 0.35 s so that it peaks as a sentence ends, gone in 0.4 s;
  not at a question (final pitch rise of 2 semitones or more). Hiyori's
  `mouth.form` already rests at its maximum, so she smiles with the eyes.
- **Blink**: an ordinary blink token at a pause of 0.35 s or more inside the
  speech and at its end, 2.5 s apart at the closest. `Autonomic` only had a
  Poisson blink (mean 4 s) and the requests of large gaze shifts; a request
  moves the next scheduled blink to the pause.

Rules: seeded and deterministic (a function of the audio, the seed and a
key); 0.8 s between two face events, also across sentences; a face event is
dropped when a model beat on the same variables (brows for a brow, eyes /
mouth / cheeks for a smile) is within 0.6 s before its onset or after its
hold (the model's beats win; `micro: ... N dropped for a model beat nearby`).
Each sentence logs `micro: N events (b brow, s smile, k blink) over X s of
voice`.

Played as `micro` tokens, parameter offsets that the body model solves once
per body (the canonical variable at its full range, scaled; face parameters
only, no arm drawings, nothing the lip sync owns; a variable the body lacks
is skipped). `flow.micro.MicroLayer` adds the eased offset to the frame
before moves and poses (`MotionStream.frame` and `runtime.main`'s loop), so
a pose on the same parameter takes over. The offline director writes the
tokens into `actions.json` (`micro`) and so into `tokens.json`: `--tokens-in`
replays them, and the live stream draws the same frames (tested).

Flags (the director's parser, shared by `flow.live`): `--micro` (default on),
`--no-micro`, `--micro-scale X` (amplitude, x the defaults; 0 is off). On by
default in the offline director render and in `flow.live` (`--avatar` live2d,
vrm and both) in body mode with a TTS; tag mode and a silent render have no
body model or audio, so none. The dataset's trajectories and the action
head's data never get it (`dataset.director_args` sets it off; only
`director.plan`, `vrm.plan` and `live.Performer` build a `Micro`).

### Framing and motion budget

*Camera follow and tall figures.* Two render rules sit on the framing
below. **Camera follows whole-body moves** (`framing.follow_tokens`): a
`[move]` of `dance`, `twirl`, `shimmy`, `bounce` or `cheer` in bust would
be cropped, so the camera eases to full 0.3 s before it (the usual 0.6 s
ease) and back to the framing it left 0.5 s after it ends (the turn's home
in the usual case); moves closer than that share one trip. `wave`, `clap`,
`sway`, `head_bob` and `look_around` are hand or head moves: no switch. Not
from full, and not when a `[frame]` marker of the model falls in the
window (from lead to tail): the model's markers win. The frame tokens are
made where the model's own are (the runtime's `--direct`, `flow.live`), so
both renders get it; it logs `framing: camera follows dance: full from ...`
when it triggers. `--no-camera-follow` (the director's parser, shared by
`flow.live`; the runtime's too) turns it off. **Hat window**
(`framing.bust_math`): the bust window's top is `ABOVE` hu over the chin
unless the figure is taller at rest (a witch hat): then it is the rest
render's alpha top plus 0.1 hu, at most `HAT_CAP` = 1.0 hu above `ABOVE`
(very tall ears do not shrink the face further). `BELOW` and the chin's
horizontal centring stay, and the prompt's camera line (hand visibility,
`BELOW`) does not change. Logged as `framing: bust, 2.60+1.0 hu ...`.


The r2 beats prompt first told every avatar to keep its hands "out of the
picture". That was true of one framing and of some bodies only: lafei's
resting hands (y -1.10 hu) are inside the bust picture, Hiyori's (-2.00)
are not, and a streamer switches scenes (face cam, full body for a dance).
The beats prompt (`v1` is untouched, byte for byte) now says how the avatar
is framed, lets it switch, and sizes its motion by what its body can do.
`flow.framing` has the numbers, `flow.pose` the wording.

- **Camera line.** The bust picture is the window of `framing.ABOVE` /
  `BELOW` (1.5 hu over the chin to 1.6 under it, as wide as it is high,
  centred on the chin). The system prompt (byte-stable per body, so the
  cached prefix and `avatars.json` stay as they are) says what each framing
  shows, worded from that window against where this body's hands rest and
  how low they reach (`pose.camera_rule`): Hiyori "your hands rest at y
  -2.00 and -2.09, below the picture, and a hand shows only once raised
  above y -1.60 ... in bust a hand move below y -1.60 is not seen: the face
  carries the performance"; lafei "in the lower part of the picture ... all
  your hand moves stay in that picture". The framing *now* is a line after
  `(your body now: ...)` in each user turn: `(camera: bust, head and chest;
  hands below y -1.60 are out of frame)` or `(camera: full, the whole body
  in frame)` (`pose.camera_now`). It is in `user_content`, so history,
  the SFT export and `evalsft` carry it for free; v1 data has no such line.
- **`[frame bust]` / `[frame full]`.** A marker like any (`[frame:full]`
  and quotes also parse; anything else is a warning "'frame' takes
  bust/full; dropped", counted as an unplayed `frame`). Only a parser built
  with `frames=True` takes it (`parser_for(..., frames=True)`, the
  director's `--body-prompt beats`, the dataset's beats runs, the export
  parsers): the v1 prompt and parser drop it as before. A VRM body never
  takes it (`framing.can_frame`): the browser page draws it at one camera
  (`vrmbody.camera`), so a VRM is always full and the prompt says so; the
  transition is not rendered there. The prompt lists it with the markers:
  "Go full only for a whole-body moment: a dance, showing an outfit or a
  pose, a big celebration with the body; not for a greeting wave or a single
  hand move ... eases over in about 0.6 seconds ... lasts until the end of
  this reply; the next reply starts back in your home framing. At most once
  in a reply." The pilot's teacher went full at the first hand moment (often
  a greeting wave) and stayed: Hiyori had 9 of 47 turns in bust although 70%
  of sessions start there, all 4 of her switches bust to full and none back.
  The camera line says the same in the other direction ("in bust a hand move
  ... is not seen: skip it and let the face carry the moment ... instead of
  switching to full"; "a switch lasts for this reply, and the next one
  starts back in your home unless you write the marker again") and the
  budget line adds "switch to full only for a whole-body moment ..., never
  for a single hand move". The home itself is not in the system prompt
  (byte-stable per body): the user turn's camera line shows it.
- **The ease.** A marker becomes a token `{"type": "frame", "view",
  "duration": 0.6}` at its time in the speech (`framing.frame_tokens`).
  `framing.Camera` holds the two `set_view` states (`bust_view` and the
  whole canvas) and eases scale and both offsets between them with minimum
  jerk from where the view is when the switch begins (a switch back mid-ease
  turns around smoothly); `Camera.at(t)` is pure in t. Live:
  `Performer.sentence` adds the tokens (stream time = the sentence's audio
  start + the marker's place), `MotionStream` commits them to its `camera`
  and `RenderLoop` calls `backend.set_view` before a frame whose view
  differs; `--renderer none` uses a stand-in zoom so the markers still run
  through the pipeline. Offline: `runtime --direct` appends the tokens to
  the directed ones (so `--tokens-in` replays them), the render subprocess
  builds the camera from them and sets the view per frame; the director
  starts a turn in `a.frame` (`--frame`, bust), and the next turn back in
  it (the home framing, below). Without a body frame (no `bust_view`) the
  render stays full and says so.
- **State across turns: a home framing.** A session or live chat has a home
  framing and every turn starts in it; a `[frame]` switch, to full or to
  bust, lasts until the end of its reply and the next turn is back home
  unless the model writes the marker again. Data: a session's home is bust
  with probability `--bust-share` (0.7; seeded by run seed and session id
  alone, `framing.start_frame`, so no other draw moves); each turn starts
  there (the camera line of `user_content` shows it), the teacher's markers
  move it within the reply, and each record has `framing: {start, home, end,
  switches}` (beats runs only; `start` equals `home`, `end` is where the
  reply left the camera). Offline director `--chat`: `--frame` is the home;
  `a.frame` is reset to it after each turn (the render of a turn starts in
  `a.frame` and replays its tokens). Live: `Performer.home` (`--frame`,
  full when the body has no bust view); when a reply ends away from home
  the Performer commits a `frame` token back home at the end of its speech
  (stream time `next_frame / fps`) and shows home at the next line.
- **Motion budget.** `framing.budget(bm)` measures the body: the share of a
  gesture zone (x 0.15-1.1 hu out from the midline, y from where the hands
  hang to 0.6 hu over the head, 8 x 6 points a side) that some arm drawing
  reaches, taking each drawing's per-variable ranges (`body.limits`) as a
  box, so it is an upper bound; the share of those points the bust picture
  shows; the lowest / highest hand y; the arm drawings; the face controls of
  `pose.offered` (brow, eye, mouth, cheek, gaze); the kind. Tiers: below 35%
  of the zone `limited`, from 80% `capable`, else `moderate`; the hand moves
  a reply per framing follow (limited 1 / 1, moderate 1 / 2 full, capable
  1 / 3 full; bust 0 when no reached point is in the picture). The prompt
  line (`pose.budget_line`) is e.g. "your arms are limited (a hand reaches
  about 14% of the places a hand goes when talking, through 2 arm drawings)
  and you have 16 face controls offered: carry the performance with the face
  and head, and move a hand rarely, at most 1 hand move in a reply". It is
  stored under `budget` (with the landmarks) in each avatar of
  `avatars.json` and printed by `dataexport stats` ("motion budget").
  Measured: Hiyori limited (14%, 38% of it in bust, 2 drawings, 16 face
  controls of 27), lafei limited (23%, 1 drawing, 10 of 20), Natori moderate
  (72%, 12 drawings, 15 of 28), seed-san capable (100%, 11 of 24); the same
  numbers for the other VRM samples. `python -c "from flow import framing,
  body; print(framing.budget(body.load('rigs/Hiyori.body.json')))"` checks
  one.
- **Manners for beats data.** The teacher now chooses its moments for a
  hand (a greeting, a celebration, a size, pointing, an emphasis), so for a
  dataset written with the beats prompt manners no longer strips unasked
  hand targets. `--max-hands N` (default 1 there; `-1` and other data: strip
  as before, same export and data hash as r1) keeps hand and arm targets and
  caps the markers that move a hand (a pose with a hand target, a `[move]`
  other than the head ones, a `[gesture]`) at N a reply where nothing was
  asked, the first ones; the later ones lose their hand targets or go.
  Shoulders and a big torso move still go. In bust (the record's start
  framing, followed along the reply by its `[frame]` markers) hand targets
  that would end out of the picture are stripped, asked or not: where the
  hand ends up (the marker's x and y, or the resting position for the one
  it leaves out; `@` references resolved with the avatar's landmarks)
  against the window. `frame` markers are kept, and a reply with only a
  frame marker still counts as having no body marker. The trainer takes
  `--max-hands` too.
- **Stats.** `dataexport stats` adds, for records with a framing, per avatar
  and per framing a turn starts in: turns, beats per sentence, face and hand
  share of targets, hand-moving poses and `[frame]` markers per reply, the
  switches (`framing` in the json); and the budget of each avatar (`budget`).
  `evalsft` counts `frame` markers per turn in its own column and leaves them
  out of the body markers. Against the session's home (`framing.home`; for
  records from before it, the framing the session's first turn started in)
  each framing row also gives the switches away from home, the switches back
  and the turns spent away (started away, or went away in the reply), and
  beats per second of speech (`timing.speech_seconds` of the record; in the
  json `away`, `returns`, `turns_away`, `timed_beats`, `speech_seconds`).
  The pilot's problem reads as many `away` and no `returns`.

### Reply language, new rigs, face vocabulary, r2 avatars (2026-10)

**Reply language.** A pilot had about 7% of teacher replies in another
language than the user's latest line (whole stretches of a session, after
the simulated user switched: `switch` / `mix` in `datagen.personas.json`);
the trained students did the same on about 11% of eval turns
(`flow.dataset.detect_lang`, mixed lines included). `scripts/lang_check.py`
lists them.

- *Prompt.* The beats prompt's rules gain (`pose.LANG_RULE`): "Answer in the
  language of the user's latest message, even if your earlier replies used
  another language. When the user mixes languages, answer in their main
  language (a mixed-in word or two may be mirrored)." It reaches the
  dataset teacher and the director / live character served with
  `--body-prompt beats`. `v1` and the director's tag prompt are untouched.
- *Export.* `dataexport stats` / `sft` and the trainer drop a turn whose
  reply language (`lang.reply`) differs from its user line's (`lang.user`),
  unless the record's `lang.mix` is set or either language was not
  detected. It is a hard flag (`language`) like the others: a session
  export is cut before the first one (the replies after it would carry the
  wrong language as history), a `--per turn` export leaves out the turns
  that would show it in their history. Stats and the sft command print
  "language: N turns dropped, by pair: en->ja 3, ...". `--lang-filter` /
  `--no-lang-filter` (also on `flow.train.sft`) force it; default: on for a
  dataset written with the beats prompt, off otherwise, so the r1 export
  and its data hash do not change.

**Rigs with no visible arm drawing / no chest.** `flow.body` measured every
arm variant by its visible drawables; a variant with none at rest (Lisette)
raised "need at least one array to concatenate". It is now skipped with a
logged caveat (`caveat: variant NAME has no visible drawable at rest`), the
rest are measured, and the file lists them under `skipped_variants`.
`flow.framing.hand_reach` read `landmarks["chest"]` even when a hand rests
somewhere (huonululu_5 has no chest landmark): it now goes through
`framing.chest_y`, which falls back to 0.4 hu under `shoulder.y` (else under
the default shoulder line, -0.25) and logs it once.

**Face-tracking rigs.** A VTube-Studio-style rig (八千代辉夜姬) offers 18
face controls, of which the body builder mapped 8. `body.DECLARED` gains,
read from the standard ids (underscores and case ignored, `body.ALIASES`
for the other spellings; sided ones follow the rig's side naming):

| variable | id | range | meaning |
|---|---|---|---|
| `eye.L.squint`, `eye.R.squint` | ParamEyeLSquint / RSquint | 0..1 | + squints |
| `cheek.puff` | ParamCheekPuff | 0..1 | + puffs the cheek |
| `mouth.pucker` | ParamMouthPuckerWiden | -1..1 | - purses the lips, + widens (as `mouth.form`: + is the smile side) |
| `mouth.funnel` | ParamMouthFunnel | 0..1 | + rounds the lips into an O |
| `tongue.out` | ParamTongueOut | 0..1 | + tongue out |
| `eye.tears` | ParamTear (ParamTears, ParamTearDown_1) | 0..1 | + tears |

`eye.tears` has the first of the rig's three `ParamTearDown_n` levels. They
are in `body.EXTENDED`: offered (`pose.offered`) and worded in the beats
prompt ("small tricks of the face", a squint for doubt) only for a body that
has them, and left out of the "does not exist on this body" line, so the
prompts of Hiyori, lafei, Natori and seed-san stay byte for byte (digests
pinned in `tests/test_facevocab.py`). They count as face for the beats
(`tongue.` joins `eye.`, `mouth.`, `cheek.` in the face prefixes of
`dataexport`, `framing`, `beat2`, and in the micro layer's smile group, so a
model beat on them keeps a micro smile away) and for the motion budget.
Also read from the ARKit-style names (EyeSquintLeft / Right, CheekPuff,
MouthFunnel, TongueOut, with or without `Param`). ARKit's MouthPucker is
not mapped to `mouth.pucker` (its sign is the other way round). Not mapped
(no canonical meaning yet): ParamMouthShrug, ParamMouthPressLipOpen,
ParamEyeSmile_Happy_*, ParamTearDown_2/3, ParamExpression_*. A body file
built before this change does not have them: rebuild it.

**r2 avatars** (Hiyori, lafei, Mao, huangjiafangzhou_3, 八千代辉夜姬,
seed-san). The default `--avatars` stays `Hiyori,seed-san,lafei`. The body
files of Mao, huangjiafangzhou_3 and 八千代辉夜姬 are only on the CPU
server: copy `rigs/NAME.body.json` (and the rig `.npz`) to the machine that
generates, or the run stops at the start with "avatar NAME: no body file
...". Rebuild 八千代辉夜姬's body for the new variables:

    python -m flow.body --model-json M --rig rigs/NAME.npz --measure

then

    python -m flow.dataset gen --avatars Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san \
        --teacher-prompt beats --out data/datagen/r2 ...

(NAME entries take `NAME:model=PATH,rig=PATH,body=PATH` where the defaults
do not find them.) 八千代辉夜姬 and the Azur Lane rigs are for training and
tests only, never streamed.

### beats2 (r3 data)

q8r2 (trained on r2, served with `--body-prompt beats`) showed in its eval
and 30 demo clips: about 20% of replies open with exactly the one dense
example of the prompt (brows 0.7, pitch +5, yaw -5, dur 0.2, hold 0.4) and
about 40% with a brow lift + pitch + yaw; dances played in bust without
`[frame full]` while `[frame full]` went to a single wave; about 8% of the
teacher turns answered in another language than the user's latest line. The
`beats` prompt stays byte for byte (q8r2 is served with it; the digests in
`tests/test_facevocab.py` pin it). The changes are a new prompt version,
`beats2` (`pose.PROMPTS`), which is now the dataset default. After the
second live trial (left and right mirrored in 4 of 6 look requests, a wave
request without a wave) beats2 also has two request topics of its own in
`flow/datagen.personas.json`, `look` and `wave` (`"prompts": ["beats2"]`:
v1 and beats draw the old topic list, so r2 regenerates as before; `wave`
is in `dataexport.ASK_TOPICS`), and its look rule gives the sign: "look to
your right" is a negative `head.yaw` / `gaze.x`. The rest of the changes:
`--teacher-prompt beats2` (`beats` regenerates r2, `v1` r1), served with
`--body-prompt beats2` (`flow.director`, `flow.live`; `flow.train.evalsft`
replays the recorded user turns with the registry's prompt, so it needs no
flag). `dataexport` recognises it as beats data (the prompt says
"expression beats"): the same defaults, manners cap `--max-beats`, hands
cap and language filter.

- *Openings.* Six dense examples (`pose.dense_variants`, built per body from
  `pose.offered` like the old one), each eight beats in four to five
  sentences with its own numbers and order. Hiyori's first beats: an eye
  smile with a tilt ("Hehe, you remembered?"); a glance away (gaze.x -0.5,
  yaw -7); a laugh with closed eyes (eye.*.open 0, mouth 1); worried brows
  with a pout (brows -0.6, mouth -0.5); a blush with chin down; a calm slow
  tilt (dur 0.45, hold 1). The two short examples rotate with the variant
  too. The data picks one variant per session from the session's seed
  (`pose.variant_for`; reproducible, recorded as `avatar.variant` in each
  turn record, and every variant's prompt is in `avatars.json` under
  `PERSONA#K`); a served prompt is variant 0 (`system_prompt(variant=0)`),
  and the student saw all of them. One more rule: "Vary how your replies
  open: never start every reply with the same expression, and do not take
  the opening of an example; begin with what this line makes you feel."
- *Camera.* The `[frame]` entry now says: "Write [frame full] right before a
  dance or another whole-body move ([move dance] ... the ones
  `framing.WHOLE_BODY` has that this body does) and before showing an
  outfit, so the whole figure is in the picture; never for a wave or a
  single hand move, which stay in bust." The move example shows
  `[frame full][move ...]`. (The runtime follows whole-body moves anyway;
  this keeps the data consistent.)
- *Language.* Every user turn of beats2 ends with the reply language next
  to the camera line, from the user's latest line:
  `<line>\n(your body now: ...)\n(camera: ...)\n(reply in: English)`
  ("Japanese", "Chinese", "Korean"; left out when the line settles nothing,
  like "OK!"). One function writes it everywhere, `pose.turn_tail`
  (`pose.detect_lang`, which `dataset.detect_lang` is: the main language of
  a mixed line, so English with a kana word is English). The dataset
  (`user_content`), the director (`--say`, `--chat`, VRM too) and the live
  loop call it, so training and serving match exactly. The rule in the
  prompt points at it.
- *Fewer switches.* `datagen.personas.json` is not edited (r2 reproduces
  with it); `--lang-switch P` sets the chance a turn's language is drawn
  afresh, default `dataset.LANG_SWITCH` = 0.03 for beats2 (the personas'
  own 0.10-0.15 for `beats` and `v1`). `mix` is unchanged and each persona
  keeps its `languages` weights, so the balance stays about 40% en, 37% ja,
  23% zh over a run. `dataset.run_opts` reads a run that did not record
  `teacher_prompt` as `beats`.

- *Quiet chat.* The first live trial of q8r2 showed a student that only
  speaks when spoken to. Live now sends `(chat is quiet)` after a while
  without a message (below) and the simulated user does the same on 10% of
  the turns after the first (`--quiet-share P`, `dataset.QUIET_SHARE`; 0
  none; beats2 only, the older prompts regenerate as they were). One
  constant, `pose.QUIET_CHAT`, and one function, `pose.quiet_user`, write
  the line in both. The turn is not a user line: no simulator call, topic
  `quiet`, `quiet: true` on the record, `lang.asked` / `lang.user` the
  session's current language (the language of the last real line), and the
  `(reply in: ...)` line read from that line (`turn_tail(...,
  lang_text=)`). The draw comes from its own generator
  (`dataset.quiet_turn`, from the session seed and the turn), so nothing
  else of a session moves. `dataexport` keeps these turns: `dedupe` leaves
  them out (the line is the same by design) and `asked_for` says they asked
  for nothing, so the manners strip unasked hands as for any chat turn. The
  prompt gets, in its rules (beats2 only): "A user message that is just
  "(chat is quiet)" means nobody has written for a while: keep talking as a
  streamer does when the chat is quiet. Start a small topic of your own,
  react to something you said or saw, ask the viewers a question, or carry
  on with the last topic. Keep it short and natural, and never say that the
  chat is quiet."
- *Looks hold.* Two sentences under the head move: "Your head stays at the
  angle a marker gave it until your next marker sets that angle again (and
  settles back when your reply ends), so tilt from one side to the other as
  you talk and do not count on the idle to bring it back." and, from the
  variables the body has, "When the user asks you to look or turn somewhere
  (left, right, up, down, at the camera), do it in one marker that holds
  that direction: "head.yaw" and "gaze.x", with "hold" 2 to 4. The head stays
  turned until your next head move or the end of the reply, so do not glance
  back after a moment." They describe the playback of the next section; the
  data itself is still played by the dataset as before (it does not carry:
  the trajectories and the action head's targets are unchanged).
- *Longer sessions.* Nothing caps a session: `--turns` / `TURNS` is the
  session length, the personas and topics have no turn count, the
  simulated user sees the whole transcript, and the character sees every
  earlier turn. r3 runs 10 turns, 1600 sessions (16000 turns; r2 was about
  15000). The live window (`--history-turns 5`) is shorter than the
  longest history the character saw (9 exchanges) and as long as a
  6-turn session's, so a request to the student never holds more history
  than it trained on. A session example of 10 turns is about 1.7x a
  6-turn one: `train_sft --max-len auto` (99th percentile, 256-token
  steps, at most 16384) cuts the rest after the last reply that fits.

- *Requests win (rev 2).* The first 675 sessions of r3 (API credit ran out)
  failed `body_request_obeyed` 35% and `wave_obeyed` 21%. Of 25 failing turns
  about 15 were checker false positives (invitations, teases, a compliment
  on a peace sign photo, a look request counted by topic), and the real
  skips were almost all in bust: the camera line said "a hand move below y
  -1.00 is not seen: skip it and let the face carry the moment", which the
  teacher applied to "point at the tree" and "make a peace sign" too.
  `flow.handask` is the new request detector (en / ja / zh; shared by
  `flow.checks` and the export; `dataexport.asked_for` is unchanged, it
  decides what the manners keep). beats2 now says "an unasked hand move
  below y ... is not seen: skip it" (`camera_rule(bm, True)`) and ends its
  rules with `pose.request_rule`: when the user asks for a hand or arm
  action, always do it with the hand in that reply, whatever the budget says
  (pointing is a hand pointing, the eyes may follow but not replace it; a
  peace sign is the hand up); in bust raise the hand into the picture, above
  y -1.00, or write `[frame full]` first when it cannot be seen there; a
  limited rig does the closest its arm can; a body whose hands reach nothing
  says so naturally. The short student prompt carries the same. `beats` is
  byte for byte as before (`pose.BEATS2_REV` = 2; `system_prompt(...,
  rev=1)` rebuilds the first wording).
- *Skipped requests are dropped.* `dataexport --request-filter` [on for
  beats2 data, `--no-request-filter`; r1 and r2 exports unchanged]: a turn
  whose line explicitly asks for a hand action, of a body whose budget has
  `hand_moves.full` above 0, and whose reply has no hand target, `[move]`
  or `[gesture]` gets the hard flag `skipped_request` and is dropped like a
  wrong language (a session export is cut before it, a `--per turn` export
  leaves out the turns that would show it in their history); `stats` and
  `sft` print "requests: N turns dropped (skipped hand request)".
  `flow.train.sft` has the same flag.
- *Resuming r3 after the prompt changed.* `flow.dataset` registers the new
  prompt's sha in `avatars.json` next to the old one (`earlier` lists the
  old ones), new sessions carry the new `prompt_sha` and the 675 old ones
  keep theirs. The export, `evalsft` and `hidden` look the prompt up by each
  record's sha; `dataexport.student_system` rebuilds the teacher prompt to
  check it byte for byte and now tries each wording revision (newest
  first), so both versions of an avatar and persona give a short prompt.

r3 generation, the r2 avatar list, 10 turns a session, 1600 sessions (about
the turns of r2):

    SESSIONS=1600 TURNS=10 CONC=8 BUDGET_USD=20 OUT=data/datagen/r3 \
    AVATARS=Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san \
        bash scripts/datagen.sh

or `python -m flow.dataset gen --avatars Hiyori,lafei,Mao,huangjiafangzhou_3,八千代辉夜姬,seed-san
--sessions 1600 --turns 10 --teacher-prompt beats2 --out data/datagen/r3 ...`
(`--teacher-prompt beats2` is the default; the personas file is the same).
Export and train as for r2; serve the student with `--body-prompt beats2`.
八千代辉夜姬 and the Azur Lane rigs are for training and tests only.

### Live after the first q8r2 trial (2026-10)

The student (tsk-q8r2, Mao, `--body-prompt beats`) was talked to for 19
turns. From turn 12 every reply was the same three beats, requested hand
poses did nothing, the first-text latency grew from 0.9 to 2.7 s, the face
went blank between beats, Mao seemed to look left all the time, a requested
look right lasted under a second, the character never spoke unprompted, and
her head tilted in discrete steps. What changed, all in `flow.live` (and
the offline director where it plays the same tokens), none of it in what
the dataset and the action head learn from:

- **History window** (`--history-turns K`, default 5; 0 keeps all). The
  system prompt and the last K exchanges and the current line are sent: a
  request is at most 6 turns, as long as the sessions of r1/r2 (6) and shorter
  than r3's. The model copies its own earlier replies once the chat is
  longer than anything it trained on. `Performer.trim_history` drops the
  older messages for good and logs `history trimmed to the last K
  exchanges (n older messages not sent)`.
- **Context budget** (always on; `--ollama-ctx N` is the window). The
  K-exchange cap is not enough for long, marker-heavy replies: a live run
  died after a prompt of 7230 tokens in an 8192 window (the Modelfile's
  `num_ctx`). `Performer.trim_context` also drops the oldest exchanges until
  the estimated prompt fits `window - reply reserve (1024, or `--max-tokens`
  if smaller) - 256`; never the system prompt or the current line. The window
  is `--ollama-ctx`, else the model's `num_ctx` from Ollama's `/api/show`,
  else 8192. Tokens are estimated from characters: 2 characters per token
  before a count is known, then the ratio of the last reported
  `prompt_eval_count` to the characters sent (it only ever goes down, as
  Ollama may leave cached prompt tokens out of the count). Logged like the
  history trim, with `(context budget: ~N of M tokens, window W)`; a reported
  prompt over 85% of the window logs a WARNING. To keep more history raise the
  window: `--ollama-ctx 16384` fits the 3090 Ti (and the Modelfile's num_ctx
  should match).
- **Backend watchdog** (`--first-token-timeout S`, default 20; `--stall-timeout
  S`, default 30; 0 disables each). A reply that gives no token in S seconds is
  abandoned (its HTTP response closed where the backend can: Ollama's
  `cancel()`), logged `no token from the backend in 20 s; retrying`, and made
  once more; a second stall drops the turn (its user line is popped from the
  history, nothing is added, the loop goes on). A stall between tokens
  mid-reply ends the reply where it is: what was said and the words still held
  by the sentence cutter are spoken and kept in the history. Each try has its
  own queue, so an abandoned thread can never feed the next turn. With
  `--backend ollama` live also uses `--ollama-timeout 120` (socket, per read;
  other callers keep 300) and `--ollama-keep-alive 30m`.
- **Backend warm-up** (`--no-warmup` skips it; `--warmup` is the physics one).
  Before serving, an Ollama backend gets a load-only request (no messages) so
  the first turn is not the ~9 s cold load; the time is logged, a failure only
  warns.
- **Chat batching.** Lines that queued up while she was talking are one turn
  (`Performer.next_turn`): oldest first, one per line (`"A\nB\nC"`), no
  names (the inbox holds plain text), the last `--batch-max 5` kept and the
  dropped count logged. The student's data has single-message turns only (one
  to three short sentences), so this is a small step off the data; a quiet
  turn happens only with an empty inbox, a real message always wins. The user
  event and recorded turn carry `batched: n`.
- **Head carry** (`--head-carry`, default on; `--no-head-carry`). A pose
  used to hold its target, then release to the idle's value, so each beat
  went out and back to centre. A beat's head and gaze angles now stay where
  it put them: `pose.tokens(..., carry=True)` names them in the token
  (`carry`) and `PoseLayer` holds them until the next token that sets that
  angle, which blends straight from there (min-jerk, the pose-ease times),
  or until an end token. Live sends the end token (`pose.release_token`) at
  the end of each reply's speech: the carried angles relax over
  `CARRY_RELEASE` = 1.5 s (never under 1 s); the baseline below starts to
  fade. Without an end (the reply failed, a quiet stream) they relax after
  `CARRY_MAX` = 20 s. HeadDrift stays under all of it. Carried: pitch and
  tilt always; yaw, gaze.x and gaze.y unless the beat is a reaction
  (`|yaw| <= 6` degrees, `|gaze| <= 0.4`, hold <= 0.6 s), which relaxes
  after its hold as before. The q8r2 replies open with a yaw of -4 or -5 (20
  of the 21 replies in `runs/demo_q8r2` that open with one; 11 -5, 9 -4,
  mean -4.1); carried, that would have been a lasting turn. A look
  (`head.yaw: 25, hold: 3`) stays to the end of the reply and relaxes over 1.5
  s. A reaction in the middle of a look relaxes to the idle, not back to the
  look. The offline director and the VRM render take the same flags
  (`--head-carry` is passed to `flow.runtime`, which writes the end token at
  the speech's end into the tokens); the dataset sets it off.
- **Expression baseline** (`--baseline-scale X`, default 1; 0 is off).
  Each face beat leaves a resting share of itself: `BASE_FRAC` 0.375 of
  how far it moved the face parameter from where the idle had it (`face_params`:
  brows and the unbound eye, mouth and cheek shapes; not angles, hands,
  switches or openings), 0.25 to 0.5 with the reply's mood (`pose.valence`
  of each beat, from its smiles, mouth form, blush and brows; a beat that
  agrees with the mean of the reply so far keeps more, one that does not
  keeps less). It is reached by the end of the beat's hold, kept to the end of the reply
  (`BASE_HOLD_MAX` = 8 s with no end token) and faded over `BASE_DECAY` = 5 s,
  or replaced by the next beat on that parameter, which starts from where
  the face is. It is added to the frame after the drift and before the
  micro-expressions and the beats (`PoseLayer.base_apply`), so both ride on
  it and a beat's release returns to the resting face, not to a blank one.
  Scale 0 is the frames of before, bit for bit.
- **Quiet chat** (`--quiet-after S`, default 20; 0 is never; `--quiet-max
  N`, default 3). S seconds after a reply's speech ends with nothing waiting
  the Performer sends the model `(chat is quiet)` (`pose.QUIET_CHAT`, the
  string the dataset's simulated user sends) with the usual body, camera and
  `(reply in: ...)` lines, the language being that of the last real message.
  At most `--quiet-max` in a row; any real message resets the count, and a
  message that arrives while a quiet turn is spoken waits its turn as
  ever. The voice keeps the conversation's language (a quiet turn does not
  re-detect it from the English line). Logged as `live: turn n: chat quiet
  S s after the last reply; quiet turn i/N`; the page's user event and the
  recorded turn carry `quiet`. The student of r2 (beats prompt) has never
  seen the line; it is only meaningful with the beats2 student.

Why Mao looks to the left. Measured with the real Mao rig (the Cubism
sample, `rigs/Mao.npz`, the calm idle, no GL) through `MotionStream`, 40
seeds of 60 s each, drift on: mean head yaw (ParamAngleX) +0.37 degrees
(the spread between seeds is 2.65, 22 of 40 negative), mean eye-ball X
(ParamEyeBallX) +0.002 (22 of 40 negative), pitch +0.40. The idle has no side.
Mao's own motions do have one (mean ParamAngleX: mtn_04 -4.1, special_01
-9.8, special_02 -20.4, special_03 -5.7; EyeBallX -0.1 to -0.22) but the
gesture library keeps out every parameter a role is bound to, head and
gaze included (checked: none of the two idle gestures writes either), so none of it
reaches the head. What does have a side is the student: in the 30 replies
of `runs/demo_q8r2`, 43 beats set a head yaw, 58% negative, mean -0.72
degrees; 20 of 21 first beats of a reply have -4 or -5. `head.yaw` is
positive toward the character's left (the viewer's right), so -5 is a turn
toward her right, the viewer's left: the same screen side every reply, a
brief turn that starts every answer. The scripted Mao_14 reply (yaw -4,
none, +5, 11 s of speech) averages -0.34 degrees of yaw over its speech
under either playback; in a reply with only the opening yaw -5 the 8 s after
the beat average -0.69 degrees (12 seeds, the idle's own spread) because the
reaction relaxes, where a carried -5 would have stayed at -5. A requested look right (yaw +25, hold 3) was at
+24 at 2 s and back at -0.8 by 5 s; it is now +25 until the end of the
reply at 9 s and back by 11 s.

### Memory (live, 2026-11)

In a long session she forgets. In a number game "15 is too big" had left the
5-exchange history by the time it mattered, and she kept saying うますむめ
for ウマ娘 after being corrected, because her own wrong replies stayed in the
history while the correction aged out. `flow/memory.py` keeps a small state
beside the history. It is modelled on the DeepSeek Harness memory plugins:
curated small facts, a cheap gate and a restricted consolidator after the
turn (deepseek-memory); a structured summary in place of dropped turns
(compaction-basic); injection near the tail, so the prefix cache holds
(dsh-agent-memory). The student was never trained with it: it is read as one
more context line, like `(camera: ...)`.

Flags (`flow.live`, off by default):

- `--memory off|notes|summary`: `notes` is the three sections below;
  `summary` adds a rolling summary of the exchanges `--history-turns` and the
  context budget drop.
- `--memory-backend deepseek` (also siliconflow, vllm): the consolidator's
  OpenAI-compatible service, built apart from the student's backend (it does
  not inherit `--api-base`, `--max-tokens` or `--thinking`); the key is read
  from `DEEPSEEK_API_KEY` (checked at start). `--memory-model` (default the
  backend's: deepseek-chat), `--memory-chars 800` (cap on the whole block),
  `--memory-timeout 20`, `--memory-place tail|head`.

The notes (kept in the Performer, written as JSON to `<record>/memory.json`):

    {"game": ["number game 1-20, she guesses: 10 too small, 15 too big"],
     "corrections": ["ウマ娘 (umamusume), never うますむめ"],
     "facts": ["viewer Ken likes cats"],
     "summary": ["Ken introduced himself, we played rock paper scissors"]}

Each entry is one plain line of at most 140 characters (a longer one is
rejected, never cut); game keeps 3 entries, corrections and facts 8, the
summary 3 lines and 400 characters. The rendered block stays within
`--memory-chars`: over the cap the oldest entries go first, in the order
summary, facts, game, corrections (corrections last: they stay unless the
viewer contradicts them). Brackets, braces, angle brackets (full-width too)
and anything between square brackets are removed from every note, and `;`
`|` and line breaks are replaced, so a viewer cannot plant a `[pose ...]`
or `[move ...]` marker in her context through a note.

The gate (`memory.gate`, local, no model call) runs after every completed,
non-quiet turn and asks for a consolidation on: an explicit remember request
(remember that, don't forget; 覚えて; 记住), a correction
(`flow.checks.corrections` against her previous reply), a game or number cue
(any digit, too big / small, your turn, rules, score, guess; 大きい 小さい
当てて ゲーム; 大了 小了 猜 游戏), a name introduction (I'm Ken, my name is,
call me; 私はケンです; 我叫), a decision or promise (let's, I'll, I promise,
tomorrow; 約束 明日; 说好 明天), and every 6th turn anyway. In `summary` mode
a turn that dropped exchanges also asks, to fold them.

The consolidator runs in a background thread after the turn, so it never
delays her reply or the next first token. It is given the current notes
JSON, the last 4 exchanges (markers stripped) and, in `summary` mode, the
dropped exchanges, and answers with the new notes as one JSON object. A
reply that is not that schema, a backend error or the timeout keeps the old
notes (logged). One call is in flight at a time: gates that fire meanwhile
are coalesced into one more run on the latest exchanges. Until a result
lands the turns use the notes they have.

Injection: the block goes at send time on the last line of the latest user
message only, after the `(pose now ...)`, `(camera: ...)` and
`(reply in: ...)` lines of a body-mode turn:

    is it 12?
    (pose now: ...)
    (camera: full)
    (reply in: English)
    (stream notes: game: number game 1-20, she guesses: 10 too small, 15 too
    big; corrections: ウマ娘 (umamusume), never うますむめ; facts: viewer Ken
    likes cats; earlier: Ken introduced himself, we played rock paper scissors)

Why the tail and not the head: the user line stays the first line (the system
prompt's `(chat is quiet)` rule names that message by its exact text), the
student reads parenthesised lines after the user line as context (it was
taught `(pose now)`, `(camera)`, `(reply in)` there and never speaks them),
and the newest tokens are the ones a long-context model attends to best.
`--memory-place head` puts the block first, to compare. The history stores
the plain user text; the block is added to a copy of the request, so older
user messages never carry it, the prefix stays the same from turn to turn,
the system prompt (and its Ollama prefix cache) is untouched, and empty notes
add nothing. The context budget counts the block (`Performer.note_block`,
one snapshot per turn).

Data for later: every consolidation is a `{"kind": "memory", "turn", "reasons",
"before", "after", "diff", "ok", "seconds"}` line of `turns.jsonl` (a failed
one has `"ok": false` and `"error"`), and a `memory: ...` line of the log.
`flow.checks.load` skips those lines. Try it:

    export DEEPSEEK_API_KEY=...
    python -m flow.live ... --memory summary --record ~/runs/mem1

### Live history: markers, chunked trims, trailing poses, glances (2026-11)

From a live q8r3 session on Mao (`--history-turns 12 --ollama-ctx 16384`).

**Old markers (`--history-markers N`, default 2).** From turn 12 to 30 almost
every reply used the same four-pose sequence, copied from her own history.
What is sent now keeps the `[pose ...]` markers only in the last N assistant
messages; older ones are sent as their spoken text (the turn's own parser,
`Performer.spoken_of`). The stored history is unchanged (the strip is done
at send time, `sent_view`), and `trim_context` counts what is sent. 0 keeps
every marker. The boundary does not slide by one reply per turn (that would
change a message already in the prefix cache every turn): it moves when the
messages that still have markers reach N + `--trim-chunk`, and at every trim,
so a message is sent with the same bytes for turns on end.

**Chunked trimming (`--trim-chunk C`, default 4).** First text took 0.27-0.41
s on turns 12-13 and 2.35 s from turn 14, exactly when "history trimmed"
started: dropping the oldest exchange every turn moves the start of the
prompt, so Ollama's prefix cache is useless every turn. Over the
`--history-turns K` cap the history now drops down to K - C + 1 exchanges
(C is at most K/3, so K=5 still trims one at a time; C=1 is the old
behaviour); over the context budget it drops until C - 1 more average
exchanges fit. The log says `history trimmed to the last 9 exchanges (8
older messages not sent; a chunk of 4 exchanges, the next trim in about 4
turns, the sent prefix unchanged until then)`.

How the prefix stays byte-identical between trims: the request is the system
prompt, the stored messages, the current user message. (1) The stored
messages only grow at the end between trims. (2) Markers are stripped by a
boundary that moves only every ~C turns and at a trim (a trim moves the
prefix anyway). (3) The notes (`--memory`) and nothing else are added to the
current user message, as its last line, so in the next turn that message is
its plain text again, which is a byte prefix of what was sent: the cache is
valid up to the end of the text. A test (`tests/test_history_glance.py`)
asserts that, over 34 turns with markers, notes and a window, the sent
messages minus the last exchange equal the previous request's, except at the
marker boundary / trim turns (at most one in four).

**Trailing marker-only segment.** A reply that ended on a pose with nothing
after it (`turn N.4 ... plan 0.00s: `) planned an empty sentence for it.
With the head carry or the expression baseline, the end of the speech
releases both at once, so such a pose is a flicker: it is dropped (logged).
Without them it holds: it is merged into the previous sentence (applied at
the end of its speech) and logged as "trailing markers merged", never as a
sentence of its own. A reply of markers only is planned as before.

**Glance reflex (`--glance auto|off`, default auto).** When the user line puts
something near her without asking her to look ("I put the candy basket at
your right foot", "behind you", "on your left"; ja 足元, 右足, 後ろ, 左側に
置いた; zh 你脚边, 你右边, 身后), her reply starts with a glance there:
`flow.checks.mentioned_place` reads the line (not an explicit look request,
not the user's own side: "my right hand hurts", "turn right at the station",
"on my right"), `glance_targets` gives the targets (feet: head.pitch -12 and
gaze.y -0.5; a side: head.yaw +-14 and gaze.x +-0.5, her right negative;
above: up; behind: over her right shoulder, yaw -25), restricted to the
variables her body has. It is a pose beat inserted before her first pose in
the first sentence (dur 0.3, hold 0.45, 0.75 s): the pose solver starts her
own pose on the same variables when its hold ends; with the head carry a
pose sending the glanced variables home is added. It is skipped when her
first pose already looks that way. Log: `glance: turn 3 at her right, down:
{...} for 0.75s, then her own first pose`.

**Training side.** The model should do this unprompted:
- checker `glance_at_mention` (`flow/checks.py`, `scripts/checks.py
  --checks glance_at_mention`): on such lines the first or second pose marker
  must move head or gaze toward the place (fail: no_pose / no_glance);
- DPO kind `glance_at_mention` (`flow/dpo.py`, in `--kinds`): chosen opens
  with the glance beat added to the teacher's reply when it has none,
  rejected is the teacher's reply as it is; when the teacher already glances,
  rejected is that reply with the signs reversed; speech identical;
- datagen topic `place` (`flow/datagen.personas.json`, beats2 only like
  `look` and `wave`): the user puts or points at things around her without
  asking her to look;
- the beats2 prompt, revision 3 (`pz.BEATS2_REV`, `GLANCE_RULE`): one line on
  opening with a glance at a mentioned place. Revisions 1 and 2 are byte for
  byte as before (`system_prompt(..., rev=2)` rebuilds them, and
  `flow.dataexport` tries 3, 2, 1 for a resumed run); data generated from now
  on is rev 3.

### The short student prompt (r3 students)

The beats2 system prompt is 12-14k characters, most of it the same for every
avatar and every session (the beats rule, the dense examples, the style
rules), which a student learns into its weights anyway and then re-reads on
every turn. `short` (`flow.pose.short_prompt`) is a student-only prompt that
keeps what varies and drops the rest. It is never a teacher prompt:
`flow.dataset` refuses it (`--teacher-prompt` does not list it;
`pose.check_teacher_prompt` raises for a programmatic caller), so
`pose.PROMPTS` stays the three teacher prompts and the pinned digests of
`v1` / `beats` are untouched; `pose.STUDENT_PROMPTS = ("short",)`.

Characters, same persona, the `[move]` menu on, no gestures (the live
prompt of a rig adds its gesture entry to both):

| body | beats2 | short | share |
|---|---|---|---|
| Hiyori | 13848 | 6595 | 48% |
| lafei | 13015 | 5985 | 46% |
| Natori (a Cubism sample; Mao's body file is not in the repo) | 14126 | 6835 | 48% |
| seed-san (VRM) | 12439 | 5716 | 46% |
| face-tracking body (`tests/test_facevocab`) | 12277 | 5989 | 49% |

The body description (3.5-4.5k) and the camera rule (about 1.2k) are kept
byte for byte, which is why the share stays near half; the rest is 1.3k.

- **Kept:** the persona; `describe_body` in full (variables, units, ranges,
  limits, "left and right are always your own", hand and arm notes); the
  motion budget line (`budget_line(bm, True)`); the camera rule
  (`camera_rule`); the marker syntax in compact form, one tiny example each
  (`[pose]` made of variables this body has, `[move]` with the primitives
  this body does, the rig's `[gesture:NAME]` entry as given, `[frame]`,
  `[blink]`, `[pause:0.5]`); the meaning of the user turn's lines
  (`(your body now: ...)`, `(camera: ...)`, `(reply in: English)`,
  `(chat is quiet)`).
- **Dropped:** the long beats rule, the dense and placement examples, the
  `[frame]` paragraph (a line is left), the general style rules. In their
  place a "How to move" list of a few lines:

  ```
  - Expression beats in every sentence: a quick face marker with a few
    degrees of head, right before the words it reacts to (dur 0.15 to 0.4,
    hold short); vary them and how your replies open. The rest of the body
    only when it fits or is asked.
  - Asked to look or turn: one marker holding it, "hold" 2 to 4. Sides are
    your own: "to your right" is a negative "head.yaw" and "gaze.x" (about
    -15 and -0.6), "to your left" a positive one.
  - (camera rule) (motion budget)
  - Listed variables only, inside their ranges, valid JSON; no other markup
    or emoji; never mention the markers.
  ```

  (The look line is written only for bodies with `head.yaw` / `gaze.x`.)
- **Parsing and turns:** `short` is `beats_like` (frames on, the beats2
  parser) and its user turns are beats2's: `turn_tail` writes the same
  camera and reply-language lines.
- **Export and training:** `python -m flow.dataexport sft DATA --out F
  --student-prompt short` and `python -m flow.train.sft train|stats --data
  DATA --student-prompt short` (default `teacher`: unchanged). Each example's
  system prompt is rebuilt by `dataexport.student_system` from the avatar's
  body file (`avatar.body`, found as given or under the repo) and from what
  the teacher's prompt in `avatars.json` was written with: the persona (its
  first paragraph), the `[gesture:NAME]` entry (verbatim) and whether the
  `[move]` menu was on. It first rebuilds the teacher prompt from those and
  the record's `variant`, and stops if it is not the registry's bytes (the
  body file or `flow.pose` is not the one the data came from). User turns,
  history, replies, manners and the language filter are as before, and the
  beats defaults (`default_max_beats`, `default_max_hands`,
  `default_lang_filter`) still read the teacher prompt in `avatars.json`, so
  the data is still recognised as beats data. beats2 data only, and a dataset
  folder (an `sft.jsonl` holds its prompts already). The trainer's data hash
  includes the choice (`data_hash(rows, student_prompt)`; unchanged for
  `teacher`) and `data.json` records it: another prompt is another run.
- **Serving:** `--body-prompt short` on `flow.director` and `flow.live`
  (choices v1, beats, beats2, short), with the same persona, gestures and
  `--body-moves` the data had. `flow.train.evalsft run --student-prompt
  short` sends the rebuilt prompt on the same held-out turns;
  `STUDENT_PROMPT=short scripts/eval_sft.sh` passes it to the student run
  only (the recorded and teacher runs keep the data's prompt).
- Untested on a real run: whether a student reads `short` as well as the full
  prompt; a head-to-head eval of the two is the check.

### Repair layer (2026-11)

`flow/repair.py`, wired into `flow/live.py` (`--repair auto|log|off`, default
`auto`; body mode). The design brief is `docs/research.md` section 4: a
checkable failure is fixed deterministically, never with a second model call
(live speech cannot wait for a re-ask), logged, and turned into preference
data. The checkers are `flow/checks.py`'s own functions, run on the user line
and the reply so far.

**Per sentence.** A reply streams as sentences (`SentenceCutter`);
`Performer.sentence()` hands each one to `repair.Turn.sentence()` before its
markers are parsed and planned. The Turn keeps the sentences already
performed; the checker sees `performed + this sentence`, and a fix may touch
this sentence only. A failure whose marker is in a sentence already spoken is
logged (`not repaired: already spoken`), never edited. The same hook covers
the merged trailing markers and the stall path. Quiet-chat turns, a body
mode off and `--repair off` skip all of it.

| check | what it does | repaired? |
| --- | --- | --- |
| `look_sign` | the user asked for a side and the first turning marker (head.yaw >= 7 degrees or gaze.x >= 0.3) of the sentence has the wrong sign: head.yaw and gaze.x there get the asked side's sign, magnitudes kept (her right is negative; a marker whose two disagree comes out agreeing) | yes |
| `look_at_me` | "look at me / the camera" and the reply so far ends sideways: in the sentence's pose markers every head.yaw >= 7 degrees and gaze.x >= 0.3 becomes 0 | yes |
| `frame_for_dance` | a whole-body move (`dance`, `twirl`, `shimmy`, `bounce`, `cheer`) in a reply that started in bust, on a body that switches camera, with no `[frame full]` before: `[frame full]` is inserted right before the move (a later one in the same sentence moves up) | yes |
| `frame_before_wave` | a `[frame full]` before a wave: removed | yes, see below |
| `wave_obeyed` | an explicit wave request, a body whose hands move, `[move wave]` on its menu, and no wave or hand pose in the FIRST sentence: `[move {"name": "wave"}]` is put at its start | yes, first sentence only; otherwise logged at the end |
| `body_request_obeyed` | a hand action was asked for and there is none | log only |
| `glance_at_mention` | the glance reflex (`--glance`) already does it | not touched |

`frame_before_wave` goes the other way from `frame_for_dance`: the checker
fails a `[frame full]` before a wave, because the prompt says a wave stays in
bust (`FRAME_ENTRY2`: "never for a wave or a single hand move";
`flow.dpo.wave_frame` makes the full-frame wave the *rejected* side). So a
wave never gets a full frame inserted; one the student wrote is taken out, and
only when no whole-body move sits between the frame and the wave (then the
frame is the dance's, and it is logged, not touched). A dance after the wave in
the same sentence gets its own frame from `frame_for_dance`.

**Choices worth knowing.**

- `look_at_me` cannot know which pose will be the last of the reply, so it
  centres every sideways value of the current sentence while the reply so far
  ends sideways; it does nothing when the checker passes.
- `wave_obeyed` is decided at sentence 1, when the later sentences are not
  known: a wave she makes in sentence 3 as well is a second wave. A body whose
  hands reach nothing (`flow.framing.budget`) or a menu without `wave` is not
  given one; the end of the reply says so in the log.
- A marker whose JSON is not strict (single quotes ...) is not edited; it is
  logged.
- `--repair log` computes every fix, logs it (`(log mode: not applied)`) and
  writes the `{"kind": "repair"}` record with `"applied": false`, and changes
  nothing: no pairs, history as the model wrote it.

**What is kept.** The history keeps the reply *as she performed it*, the
repaired one (`Turn.rebuild()`: each repaired sentence replaced in the model's
reply, in order). She must not see, next turn, a look she never made: a
history of her own wrong replies is what she copies (`pose_copy`), and a
repaired one teaches the right form. The model's raw reply is kept as
`reply_unrepaired` in the turn's `turns.jsonl` line (with `repaired`: the
checks), so `scripts/checks.py runs/live/X/turns.jsonl` scores the performed
reply. The `reply`-less `{"kind": "repair"}` lines are skipped by
`flow.checks.load`.

**Logs and records.** Every fix is `repair: turn N.k <check>: <before> ->
<after>` (N the turn, k the sentence; for example `repair: turn 3.1 look_sign:
[pose {"head.yaw": 15}] -> [pose {"head.yaw": -15}]`), the same as a
`{"kind": "repair", "turn", "k", "check", "before", "after", "applied",
"repaired", "mode", "user"}` line of `turns.jsonl`.

**DPO pairs.** With `--record DIR` and `--repair auto`, a turn with a repair
whose raw whole reply fails the repaired check and whose performed reply
passes it appends one pair to `DIR/repair_pairs.jsonl`, with its meta line in
`DIR/repair_pairs.meta.jsonl` (the layout `flow/dpo.py` writes and
`flow/train/dpo.py` `load_pairs` reads line for line): `prompt` = the system
prompt and the messages exactly as sent (history, the user turn with its tail
lines, the memory notes), `chosen` = the performed reply, `rejected` = the
model's raw reply; meta `id` `live/<record folder>/tNN/repair` (the trainer's
session is the id without its last part and it holds whole sessions out for its
eval, so each turn is a session of its own: with one per recording, a single
recording would be held out whole and `train` would find no training pair), `avatar`, `kind`
`repair_<check>` (the first repaired check whose raw reply fails and whose
repaired reply passes), `src` `live`, `checks`, `verdicts` (raw, repaired)
and `fixes`. A turn that repaired two checks is one pair, not two. Train on
it with `python scripts/train_dpo.py train --sft-run runs/sft/q8r3 --data
runs/live/X/repair_pairs.jsonl[:weight][,more.jsonl]` (a file path, not the
folder: the folder form reads `dpo.jsonl`). These are on-policy pairs (the
student's own sample, rejected; a deterministic fix, chosen), the live
version of OPCD's "own generations, a better teacher".

Code: `flow/repair.py`; tests: `tests/test_repair.py` (en / ja / zh for each
fix, the no-op cases, log mode, the pairs file read back by the trainer's
`load_pairs`, the live loop).

### Rule graph and reminders (2026-11)

`flow/rules.py` and its data `flow/rules.json`, wired into `flow/live.py`
(`--reminders auto|off`, default **off** until the prompt A/B below says
otherwise). The beats2 prompt is 3-4k tokens and the student still breaks
rules it states; a rule that matters once in twenty turns sits in the middle
of it, the worst place to be read from (`docs/research.md` section 4). So the
rules the turn needs are said again at the end of the context.

**Nodes.** `{id, trigger, text, needs}`: `trigger` is the name of a detector
function in `flow/rules.py` (`DETECTORS`), reusing `flow/checks.py`'s
(`asked_side`, `asked_at_me`, `asked_wave`, `mentioned_place`,
`corrections`) plus `asked_dance`, `corrected`, `game_on`, `chat_quiet`,
`language_switch`; `never` marks a node that is only pulled in by another
node's `needs`. `text` is one short line, in the wording of the beats2 prompt
(`flow/pose.py`: `LOOK_RULE`, `FRAME_ENTRY2`, `GLANCE_RULE`, `QUIET_RULE`,
`LANG_RULE2`; the game line from `flow/memory.py`'s consolidator prompt); a
test checks the phrases against those constants. The nodes, in priority order
(the order of the file):

| id | trigger | needs | text |
| --- | --- | --- | --- |
| `look-side` | `asked_side` | `her-right` | Asked to look or turn somewhere: one marker that holds that direction, "hold" 2 to 4; the head stays turned until your next head move or the end of the reply. |
| `her-right` | `never` | - | Sides are your own, never the viewer's: "look to your right" is a negative "head.yaw" and "gaze.x" (about -15 and -0.6), "to your left" a positive one (+15, +0.6). |
| `look-at-me` | `asked_at_me` | - | Asked to look at the camera or at the viewer: one marker that holds it, "head.yaw" and "gaze.x" at 0, "hold" 2 to 4; do not turn to a side. |
| `wave-move` | `asked_wave` | - | A wave is a move, not a held pose: [move {"name": "wave"}]; it stays in bust, never [frame full] for a wave or a single hand move. |
| `dance-frame` | `asked_dance` | - | Write [frame full] right before a dance or another whole-body move ([move {"name": "dance"}]), so the whole figure is in the picture. |
| `place-glance` | `mentioned_place` | - | The user put something near you without asking you to look: open with a short glance there, your first marker turning the head and eyes toward it (down for your feet, up for above you; your right is negative "head.yaw"), "dur" 0.3, "hold" 0.4. |
| `correction` | `corrected` | - | A word was corrected: use the corrected form, never the wrong one it replaced. |
| `game-state` | `game_on` | - | A game is on: play the next move the stream notes spell out (the "next:" in game); do not work the numbers or letters out yourself. |
| `quiet-chat` | `chat_quiet` | - | A user message that is just "(chat is quiet)" means nobody has written for a while: keep talking as a streamer does, start a small topic or ask the viewers a question, keep it short, and never say that the chat is quiet. |
| `language` | `language_switch` | - | Answer in the language of the user's latest message (the "(reply in: ...)" line), even if your earlier replies used another language. |

The wave and the dance are two nodes, and the wave one says what the checker
and the prompt say: `FRAME_ENTRY2` is "never for a wave or a single hand
move, which stay in bust", `frame_before_wave` fails a full frame before a
wave. Only the dance gets "[frame full] first".

**Per turn.** `Graph.select()`: the nodes whose detector fires on the user
line, her previous reply and the memory notes; for each, in priority order, it
and its `needs` (recursively); at most 3 nodes in all, a node with its needs or
not at all (a group that no longer fits is skipped and a later, smaller one
may still show), deduplicated, shown in file order. `look-side` fires with
`her-right`: two of the three slots. The game and correction nodes read the
notes, so they are silent without `--memory` (and in the eval, which has none);
the language node fires on a *switch* only (the user's language differs from
her previous reply's), or it would take a slot every turn.

**Where it goes.** `rules.block()` renders the chosen texts as one line,
`(reminders: <text> <text> <text>)`, and `Performer.tail_block()` puts it
after the memory notes in the same `note_block`: the last lines of the latest
user message only, at send time, by `flow.memory.inject()` (so `--memory-place
head` moves both); the history keeps the plain message, the system prefix and
the older messages stay byte-stable, and `trim_context` / `sent_chars` count
the block. Nothing is added when no node fires, and body mode only (the rules
are about markers). Log: `rules: turn N: look-side, her-right, game-state`.

One tail block, from `tests/test_rules.py` (a number game in the notes, the
viewer asking her to look right):

    Look to your right
    (your body now: head.yaw -15, hand.L (0.75, -1.17), hand.R (-0.77, -1.28))
    (camera: bust, head and chest; hands below y -1.00 are out of frame)
    (reply in: English)
    (stream notes: game: number game, 1-20: 15 too big, next: her turn)
    (reminders: Asked to look or turn somewhere: one marker that holds that direction, "hold" 2 to 4; the head stays turned until your next head move or the end of the reply. Sides are your own, never the viewer's: "look to your right" is a negative "head.yaw" and "gaze.x" (about -15 and -0.6), "to your left" a positive one (+15, +0.6). A game is on: play the next move the stream notes spell out (the "next:" in game); do not work the numbers or letters out yourself.)

The reminders are also what the eval's `full+rem` and `short+rem` conditions
append (next section), through the same `rules.block()` and `inject()`.
`--reminders off` (the default) changes no byte of any request.

Code: `flow/rules.py`, `flow/rules.json`; tests: `tests/test_rules.py`
(triggers in en / ja / zh, the cap and the needs closure, the block, the
injection with notes, the prefix, the budget, off).

### Prompt A/B: full, full + reminders, short + reminders (2026-11)

The third step of `docs/research.md` section 4: does the student need the
long beats2 prompt, or do the short core prompt and a few reminders at the
end of the context do as well? `flow.train.evalsft run` takes
`--prompt-cond full|full+rem|short+rem` and runs the held-out sessions
(`$RUN/split.json`) three ways, each turn scored by the failure checkers
(`flow/checks.py`):

| condition | system prompt | user turn | file |
| --- | --- | --- | --- |
| `full` | the data's own (`avatars.json` bytes; the current behaviour) | the held-out turn's | `student.jsonl` |
| `full+rem` | the same | and `(reminders: ...)` on its last line, as `flow.live --reminders auto` | `student.full+rem.jsonl` |
| `short+rem` | the short student prompt (`flow.dataexport.student_system`: the bytes of `--student-prompt short` and of `flow.director --body-prompt short`) | the same as `full+rem` | `student.short+rem.jsonl` |

The reminders come from the same function as live's: `rules.block(user line,
her previous reply)` for the held-out viewer line and the model's own previous
reply, put on the last line of the turn being sent by `memory.inject` (the
history stays plain, so it is the live format byte for byte). The eval has no
memory notes, so the game and correction nodes fire on the line alone (a
correction in the line) or not at all. Each condition has its own
`<name>.<cond>.summary.json` (`compare` lists them as `student`,
`student.full+rem`, `student.short+rem`) and its rows carry `cond` and the
`rules` ids shown. `--prompt-cond full+rem` with `--student-prompt short` is
refused (use `short+rem`), as is a condition with `--recorded`.

`scripts/eval_sft.sh` runs the student once per name in `PROMPT_CONDS`
(default `full`; space or comma separated). On the GPU machine (WSL,
`.venv`; the shell goes through a local HTTP proxy, so `NO_PROXY` must list
127.0.0.1,localhost):

    cd Tsukuyomi && source .venv/bin/activate
    export NO_PROXY=127.0.0.1,localhost

    # Ollama, as live serves it (the merged Q4_K_M, name from `ollama list`)
    RUN=runs/sft/q8r3 STUDENT_MODEL=tsk-q8r3 RECORDED=0 TEACHER=0 \
        PROMPT_CONDS="full full+rem short+rem" bash scripts/eval_sft.sh

    # vLLM (docs: "vLLM for student evals"), sessions side by side
    RUN=runs/sft/q8r3 STUDENT_BACKEND=vllm STUDENT_MODEL=q8r3 CONC=8 \
        RECORDED=0 TEACHER=0 PROMPT_CONDS="full full+rem short+rem" \
        bash scripts/eval_sft.sh

    # or one condition by hand (vLLM shown; --backend ollama --model tsk-q8r3
    # for Ollama), the same --out for all three
    for c in full full+rem short+rem; do
      python -m flow.train.evalsft run --data data/datagen/r3 \
        --split runs/sft/q8r3/split.json --out runs/sft/q8r3/eval \
        --name student --backend vllm --model q8r3 --conc 8 --prompt-cond $c
    done

    # the three side by side: the folder gives one column per file
    python scripts/checks.py runs/sft/q8r3/eval --data data/datagen/r3 --by avatar
    # or the files, in the order wanted
    python scripts/checks.py runs/sft/q8r3/eval/student.jsonl \
        runs/sft/q8r3/eval/student.full+rem.jsonl \
        runs/sft/q8r3/eval/student.short+rem.jsonl --json ab.json

The labels are `eval/student`, `eval/student.full+rem`,
`eval/student.short+rem` (for the files) and `eval:student` ... (for the
folder); the same file of two runs gets its path as the label. Read the rows
the reminders are about: `look_sign`, `look_at_me`, `glance_at_mention`,
`wave_obeyed`, `frame_before_wave`, `frame_for_dance`, `correction_obeyed`,
`language`, `quiet_says_quiet`, and `beats` / `opening` / `pose_copy` as the
guard that a shorter prompt does not flatten the style.

Two cautions. (1) A student trained on the full prompt (q8r3) was never shown
the short one, so `short+rem` on it measures how much of the long prompt its
weights already hold; the fair test of the short core prompt is a student
trained with `--student-prompt short` (r3s) in the `short+rem` column. (2) The
held-out sets hold few turns that trigger a given node (a look request,
a place): count the applicable replies in the table (`n`) before reading a
difference. `--reminders` stays `off` in live until this says otherwise.

Code: `flow/train/evalsft.py` (`resolve_cond`, `with_reminders`),
`flow/checks.py` (`load`: `student.<cond>.jsonl` files and the labels),
`scripts/eval_sft.sh`; tests: `tests/test_prompt_ab.py` (fake backends: what
each condition sends, byte for byte, the files, the labels).

### Failure checkers (2026-11)

`flow/checks.py` (`python scripts/checks.py PATH [PATH ...] [--by avatar]
[--json out.json]`) scores a model's replies on the failures seen live, to
compare models side by side and to pick the turns a preference tuning should
push on. Inputs: an evalsft `student.jsonl` (a folder with `recorded.jsonl`
gives two columns), flow.dataset shards (or the folder), a flow.live
`--record` `turns.jsonl` (it keeps the raw reply, `quiet` and the framing,
but no avatar: `--avatar NAME`, else the folder's name). Hands of an avatar
come from `--data`'s `avatars.json` budget, else `rigs/NAME.body.json`,
else assumed. Every check gives each reply pass / fail / na and the table
shows the fail rate among the applicable ones (lower is better); `--json`
keeps counts, per-avatar numbers and the first 50 failing ids.

| check | fails when | na |
|---|---|---|
| look_sign | the line asks to look / turn to your right or left (en "your right", "to the left" with look, turn, glance; ja 右 / 左 near 見 / 向; zh 右 / 左 near 看 / 转 / 轉, 右邊) and the first pose with \|head.yaw\| >= 7 or \|gaze.x\| >= 0.3 has the wrong sign (right is negative); also no turn (told apart: wrong sign / no turn) | no request, both sides, a body part ("your right hand") |
| look_at_me | the line asks her to look at me / the camera / the chat (en "look at me", "face the camera"; ja こっち見て, 私を見て, カメラ見て; zh 看着我, 看我, 看镜头; not a negation, not a line that names a side) and the head.yaw / gaze.x she ends on (the last pose marker setting each) is clearly sideways: \|head.yaw\| >= 7 or \|gaze.x\| >= 0.3 (told apart: yaw / gaze / both) | no request; quiet turns; a reply that sets neither passes |
| correction_obeyed | the user corrects a word of hers and the reply still says the wrong form X: "not X but Y", "it's Y, not X", "you said X, it's Y"; ja XじゃなくてY, Yだ、Xじゃない; zh 不是X，是Y, 是Y不是X, 你说X不对; or a correcting line (違う, "no,", 不对, wrong) with a word close (difflib 0.6-0.95) to one of her last reply. For ja / zh the wrong form is the clause before the particle, so a tail of it (4+ characters) also counts, unless the right form contains it | no correction; quiet turns |
| pose_copy | the reply's pose markers (variables with rounded values, dur / hold aside) equal, or at least 3 of 4 (ceil 75%, and 2 of 2) equal in place, those of one of her previous 3 replies in the session (live: the same 4 poses on turns 17-21) | fewer than 2 poses; first reply of a session |
| wave_obeyed | the line asks to wave (wave, 手を振, バイバイ, 挥手, 揮手) and the reply has no wave move or hand target | no request; no hand reach (`budget` hand_moves full 0) |
| body_request_obeyed | the line explicitly asks for a hand or arm action (`flow.handask`: point, wave, raise a hand, peace sign, clap, hands on hips ... as a request; not by topic, not a head request, not a past mention or compliment) and no hand target, move or gesture | not asked; quiet; no hand reach |
| opening | the first pose is exactly the q8r2 template (brow.L/R.lift 0.6-0.9, head.pitch 4-6, head.yaw -4/-5); also the top opening's share by variable set and with rounded values | no pose |
| template_repeat | the marker sequence (variable set of each pose, tag of the rest) equals the previous reply's in the session | first reply, no marker |
| language | `detect_lang` of the spoken text differs from the line's | mixed line, quiet turn, undetected |
| frame_for_dance | a whole-body move (`WHOLE_BODY`) in a reply starting in bust without a [frame full] before it (informational: the camera follows anyway) | none, VRM, starts in full |
| frame_before_wave | a [frame full] before a wave (should be ~0) | no wave |
| beats | fewer beats than sentences (`dataexport.beat_counts`); beats per sentence and per reply | no speech |
| quiet_says_quiet | a quiet-chat turn whose speech says the chat is quiet (en, ja, zh) | not a quiet turn |

The held-out evals of q8m and q8r2 (`python scripts/checks.py
runs/eval_q8m/student.jsonl runs/eval_q8r2/student.jsonl`; q8m is on three
avatars, q8r2 on six):

```
                            eval_q8m/student  eval_q8r2/student
replies                                  504                630
look_sign                                  -                  -
  wrong sign / no turn                 0 / 0              0 / 0
wave_obeyed                         0% (0/5)         18% (3/17)
body_request_obeyed              8% (10/127)       28% (47/166)
opening = q8r2 template           0% (0/504)      26% (161/630)
opening top set share                   0.13               0.39
opening top exact share                 0.03               0.04
template_repeat                   1% (3/420)        4% (21/525)
language                        11% (55/496)        4% (26/624)
frame_for_dance                 100% (10/10)        91% (20/22)
frame_before_wave                   0% (0/9)          8% (2/24)
beats < sentences              90% (453/504)      47% (293/630)
beats per sentence                      0.41               0.87
beats per reply                         1.12               2.97
quiet_says_quiet                           -                  -
```

Reading it: those held-out sets predate the r3 look and quiet topics, so
look_sign has no applicable line and quiet_says_quiet no quiet turn yet (the
checks exist for the r3 students); q8m has no [frame] markers (frame_for_dance
100% is its age). q8r2 opens 26% of its replies with the q8r2 template (39% of
them with one variable set) against 0% for q8m, but writes about 0.9 beats
per sentence (q8m 0.4) and keeps the reply language far better (4% against
11% wrong). It also skips a requested body action more often (28% against 8%;
the two sets' user lines differ). The wave counts are tiny.

### Style checks (2026-11)

The failure checkers say what broke; they say nothing about whether a reply
is worth hearing. q8r3's held-out replies read as "silly / too simple", and
its teacher is little better: about five tiny exclamatory sentences, a marker
per sentence, reactions without content. `flow/style.py` (shown under the
failure checks by `scripts/checks.py`, and by `scripts/style_ref.py` on a file
of real streamer replies) measures it per reply on the spoken text, split by
language where it matters. These are numbers, not pass / fail: there is no
right value, only "near the real streamers" (section "Real streamer data" in
`docs/research.md`) and "not worse than the teacher".

| number | definition |
|---|---|
| replies by language | ja: kana in it; zh: Han and no kana; en: no CJK and 3 or more Latin words |
| words/reply (en) | runs of Latin letters (inner `'` and `-` kept), English replies |
| chars/reply (ja), (zh) | characters without whitespace, per language |
| chars/reply (all, len) | `len(spoken)` over every reply, spaces included (the number of the first q8r3 comparison) |
| sentences/reply | stretches with a letter between `. ! ? 。！？ …` and newlines (`dataexport.sentence_count`) |
| words/sentence (en), chars/sentence (ja, zh) | the same, per sentence |
| distinct-1/2 (en) | unique 1-/2-grams of English words over the English replies, none across a reply |
| distinct-2 (ja+zh chars) | unique character bigrams over the ja and zh replies |
| unique words (en) | distinct lower-cased English words in the file |
| echo of the user line | the first sentence shares 2+ content words with the user line (English: stopwords and words under 3 letters out, a plural s dropped), or a 4+ character substring (ja, zh); quiet-chat turns left out |
| ends with a question | the spoken text ends with `?` or `？` |
| top opening share, top openings | the first token (English word, or up to 3 CJK characters up to the first punctuation mark); share of the most common, and the 3 most common |
| interjection opening | the reply opens with oh / ooh / hmm / wow / okay / ah / eh / well ..., えっ / あ / うん / わあ / ふふ ..., 嗯 / 哇 / 哎 ... (`flow.style.INTERJ`) |
| length short/medium/long/over | shares of the non-quiet replies by size (up to 20 words / 35 characters, 40 / 80, 60 / 130, 10 % slack) |
| length target hit rate | with `length_target` on the records (beats3 revision 2): the share whose class is the target asked for, in total and per target |
| similes/reply, replies with a simile | a literary-register proxy (below); a number, not a check |
| substance (any) | share of replies with at least one concrete item, listed next |
| number | digits, a spelled number above one ("two", "hundred"; not "one"), a CJK numeral with a counter (三个, 5年) |
| named thing | an English capitalised word inside a sentence (not the first word, not I), a katakana run of 3+, a quoted title (「」『』《》 and curly quotes) |
| opinion / experience | "I think / love / hate / prefer / used to / once / always / tried / went ...", "when I", "my favorite / first ...", "personally"; ja 私は / 僕は near 好き / 思う / 苦手 ..., 昔 / この前 / 前に; zh 我觉得 / 我喜欢 / 我以前 / 小时候 / 上次 ... |
| callback | a cue ("earlier", "you said", "last time", さっき, 前に言, 刚才, 你说过 ...) or 2+ content words (or a 4-character substring) of an earlier user line of the session that the current line does not have |

The substance proxy is shallow on purpose: it counts a reply that names a
number, a thing, an opinion or an earlier turn, not one that says something
true or interesting. Use it to see a prompt change move, not as a score. The
numbers on the 820 held-out turns of q8r3 and its teacher (`python
scripts/checks.py runs/eval_q8r3`, the style rows; reproduced by
`tests/test_style.py`; the first comparison's figures for 78 / 72 chars, 24.7
/ 22.8 words, 28% / 38% echo, 24% / 21% questions come out of these
definitions, the sentence counts and distinct-2 of that comparison used
other definitions, so the documented ones are given here):

```
                                 q8r3 student   teacher (recorded)
  words/reply (en)                      22.77        24.58
  chars/reply (ja)                      48.85        53.19
  chars/reply (zh)                      39.83        42.92
  chars/reply (all, len)                71.86        78.26
  sentences/reply                        3.43         3.89
  words/sentence (en)                    6.42         6.26
  chars/sentence (ja)                   14.93        13.75
  chars/sentence (zh)                   12.83        12.88
  distinct-1 (en)                        0.17         0.19
  distinct-2 (en)                        0.69         0.74
  distinct-2 (ja+zh chars)               0.35         0.38
  unique words (en)                      1032         1276
  echo of the user line                  0.38         0.28
  ends with a question                   0.21         0.24
  top opening share                      0.06         0.04
  top openings           oh 6%, えっ 5%, あ 4%  えっ 4%, oh 4%, 嗯 3%
  interjection opening                   0.48         0.37
  substance (any)                        0.37         0.41
    number                               0.10         0.14
    named thing                          0.11         0.11
    opinion / experience                 0.10         0.09
    callback                             0.11         0.13
```

Reading it: the student is a little shorter, less varied (fewer unique
words, lower distinct-n), echoes the user line more and opens with an
interjection more often than its teacher, which is itself short: 6 words a
sentence, interjections on more than a third of its replies, and a
substance share of 0.41 in which an opinion or a story of her own shows up in
under one reply in ten. So r4's teacher (`beats3`, below) is where the fix
starts; the student only inherits.

`--json` keeps the style numbers under `"style"` for each input (and each
avatar with `--by avatar`). The style block is not in `CHECKS`, so
`--checks`, the DPO pair kinds and the live repair layer never see it.

### beats3 (r4 data, 2026-11)

q8r3's replies are "silly / too simple" because its teacher's are: the beats2
rules want a marker per sentence and short streaming replies, so the teacher
writes about five tiny exclamations, the dense examples it is shown are eight
fragments each, and the simulated viewer (also an LLM) keeps the talk shallow
(table in "Style checks"). Revision 1 below is the r4p pilot; revision 2 (`### beats3 revision 2`) follows it. `--teacher-prompt beats3` (`flow.pose`, alias
`--prompt`; the rules are below) is the r4 teacher. Everything the
checkers and reminders depend on is beats2's, word for word, and a test
asserts it (look sign, look at me, glance, camera rule, motion budget, hand
requests, `[frame full]` entry, language rule, quiet chat, head stays); the
bytes of beats2 (revisions 1-3), beats, v1 and short are pinned by hashes.

What beats3 changes, and nothing else:

- "one to three short spoken sentences" becomes "a quick line, two or three
  sentences, or a short story of up to 60 words";
- "every sentence gets at least one expression beat, a longer sentence two or
  three" becomes "one in every sentence, and a longer sentence one more at
  each turn of its clauses (the marker goes where the thought turns; never
  cut a sentence into pieces to make room for one), something changes about
  every 2 to 3 seconds of speech";
- the rules block (`SENTENCE_RULE["beats3"]`), verbatim:

> - Every reply carries one concrete thing, so it is never only a reaction: your own opinion, a small story or memory of yours, a specific detail (a name, a number, a place, a food), a callback to something said earlier in this conversation, or a real answer to the question you were asked. Never just react to the line, and never echo it back.
> - Vary the length on purpose: some replies are one quick line, many are two or three sentences, and now and then one runs 40 to 60 words when you have something to tell. Do not write every reply as four or five short fragments. Keep the first sentence short enough to start speaking quickly.
> - Make expression beats where the face would move: one in every sentence, another at each turn of a longer sentence, each in its own marker right before the words it reacts to. Do not chop a sentence into pieces to make room for markers.
> - Do not open by repeating the viewer's words. Vary how your replies open; an interjection (Oh, Hmm, えっ) is fine now and then, not by default: start with the substance when you can, and never take the opening of an example.
> - Several chat lines can arrive in one message, one per line: that is normal. Answer the one that is most interesting, or a few of them; you do not have to answer all.

  followed by beats2's language rule (`LANG_RULE2`) and quiet-chat rule;
- new worked examples (`_variants3`, six variants, one per session by its
  seed like beats2's dense examples): each prompt shows a quick line, a
  medium reply and a 45-60 word one, in an order that turns with the variant,
  each with a marker per clause and something in it (an opinion, a memory, a
  specific detail, a callback, an answer), none opening on the viewer's
  words. Their beat density passes `check_beats` (tests);
- the character bible (next).

**Character bible.** `flow/datagen.personas.json`: every persona has
`"bible"`, a list of lines (backstory, likes, dislikes, opinions, running
bits, what she is bad at); `flow.dataset` writes it behind the persona text
(`pz.BIBLE_SEP`) for beats3 and `system_prompt` shows it after the rules as
"Your background, to draw on when it fits ... never recite it, and never say
you were given it". The simulated viewer is never given it (it gets the plain
persona). The student: `--student-prompt teacher` (the default) puts the
teacher's whole prompt, bible included, in the student's system prompt;
`short` leaves it out (the student learns the character from the replies);
`short+bible` is `short` with the bible after the persona (serve it with
`--body-prompt short+bible`). All the characters are original; none is
based on a real streamer or person. Example (cheerful):

> - Backstory: grew up in a seaside town where the ferry was always late; moved to the city at fifteen for school and still keeps every ferry ticket in a biscuit tin.
> - Likes: strawberry milk bread from the bakery by the station, thunderstorms watched from a window, drawing tiny maps of places she has never been, group chats that run until midnight.
> - Dislikes: cold rice balls, being told to calm down, horror films (she says she is fine and is not).
> - Opinions: breakfast is the best meal, and cereal for dinner is a perfectly good life choice; a map beats a photo because you have to imagine the rest; a sequel is fine if the dog survives.
> - Running bits: she measures every wait in ferries ("that is about a three-ferry problem"); she names every plant she buys, and the current fern is Gerald; she starts an opinion with "tiny announcement!".
> - Bad at: cooking rice (always a little crunchy), telling left from right quickly, keeping a birthday surprise secret, waking before nine.

**Viewer modes** (topics with `"prompts": ["beats3"]` in the personas file;
the beats2 topics look / place / wave apply to beats3 too, by family, and the
beats2 draws are unchanged):

| topic | what the simulated viewer does |
|---|---|
| `ask_her` | asks about her: likes, her day, a memory, what she thinks; needs a real answer |
| `story` | tells a short story of their own, with one concrete detail |
| `disagree` | disagrees or teases; takes the other side of a common opinion |
| `correct` (from turn 2) | corrects a word or fact in her last message |
| `callback` (from turn 3) | refers back to something said earlier and follows up |
| `chat_burst` | 2-4 viewers write at once: the simulator writes one line each, each line cleaned on its own and joined with a newline, as `flow.live.batch_text` does (at most 5) |
| `place` (beats2) | puts things around her, without asking her to look |

**Long games** (`flow/datanotes.py`; `--game-share P`, default 0.2 for
beats3, `--game-kinds`): a session of 6+ turns may carry one game, planned
from its seed alone (so no other draw moves): number guess 1-50 (she picks a
secret, the viewer guesses), 20 questions about a secret object of hers,
an English word chain, would-you-rather. It starts on turn 2 or 3 and lasts
4-8 turns; on those turns the simulator's hint is the game's (its guesses so
far, her higher / lower answers), the topic id is `game_<kind>` and the record
has `game: {kind, turn}`. The two games with a secret of hers only run with
`--notes auto`.

**Reminders and notes in the data.** Two flags, both off by default and for
beats prompts only:

- `--reminders auto`: appends the `(reminders: ...)` block of `flow.rules`
  (the nodes whose detectors fire on the user line, her previous reply and
  the notes; at most 3) to the last line of the teacher's latest user turn;
- `--notes auto`: the `(stream notes: ...)` line, rendered by
  `flow.memory.Notes` (the live renderer, fitted to its 800 characters), from
  `flow.datanotes.Keeper`: no model is asked. The game's entry (her secret,
  the guesses with "too small / too big / correct", the questions asked, the
  words used, the picks), corrections ("Y, never X" from
  `flow.checks.corrections`) and a viewer's name. What the notes say at turn
  t is what was known after turn t-1, except a game's setup, which is there
  from its first turn (she has to be able to pick).

Both go through `flow.rules.tail_block`, the function `flow.live`
(`Performer.tail_block`) calls, and are put on the last user message with
`flow.memory.inject`, so the bytes are the live ones (a test builds both from
the same inputs). The history keeps the plain message, as live does. In the
record: `user_content` stays plain, `tail` is the block, `notes` the Notes
object, `rules` the node ids. `flow.dataexport` appends `tail` to that turn's
user message: in `--per turn` examples the history stays plain and only the
last message carries it (exactly what the live student gets); in
`--per session` examples (the default) each turn carries its own block, since
each is the latest turn for its own reply, which the live history does not
(live shows the plain history), so prefer `--per turn` when notes are on.
`--history-turns N` (default 0 = all; live keeps 5) gives the teacher the last
N exchanges only, so the notes decide a game; the record's `history` is what
it saw, which `--per turn` reproduces. What the r4 student sees, then: the
beats3 system prompt (or `short` / `short+bible`), the plain history, and a
last user line of the form

```
<viewer line(s)>
(your body now: ...)
(camera: ...)
(reply in: English)
(stream notes: game: number guess 1-50, her secret is 37, guesses: 25 too small, 40 too big; corrections: Kobe, never Kyoto)
(reminders: ...)
```

with the notes and reminders lines present only when something fired; serve
it with `flow.live --memory notes --reminders auto`.

**Real chat** (`flow/realdata.py`, docs/research.md section 5; all off):

| flag | effect |
|---|---|
| `--real-viewer PATH` | the simulated viewer gets `--real-refs` (4) cleaned real chat lines of the turn's language as references for the way people type, and a turn (not a game, a burst or the first) opens with one verbatim with chance `--real-verbatim` (0.15) |
| `--style-seeds PATH` | beats3 only: `--seed-n` (4) streamer (viewer, reply) pairs per session go into the teacher's system prompt as "how real streamers answer chat, for rhythm only ... do not copy their content, names or facts" (not in the record's `prompt_sha`; the student's prompt is the base one) |
| `--real-source` | `livechat`, `oshichats`, `processed_chats` or `custom` [guessed from the path]: the attribution recorded |
| `--real-licence-ok` | required for the three named sources: you read the licence at the source |

Formats: JSONL with a `message` / `text` / `content` / `body` / `msg` /
`comment` field (author, channel and id fields are never read) and an optional
`lang`; `.json` as `{"streamer": [[comment, reply], ...]}` (LiveChat's
Subset) or lists of `[streamer_id, comment, reply]`; LiveChat's `.pk` files
through a restricted unpickler; `.txt`, a line a line. Cleaning and the
keyword filter are in the docstring of `flow/realdata.py` (crude: read the
lines, `scripts/style_ref.py --show 50`). Each turn's record gets `real`
(source ids, the reference lines, the verbatim line; the seed pairs on turn
1), and `<out>/README.md` and `avatars.json` "sources" carry the attribution
each licence asks for. `scripts/style_ref.py FILE --compare data/datagen/r3
--sample 820` puts the style numbers of the real replies next to the
teacher's, to tune r4 against.

### beats3 revision 2 (r4p2, 2026-11)

The r4p pilot (600 turns, DeepSeek teacher and viewer, `--teacher-prompt
beats3`, revision 1 below) moved what it was built for: substance 0.40 to
0.79, interjection openings 0.37 to 0.17, the motion checks held. It did not
fix four things: **length did not vary** (non-quiet English replies <=10 / <=25
/ <=45 / <=70 / >70 words: 0 / 9 / 47 / 42 / 1 %, mean 44 words against r3's
24; no quick lines), the **register is literary** ("colour as faint as an old
letter"; streamers talk plainly), the **simulated viewer is unrealistic**
(crafted paragraphs, three or four body requests in one message, objects
placed around her in many turns, a `callback` viewer that invented a promise
never made in the session) and the glance at a mentioned place was still
missed in 39 % of cases (7/18, as in r3). Her text also had line breaks
between sentences, and the sleepy persona answered with the same "old books"
image in every session (the bible had a bookshop backstory, paperbacks and a
notebook, and nothing else to reach for).

Revision 2 is `--beats3-rev 2` (`flow.pose.BEATS3_REV`, the default; `1` is
the r4p wording). **Revision 1 is the pilot byte for byte**: its prompts are
pinned by hashes, its sessions by a digest of every record and every request
to the character and the viewer (`tests/test_beats3_rev2.py`), and
`flow.dataexport.student_system` tries `BEATS3_REV .. 1` when it rebuilds a
teacher prompt, as it does for beats2. Every new draw has a generator of its
own keyed by the session seed and the turn and runs under revision 2 only.

**Length plan.** `--length-mix S,M,L` (default 35,45,20; `off`): each non-quiet
turn draws `short`, `medium` or `long` (`flow.dataset.length_target`,
generator `crc32(seed:turn:len)`). Short is one sentence or two very short
ones, up to 20 English words or 35 CJK characters; medium 2-3 sentences, 20-40
words or 35-80 characters; long a small story or a real explanation, 40-60
words or 80-130 characters (`flow.style.BANDS`, defined once and used by the
line, `scripts/lengths.py` and the style rows). Topics lean: `chat_burst`,
`greeting`, `show`, `pose`, `look`, `place`, `wave`, `tease`, `apology`, `game`
and the turns of a long game lean short (tilt 1.8 / 0.9 / 0.35); `story`,
`ask_her`, `question` and `disagree` lean long (0.4 / 1.0 / 2.2); the
other topics' mix is solved from the topic weights (and the share of game
turns) so the whole stays at `--length-mix`, a test draws 24000 turns and
checks it within 4 points for three different mixes.

*The target is told to the teacher only.* It is one more last line of the
teacher's copy of the latest user message, after the reminders (the same
`flow.memory.inject` the tail uses):

```
<viewer line(s)>
(your body now: ...)
(camera: ...)
(reply in: English)
(stream notes: ...)          only when something fired
(reminders: ...)             only when something fired
(length for this reply: short — one quick sentence, or two very short ones, 20 words at most)
```

with `medium — two or three sentences, 20 to 40 words` and `long — a small story
or a real explanation, 40 to 60 words`, and `characters` with the 35 / 35-80 /
80-130 bands for a Japanese or Chinese reply (`flow.style.length_line`; the
language is the one of the user's line). It is not in `user_content` (so not
in the history the teacher sees on later turns, nor in an export), not in
`tail` (the export appends that), and not in the system prompt (a per-turn
system line would break DeepSeek's prefix cache, and the rules block does not
mention the line at all, so a student taught with `--student-prompt teacher`
reads no rule about a line it is never sent). The record gets
`length_target`. A test exports `--per session` and `--per turn` with the
`teacher`, `short` and `short+bible` prompts and finds no trace of it, and
asserts the teacher *was* sent it (so the test is not empty).

`python scripts/lengths.py DIR` prints the five r4p buckets (comparable with
the pilot), the four classes short / medium / long / over (an upper edge has
10 % slack, `flow.style.SLACK`) against the mix, and, for a dataset with
targets, the **hit rate**: the share of replies whose class is the target, in
total and per target. The checks table (`scripts/checks.py`) has the same
numbers as style rows ("length short/medium/long/over", "length target hit
rate"), and `flow.checks.Rec` carries the target.

**Register, glance, motifs: the rules block of revision 2**
(`SENTENCE_RULE2`, verbatim). Substance, clause, opening, chat and language
and quiet rules are revision 1's, and so is `GLANCE_RULE` inside the beats
rule; revision 2 replaces the length rule and adds three:

> - Every reply carries one concrete thing, so it is never only a reaction: your own opinion, a small story or memory of yours, a specific detail (a name, a number, a place, a food), a callback to something said earlier in this conversation, or a real answer to the question you were asked. Never just react to the line, and never echo it back.
> - Reply lengths differ and a reply is exactly as long as it is meant to be: short is one quick sentence, or two very short ones (about 20 words, 35 Japanese or Chinese characters, at most); medium is two or three sentences (20 to 40 words, 35 to 80 characters); long is a small story or a real explanation (40 to 60 words, 80 to 130 characters). Never stretch a short reply, and keep the first sentence short enough to start speaking quickly.
> - Talk the way a streamer talks live, to friends: plain everyday words and short clauses, casual phrasing in the reply's language (Japanese in casual speech that fits your character, Chinese in spoken, colloquial style). No literary similes, no written-style imagery, nothing that sounds like a novel or an essay. A detail is concrete and ordinary: a food, a time, a number, a place, a name. Say "it was midnight and I was eating cold ramen", not "the night was quiet like an old letter".
> - Make expression beats where the face would move: (revision 1's clause rule) ...
> - Do not open by repeating the viewer's words. (revision 1's opening rule) ...
> - Several chat lines can arrive in one message ... (revision 1's chat rule) ...
> - Pick a different detail from your life than in your last replies, and do not use the same image, object or running joke twice in a conversation: your background has several interests, so draw on another one each time. A running bit is fine now and then, never in every reply.
> - When the user puts something near you or points out where it is (at your feet, behind you, on your left or right, above you), your very first marker is a short glance there, before any face beat and before the words: head and eyes toward it (your right is negative "head.yaw" and "gaze.x", down is a negative "head.pitch" and "gaze.y"), then say something about it in your own voice. Do this even when the line is also a question or a joke, and when you answer more than the placement. It does not apply when they ask you to look: that is a request.

followed by the language and quiet-chat rules. "Spoken aloud" becomes "plain
spoken sentences, from one quick line to a short story of up to 60 words". The
glance rule is the checker's (`glance_at_mention`: the first or second pose
marker turns head or gaze toward the place; her right negative, down a negative
pitch) and the reminder `place-glance` in `flow/rules.json` says the same in
fewer words; the new example list shows the glance as a fourth placement
example ("A basket at my right foot? Perfect, ..."), which the test checks
with the checker's own `looks_toward`. The worked examples have the revision 1
markers with plain, spoken words (no similes; short <=20, medium 20-40, long
40-60 words, a test counts).

**Motifs.** Each persona has `"bible2"` next to `"bible"` in
`flow/datagen.personas.json`: the same lines plus "Other interests (draw on a
different one each time ...)" with seven to ten of them; revision 2 shows
`bible2`, revision 1 `"bible"` (untouched). The sleepy bookshop backstory
is one place among many and the paperbacks are gone from her likes.

**The simulated viewer** (revision 2 only; `USER_SYSTEM2`, `limit_viewer`):
chat lines are 3-20 words (CJK about 5-35 characters), a message may run to
40 words / 70 characters with chance `--viewer-long-share` (0.15); at most
one body request (look, point, wave, show, hands) per message, two with
chance 0.15; `place` mentions in at most `--max-place-turns` (2) turns of a
session; a callback only to a line that is really in the session; bursts of
2-4 lines of at most 12 words, each from a different "person" in a different
style. Each is said to the simulator (system prompt and the turn's request)
*and* enforced on what it writes, because an LLM ignores a length rule
exactly as the teacher did: `cap_length` drops trailing sentences or clauses,
`cap_requests` keeps the first N request clauses (`handask`, the look
detectors and a verb list), `cap_place` drops clauses `checks.mentioned_place`
reads as a place and a `place` topic becomes `small_talk` after the second.
`callback` turns pick a real earlier line of the transcript (not the latest
exchange, generator `crc32(seed:turn:cb)`) and quote it in the topic hint;
with nothing to go back to the turn becomes `small_talk`. The `greeting` and
`apology` hints of the personas file ("it's been a while", "forgetting a
promise") invent a past, so revision 2 uses its own (`VIEWER_HINTS2`) and the
file's strings stay as they were.

**Line breaks.** The reply text is cleaned after the model's reply in
revision 2 only: every run of whitespace with a line break outside a
`[marker]` becomes one space in the raw reply (what the student is taught),
and the reply is parsed again, so `spoken` and the markers' offsets agree
(`flatten_newlines`, `reflow`; the record gets `newlines: N`). Nothing relies
on a newline: the parser's offsets index characters, markers are single-line,
and the one place that read a newline as a sentence end (`speech.SENT_END`,
`dataexport.sentence_count`) now sees the full stop or the question mark that
was there. A break with no mark before it ("Hello\nHow are you") is no longer
a sentence end; the beat-per-sentence number can move a little for that, and
the r4p2 pilot says how much. `speech.speak` also turns a `\n` into a space
(one for one, so offsets stay) before the TTS, for every prompt.

**Style numbers added**: the length mix and hit rate above, and a literary
proxy: simile markers per reply (en "like a / an", "as if", "as though", "as X
as a"; ja ような / のようで / みたいな; zh 像...一样 / 的那种 / 仿佛 / 宛如) and the
share of replies with one, in `flow.style` and the checks table. A number, not
a check.

### A 14B student (Qwen3-14B, planned; nothing trained here)

What was checked (code and arithmetic: `flow/train/vram.py`, `tests/test_vram.py`;
no GPU and no Hugging Face access in the session that wrote it, so the
architecture numbers are the model card's and reproduce its 14.8B parameters;
check them against the base's `config.json` the first time):

- **Family and templates.** `chat.family_for("Qwen/Qwen3-14B")` is `qwen3`
  (the regex is `qwen-?3`): same ChatML with thinking off, same Modelfile,
  stops and sampling. The trainer, `merge`, `gguf`, `modelfile` and evalsft
  take `--base Qwen/Qwen3-14B` with no change; nothing hard-codes 8B except
  `DEFAULT_BASE`, doc estimates and two defaults below.
- **Fixed: merge.** `merge` loaded the base in fp16 on the card ("8B in fp16 is
  ~16 GB"); a 14B is 29.5 GiB and would not load. `--device auto` now merges on
  the CPU when the fp16 model does not fit the card (needs ~30 GB of RAM).
- **Fixed: fp32 embedding and head.** peft's `prepare_model_for_kbit_training`
  casts every 16-bit parameter to fp32, the embedding and the output head
  included: 2 x 152k x 5120 x 4 bytes = 6.2 GB on a 14B (the 8B: 5 GB). The new
  `--io-dtype auto|fp32|bf16` puts them back to bf16 after it for bases of 10B
  or more (auto), 3.1 GB less; the 8B keeps its fp32 so q8r3 stays reproducible.
- **Memory (QLoRA r32, batch 1, accum 16, gradient checkpointing, PEFT):**

| | Qwen3-8B | Qwen3-14B |
|---|---|---|
| 4-bit layers + embedding and head (fp32) | 8.0 GiB | 12.2 GiB (9.1 with bf16 io) |
| LoRA r32 (87 M / 128 M parameters, fp32 + grads + 8-bit Adam) | 0.8 | 1.2 |
| activations at 4.5k / 6k / 8k tokens | 2.7 / 3.3 / 4.1 | 3.3 / 4.2 / 5.2 |
| logits at ~1000 labelled positions | 1.7 | 1.7 |
| total at 4.5k / 6k / 8k | 13.2 / 13.8 / 14.6 | 18.4 / 19.3 / 20.3 (15.5 / 16.4 / 17.4) |

  So r3-sized sequences (4-5k tokens) fit with 5+ GiB to spare, 8k fits, and
  WSL / the display take a GiB or two: set `--max-len 6144`. beats3 sessions
  are longer (replies ~60% longer, the bible adds ~300 tokens to the system
  prompt): `--student-prompt short` (system prompt ~1k instead of ~3.5k) brings
  an example back to ~3k and the epoch to about half. `stats --base
  Qwen/Qwen3-14B` prints these lines for the data's real length.
- **Time.** 14.8 / 8.2 = 1.8x the 8B's per-token cost: ~0.55-0.85k tokens/s with
  PEFT; 11M tokens an epoch is 3.6-5.6 h, two epochs 7-11 h (with the short
  prompt about half). Run `--max-steps 20` first as for the 8B; Unsloth
  (when it imports) supports Qwen3-14B 4-bit and is faster.
- **Serving.** Q4_K_M of the 14B is ~9.0 GB (8.4 GiB). KV cache at `num_ctx`
  16384: 2 x 40 layers x 8 KV heads x 128 x 2 bytes = 160 KiB a token = 2.5
  GiB (8B at 8192: 1.1 GiB); `OLLAMA_FLASH_ATTENTION=1` with
  `OLLAMA_KV_CACHE_TYPE=q8_0` halves it. Model + KV + buffers is ~11.5 GiB on
  the 24 GiB card, leaving ~12 GiB for TTS and the Live2D / VRM renderer
  (a GPT-SoVITS-class TTS 2-3 GiB, the GL renderer ~1-2 GiB). Decoding is
  bandwidth-bound: expect ~60-70 tokens/s against the 8B's ~125 (q8r3), a
  200-token reply in ~3 s, the first token slower with a cold 3-4k prompt.
  Create the Ollama model with `--num-ctx 16384` and run live with
  `--ollama-ctx 16384`; `scripts/eval_sft.sh`'s base-model line still sends
  8192.
- **Hidden-state tools** (`flow/train/hidden.py`) read the width from the model
  (5120 instead of 4096: 10 KB a token in fp16, not 8).

Commands: docs/todo.md, "q14r4 (after r4)".

### vLLM for student evals (2026-11)

Step 1 of the vLLM plan (`docs/todo.md`): the eval's student through a vLLM
server with the adapter as a LoRA module, sessions run side by side. For
evaluating and comparing students only; live stays on Ollama. On the GPU
machine (WSL), in a venv of its own (vLLM pins its own torch):

    nvidia-smi        # the driver must support CUDA 13 (the wheels are 13.0 only)
    python -m venv ~/vllmvenv && source ~/vllmvenv/bin/activate
    pip install vllm==0.33.0
    cd Tsukuyomi
    python -c "from flow.train import chat; open('/tmp/qwen3.jinja','w').write(chat.jinja('qwen3'))"
    vllm serve Qwen/Qwen3-8B --enable-lora --max-lora-rank 32 \
        --lora-modules q8r2=<run>/adapter q8r3=<run>/adapter \
        --chat-template /tmp/qwen3.jinja --max-model-len 10240 \
        --gpu-memory-utilization <tuned>

The template is `flow.train.chat.jinja(family)` (the adapter's tokenizer
carries the same text): it writes the empty think block itself and never
reads `enable_thinking`, so the requests send no `chat_template_kwargs`;
without `--chat-template` the server uses Qwen3's own and the prompts differ
from training. `--max-model-len`: the ~3k-token system prompt and a 10-turn
beats2 session are about 7.7k tokens (6 turns ~4.5k, 10 turns ~1.7x of
that for the history); 10240 leaves room for the reply. `--lora-modules`
names are what `STUDENT_MODEL` / `--model` take. Then, in the project's
venv (`.venv`):

    STUDENT_BACKEND=vllm STUDENT_MODEL=q8r3 CONC=4 bash scripts/eval_sft.sh

`--backend vllm` (`flow.director`) posts to `http://localhost:8000/v1`
(`--api-base`, `VLLM_URL` in the script; a key from `VLLM_API_KEY` only if
the server was started with `--api-key`, else a dummy one) with the
student's sampling in the body, as the Modelfile has it
(`flow.train.chat.SAMPLING`: temperature, top_p, `top_k`, and
`repetition_penalty` where a family sets `repeat_penalty`); the family is
`--vllm-family`, which the script reads from `$RUN/train.json` (qwen3
otherwise). Shell traffic goes through a local HTTP proxy: `NO_PROXY` must list
127.0.0.1,localhost.

`--conc N` / `CONC=N` (`flow.train.evalsft run`, student and live teacher;
the recorded run is local): N held-out sessions at once in threads, the
turns of a session in order; the records are written in the serial run's
order, the log lines are prefixed with the session id. vLLM batches the
requests and computes the shared system prompt once (prefix caching). With
the Ollama backend `CONC` only helps if the server runs
`OLLAMA_NUM_PARALLEL=N` (it also multiplies the context memory by N).

Numbers from vLLM are the bf16 base plus the adapter; live serves the
merged Q4_K_M in Ollama. The chosen model is re-evaluated through Ollama
(`bash scripts/eval_sft.sh` as before) before it counts.

### VLM judge and DPO pairs (2026-11)

The text checkers (`flow/checks.py`) see the markers, not the picture: a hand
out of the bust window, a wave the wrong way, a stiff run of identical beats
pass them. A vision-language judge looks at the avatar as rendered; the pair
builder combines both and also writes minimal-edit pairs that need no judge.
The trainer is the next section ("DPO training"). Code: `flow/judge.py`
(keyframes, sheet, judge, `python -m flow.judge`), `flow/dpo.py` (pairs,
calibration; `scripts/dpo_pairs.py` runs it), `tests/test_judge.py`,
`tests/test_dpo.py` (fakes only: no GL, network or model).

**Keyframes.** One reply on one avatar becomes a handful of stills, not a
video, so a reply costs about a second of GL:

- The reply's markers are parsed with the runtime's parser
  (`pose.parser_for(bm, moves=True, frames=True)`), timed as the offline
  director times speech (`director.timeline`, no TTS), solved (`pose.
  solve_poses`, `moves.solve_moves`) and played frame by frame through the
  layers the runtime uses (`MoveLayer`; `PoseLayer` with head carry, pose
  ease, the expression baseline and the end-of-speech release). So the state
  carries from marker to marker as it does live: a head angle stays until a
  marker sets it again, the face keeps its baseline, a hand comes back, a
  short reaction relaxes.
- A still is taken after each `[pose]` at its peak (start plus the longest
  eased move of its parameters, where the hold begins) and after each
  `[move]` at the frame of the move where its parameters are furthest from
  rest (a wave's raised hand, a nod's lowest head). A `[frame]` marker sets
  the camera of the stills after it; one nothing follows gets a still of its
  own; in bust the camera follows a whole-body move (`framing.follow_tokens`,
  `--no-follow` turns it off) and the still is full.
- The turn starts in the record's home framing (`framing.start`, else the
  camera line) and in its body state when the user turn's "(your body now:
  ...)" says it is away from rest (head, face and body angles; hands start at
  rest). A rest still of that start state is always the first tile.
- Not in the still: lip movement (the mouth stays at rest), the idle's drift,
  physics. A marker the body cannot play (unsolved pose, a move it lacks) makes
  no still: the judge sees what the viewer would see.

**Renderers.** `Live2DRenderer`: one `Live2DBackend` per avatar, calibrated
once with its `bust_view`, reused across replies (needs GL; `xvfb-run -a`);
the model comes from `avatars.json` or `<avatar>.model3.json` under
`--models`, `live2d-models`, `CubismWebSamples`. `FakeRenderer` (`--renderer
fake`) for tests. **VRM is not supported** (the judge returns NA, the pair
builder scores such turns on text alone): the browser page
(`flow.vrmrender.Page.frame`) can draw a posed frame headless, but it needs a
whole per-frame state from `flow.vrmbody`'s stream, a Chromium and WebGL on
SwiftShader, and nothing of that could be checked here. A Live2D avatar whose
model is not on the machine is NA too.

**Contact sheet.** At most `--tiles` 5 stills (the rest still plus the four
that tell most: moves and hand poses first, then a turn of head or eyes, then
the rest, in reply order), each `--tile` 320 px with a caption: its number
and camera, the first ~5 words the marker precedes (9 characters for
Chinese / Japanese; the caption font is a Noto/Windows CJK font when one is
found, else ASCII), the key targets ("head.yaw -15, hand.R.y @shoulder", "move
wave", "frame full"). The sheet is about 1600 x 420 px (it was 6 x 256 px: the same pixel
count within a third, so about the same image tokens, with a bigger face).
`python -m flow.judge render --data DIR --id SESSION/tNN [--reply "..."] --out
sheet.png [--renderer fake]` writes one (`score` also asks the judge).

**The judge.** One request: the sheet, the user turn with its tail lines
(body now, camera, reply language), the spoken text, the body's motion budget
(`avatars.json`: can its arms move, hand moves per framing) and a list of the
tiles. Temperature 0, images as base64 data URLs, JSON back, parsed leniently
(code fences, `<think>` blocks, "NA", out-of-range values clamped), one retry
on bad JSON, results cached on disk by (sheet hash, prompt hash, model,
backend): a rerun costs nothing. The rubric (the system prompt, verbatim in
`judge.RUBRIC`):

    Sides: "your right" / "your left" in a request mean the CHARACTER's own
    right and left. The character faces the viewer like a mirror: its right is
    on the VIEWER's LEFT in the picture. ...
    - request_done (0-2): did the viewer ask the character for a body action
      (point, wave, raise a hand, a pose, look or turn left / right / up /
      down)? 2 = visibly done in the frames, by the right part (a hand request
      needs the hand; a look needs the head or eyes), toward the right side;
      1 = partly; 0 = not done, the wrong side, or only the face moved. null
      when no body action was asked.
    - visible (0-2): are the body moves of the sheet (hands, arms, body)
      inside the picture of their camera and readable? 2 clearly; 1 partly
      cropped or tiny; 0 out of the picture (a hand below the bottom edge of
      a bust picture is not seen) or no visible change. null when the reply
      makes no hand, arm or body move.
    - expression_fit (0-2): does the face match what is being said?
    - natural (1-5): a natural streamer? 5 lively, varied, relaxed; 1 stiff,
      the same pose again and again, jerky or extreme.
    - issues: up to four short strings; JSON only.

Backends (`--judge-backend`): `siliconflow` (OpenAI-compatible, key in
`SILICONFLOW_API_KEY`, default model `Qwen/Qwen2.5-VL-72B-Instruct`), `vllm` (a
local server, `--judge-api-base`, default model `Qwen/Qwen3-VL-8B-Instruct`),
`anthropic` (the director's backend; no temperature control), `fake` (tests:
reads the sheet's own metadata). The OpenAI-compatible ones use
`director.OpenAICompatBackend` with `temperature: 0`, not the student's
sampling. The model names are from memory, not checked against the providers'
lists (nothing was called): check SiliconFlow's model list (a Qwen3-VL 32B or
235B instruct and Qwen2.5-VL 72B / 32B instruct were there when last seen) and
the Hugging Face id before a run. Cost: ~1.8k input tokens (rubric ~550, user
turn and tiles ~600, image ~500-700) and ~70 output a call, so about 1.8M
input + 0.07M output tokens per 1000 calls: a few US dollars on a 72B-class
VL at SiliconFlow-like prices (~$0.5-1.5 per million tokens, unchecked), well
under a dollar for a 30B-class model, nothing but GPU time on a local
Qwen3-VL-8B (an RTX 3090 Ti serves it at 24 GB with room for batches; do not
run it next to the student's server). `--max-calls N` stops the run when the
budget is spent (cache hits are free; rerun with the same `--judge-cache`);
the run prints calls, retries, unparsed answers, cache hits, tokens and, with
`--price IN OUT`, a cost.

**Pairs** (`scripts/dpo_pairs.py`). Per turn of a data folder (r3, or one
side of a training run's `split.json`: `--split ... --side eval`), the prompt
is exactly the SFT export's for that turn (`dataexport.sft_examples`, per
turn: system prompt, history, user turn; same corrections, manners, language
and request filters via `dataexport.prepare` (which `dataexport sft` calls
too), the same flags, and like the export no turns of live2d-lite avatars
unless `--keep-lite`; `--student-prompt short` for a student trained on it) with the final assistant message cut; the teacher's
reply is the export's target. Candidates are K student samples (`--k 4
--backend vllm --model q8r3 --conc 8`, the student's sampling, a seed per
sample) or `--candidates FILE.jsonl` ({"id", "reply"}; an `evalsft`
`student.jsonl` works). `--teacher-chosen` adds the teacher's reply to the
pool. Every candidate gets the text checks (`flow.checks`, as its own session
after the history's last reply, so `template_repeat` is real) and, unless
`--no-judge`, a sheet and a judgment. One score (higher is better):

    score = -25*H - 1*S - 5*(2 - request_done) - 5*(2 - visible)
            + 1*expression_fit + 0.5*(natural - 1)

H: hard text failures, each failing check once: `look_sign` (wrong side or no
turn), `body_request_obeyed` / `wave_obeyed` (a skipped request), `language`,
`template_repeat`. S: soft: `opening` (the q8r2 template), `beats` <
sentences, `quiet_says_quiet`, `frame_before_wave`. The judge's NA counts as
2 (no penalty); with no judgment the score is the text part. Each level
outweighs everything below it: one hard failure (25) more than both primary
scores can move (20) plus the secondary range (4); one primary step (5) more
than the whole secondary range. chosen = the best candidate with no hard
failure and no visibility flag, rejected = the worst of the rest, kept when
the gap is at least `--min-gap` (5: a pair rests on a hard failure or a
request / visibility difference; lower it to learn from expression and
naturalness too), the replies differ and (`--min-natural`) the chosen one
looks natural. **Contradictions**: (a) a request check failed and the judge
says `request_done` = 2, or (b) the request checks passed and the judge says
`request_done` = 0 while it sees a visible body move (`visible` 1 or 2): one
of the signals is wrong and we cannot tell which, so the candidate is out of
the turn (neither side). When the checks pass and `visible` is 0 or null (a
hand marker the viewer does not see, e.g. a hand below the bust window: the
judge may call that "not visible" or "no move") with `request_done` = 0, or
`request_done` is not 0 but `visible` is 0, visibility is the reason: the
candidate may be the rejected reply, never the chosen. The teacher's reply
(`--teacher-chosen`) is a chosen candidate only: it is the SFT target, so it
is never the rejected one, and a student reply identical to it counts as it. Dropped turns are counted by reason
(no_candidates, contradictions, no_clean_chosen, identical, gap, natural,
judge_budget).

Output in `--out`: `dpo.jsonl` (TRL's conversational preference format:
`prompt` the chat messages, `chosen` / `rejected` one assistant message each),
`dpo.meta.jsonl` (same line order: id, avatar, kind, scores, text verdicts,
judge outputs, flags), `candidates.jsonl` (the samples, to rerun without
sampling), `calibration.json`, `summary.txt`; the summary printed: pairs by
kind and avatar, drops, contradictions, judge agreement, calls.

**Minimal-edit pairs** (`--synthetic N` per kind; the teacher's reply is
chosen, a copy that breaks one thing rejected; no judge; the spoken text and
every other marker are untouched): `look_flip` (the first turning marker's
`head.yaw` and `gaze.x` change sign; on look requests the teacher got right),
`hand_drop` (hand, move and gesture markers removed or their hand keys
stripped; on hand and wave requests it fulfilled), `template_open` (the first
beat becomes the old q8r2 opening: brows 0.7, pitch 5, yaw -5, dur 0.2, hold
0.4), `beat_dup` (a beat repeated before the next beat or after the next
sentence), `wave_frame` (a `[frame full]` inserted right before the first
wave move that has none before it: q8r3 does that 7 of 22 times, the teacher 1
of 22; `frame_before_wave` sees it), `correction` (below) and `pose_copy` (below), `look_at_me` (on "look at me / at the camera" lines: chosen has head.yaw within 3 degrees and gaze.x within 0.15 in every pose marker that sets them, the teacher's reply when it does, else with the off-centre values set to 0 or, with none set, `head.yaw: 0, gaze.x: 0` added to its first pose; rejected turns every such marker 12-20 degrees sideways with gaze.x 0.03 per degree on the same side; the chosen meta `src` is `teacher+edit` when the reply was edited; `look_at_me` is also a hard failure in the candidate score), `opening_repeat` (the first beat becomes
the student's most common opening, `--opening-from runs/eval_q8r3/student.jsonl`,
from `flow.checks`' opening stats; where the teacher's opening differs; q8r3's
top opening share is 0.20, the teacher's 0.08). Each kind is the one thing its
checker sees (`look_sign`, `wave_obeyed` / `body_request_obeyed`, `opening`,
`signature`, `frame_before_wave`, the exact opening). `--kinds` picks kinds,
`--kind-n look_flip=100,wave_frame=40` a count each (the summary prints the
pairs made per kind, and the user lines' language: ja / zh / en, also in each
pair's meta as `lang`), so `look_flip` can be run apart, e.g. for Japanese
lines, and mixed in with a weight by the trainer. The look detector
(`checks.asked_side`) also reads the character's own side named without a verb:
そっちの / 君の / あなたの + 右側 / 左側, 你的 + 右边 / 左邊 (and 〜から見て右側,
which it read before); the convention is the same (the character's right is
negative). The judge is calibrated on the first four kinds only: it cannot see
a frame or an opening.

`correction` (live: the user typed "umamusume", then ウマ娘だ！うますむめじゃな
い, and she kept saying うますむめ, copying her own history). Chosen: the
teacher's reply, which says the right form Y and not the wrong X; rejected:
the same reply with Y swapped back to X in its speech (markers untouched).
Two sources, in the pair's meta `info.source`: `real` (the turn's own user
line corrects a word, `checks.corrections`; there are few in the simulated
data) and `injected`, which is **synthetic**: from an ordinary turn whose
teacher reply shares a word W (a Latin word of 4+ letters, a katakana / kanji
run of 3+, a long hiragana run) with her previous reply, the previous reply in
the prompt gets W misspelled to X (two neighbouring characters swapped), the
user line is replaced by a correction in the language of the original line
("Xじゃなくて Wだよ！", "It's \"W\", not \"X\"!", "不是X，是W！" ...) and the
teacher's reply is the chosen side. The teacher never answered that
correction line; it only said W. The pair therefore teaches "after a
correction to W say W, never X", not how to word the answer; the original
line is kept in `info.user_was`. Quiet turns and first turns are not used.
`pose_copy` (live: the same four poses on turns 17-21): rejected pastes the
poses of the most recent earlier reply (of the last 3, from the turn's
history) onto this reply's speech, in place, markers only; chosen is the
teacher's reply, used only when it does not itself copy one of the last 3 and
the paste is one the `pose_copy` checker flags. Both checkers are hard
failures in a candidate's score (`HARD`).

**Calibration** (`--calibrate N`): N minimal-edit pairs, balanced over the
kinds, judged on both sides; per kind how often the judge scores the intact
reply higher. Output:

    kind             n  na  overall w/t/l  rate own dimension w/t/l  rate
    look_flip        n  na    w/  t/  l     %    w/  t/  l     %  (request_done)
    hand_drop        n  na    w/  t/  l     %    w/  t/  l     %  (request_done)
    template_open    n  na    w/  t/  l     %    w/  t/  l     %  (natural)
    beat_dup         n  na    w/  t/  l     %    w/  t/  l     %  (natural)
    win: the judge scores the intact reply higher; tie: equal; loss: it prefers the broken one

(The layout only; no real run yet.) rate = (wins + half the ties) / n;
`overall` is the vision part of the score, "own dimension" the one rubric
item the kind breaks (request_done for look_flip and hand_drop, natural for
template_open and beat_dup). `calibration.json` has the counts.

**On the GPU machine** (WSL, `.venv`, repo at the repo; Live2D
renders need GL, so `xvfb-run -a` as for the other renders; `NO_PROXY` must
list 127.0.0.1,localhost; the 3090 Ti has room for the student on vLLM and
the judge on a second vLLM only if each gets about half of the memory
(`--gpu-memory-utilization`), otherwise sample first, stop the student's
server, then judge):

    # 1. serve the judge (a second venv or the same vllm venv)
    vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 8192 \
        --limit-mm-per-prompt '{"image": 1}' --port 8001
    # 2. calibrate: can the judge be trusted?
    xvfb-run -a python scripts/dpo_pairs.py --data data/datagen/r3 \
        --split ~/runs/sft/q8r3/split.json --out ~/runs/dpo/cal \
        --calibrate 80 --judge-backend vllm \
        --judge-api-base http://127.0.0.1:8001/v1 \
        --judge-model Qwen/Qwen3-VL-8B-Instruct \
        --judge-cache ~/runs/dpo/cache --conc 8
    # 3. pairs from student samples (the student's vllm at :8000; --max-calls
    #    caps a SiliconFlow run: --judge-backend siliconflow --judge-model ...)
    xvfb-run -a python scripts/dpo_pairs.py --data data/datagen/r3 \
        --split ~/runs/sft/q8r3/split.json --out ~/runs/dpo/q8r3 \
        --backend vllm --model q8r3 --k 4 --conc 8 \
        --judge-backend vllm --judge-api-base http://127.0.0.1:8001/v1 \
        --judge-cache ~/runs/dpo/cache --synthetic 50 --teacher-chosen

`--candidates runs/.../student.jsonl` instead of `--k` takes samples made
elsewhere. Look at a sheet by hand first (`python -m flow.judge render ...
--out sheet.png`): the stills are what the judge sees.

**The judge is trusted only after calibration.** Minimal-edit pairs are the
easy case (one thing differs, the teacher's reply is the intact side): a judge
that cannot rank them cannot rank student samples, and one that can has still
not been checked on subtler differences. Use the judge's part of the score
only for the kinds it wins clearly (a hand or look the sheet shows, which is
what `request_done` and `visible` measure); a judge that ties or loses on
`look_flip` has the left / right convention wrong (check the sheet's captions
and the prompt's mirror sentence first) and must not feed pairs. The judge
is blind to the markers' targets by default: the tile captions and the tile
list give each still's number, camera and the words it comes before, never
`head.yaw -15` or `move wave`, so it has to read the frames (the text
checkers already score what the targets say). `--show-targets` puts them
back, for comparing. What a judge can still read without the image: a
dropped hand marker makes one still fewer, and `beat_dup` repeats a tile, so
read `hand_drop` and `beat_dup` as an upper bound on what the image gives;
`look_flip` is the clean vision test. Expression
and naturalness are the least checkable: the default `--min-gap` keeps them
from deciding a pair alone. Untested here: the Live2D renders (no GL, no
model in this container), the VRM path (unsupported), any real VLM answer.

**Calibration results and settings (2026-11).** 80 minimal-edit pairs
(`--calibrate 80`), judge win rate = intact reply scored higher (ties half):

| judge | hand_drop | look_flip | template_open | beat_dup |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct (SiliconFlow) | 91% | 47% (13 of 16 ties) | 67% (mostly ties) | 68% (mostly ties) |
| Qwen3-VL-8B (vllm) | 59% | - | - | - |
| GLM-5V-Turbo | every call failed, 0 tokens | | | |

Decisions. (1) The judge is the 32B instruct, for `request_done` and
`visible` only: a dropped hand is seen (91%), the rest are ties. (2)
`--judge-dims` (default `request_done,visible`) are the only judge fields in
the pair score; `expression_fit` and `natural` stay in the meta (and
`--min-natural` still reads `natural`) and add nothing to the score. Add them
back with `--judge-dims request_done,visible,expression_fit,natural` for the
old formula. (3) The look side is the text checker's: ±10-15 degree head
turns are hard to see in a tile, and `look_sign` checks the sign exactly, so
for a look request the judge's `request_done` is neither scored nor compared
with the checker (no contradiction is raised; `visible` still is). The
calibration's `look_flip` overall column is therefore all ties by design; its
own-dimension column still shows what the judge says. (4) Tiles are 320 px,
at most 5 (in bust the tile is already head and chest, so the face is about
2x the pixels it had). (5) A failing judge call prints the provider's status
and the first 300 characters of its body to stderr once per distinct message,
an empty 200 answer prints its stop reason and usage, and the summary counts
`failed` (raised) apart from `unparsed` (answered, no JSON) and lists the top
errors: run GLM-5V-Turbo once with `--max-calls 2` to read why it fails.

**Eye smile only shows on a closing eye (2026-11).** On Cubism rigs
ParamEyeLSmile / RSmile reshapes the *closing* eye: Hiyori with smile 1 and
open 1 renders as rest; the arc appears as open falls (0.5: a slight happy
squint, 0: the closed happy arcs). The body drove `eye.*.smile` alone, so the
model's most used face beat (~4k targets in r2) and the micro layer's smile
were invisible on such rigs, live included. `flow.squint` fixes it on the
render side only (the solver's targets and reach, the dataset, the action
head's data and the SFT export are unchanged): where Live2D parameters are
drawn (`flow.stream`'s frame for live, `flow.runtime`'s render loop for the
director's offline renders, the judge's `Live2DRenderer`), after the pose
layer, the expression baseline, the micro layer and the blink,

    open = min(open, rest - 0.6 * s * (rest - closed))

with s the smile in 0..1 of its range, per side: s = 0.8 is about half
closed (a clear squint), 0.3 a gentle one, 1 never fully closed, and a blink
(or any lower open) wins by the min. Detection is on the Cubism Core, no GL
(`l2dprobe.core`): the summed per-drawable vertex shift of smile 0 -> 1 at the
open eye's rest value against at 30% of the open range; the coupling is on
when the first is under 20% of the second. It is computed lazily from the
model3.json (cached per process; the body file's declared ids are used when
there is one), not stored in the body, so existing body files work; no Core or
model means no coupling. Sample rigs: Hiyori, Mao, Natori, Haru and Ren need
it (shift at open 0-0.004, at 30% 0.03-0.08 model units); Mark, Rice and
Wanko have no eye smile parameter. `--no-smile-squint` turns it off (director,
live, runtime, judge). Before / after strips (s = 0, 0.3, 0.8, 1, and a blink
during 0.8) were rendered for Hiyori and Mao.

### DPO training (2026-11)

`flow/train/dpo.py` (`scripts/train_dpo.py {stats,train,merge,gguf,modelfile}`,
`tests/test_train_dpo.py`) trains the student on `dpo.jsonl` with the project's
own trainer code (SFT's tokenization and masking, 4-bit loading, LoRA,
checkpoints, speed line); no TRL. Why not TRL: the pinned transformers is 5.x
and TRL tracks its own range of it, its `DPOTrainer` would have to be taught
our chat template (the empty think block on every assistant turn) and the
history masking, and it adds a dependency for about 300 lines of loss and
bookkeeping. If a later TRL is wanted anyway, `dpo.jsonl` is already in its
conversational format.

**What it does.** The start is the SFT run's merged fp16 model
(`<sft run>/merged`, from `train_sft.py merge`), loaded in 4-bit NF4, with a
new LoRA (rank 16, alpha 32, the same projections). The reference is the same
model with the LoRA off (`disable_adapter()`): the SFT student, exactly, in
the same memory. With `--precompute-ref` (default) its summed reply log-probs
for every pair are computed once before training (adapter off, no grad) and
kept in `<out>/ref_logps.json`, keyed by the pair's token ids and the base's
files, so a rerun, a resume or another beta reuses them and a step is one
forward per side. The loss is -log sigmoid(beta * [(pi_c - ref_c) - (pi_r -
ref_r)]) on the summed log-probs of the reply tokens (the reply and its
`<|im_end|>`; the system prompt, the history and the user line carry no
label), plus `--sft-alpha` (0.2) times the chosen reply's mean NLL (RPO, so it
stays near the SFT distribution); `--length-norm` makes the log-ratios per
token. The prompt is cut only when a pair is over `--max-len` (auto: p99 of
the pairs): the oldest history turns go, whole, never the system prompt, the
last user line or a reply; a pair that still does not fit is left out (both
counted and printed). Pairs whose chosen and rejected are the same text are
dropped. `--data a.jsonl:1,b.jsonl:2` mixes files with weights (a pair is
repeated that many times; 0.5 is a chance of one). Held out by session
(`--eval-frac` 0.1) for the eval accuracy and margins; the log has loss,
`rewards/chosen`, `rewards/rejected`, `margin`, `accuracy` (the implicit
reward prefers chosen), `nll`, and `accuracy/<kind>` and `margin/<kind>` per
pair kind (from `dpo.meta.jsonl` beside the pairs).

Defaults: batch 1 pair, `--accum 8`, lr 5e-6, 1 epoch, beta 0.1, cosine with
10% warmup, paged 8-bit AdamW, gradient checkpointing, bf16, eval and
checkpoint every 25 steps (the last 3 kept), resume from the last checkpoint
unless the tokenized pairs changed (`data.json` holds their hash; `--restart`
to start over). The logits are computed only where a label is.

**Fair comparison.** `--sft-run runs/sft/q8r3` copies the SFT run's
`split.json` to `<out>/split.json` and the data folder into `<out>/data.json`,
which is all `scripts/eval_sft.sh` reads, so `RUN=<dpo run>` scores the
student on the SFT run's held-out sessions. Pairs from those sessions are left
out of training (counted; `--allow-eval-sessions` keeps them and the
comparison is no longer fair). So build the pairs on the TRAINING sessions
(`dpo_pairs.py --split <sft run>/split.json --side train`); the minimal-edit
pairs are the teacher's own replies and need nothing else. (The checklist in
`docs/todo.md` once said "held-out sessions": that would be both the eval and
the training set.) The DPO run's own held-out sessions are in
`<out>/dpo_split.json`.

**End to end, on the GPU machine** (WSL, `.venv`, repo at
the repo, runs in ~/runs):

    # 1. q8r3 samples on the training sessions, pairs. The judge (32B) runs on
    #    the CPU server or here; copy dpo.jsonl and dpo.meta.jsonl over with
    #    scp (PowerShell) or git if it ran there
    xvfb-run -a python scripts/dpo_pairs.py --data data/datagen/r3 \
        --split ~/runs/sft/q8r3/split.json --side train \
        --out ~/runs/dpo/q8r3 --backend vllm --model q8r3 --k 4 --conc 8 \
        --judge-backend siliconflow --judge-model Qwen/Qwen3-VL-32B-Instruct \
        --judge-cache ~/runs/dpo/cache --teacher-chosen
    # 2. the minimal-edit pairs for q8r3's failures, no judge (look_flip
    #    on Japanese lines can be a run of its own: --kinds look_flip, mixed
    #    in by weight)
    python scripts/dpo_pairs.py --data data/datagen/r3 \
        --split ~/runs/sft/q8r3/split.json --side train --no-judge \
        --out ~/runs/dpo/syn --kind-n wave_frame=300,opening_repeat=300,look_flip=300 \
        --opening-from ~/runs/sft/q8r3/eval/student.jsonl
    # 3. what the trainer sees: pairs by kind / avatar, lengths, cuts
    python scripts/train_dpo.py stats --sft-run ~/runs/sft/q8r3 \
        --data ~/runs/dpo/q8r3/dpo.jsonl:1,~/runs/dpo/syn/dpo.jsonl:1
    # 4. train (a probe first: --max-steps 10 prints tokens/s, memory and
    #    the step-0 check)
    python scripts/train_dpo.py train --sft-run ~/runs/sft/q8r3 \
        --data ~/runs/dpo/q8r3/dpo.jsonl:1,~/runs/dpo/syn/dpo.jsonl:1 \
        --out ~/runs/dpo/q8r3d
    # 5. merge into the SFT merged model, GGUF, Ollama (the SFT commands)
    python scripts/train_dpo.py merge --run ~/runs/dpo/q8r3d
    python scripts/train_dpo.py gguf --run ~/runs/dpo/q8r3d \
        --llama-cpp ~/llama.cpp --name tsk-q8r3d
    python scripts/train_dpo.py modelfile --run ~/runs/dpo/q8r3d \
        --name tsk-q8r3d --create
    # 6. eval on the SFT run's held-out sessions, then the checkers side by side
    RUN=~/runs/dpo/q8r3d STUDENT_MODEL=tsk-q8r3d bash scripts/eval_sft.sh
    python scripts/checks.py ~/runs/sft/q8r3/eval ~/runs/dpo/q8r3d/eval \
        --data data/datagen/r3 --by avatar

**Memory (estimates, not run on the GPU).** The 4-bit 8B model about 5.5 GB,
the LoRA, its 8-bit optimizer state and gradients under 1 GB, activations with
checkpointing about 1.5 GB per 4k tokens and the two sides' graphs are held
until the backward (each side is a full prompt + reply; the prompt is not
shared), the head over the reply tokens only (a few hundred tokens x 152k x 4
bytes). Expect 9-13 GB peak and, from SFT's speed, a few seconds a pair; the
speed line prints the real numbers after 10 steps. The reference pass is one
forward a side without grad, a fraction of one epoch.

**What to watch.** The first lines: `step 0: |policy - reference| ... (must be
~0)`; if it is not, the reference cache and the policy disagree (another base,
dtype or tokenization) and nothing after it means anything. During the run:
eval `accuracy` rising to about 0.8 or more is normal, with `margin` growing
slowly. Drift: the margin exploding (tens, with loss near 0), `rewards/chosen`
falling below 0 (the chosen reply becomes less likely than under the SFT
student), the `nll` climbing, or replies getting short or odd in the eval: lower
`--lr` or `--beta`, or raise `--sft-alpha` (0.5-1). `accuracy/<kind>` shows
which pairs it learns first; the synthetic kinds are easy, so a high accuracy on
them with a flat accuracy on the judged pairs is expected. After the eval:
`look_sign`, `frame_before_wave`, `opening` (top share), `body_request_obeyed`,
`wave_obeyed`, `language`, `beats` against q8r3, then the judge on the same
replies, then Ollama as live. Not run: any real GPU step, the 4-bit path, the
cache against the real merged model.

### Action head ablations on BEAT2 (2026-10)

The first BEAT2 run (text only, 54 h, 27.6k segments) came out at normalised
MSE head 1.065, rest 1.133, mean 1.075, with the eval loss best near step
2000 and rising after: absolute MSE from text alone measures each segment's
posture, not the gesture (pitch / tilt / yaw: head = mean >> rest; the
hands 5-10% under rest). So the head can now also see its own past motion
(`--history`), the speech's prosody (`--prosody`), a learned mix of the
stored layers (`--layer-mix`) and per-avatar statistics (`--norm avatar`),
and `eval` scores what a gesture is: motion around the posture, velocity,
how much motion the prediction has, and whether it lines up with the
audio. Code and tests ran on CPU on synthetic data only; none of the
numbers below exist yet.

Hidden states with the three layers, once (WSL, repo `the repo`;
the `=` after `--layers` is needed, a bare `-1,-8,18` is read as an
option; `--proj 1024` projects each layer to 1024, so 3072 numbers a token
and `--layer-mix` splits them back into three):

    cd Tsukuyomi
    python scripts/extract_hidden.py --export data/beat2/seed-san/action \
        --context none --text spoken --model ~/runs/sft/q8/merged \
        --layers=-1,-8,18 --proj 1024 --out ~/runs/hidden/beat2

The grid, one script, a log per run in `~/runs/ah/grid/<run>/`
(`train.log`, `eval.log`, `eval.json`, `head.best.pt`), the table at the
end (also `table.txt`):

    bash scripts/ah_grid.sh                       # 5 runs x 4000 steps
    STEPS=5000 PATIENCE=10 bash scripts/ah_grid.sh
    EXTRA="--layer-mix" bash scripts/ah_grid.sh   # a flag on every run

| run | flags | answers |
|---|---|---|
| a_text | (none) | the head as it was |
| b_hist | `--history 1` | what 1 s of its own motion adds |
| c_hist_pros | `--history 1 --prosody` | what the audio's prosody adds |
| d_no_text | `--history 1 --prosody --no-text` | c without the text: what the text adds is c - d |
| e_fm | c with `--objective fm` | whether sampling beats regression (it does not average several motions into a smaller one) |

Each run trains `--steps` (default 4000 in the script; 3000-5000 is the
range to try), evaluates every 100 and keeps `head.best.pt` at the best
eval loss; `eval` uses it (`--ckpt` takes the run folder, or a file:
`head.pt` is the last step). With a history the eval loss is on the
middle chunk of each held-out segment after a clean history. Every run is
scored on the same frames: the first `SKIP` s (default 1.0, the history)
of every segment are not scored for anyone, a is not credited for them and
b to e are not handed them as ground truth to be scored on. Run one by hand:

    python -m flow.actionhead train --export data/beat2/seed-san/action \
        --hidden ~/runs/hidden/beat2 --out ~/runs/ah/hp \
        --steps 4000 --history 1 --prosody
    python -m flow.actionhead eval --ckpt ~/runs/ah/hp \
        --export data/beat2/seed-san/action --hidden ~/runs/hidden/beat2 \
        --skip 1.0 --json ~/runs/ah/hp/eval.json
    python -m flow.actionhead table ~/runs/ah/*/eval.json

**Reading the summary** (columns: head, rest = the first frame held, mean =
the training mean, hold = the last frame before the scored part held; lower
is better except the ratios and correlations):

- `normalised MSE` is the old number, posture included. Expect rest and
  hold to be hard to beat; the head's win here means it knows where the
  body is.
- `relative MSE` is each trajectory minus its own clip mean: the motion
  around the posture. Rest, mean and hold are constants, so they all score
  the clip's own variance, which is the number to beat: a head under it
  predicts motion that is right, a head over it moves wrongly (worse than
  standing still).
- `velocity MSE` is the same on frame differences: smooth, right-looking
  motion that is out of step still scores badly here.
- `std ratio` is predicted over true motion std. Near 0: the head collapsed
  to the mean (what an MSE head does when it learns only the posture); near 1: it moves as much as the
  real mocap; the velocity ratio under the std ratio says over-smoothing.
  A head with a good normalised MSE and a ratio near 0 learned the posture
  and nothing else.
- `beat hands / head`: `corr` is the correlation of motion speed with the
  audio's accent strength (|d rms|), `hit` the share of accents with a
  motion-speed peak within +-0.2 s, `prec` the share of speed peaks within
  0.2 s of an accent; each followed by the ground truth's value, which is
  the ceiling (real people only partly gesture on the beat). The head
  without prosody (a, b) can only be near chance here; c should approach
  the ceiling if the audio helps, d shows how much of it is the prosody
  and the history alone.
- Compare c with d for the text, b with a for the history, c with b for
  the prosody, e with c for fm; `layer mix` prints the learned layer
  weights (layers 36, 29, 18 for `-1,-8,18`). Differences smaller than
  two runs' spread (rerun with `--seed 1`) are not results.

#### Probes: why the grid came out as it did (2026-10)

The grid says no regression head beats a constant on the motion around
the posture, text adds nothing, and prosody heads tie hand speed to
loudness far more than people do. `flow.probe` asks why, with no
controller: two questions, each with the control that would show an
answer to be speaker or recording identity. Spec in the module docstring.
Code and tests ran on synthetic data only (CPU); none of the numbers
below exist yet.

    cd Tsukuyomi
    bash scripts/beat2_probe.sh                    # prosody + match, ~/runs/probe
    MODE=prosody bash scripts/beat2_probe.sh       # numpy only, minutes
    LIMIT=2000 STEPS=500 bash scripts/beat2_probe.sh    # a quick look

or by hand (export `data/beat2/seed-san/action`, hidden states
`~/runs/hidden/beat2`, layers `[36, 29, 18]` projected to 1024):

    python -m flow.probe prosody --export data/beat2/seed-san/action \
        --out ~/runs/probe
    python -m flow.probe match --export data/beat2/seed-san/action \
        --hidden ~/runs/hidden/beat2 --out ~/runs/probe --layer 1
    python -m flow.probe match ... --split speaker --layer all   # new speakers
    python -m flow.probe all ... --sem ~/BEAT2/sem               # + semantic windows

Both write `report.txt` and `report.json` (`prosody`, `match`). `--layer`
takes an index into the stored layers (1 = layer 29, the default), a layer
number, or `all` (a learned mix, three times the RAM; estimate: the states of
one layer are ~3 GB). `--limit N` keeps whole clips until N segments are in.
The budget of `match` is `--steps 3000 x --batch 128` per variant, six
variants. Measured on CPU with random states (D = 1024, 128 windows of
~20 tokens): assembling a batch takes 10 ms for one layer, 20 ms for all
three, 60 ms at 48 tokens a window; scoring 256 windows 0.04-0.15 s. That
is the cost that stays on the CPU on a GPU machine, so a variant is about
2-6 minutes (3000 steps, 12 curve checks, the final scoring), the six of
them 15-25 minutes for one layer and up to ~40 with `--layer all`, plus
loading the shards. An estimate from those numbers, not a run on the 3090
Ti. `prosody` is numpy and takes minutes.

**prosody** (the ground truth only). Per segment, six voice features
(loudness, its range, voiced share, f0 and its range, speaking rate in
units per voiced second -- characters for zh / ja, space-separated words
for en / es --, pauses) against the hands' and head's amplitude (std and
p95 - p5 around the segment mean), speed (mean, p90) and the share of
frames with the hands moving; then 2 s windows with a lag scan, and the
frame-level accent correlation of the grid (reproduces the 0.03). Every
number has a 95% CI from resampling whole clips. Read the
`within speaker` tables (z-scored per speaker, then pooled), not
`overall`: where the two differ, speaker style was driving it. The
`no dur` table removes segment length (a longer segment has more spread by
itself). In the tables `*` marks a CI that excludes 0, the last two
columns average |r| over the four amplitude and the four speed columns.
The two hypotheses are at the end of the report, with the paired
difference of the two |r| and its CI:

- loudness moves the amplitude more than the speed: `loud` row, `hand_std`
  / `head_std` against `hand_v` / `head_v`; supported when the difference
  is positive with a CI above 0. A lag scan that peaks away from 0 says who
  follows whom (positive: motion after the voice).
- the speaking rate moves the speed more than the amplitude: `rate` row,
  the same columns the other way round; supported when `hand_v` beats
  `hand_std` with a CI above 0.

Amplitude and speed are not independent measures (a faster swing of the
same size is faster), so read the whole row, not one cell. If the within-
speaker numbers are small (|r| under 0.1-0.2) the voice carries little of
the motion in the mocap itself, and a head that reads loudness as speed
has learned something the data does not support.

**match** (text and prosody against motion). 3 s windows; the motion, the
tokens from 1 s before the window to its end, and the window's prosody are
encoded to 128-d and matched with a symmetric InfoNCE. Each variant is
scored on held-out clips (or speakers) as top-1 / top-5 retrieval among
K = 32 candidate motions for a text, under three ways of drawing the 31
wrong ones: `random` (another clip), `speaker` (the same speaker, another
clip) and `clip` (the same recording, a different time). Chance is 1/32 =
0.031 and 5/32 = 0.16.

| variant | trained on | tells you |
|---|---|---|
| text, prosody, text+prosody | batches of one speaker | what each carries about the motion once the speaker is no help |
| text_shuffled | the tokens of another window | the pipeline: must be at chance everywhere (a CI above chance is a leak) |
| text_random | random batches | how much of the score is speaker or language, which random negatives allow; compare its `random` column with `speaker` and `clip` |
| text_clip | batches from one recording | the hardest: speaker, recording and language the same |

How to read it:

- Text carries gesture information if `text` is clearly above chance in the
  `speaker` and `clip` columns (the CI's low end above 0.031, and more than
  `text_shuffled`) on held-out clips; the `clip` column is the strictest.
  Better still on `--split speaker` (speakers never seen in training).
- Text beyond the voice: `text+prosody` above `prosody` in the same column.
  `prosody` above `text` means the audio is the signal and the LLM states
  add nothing, as the grid's d - c suggested.
- Everything at chance in `speaker` / `clip` but high in `random` for
  `text_random` is the speaker-identity finding: the states say who is
  talking and in what language, not what the hands do.
- All variants near chance with train top-1 rising in the curves is
  overfitting of a probe that is too large for the signal (the text is
  1024-d per token): lower `--steps`, or `--dim 64`, before concluding.
  The report is the last step; nothing is chosen on the held-out windows.
- `--center` removes the window's mean pose, leaving only the gesture
  (posture identifies a recording and a speaker; it is what the MSE grid
  mostly scored).
- `--sem DIR` (BEAT2's semantic annotations; the format is detected, not
  known: a file `<clip>.txt|csv|tsv|json` with start and end seconds, and
  a gesture type or a relevance score) splits the accuracy into windows
  that overlap a semantic gesture and those that do not; if nothing parses
  it says so and goes on. Text helping only on the semantic windows would
  be the case for a text-driven gesture layer for the rare semantic ones.

## E4: BEAT2 retargeted (2026-10)

The head learned little from the teacher's trajectories (normalised MSE
1.07 against 1.18 for holding still): the markers' motion choices are
partly arbitrary. `flow/beat2.py` puts real human co-speech motion in the
same files. Built and tested on synthetic BEAT2-shaped clips
(`tests/test_beat2.py`); it has not seen the real data, so the first run
is the check (below).

- **Retarget by rotation transfer.** SMPL-X's template and three-vrm's
  normalized rig (what `flow.vrmbody` drives) are both T-poses whose bones
  have the identity rotation at rest, y up, facing +Z, the character's left
  on +X: one frame, so a joint's world rotation is the same rotation on
  both, no SMPL-X body model (its mesh and licence are not needed) and no
  VRM 0 special case (the body file and the page already turn VRM 0 into
  the 1.0 frame). The world rotations are fitted to the body's own
  parameters top down (torso, head, shoulders, upper arm from its direction,
  the elbow from the angle to the forearm, the twist chosen so the forearm
  lands on the mocap forearm, the wrist from the rest; forearm roll goes to
  the wrist), each joint measured against what the joints above it achieved
  after their limits. The avatar's own T-pose droop (`align`) drops out.
- **Root.** Everything is taken in the actor's own pelvis frame (global
  orientation, per frame) turned by the clip's habitual spine posture (median
  pitch / tilt of spine3 against the pelvis), so the torso values are the
  spine's motion against the pelvis, not the mocap's global frame: on the
  first real clip (Seed-san) the torso sat at pitch -22 / -12 / -5 deg
  (p5 / p50 / p95, range +-15) and yaw -40 / +5 / +29 (range +-25) until
  this, a tilted pelvis and whole-body turns read as torso lean and twist;
  now -8 / 0 / +7 and -5 / +1 / +8. The translation, in the frame of the
  torso's heading (circular moving mean over 20 s, `--facing-window`; 0 =
  the clip's mean), high-passed, is `hips.x` / `hips.y` (clipped to 0.15 /
  0.08 hu). A z-up clip is detected from the spine and turned. Legs unused.
- **Variables.** The same columns the data engine records for the avatar
  (`flow.dataset.traj_vars`), from the body's forward model; gaze, eyes,
  brows, mouth and visemes are NaN (the loss masks them; the voice owns the
  mouth; no FLAME mapping). `<key>.joints` keeps the 23 fitted parameters.
- **Filter** (GMR's lesson). Per frame, before the limits clamp: arm in
  trunk / head and arm in arm (the collision model), palm speed, joint past
  its limit. Bad frames (defaults: 0.10 hu, 25 hu/s, 25 deg, 0.15 s around)
  cut the utterances; pieces under 3 s are dropped. The wrists are clamped
  but not counted as past a limit (their roll is the forearm's twist, the VRM
  has no forearm bone for it; on the real clip they were most of the "52 deg"
  limit excess).
- **Contacts.** Rotation transfer keeps angles, not contacts, so a thin
  actor's hand on the belly is inside a bulkier avatar (real clip: arm in
  body 0.05 / 0.17 / 0.25 hu at p50 / p90 / p99, 30% of frames over 0.10). Arms
  deeper than `--contact-pen` (0.04 hu) are solved again with the body's
  collision-aware IK for the palm where the mocap put it (onto the surface
  when the palm is inside), the swivel nearest the mocap arm, on keyframes
  every 3 frames, the change interpolated and faded in and out over 0.2 s
  (contact-aware retargeting, docs/research.md). Real clip: 0.01 / 0.02 /
  0.03 hu, palm moved 0.04 hu (median) where it was changed, palm speed p99
  unchanged. `--no-contact-fix` turns it off. Look at `report` and tune
  `--max-pen` before training.
- **Segments**: one utterance per row (cut at 0.35 s silences between
  words, 3-15 s, 0.3 s of motion either side); `spoken` = the words joined,
  per-character times from the word times, relative to the segment.
- **Prosody** for the head (`flow.actionhead --prosody` reads it):
  `<key>.prosody` [frames, 3] = RMS, f0 (Hz, YIN), voiced, at the frame
  times; the row names the columns (`prosody`).

On the GPU machine (WSL, repo at `the repo`). BEAT2 is the Hugging
Face dataset `H-Liu1997/BEAT2` (check the page for the files and the
licence; the layout assumed is `smplxflame_30/`, `wave16k/`, `textgrid/`
next to each other, and `--beat2` is the folder that holds them, or one
level above it when only one subfolder does). Retargeted motion is a derivative of it: **keep the
outputs out of git** (`--out` must be under `data/`, which is ignored).

    cd Tsukuyomi
    # 0. the data (check the dataset page; this needs git-lfs or huggingface-cli)
    huggingface-cli download H-Liu1997/BEAT2 --repo-type dataset \
        --local-dir ~/BEAT2
    # 1. a body file for the avatar (once; flow.train.hidden reads rigs/<name>.body.json)
    python -m flow.vrmbody --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \
        --out rigs/seed-san.body.json
    # 2. a small run first, then all of it (resumable; --jobs = processes)
    python -m flow.beat2 retarget --beat2 ~/BEAT2 \
        --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \
        --body rigs/seed-san.body.json --out data/beat2/seed-san \
        --speakers 2 --limit 5 --fingers
    python -m flow.beat2 report --out data/beat2/seed-san
    python -m flow.beat2 preview --out data/beat2/seed-san --beat2 ~/BEAT2 \
        --id beat2.<clip>/s00                  # data/beat2/seed-san/preview/
    python -m flow.beat2 preview ... --also-vrm assets/vrm/three-vrm-girl.vrm \
        --also-body rigs/three-vrm-girl.body.json    # side by side
    python -m flow.beat2 retarget --beat2 ~/BEAT2 --vrm ... --body ... \
        --out data/beat2/seed-san --jobs 8
    # 3. hidden states of the text alone, then the head, then its eval
    python scripts/extract_hidden.py --export data/beat2/seed-san/action \
        --context none --text spoken --model ~/runs/sft/q8/merged \
        --layers -8 --proj 1024 --out ~/runs/hidden/beat2
    python -m flow.actionhead train --export data/beat2/seed-san/action \
        --hidden ~/runs/hidden/beat2 --out ~/runs/ah/beat2 --steps 3000
    python -m flow.actionhead eval --ckpt ~/runs/ah/beat2 \
        --export data/beat2/seed-san/action --hidden ~/runs/hidden/beat2

What to check first on the real data: that a clip's report says `up_axis:
y-up` and `fps_source: mocap_frame_rate` (otherwise the layout differs from
the assumption), the preview of two or three clips (arms where they should
be, not crossed or behind the back, the torso not twisting as the person
turns), and `report`'s distributions (kept hours, the penetration quantiles;
if most is dropped, raise `--max-pen`). `--context none` is the text alone,
the distribution the head sees when the speech is not a marked reply; the
`--text spoken` matches the rows (`reply_raw` = `spoken`).

#### Grid results (2026-10-28, runs/ah_grid/)

BEAT2, 3,068 held-out segments, the first second of each unscored (the
history's); normalised MSE (nmse), motion around each segment's own mean
(rel), predicted / true motion std (std):

| run | nmse | rel | std | hands speed ~ audio accents (corr) |
|---|---|---|---|---|
| hold last frame | 1.105 | 0.471 | 0 | - |
| a text only | 1.066 | 0.473 | 0.08 | 0.02 |
| b + history 1 s | 0.899 | 0.496 | 0.42 | 0.04 |
| c + history + prosody | 0.854 | 0.488 | 0.41 | 0.30 |
| d history + prosody, no text | **0.790** | **0.468** | 0.34 | 0.29 |
| e = c with flow matching | 1.652 | 0.764 | 0.84 | 0.01 |
| ground truth | | | 1 | 0.03 |

What it says:
- History is what lowers nmse: knowing the current pose. None of the
  regression heads predicts the motion *around* the posture better than a
  constant (rel 0.47-0.50 vs 0.471): the exact gesture is not predictable
  from these inputs, and regression shrinks to a fraction of the real
  motion (std 0.08-0.42).
- Text adds nothing measurable here: d (no text) beats c on every row.
  The LLM's states of the transcript do not carry the gesture beyond what
  history and prosody give, at least under per-frame error.
- The prosody heads tie hand speed to loudness far more than people do
  (corr 0.30 vs 0.03 in the mocap): a learned shortcut, not human timing.
- Flow matching (e) moves like a person in amount (std 0.84) but its
  per-frame error is large, as a sample (not an average) must be; per-
  frame metrics cannot judge it, only watching or distribution metrics
  (FGD-style) can.

So, for now: the explicit path (the LLM writes poses and moves, the solver
executes) stays the body's controller; an audio-driven generative layer
(history + prosody, flow matching) is the candidate for background gesture
between markers, to be judged by eye; LLM hidden states as the motion
source are not supported by either the teacher data or BEAT2.

#### Probe results (2026-10-28, runs/probe/report.txt)

- **Voice vs motion in people** (segment level, within speaker): loudness
  r ~0.21-0.28 with both gesture range and speed (speed slightly the
  stronger: the "loudness -> amplitude" hypothesis is not confirmed, both
  rise); pitch and pitch range ~0.15-0.20; speaking rate -> speed (0.17) >
  range (0.13), as hypothesised; pauses -> less motion. Chinese couples
  more strongly (loudness -> head speed 0.48). At the frame level the
  coupling is ~0.03-0.09: voice sets how much a phrase moves, not when
  each frame moves. Windowed lag scan is flat, hands slightly leading the
  audio (best -0.4 to -0.7 s), as gesture studies report.
- **Text carries motion information.** Contrastive matching, held-out
  clips, 32 candidates (chance 0.031): text top-1 0.153 against
  same-speaker candidates and 0.098 against the same recording; prosody
  0.149; text + prosody 0.204 (the two add up); wrong text 0.035
  (chance). The grid's "text adds nothing" was the per-frame regression,
  not the text: what text predicts is the kind and amount of motion over
  a few seconds, which per-frame error cannot score.
- So: condition on phrase-level voice features (not per-frame loudness),
  keep text, generate (sample) rather than regress, and score generators
  the way text-to-motion work does: retrieval precision and embedding
  distances with these contrastive encoders, not frame MSE.

### Gesture search (2026-10)

`flow/gesture_search.py`: between and under the model's explicit markers, a
VRM avatar plays real human co-speech motion retrieved from the BEAT2
library. It is the use the probe results point to ("condition on phrase-
level features, keep text, retrieve or sample, judge by retrieval"), built
on the contrastive encoders instead of a regression head. Built and tested
on synthetic BEAT2-shaped data (`tests/test_gesture_search.py`: tiny
encoders trained on a toy export where the text names the arm's swing); it
has not run on the real library, so the first run on the GPU machine is the
check (below).

How it works (the module docstring has the details):

- **Encoders kept.** `flow.probe match --save DIR` writes each trained
  variant (state dict, motion normalisation, columns, window / look-back /
  token settings, the layers the text encoder reads, the hidden folder's
  `meta.json`, the held-out and training clips). The report is unchanged
  without it. `flow.probe.load_encoders` embeds motion windows and
  (token states, prosody) from a saved folder.
- **Library** (`build`): every 3 s window of the export (hop 0.5 s) except
  the probe's held-out clips, embedded from the canonical `.traj` columns as
  the probe's motion encoder reads them; the 23 fitted joints (`.joints`)
  stored once per segment, a window is (segment, start); start / end poses,
  speaker, clip, language per window; the median pose and the recordings'
  median voiced RMS. About 0.35 KB per window (a 128-d fp16 embedding and
  two poses); the joints (46 bytes a frame) are stored once, not per
  window. Under `data/gesture_lib/` (gitignored).
- **Query** (per sentence of the TTS reply): text states as `flow.train.
  hidden` made them for the probe -- same model, `--context none`, the
  spoken text, the stored layers, the stored projection, fp16, then the
  probe's layer selection -- run in the director's process
  (`TextStates`; `tests/test_gesture_search.py` checks bit equality with
  what `hidden` stores); prosody from the sentence's TTS audio at 30 fps
  (`flow.beat2.prosody`) after its loudness is set to the library's (TTS and
  microphone levels differ by more than the encoder's input tolerates);
  the sentence itself is made to look like the export's texts first (BEAT2's
  words are the TextGrid's: `build` records whether they are lowercase and
  have punctuation -- `text_form` in the library's `meta.json` -- and a
  sentence is lowercased / stripped of punctuation accordingly, NFKC, the
  characters' times kept on their characters; a library without `text_form`
  leaves the text as it is);
  windows of 3 s every 2 s over the sentence plus 0.3 s either side, each
  embedded as the probe embeds a window (silence past the audio).
- **Choice**: cosine to the library's motion embeddings (optionally
  one speaker / language), the best 48 scored cosine - 0.25 x continuity -
  a small price for each recent pick from the same clip, one of the best 6
  drawn with a softmax at temperature 0.05 (`--gs-temperature 0`: the best;
  `--gs-seed`). Continuity: at the moment the cross-fade starts, the arm
  joints (share of their range) and the palms (hu, the target body's
  forward kinematics) of the candidate against the pose the track is in.
  A window overlapping a recent pick of the same recording is out.
- **Body**: the arms (upper arm, forearm, wrist, shoulders) are absolute,
  scaled about the body's rest pose (`--gs-amp`) and clamped to its limits;
  `--gs-parts full` adds the head and torso as the window's deviation from
  the library's median pose, added to the idle's head and trunk (so the
  idle's own head moves keep going). Each chosen window goes through the
  body's collision model; where an arm is deeper than 0.04 hu in the trunk,
  head or the other arm it is solved again with `beat2.fix_contacts` (the
  collision-aware IK); a window still deeper than 0.08 hu is skipped for
  the next best. The cross-faded track is checked once more (a blend of
  joint angles is not collision-free by itself). Retargeting to another
  VRM is applying the same joint names to its body (clamped, collisions
  re-checked): `tests` run a Seed-san window on the three-vrm-girl body.
  Live2D is out of scope (its arms are drawings).
- **Play**: one `"gsearch"` token per run of sentences closer than 1 s: the
  windows cross-faded over 0.4 s (minimum jerk, the later window taking
  over as it fades in), eased in and out over 0.4 s. `GestureLayer` sits in
  `MotionStream.frame` after the idle and the mouth, before moves and
  poses: a pose or move that plays an arm takes it exactly as it takes it
  from the idle, the gesture ducking out of those parameters before the
  marker starts (0.4 s) and back in after its hold; nothing of the
  gesture shows in a held pose (the layer reports the parameters it plays
  as `holding`, so the pose layer's "life" ignores them). Without
  `--gesture-search` none of this is imported: `tests` compare the plan's
  tokens, states, subtitles and actions byte for byte with the module made
  unimportable.

Director flags (`--avatar vrm`; they need `--tts`, the prosody is the
audio's): `--gesture-search LIB --gs-encoders DIR [--gs-model PATH]
[--gs-hidden DIR] [--gs-amp 1] [--gs-seed 0] [--gs-parts hands|full]
[--gs-speaker NAME] [--gs-temperature 0.05] [--gs-topk 6] [--gs-4bit]
[--gs-device auto|cuda|cpu] [--gs-baseline]`. `--gs-baseline` also writes
`states_markers.json`, the same turn without the retrieved motion
(`demo --say` uses it). The text model is loaded once per session, on the
first sentence. `flow.live` does not take the flags yet (its `Follower`
plans per sentence and has the audio: `GestureSearch.plan` takes queries,
so it is the place to hook it).

On the GPU machine (WSL, repo at `the repo`; the export
`data/beat2/seed-san/action`, the hidden folder `~/runs/hidden/beat2`
extracted with layers [36, 29, 18], `--proj 1024`,
`--text spoken --context none`, the merged model `~/runs/sft/q8/merged`):

    cd Tsukuyomi
    # 1. the text+prosody encoders again, kept (the settings of the report
    #    you want to match: --layer 1 is the stored layer 29; the split
    #    follows from --seed / --eval-frac / --limit and is recorded; --out
    #    is another folder so the earlier report.txt / report.json stay)
    python -m flow.probe match --export data/beat2/seed-san/action \
        --hidden ~/runs/hidden/beat2 --out ~/runs/probe-gs \
        --layer 1 --variants text+prosody --save ~/runs/probe/gs
    # 2. the library (BEAT2-derived: research only, stays under data/)
    python -m flow.gesture_search build --export data/beat2/seed-san/action \
        --encoders ~/runs/probe/gs --out data/gesture_lib/seed-san
    # 3. retrieval precision and motion statistics on the held-out clips
    python -m flow.gesture_search eval --export data/beat2/seed-san/action \
        --encoders ~/runs/probe/gs --lib data/gesture_lib/seed-san
    # 4. held-out segments side by side: idle | random windows | retrieved |
    #    ground truth, the recorded audio on it
    python -m flow.gesture_search ids --export data/beat2/seed-san/action \
        --encoders ~/runs/probe/gs
    python -m flow.gesture_search demo --id beat2.<clip>/s03 \
        --export data/beat2/seed-san/action --encoders ~/runs/probe/gs \
        --lib data/gesture_lib/seed-san --beat2 ~/BEAT2 --out runs/gs/held
    python -m flow.gesture_search demo --id beat2.<clip>/s03 ... \
        --vrm vrm/toukai_teio.vrm --out runs/gs/toukai     # a body of yours
    # 5. a free sentence through the director: markers only | markers + search
    python -m flow.gesture_search demo --say "Come here, I want to show you something." \
        --export data/beat2/seed-san/action --encoders ~/runs/probe/gs \
        --lib data/gesture_lib/seed-san --out runs/gs/say-oguri \
        --gs-model ~/runs/sft/q8/merged --gs-hidden ~/runs/hidden/beat2 \
        --avatar vrm --vrm vrm/oguri_cap.vrm --mode body --backend ollama \
        --tts voicevox --size 384
    # the same inside a normal director run
    python -m flow.director --avatar vrm --vrm vrm/toukai_teio.vrm --mode body \
        --backend ollama --tts voicevox --say "..." --out runs/gs/turn --render \
        --gesture-search data/gesture_lib/seed-san \
        --gs-encoders ~/runs/probe/gs --gs-model ~/runs/sft/q8/merged

`demo --say` takes the director's own flags as they are (everything it does
not know, `--mode`, `--backend`, `--tts`, `--vrm`, ...) and `--gs-model`
for the text model (`--model` stays the director's language model).
The two VRMs of yours are not in the repo: the paths above are where they
are expected (`vrm/` is ignored); their body files are built on first use
(`rigs/<name>.body.json`).

What to check first on the real library: that step 1's held-out text+
prosody top-1 (32 candidates) is the old report's 0.204 -- if not, the
flags (`--layer`, `--limit`, `--seed`, `--eval-frac`) differ from that run
and the held-out clips are not the same ones; that `meta.json`'s
`text_form` is what you expect of the transcripts (lowercase, no
punctuation: otherwise TTS sentences are read as they are); that `eval` says retrieved
windows are closer to the true motion than random ones by clearly more than
chance (0.5; the probe's text+prosody top-1 of 32 was 0.204 against 0.031,
so expect a modest margin: what text and voice predict is the kind and
amount of motion over a few seconds, not the gesture), the speaker-
normalised amplitude and speed (retrieved close to the truth, a random
window's correlation with the truth near 0), then the demo by eye -- the
cross-fades and the arms against the body -- and the prosody gain (the
library's median voiced RMS is in `meta.json`; a TTS voice far from it is
the first suspect if every sentence retrieves the same still gesture).

Cautions:
- The library is BEAT2-derived (retargeted mocap): research use under
  BEAT2's licence, kept out of git (`build --out` must be under `data/`).
- Text states: the hidden folder does not record whether `--load-4bit` was
  used; query with the same precision (`--gs-4bit` only if extraction used
  it), or the states shift. `--context none` states are what the probe was
  trained on; a hidden folder made with `--context turn` is said to differ.
- Latency: one forward of the text model per sentence in the director's
  process (a sentence is ~10-40 tokens: tens of ms on a GPU for an 8B model
  in bf16, seconds on CPU -- an estimate, not measured on the real model,
  only on a tiny one here) plus the encoders (milliseconds) and the checks
  of the chosen windows (about 40 ms per window measured on Seed-san, 0.3-
  0.5 s when the contact fix runs). Offline this is paid once per reply,
  before the render; the model load (tens of seconds) once per session. The model is 16 GB in bf16: next to Ollama on one 24 GB card use
  `--gs-4bit` (and extract the hidden states the same way) or run the
  search on another GPU / the CPU (`--gs-device`); the encoders and the
  library (200-300 MB of embeddings for ~400k windows, in RAM) are small.
- Continuity artefacts: the cross-fade blends joint angles, so two windows
  that differ much at the fade can pass through a pose neither has (the
  track is re-checked for collisions and solved again where it goes into the
  body, but a bent wrist or an elbow swinging through is not a collision);
  the continuity weight (0.25) is a first guess, untested on real data.
  The first window of a run eases in from the rest pose, so a window that
  starts with the hands high is penalised by the same cost.
