# Research survey: motion control and beyond (2026-10)

What recent work (2025 to 2026-10) offers this project: a local Qwen3-8B on
one RTX 3090 Ti (24 GB, Ampere: no FP8) driving Live2D/VRM bodies, moving
from inline markers to an action head on the model's hidden states
(docs/embodiment.md, Stage 2), and later to video, games, driving and
hands.

**How far to trust this.** Four research passes, 2026-10-26. From their
sandbox arxiv.org and huggingface.co were mostly unreachable, so most paper
claims come from search snippets; GitHub pages were read directly. Marks:
**[repo]** the GitHub page was read; **[snippet]** search results only,
not read. Speeds are the authors' claims, mostly on newer GPUs; nothing
here was run on our card. arXiv ids from 2606 on could not be cross-checked
at all. Check a paper before building on it.

## 1. Motion control (the current focus)

### The interface: what reads the LLM

Our plan (per-token hidden states, aligned to the speech, to a small head)
has close published relatives:

| Work | How the LLM drives motion | Status |
|---|---|---|
| GestureFAR (arXiv 2609.21576, 2026-09) | AR backbone, a per-token flow-matching head on its hidden states, distilled to one step; streaming, body + face | [snippet]; repo linked but empty |
| LLaMo (CVPR 2026) | LLM + motion-side parameters, flow head on the motion hidden state; claims 30+ fps streaming | [snippet] |
| UniMotion (arXiv 2603.22282) | modality-specific flow heads from LLM hidden states to motion latents | [snippet] |
| MotionGPT3 (arXiv 2506.24086) | LLM hidden states condition a diffusion head over motion-VAE latents; body only, not streaming | [repo] MIT code, HF weights |
| ELLSA 2 (arXiv 2610.21452) | full-duplex dialogue model; speaking/listening hidden states drive gestures + face in 0.8 s chunks | [snippet]; HF tsinghua-ee/ELLSA2, licence unknown |
| OpenHelix (2505.03912), RationalVLA (2506.10826) | a learned `<ACT>` token; its hidden state is the goal for a low-level policy | [snippet]; OpenHelix code open |
| FLOWER (2509.04996) | VLM hidden states into a flow transformer by cross-attention | [snippet] |

Takeaways:
- **Per-token conditioning plus a flow head is the mainstream design.**
  flow/actionhead.py already has this shape (cross-attention over token
  states, regression or flow matching).
- **One-step distillation** (GestureFAR's head distillation, SnapFlow
  2604.05656) is the known way to keep a flow head cheap enough for 30 fps.
- **The `<ACT>` token is the alternative**: one learned token per control
  step instead of reading every speech token. It's worth an ablation once
  the per-token version has a baseline.
- **Layer choice matters and nobody agrees.** Try `-8`, `-16` and the middle
  layer; the extractor takes `--layers=-8,-16`.

### Latency and streaming

- **Real-Time Chunking** (Physical Intelligence, 2506.07339, [snippet]):
  frames the actions already committed during inference as an inpainting
  constraint, at inference time only. This applies to any chunked head
  here; it's the fix for the "audio waits on motion" gap.
- **FASTER** (2603.19199, [snippet]): a flow schedule that produces
  near-term actions first, so the first action comes after one step.
- **LiveGesture** (CVPR 2026, 2604.10927, [snippet]): a causal motion
  tokenizer with hierarchical AR over body regions, with no look-ahead.
  It claims under 50 ms per 200 ms audio chunk; a good latency budget to
  copy.
- **Cheap baseline first:** OpenVLA-OFT (2502.19645) reports an L1
  regression head matching diffusion heads. Our head's regression mode is
  that baseline; flow matching has to beat it to be worth its cost.

### Two layers: plan, then track

Helix (Figure: a 7B VLM at 7-9 Hz over a fast visuomotor policy at
200 Hz), Gemini Robotics 1.5 (planner then actor) and HOVER (one tracking
policy for several command types) all split slow intent from fast control.
That is what the marker path already is: the LLM gives targets, and the
solver and motion stream track them. The action head should keep that
split: predict targets or short chunks at token rate, and leave
smoothing, idle and breath to flow.stream.

### Data beyond the teacher

The head can only learn what its targets hold, and today those are the
trajectories DeepSeek's markers produced. Real motion would raise that
ceiling:

| Source | What | Licence notes |
|---|---|---|
| BEAT2 | conversational co-speech mocap (face, body, hands), 25-30 speakers, ~60-76 h | Apache-2.0 per one secondary source; check the HF tag |
| FloodDiffusion (CVPR 2026) | streaming text-to-motion, ~50 fps output | [repo] Apache-2.0, HF weights; trained on HumanML3D/BABEL |
| FloodDiffusion 2 (2609.33167) | adds root-path conditioning | [repo] MIT, HF weights, run on an RTX 4090 |
| SemTalk (ICCV 2025) | holistic face/body/hands, per-part RVQ tokenizers | [repo] weights on HF, code licence not stated |
| HumanML3D / AMASS | text-motion | the AMASS terms forbid redistribution: rebuild it locally, research use |
| Seamless Interaction (Meta, 2506.22554) | 4,000+ h of dyadic video with motion/expression models | licence not found |

The VRM body is a standard humanoid, so SMPL-X mocap retargets to it
directly; that's step E4 (GMR, 2510.02252, is a reference retargeter).
Live2D only takes what the canonical variables can express: head, torso,
arms and face.

### Front ends (for comparison)

Open-LLM-VTuber ([repo] MIT, Live2D, works with any OpenAI-compatible
server) and Project AIRI ([repo] MIT, Live2D + VRM) are the popular open
VTuber stacks. Neither has body control beyond blinks, look-at and idle.
Our solver, poses, moves and data engine are further along than both; at
most, borrow their UI or plugins.

### RPent

github.com/RLinf/RPent, "Recursive Physical Agent" ([repo] per the
research pass, Apache-2.0, updated 2026-09): an agentic planner (Claude
Code, Codex or a custom planner) calls VLAs such as pi0.5 as tools, in
simulators (LIBERO-PRO, RoboCasa, RoboTwin) and on a Franka arm. It's a
planner above the VLAs, not hidden states driving a body: the reference
for a future planning layer on top of our controller, not for the head.

### Read in full (papers/, 2026-10-26): what changes for flow/actionhead.py

The nine priority-1 papers are in papers/; all the arXiv ids above for them
are correct. What they say, against the head as built (frames cross-attend
to the token states up to a look-ahead, chunks with half overlap
cross-faded, MSE or flow matching, no motion history, no audio):

- **Condition on past motion.** GestureFAR (a causal transformer over its
  own past motion latents + audio so far, an MLP flow head per latent at
  7.5 Hz) and LiveGesture (region-expert AR transformers over causal motion
  tokens) both autoregress on the motion already emitted. Our head sees
  only text, so continuity comes from the cross-fade. LiveGesture's
  **autoregressive masking** (training on corrupted history) is the fix for
  the drift that history feeding brings.
- **Replace the cross-fade.** RTC shows that temporal ensembling (what the
  cross-fade is) is less smooth than **inpainting**. The next chunk is
  generated with the frames already committed frozen, and a soft mask over
  the overlap. It's training-free for any flow head, so it applies to
  `--objective fm` as is.
- **Timing comes from speech, not the LLM.** OpenHelix's probe found the
  MLLM latent carries task semantics and is insensitive to fast changes.
  GestureFAR conditions on audio (mHuBERT), and ELLSA 2 on the speech
  synthesiser's own hidden states. Our head gets per-character speech times
  but no prosody. r1 was generated with `--tts none` (rate timing), so
  adding audio features (energy and pitch from the TTS clip, which the lip
  sync already computes) needs a TTS pass over the records.
- **Which layer:** ELLSA 2 conditions on the final layer and mixes layers
  16/20/24 for its call head; OpenHelix reports both final and middle layers
  in use. So learn a weighted mix of a few layers rather than picking one.
- **Freeze, align, then (maybe) tune.** OpenHelix: joining the LLM and the
  policy without first training the projector with the LLM frozen fails
  outright (0% in their Table 6). Prompt-tuning one `<ACT>` token
  generalised best. Our order (frozen extraction, head trained on cached
  states) is the right first stage. MotionGPT3 adds that a separate motion
  branch keeps the language ability intact.
- **Keep explicit gestures explicit.** ELLSA 2 separates *whether* to make
  a semantic gesture (a binary call head on layers 16/20/24, which can run
  ahead of the speech) from *how* (a retrieved exemplar guiding the
  generator). That is our split already: markers / [move] / gestures are
  the exemplars, and the head does the co-speech motion. A call head could
  later replace the textual markers for gestures without losing precision.
- **Listening.** ELLSA 2 keeps generating listener motion (nods,
  expressions) while the user talks. We only have the idle there.
- **Making flow cheap:** first check OpenVLA-OFT's result (L1 regression ≈
  diffusion, faster) against our MSE mode. If flow wins, distil the head
  alone: GestureFAR distils only the per-token head on cached conditions
  (consistency warm-up + DMD); SnapFlow self-distils to one step with no
  separate teacher or critic, and beats its 10-step teacher on LIBERO.
  FASTER's horizon-aware schedule makes the first frames one step, the
  lever if time-to-first-motion matters more than chunk quality.
- **Runtime:** GestureFAR's stateful runtime (persistent caches, body
  anchors against drift, a short playback buffer, slerp blends between
  speaking, listening and idle) matches what flow.stream already does for
  markers. The head should feed the same stream, not replace it.

### Priority-2 papers (papers/, 2026-10-26)

- **Knowledge Insulation** (Physical Intelligence, 2505.23705): adding a
  flow-matching action expert and training it into the backbone harms
  both training speed and the backbone's knowledge. The fix: stop the
  expert's gradient at the backbone (`sg`), and co-train the backbone on
  *discrete* action tokens (FAST) plus ordinary VLM data. Freezing the
  backbone outright works worse when it has to learn the domain. **This is
  our recipe almost exactly.** The LoRA student learns the domain from
  discrete actions in text (the markers play the FAST role), and the head
  reads its states without sending gradients back. If the head is ever
  trained jointly with the LoRA, keep the stop-gradient and keep markers
  in the LoRA's training text. Whether the speech-only student should
  still be trained on markers (as KI's discrete co-training suggests)
  is a question to test.
- **SmolVLA** (2506.01844): takes VLM features from only the first half
  of the layers (N = L/2: faster, and as good). Its action expert
  alternates cross-attention (to the VLM) and self-attention (over the
  action chunk). Self-attention is what makes chunks smooth. An async
  stack predicts the next chunk while the current one plays. For us:
  layer 18 of Qwen3-8B's 36 belongs in the layer ablation, and the head's
  frames should self-attend between cross-attention blocks.
- **FloodDiffusion 2** (2609.33167): streaming text-to-motion with a
  KV-cached "partial attention" (finalised history no longer depends on
  the active window: 11x faster denoising, 2.3 ms per update) and root-path
  conditioning. Also an **FK-induced quadratic loss**: errors are weighted
  by how much they move the body (a dataset-averaged kinematic
  response), shown to keep the regression target unbiased (any Bregman
  divergence does). For our head: weight each canonical variable's error
  by its forward-model reach (flow.body already measures how far each
  variable moves the hands, head and mouth) instead of plain MSE on
  normalised variables. That's cheap, and FD2 shows it's safe.
- **GMR** (2510.02252): retargeting artifacts (foot sliding,
  self-penetration, impossible motion) carry straight through to what is
  learned; better retargeting beats reward tuning. When BEAT2 goes to VRM,
  run it through the collision model in flow/vrmbody.py and drop clips
  that penetrate rather than training on them.
- **UniMotion** (2603.22282): one LLM backbone with continuous motion and
  RGB pathways (a motion VAE aligned with vision). It has a cold-start
  problem, where text alone is too sparse to calibrate a new motion
  pathway, and fixes it with a self-supervised latent pre-training stage.
  This matters if the head ever moves inside the LLM; for now it doesn't.

### One controller, many bodies (2026-10-27)

Prompted by "Learning to Get Up Across Morphologies" (2512.12230, read in
full; it trains one get-up policy for seven small humanoids). The rest
comes from a research pass, mostly [snippet]: arXiv was unreachable.

Where we stand: a shared body-relative control space (the canonical
variables), each body's own forward model and a clamping solver, and the
body's description in the prompt (the LLM) or as an avatar embedding plus
a variable mask (the action head). 2512.12230 does the first two
(common joints, body-relative measures, the simulator enforces limits)
with no body description at all.

What the literature says, against that:
- **Body diversity beats data volume.** "Towards Embodiment Scaling Laws"
  (2505.05753, GENBOT-1K: ~1,000 procedurally generated bodies) found
  that adding bodies generalises to unseen ones more than adding data
  per body. 2512.12230: held-out bodies far from the rest (top-heavy,
  long-armed) transfer worst (17-42%), and a body that could not learn
  alone did at 81% in shared training. Our version: lafei is the outlier
  (the student's 17% miss rate there), and synthetic Live2D/VRM variants
  (scaled limbs, narrowed ranges, parameter subsets) are cheap to make
  from the body files.
- **Per-parameter tokens with a shared decoder** (URMA 2409.06366,
  CrossFormer 2408.11812, PALUM 2601.07272's part grouping), or thin
  per-body adapters over a shared core (X-WBC 2609.15213). Our head's
  fixed vocabulary plus a mask is the simplest form; tokens per variable,
  carrying its range from the body file, would let an unseen body bring
  its own ranges.
- **Text descriptions of the body transfer.** AnyTop (2502.17327: motion
  diffusion for any skeleton) embeds a text description per joint, the
  same idea as our prompt. A description embedding may generalise better
  than a learned avatar ID.
- **Keep contacts as contacts.** Contact-aware retargeting (Villegas et
  al., ICCV 2021, 2109.07431) and ReConForM (2502.21207, real-time) keep
  hand-to-face and hands-together contacts across body sizes. That is the
  answer to absolute vs reach-normalised space: absolute (shared) space,
  plus named contact targets resolved per body (`@together`, a face
  anchor), plus a contact metric in evaluation. It matters for the BEAT2
  retarget, where a human's hand-to-chin should stay a hand-to-chin.
- **Baselines:** the NerveNet critique ("My Body is a Cage", 2010.01856)
  found structure-free encoders about as good. Always run one. Report
  per-held-out-body results with variance (2512.12230, AnyBody
  2505.14986), never one average.
- **Live2D specifically:** no published method generates Live2D parameter
  curves from text or audio, or handles per-rig reach. Textoon
  (2501.10020) is the closest (ARKit-style mouth shapes for Live2D).

Read in full (papers/, 2026-10-27), the details that matter here:
- **URMA** (2409.06366): every joint has a fixed-size *description
  vector* (rotation axis, position on the body, torque/velocity limits,
  control range). The encoder pools joints by attention keyed on their
  descriptions. The universal decoder pairs the shared action latent with
  each joint's encoded description to output that joint's action. An
  unseen robot just brings its own descriptions. For our action head:
  replace the learned avatar ID with a description per canonical variable
  (its range on this body, units, body part, how far it moves the hand or
  head, all already in the body file), and decode each variable from
  (latent, its description). A new avatar then needs no new embedding.
- **Embodiment scaling laws** (2505.05753, GENBOT-1K, URMA-based):
  experts per body distilled into one policy. Within each class, going
  from 5% to 100% of the training bodies about doubles reward on unseen
  ones; across classes, 2-5x over single-class training. More data on 5%
  of the bodies saturates at 4x volume. Humanoids had not saturated at
  ~800 bodies (quadrupeds and hexapods at ~100). The real Go2 with one
  knee's range cut by 20% got a working limp zero-shot. **That is the
  lafei case.** Training across bodies whose ranges vary is what teaches
  a controller to respect a narrow range it has never seen, which
  supports the synthetic body variants in the next data run.
- **AnyTop** (2502.17327): each joint's *name*, embedded with T5, is
  added to its token, and attention is biased by joint relations and
  graph distance. Names carry semantics across skeletons ("Left Wing" ~
  "Left Arm"). Our variables already have stable semantic names
  (`hand.R.y`), which the LLM reads; a head could embed them the same
  way.
- **ReConForM** (2502.21207): key vertices rigged on the source mesh,
  transferred automatically to the target; descriptors are the key
  vertices' relative trajectories (distances, normalised by body size),
  adaptively weighted over time; the target motion is optimised to match
  them, in real time. **Contact-aware retargeting** (Villegas et al.,
  2109.07431): detect self-contacts (from geometry) and foot contacts in
  the source; an energy keeps them and penalises interpenetration;
  geometry-conditioned RNN plus encoder-space optimisation. For
  flow/beat2.py, which is rotation transfer only: a contact post-pass
  that finds frames where the source hand is near the face, the other
  hand or the torso (distances relative to the source's own limb
  lengths) and pulls the target to the same relative distance with
  VRMBody's IK (it already has `@together`-style named targets). Do it
  if the first previews show hands missing the face or sinking into
  the body.

Experiments this suggests, in order of cost:
1. **Leave-one-body-out for the LLM:** train the student without xuefeng
   (or lafei) and measure its pose accuracy there from the prompt alone.
2. **Synthetic bodies in the data engine:** vary the existing body files
   (ranges, limb scale, dropped parameters) per session, so the student
   learns to read ranges instead of memorising three bodies; then repeat
   test 1.
3. **Action head conditioning:** learned ID vs a body-description
   embedding vs per-variable range tokens, with a structure-free
   baseline, held-out bodies, and BEAT2 targets.

### Weight-space editing against repetition (2026-11)

Source: the blog post "Sparks of Weight Space Programming: LLMs can
understand and write neural network weights" (rikiyat.github.io, not
reachable from the cloud session) and its log release,
github.com/RikiyaT/lm_understand-logs (logs, data and networks CC BY 4.0,
scripts MIT). A frontier model (`gpt-6-astra`, as the post names it, through
the Codex CLI with its tools off) reads and edits network weights without
training. Eight experiments; one is close to ours.

- **Experiment 8.** Qwen3.5-0.8B, thinking on, looped until the token limit:
  93% of its MATH-500 answers were cut there. With a CPU kernel over the
  weights and a budget of forward passes (no gradients, no training), the
  editor found a "loop" direction in the residual stream. The best edit
  (50 passes, ~20 h) subtracts it from the 64 output columns per layer that
  write it most, in the MLPs of layers 15 and 19 (131,072 weights),
  W' = W - a u(u^T W). MATH-500 7.9 -> 48.1, GSM8K 13.8 -> 67.1, cut at the
  limit 93% -> 33%, and outside math MMLU-Pro 40.3 -> 45.5, IFEval 21.0 ->
  62.3. With 10 passes the edit removed the same direction from 30 output
  matrices in layers 8-22 (MATH-500 12.3). Their eval sampled with presence
  penalty 1.5 (vLLM, temperature 0.6, top_p 0.95, top_k 20).
- **The rest** does not carry over: Experiments 1-6 are small game,
  diffusion and Go networks; in Experiment 7 the weights alone did not tell
  which of two Gemma 3 270M fine-tunes was sycophantic, a tool showing
  attention patterns did. `exp3_language/python_dataset` is the file list
  and rebuild script of the Python text of a 7M-weight character-level toy
  model: nothing for us.

Why it matters here: q8r2 repeated itself (from turn 12 of the live trial
every reply was the same three beats; ~20% of replies opened with the
prompt's one dense example). r3 attacks that in the data (six variants,
the history window, quiet chat) and the DPO pairs in the preference stage.
If q8r3 still repeats, two cheap tests come before anything bigger, both
measured with `scripts/checks.py` (`template_repeat`, `opening`, `beats`):

1. **A repetition penalty at serving.** The Modelfile sets none
   (`flow/train/chat.py` `SAMPLING`; `repeat_penalty` passes through to
   Ollama and to the OpenAI-compatible backends). Careful: the markers
   repeat their own keys (`"head.yaw"`, `"dur"`, `"hold"`) every reply, so
   watch the parse-warning rate and the marker counts as well.
2. **A direction edit, done ourselves.** No frontier editor and no 20-hour
   harness: hidden states (`flow/train/hidden.py`, the merged student) over
   held-out replies the checkers flag as repeats against replies they pass,
   the difference of their means per layer as the direction u, then
   W' = W - a u(u^T W) on the output matrices (attention o_proj, MLP
   down_proj) of a few middle layers, a swept, on the merged fp16 model
   before GGUF. Score with the checkers and the eval's marker table; a
   direction that also carries "write a marker" would show as fewer beats.
   It works on the merged model only (the LoRA adapter changes with each
   run), so it is a step after merge, per student.

### Markers as their own modality; fast and slow replies (2026-11)

Two ideas from the user, with what the code says about them. Neither is
for q8r3.

**Text and markers are two modalities.** The markers are most of what the
student writes: in q8r2's eval replies (`runs/eval_q8r2/student.jsonl`)
83% of the characters are marker text (by tokens less, since Chinese is
about a token a character, but still most of the output). A marker such as

    [pose {"brow.L.lift": 0.8, "brow.R.lift": 0.8, "head.pitch": 5, "head.yaw": -5, "dur": 0.2, "hold": 0.5}]

is plain JSON to Qwen's tokenizer: every key spelled out each time, every
number in digit pieces. So the first spoken words wait behind syntax, most
of the loss is on boilerplate (`"brow.R.lift": 0.`) rather than on which
expression to make, and long near-identical strings are easy to copy (one
reason the openings collapsed). In order of cost:

1. **Measure it.** The trainer reports loss and token accuracy on marker
   tokens and on speech tokens separately (character spans from the pose
   parser, mapped to tokens), on train and eval. Free, and it says which
   half is weak before anything is changed.
2. **A marker vocabulary.** Special tokens per variable key (`<brow.L.lift>`,
   `<head.yaw>`, `<dur>`) and per binned value (e.g. 21 levels over each
   variable's range), so the marker above is about 10 tokens. The export
   rewrites r3 deterministically and the parser maps it back to JSON: no
   new teacher data. Costs: the new embeddings must train (LoRA with
   `embed_tokens` and `lm_head` in `modules_to_save`); GGUF and Ollama take
   added tokens; binning loses precision (check the reach rates). Keep the
   interleaving: where a marker sits in the text is what times it against
   the speech; only its encoding changes. A q8r4 data and tokenizer change.
3. **The full version is the action head** (`flow/actionhead.py`): hidden
   states to a continuous trajectory, the "plan in text, track in a
   continuous model" design above. Compact markers make the text plan
   cheaper and give the head a cleaner signal.

**Answer simple lines at once, think on hard ones.** The student already
answers every line instinctively: thinking is off in the template (the
empty think block, `flow/train/chat.py` `THINK_OFF`) and its first text
came in ~0.06 s in the eval (the live slowdown to 2.7 s was the history,
fixed by the window). The missing half is the slow path, and silence is
not it: a streamer who stops for five seconds reads as frozen, not
thoughtful.

- **Cover the wait as a person does.** On a hard line a person says "hmm,
  good question...", looks up, frowns a little, then answers. In
  `flow/live.py`: a quick acknowledgement with a thinking beat (gaze up or
  away, brows down) plays at once, while the line gets a thinking budget
  (the same model with thinking on, or a larger model), and the answer
  follows. A live-runtime prototype, not a training run, and the part a
  viewer notices.
- **Who decides a line is hard.** A small classifier on the user line, or
  the student trained to open either an empty or a short think block. The
  second is neater, but r3's persona topics are mostly chat with few hard
  questions: there is little to learn it from yet (it would need hard
  questions in `flow/datagen.personas.json` and teacher reasoning).
- **Does thinking still work.** The student trained with thinking off only.
  Before relying on it: Qwen3's thinking mode with the adapter, and whether
  the marker format survives a think block.

Order: the separate marker / speech loss report now (free); compact marker
tokens as a q8r4 change; the thinking pause as a live prototype. The two
are independent.

### Separate streams for text and motion (2026-11)

A proposal from the user's reading, checked against what we measured. It
argues that text and motion, even both serialised as tokens, should not
share one representation, one attention pattern or one temporal model, and
splits the question four ways: representation (separate tokenizers and
encoders), attention (how information flows between the two), temporal
modelling (motion keeps its own memory) and control (actions against the
measured physical state). It proposes a dual-stream model (a text stream,
a motion stream with temporal self-attention over its own history, and
cross-attention from motion to text whose strength is learned), a third
stream for proprioception (joint positions, velocities, torques, contacts:
the next action follows the measured state, not the last prediction,
A_t = f(L, M_1:t-1, S_t)), a three-level hierarchy by timescale (semantic
planner at seconds, motion generator producing ~0.5 s chunks, motor
controller at milliseconds), block-structured attention masks, and a
robot-specific adapter that maps a body-independent motion latent to each
body's joints. Related work it names: RT-2 / PaLM-E, OpenVLA, Diffusion
Policy, ACT (action chunking).

**What we already measured says the same thing.**

- Reading motion frame by frame off the student's hidden states failed: in
  the action-head grid on BEAT2 no regression head beat a constant on
  motion around the posture (relative 0.47-0.50 against 0.471, history and
  prosody included), and the flow-matching head moved plausibly (std ratio
  0.84) but far off frame by frame.
- The information is there at the window scale: the probe matched held-out
  motion windows among 32 candidates (chance 0.031) at 0.153 from text,
  0.149 from prosody and 0.204 from both, 0.035 with shuffled text; within
  a speaker loudness goes with amplitude and speed at r ~0.21-0.28, but the
  frame-level coupling is ~0.03.

Text and voice say what kind of motion fits the next second or two, not
what each frame is. That is the proposal's timescale argument, on our own
data: motion wants its own temporal model and chunked, generative output;
per-frame error is the wrong objective and the wrong metric for it.

**How Tsukuyomi maps onto the hierarchy.**

| Proposal's level | Tsukuyomi now | Learned? |
|---|---|---|
| Semantic planner (seconds) | the student writing markers: intent, target, timing, per clause | yes |
| Motion generator (~0.5 s chunks) | flow.stream and the runtime layers: eases, head carry, expression baseline, micro-expressions from the voice, drift, moves, camera | no: rules |
| Robot-specific adapter | the body file and solver: ranges, reach, rig mapping, squint coupling | measured, not learned |
| Proprioception | the "(your body now: ...)" line; the solver clipping to reach | minimal |

The markers are the planner's language, not continuous motion squeezed
into text: no frame of motion passes through the LLM. "Two layers: plan,
then track" above already took this split from Helix and Gemini Robotics.

**Where it does not carry over as written.**

1. **Separate tokenizers or block-sparse attention inside the LLM.** With
   a pretrained 8B and ~14k examples, re-wiring its attention is not
   trainable. The practical dual stream is an action expert beside the
   LLM: a small motion transformer with its own temporal self-attention
   over motion history and prosody, cross-attending to the student's
   hidden states (or KV), trained with flow matching on 0.5-1 s chunks,
   with the gradient stopped into the LLM (Knowledge Insulation, above).
   pi0's recipe, sized for the 3090 Ti (SmolVLA's expert is the scale).
2. **Proprioception.** Essential for a robot, close to irrelevant for an
   avatar: a Live2D rig does what it is told within its ranges, which the
   solver already enforces. Keep the stream in the design for robot hands
   (section 2), do not build it for avatars.
3. **Data, not architecture, is the constraint.** A learned motion
   generator trained on our own rendered output learns our rules back. It
   adds something only with real human motion: BEAT2 for the body (54 h,
   already retargeted to VRMBody by flow/beat2.py) and, for the face and
   head, streamer video through face tracking into Live2D parameters.
4. **Tokenization matters less than timescale** for us (the proposal's
   section 8 ranks them the same way). Compact markers (previous section)
   are about cost and copying, not about modelling motion.

**What to do, after q8r3 and DPO.**

1. Keep the markers as the planner's output: they are checkable, and the
   checkers, the judge and DPO act on them.
2. Replace rule layers with learned ones, one at a time, each scored with
   window-level metrics (retrieval among candidates as in flow/probe.py,
   beat alignment, diversity, the VLM judge), never per-frame error:
   - **VRM body gestures first.** BEAT2 is ready and flow/gesture_search.py
     is the non-learned version of this layer. Next: a flow-matching motion
     expert conditioned on the marker intent, prosody and motion history,
     compared against gesture search and the rules.
   - **Live2D face second.** Extract head and face tracks from streamer
     video (the reference clips' kind of footage), train a face-motion
     generator on voice plus the beat markers, compare against flow/micro.py.
3. The robot adapter generalises the body files: a shared motion latent
   mapped per body, the cross-morphology direction of "One controller, many
   bodies" above, once there are several very different bodies or a robot.

## 2. Later directions

### Live co-watching and long video

| Work | Why | Notes |
|---|---|---|
| MiniCPM-o 4.5 (2604.27393) | full-duplex video/audio/speech; **its LLM core is Qwen3-8B**, the same as ours | [snippet]; GGUF builds exist; check the licence |
| JoyAI-VL-Interaction (JD, 2026-06) | each second: speak, stay silent, or hand off; initialised from Qwen3-8B | [snippet]; INT4 variant |
| LiveCC-7B (CVPR 2025) | continuous commentary at ~2 fps; the cheapest working baseline | [repo] code + HF checkpoints, licence not stated |
| Proact-VL (2603.03447) | a per-second "speak or stay silent" head; gaming commentary benchmark | [snippet] |
| StreamingVLM (2510.09608) | an attention-sink + recent-window KV cache for hours-long streams | [snippet]; code not confirmed |

First experiment: LiveCC-7B (or MiniCPM-o 4.5) on a 1-2 fps screen capture
of a recorded stream. Log time to first token, and how often it speaks or
stays silent; pass what it says to the character as a user line.

### Games from the screen

- **Labelling first.** No open keyboard-and-mouse FPS agent turned up.
  Record your own play with an input log and train an inverse dynamics
  model, as in "Pixels to Keys" (2609.37907, [snippet]: score each key by
  macro-F1, not accuracy).
- **NitroGen** (NVIDIA, CVPR 2026, [repo]): a 500M DiT that sees only the
  last frame and outputs gamepad actions; runs on Windows games only, for
  research use. A behaviour prior for gamepad games, not an FPS policy.
- **JARVIS-VLA** (Qwen2-VL-7B, Minecraft keyboard and mouse) and **Lumine**
  (30 Hz keyboard and mouse, reasons only when needed) are the
  architecture references. Lumine's "think only when needed" loop is the
  same idea as a slow planner over a fast body.
- **Matrix-Game 4.0** (Skywork, [repo] Apache-2.0, 2026-10) and 2.0 are
  playable world models (FPS and driving scenes) usable as a sandbox. The
  quoted speeds are FP8 on RTX 50-series cards, so measure on ours.

### World models

V-JEPA 2 ([repo] MIT code, encoders 300M-1B, an action-conditioned
predictor released) is the most runnable option. Freeze the encoder and
train a small action-conditioned predictor on your own frames plus inputs,
then compare it against copying the last frame. Genie 3, SIMA 2 and Cosmos
3 Super are closed or too large; Cosmos 3 Nano's real size needs checking.

### Driving with a pan-tilt camera

- **CARLA + Bench2Drive** (closed-loop, 220 routes) as the test harness.
  **SimLingo** (CVPR 2025, camera-only VLM driver, code and data open) as
  the first model to run on one GPU.
- **EyeVLA**, "The Robotic Eyeball" (2511.15279, [snippet]): a fine-tuned
  Qwen2.5-VL-7B aims a pan-tilt-zoom camera. It's the closest precedent for
  treating the camera's direction as an action. First experiment: SimLingo
  on a few Bench2Drive routes, then add camera pan as an extra action.
- Alpamayo 1 (NVIDIA, 10B) needs the whole 24 GB card for inference; use it
  as a reference only. A Donkey Car is the cheapest real camera-only car
  (about US$500 plus a Pi; a listing price).

### Robot hands

- **Simulation first, at no cost:** MuJoCo Playground has a LEAP-hand
  in-hand reorientation task with sim-to-real.
- **Cheapest real training loop:** an SO-101 arm (LeRobot, about US$100
  DIY base) with **SmolVLA** (450M, a flow-matching action expert, trains
  on one GPU from about 50 demos). SmolVLA's action expert is also a good
  reference for our own head.
- **Hands:** LEAP Hand (v1 about US$2,000 in parts; a cheaper v2), ORCA
  (tactile, under 2,000 CHF). DexGraspVLA (VLM planner over a diffusion
  controller) is the grasping template.
- pi0/pi0.5 LoRA needs more than 22.5 GB (openpi README, read directly),
  so it can't share the card with the LLM; use them as teachers or
  references.

## 3. What to do with this, in order

1. **The scheduled offline action-head test**, with the regression head
   first as the OpenVLA-OFT-style baseline, then flow matching, and three
   layers (`-8`, `-16`, middle).
2. **If flow wins:** distil it to one step (GestureFAR / SnapFlow) and add
   Real-Time Chunking in flow.live, so motion never holds back the audio.
3. **Raise the data ceiling** for VRM avatars: retarget BEAT2 (once its
   licence is confirmed) and use FloodDiffusion 2 as a body prior.
4. **Serving hidden states** (not done yet): llama-server doesn't stream
   intermediate layers. The options are a llama.cpp evaluation callback,
   llama-cpp-python, or transformers with hooks; GestureFAR and ELLSA 2 are
   the designs to read first.
5. **Later, in any order:** a co-watching baseline on MiniCPM-o 4.5 or
   LiveCC (MiniCPM-o shares our Qwen3-8B core); an inverse dynamics model
   from your own game recordings; SimLingo in CARLA with camera pan as an
   action; the LEAP hand in MuJoCo Playground before buying hardware.

## 4. Long prompts, wrong markers and memory: how others handle it (2026-11)

The student is told what to do by a ~3-4k-token beats2 prompt, and live it
still writes markers that break rules the prompt states (left/right flipped,
a wave without `[frame full]`, a sideways head on "look at me", no glance at
a place the viewer mentions). Two questions: why rules in a long prompt get
dropped, and what others do about it, including graph-shaped memory.

### Why rules in a long prompt get dropped

- **Many rules at once.** [IFScale](https://arxiv.org/abs/2507.11538)
  (500 keyword instructions): the best frontier models reach 68% at the
  highest density, with a bias towards earlier instructions.
  [Instruction Stacking Collapse](https://arxiv.org/pdf/2608.02639): about
  96% compliance at one verifier-checked constraint, 20% at twenty, much of
  the drop from conflicting pairs (abstract only read).
- **Position.** [Lost in the Middle](https://arxiv.org/abs/2307.03172)
  (TACL 2024): information at the start or end of the context is used best,
  the middle worst (a U-shaped curve), also for long-context models. A rule
  that matters once in 20 turns, in the middle of the beats2 prompt, is in
  the worst place. The end of the context is the strong spot: where the
  memory notes already go.

### What people do

1. **Bake the prompt into the weights (context distillation).**
   [Learning by distilling context](https://arxiv.org/abs/2209.15189),
   [Prompt Baking](https://arxiv.org/html/2409.13697v1) (KL between the
   prompted and the baked model), Microsoft's on-policy OPCD (the student
   trained on its own generations with the prompted teacher scoring;
   [coverage](https://venturebeat.com/orchestration/microsofts-new-ai-training-method-eliminates-bloated-system-prompts-without)),
   [TokMem](https://iclr.cc/virtual/2026/poster/10009482) (each procedure
   one trained memory token). Ours: SFT already does part of this, and
   `--student-prompt short` is context distillation; OPCD's on-policy step
   is DPO on q8r3's own samples.
2. **Fetch the rules only when needed (procedural memory).**
   [Voyager](https://80.lv/articles/voyager-gpt-4-powered-lifelong-learning-agent-for-minecraft/?amp=1)
   retrieves the top-5 skills by embedding;
   [Agent Workflow Memory](https://arxiv.org/pdf/2409.07429) (ICML 2025) and
   [LEGOMem](https://www.microsoft.com/en-us/research/publication/legomem-modular-procedural-memory-for-multi-agent-llm-systems-for-workflow-automation/)
   give the agent only the workflows that apply;
   [Hyperspell](https://docs.hyperspell.com/usage/procedural-memory)
   extracts procedures from session traces. No benchmark found compares
   this against one long static prompt: we have to measure it ourselves.
3. **Validate, then fix (and only then re-ask).**
   [Guardrails AI](https://guardrailsai.com/docs/hub/concepts/on_fail_policies):
   on failure, fix deterministically, re-validate, re-ask the model only if
   that still fails; it works only where the requirement is checkable
   ([writeup](https://latent.space/reading/guardrails-structured-output-quality/)).
   Robotics does the same for LLM-written behavior trees: a runtime
   validation gate that only lets through skills the robot has
   ([arXiv 2607.12220](https://arxiv.org/abs/2607.12220)),
   [BTGenBot](https://arxiv.org/abs/2403.12761v2) (schema check plus
   static-analysis cleanup gave the best results), BETR-XP-LLM (the LLM
   proposes the missing precondition after a failure). Re-asking costs a
   second generation, which live speech cannot afford: we fix and log.

### Graph-shaped memory

- [Zep / Graphiti](https://arxiv.org/abs/2501.13956v1): a temporal
  knowledge graph in three layers, episodes (raw), entities and facts,
  community summaries; a conflicting new fact marks the old edge invalid
  instead of deleting it. Reported: DMR 94.8% vs MemGPT 93.4%, LongMemEval
  up to +18.5% with 90% lower latency.
- [A-Mem](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html)
  (NeurIPS 2025): Zettelkasten notes with keywords and tags, linked to
  related notes; a new memory can update the old ones it links to.
- For one stream a graph database is too much. What we take: the three
  layers (`turns.jsonl` = episodes, the notes' facts and corrections =
  entities, the summary = communities, which flow/memory.py already has),
  and "invalidate, don't delete" for corrections and game state.

### Avatars specifically

The nearest work has an LLM pick gestures from a fixed list
([arXiv 2508.21087](https://arxiv.org/pdf/2508.21087)) or filter candidate
actions against scene and kinematic constraints
([SARGes](https://arxiv.org/pdf/2503.20202) related work). Nothing found
repairs inline motion markers, or turns the repairs into preference data.

### What we build (in this order)

1. **A repair layer** in flow.live: every marker goes through the checkers
   (flow/checks.py) before it is planned; a checkable failure gets a
   deterministic fix (flip the side of a look that went the wrong way, add
   `[frame full]` before a dance, remove one before a wave, which stays in
   bust, centre the head on "look at me"),
   logged as `repair:`, and the raw / fixed pair written as a DPO pair. No
   re-ask live.
2. **A rule graph**: the beats2 rules as nodes (trigger = an existing
   detector, text, links to the nodes it needs, e.g. look-side -> "her
   right is negative yaw"); the nodes a turn triggers (3 at most) go into
   the tail block with the memory notes as `(reminders: ...)`. A small
   hand-written graph, not embedding retrieval: a dozen rules, each with its
   own detector.
3. **An eval A/B** on the held-out sessions, scored by the checkers: the full
   prompt, the full prompt + reminders, the short core prompt + reminders.
   It decides whether r4 uses the short core prompt.

Later: r4 teacher data made with the core prompt plus reminders, so the
student learns to rely on them (context distillation), and DPO from the live
repair log (on-policy, as OPCD).

## 5. Real streamer data (2026-11)

Context: the student's replies read as "silly / too simple", and the teacher
and the simulated viewer are both LLMs. Advice, as given: no crawling of VODs
or clips (rights, platform terms, imitation of a real talent); use open
datasets of chats through the teacher, never as direct SFT data, so that no
real person's words or voice end up in the student. What was built:
`flow/realdata.py` (loaders and filters), `flow.dataset --real-viewer /
--style-seeds` (off by default), `scripts/style_ref.py` (the style numbers of
`flow/style.py` on a real file, for tuning r4). Nothing here downloads
anything: the files are fetched on the CPU server by hand, after the licence
of each has been read at its source.

### What could be read from here, and what could not

The session this was written in had GitHub but not huggingface.co (DNS and
the proxy both refuse it: "has to be allowed in the session's network
settings"). So only LiveChat's repository could be read first-hand (cloned,
`README.md`, `Dataset/README.md`, `LICENSE`, `Dataset/Subset/*` read). The
two Hugging Face cards were **not** read; what is written for them below is
what a web search summary of their cards reported, second hand, and is
marked as such. A licence that is not read from the source is not confirmed.
(Allowing huggingface.co in the session's network settings would let a
session verify them.) Before any download on the CPU server: open the card, copy its licence line
and attribution sentence into this section, and only then use
`--real-licence-ok`.

### LiveChat (Gao et al., ACL 2023)

- Source: https://github.com/gaojingsheng/LiveChat (read first-hand).
  1.33 million Chinese dialogues from live streams (Douyin) across 351
  streamers ("personas"), built automatically from videos: the streamer's
  speech (transcribed) against viewers' comments. Citation requested in the
  README (`gao-etal-2023-livechat`, ACL 2023, pp. 15387-15405).
- Licence: the repository has `LICENSE` = MIT, "Copyright (c) 2022
  Xiaobing.AI". That covers the repository (the code of the three tasks). The
  data is a Google Drive download (`Dataset/README.md`); **no licence or
  terms of use for the data is stated** in the README, the dataset README or
  the LICENSE, and the repository says nothing about the streamers' or
  viewers' consent. The sidebar licence (MIT) is not a data licence.
- Status: **do not use until confirmed.** Unclear is not permitted. What would
  confirm it: a reply from the authors (the paper's contact address), or the
  paper's data statement (not read here).
- If confirmed, what we would do: private research use of the cleaned lines
  as style references (never as SFT targets, never redistributed), the
  citation above in the data folder's README and in the docs, which
  `flow.dataset` writes (`README.md` and `avatars.json` "sources").
- Format: `Dataset/Subset/subset.json` is `{"streamer1": [[audience_comment,
  streamer_response], ...]}` (a 5 KB sample in the repository); the full
  processed data are pickles (`400kDialogueInPaper`, `AddresseeRecognition`,
  `RawDialogueData`: `train_data.pk` / `dev_data.pk` / `test_data.pk`), each
  sample `[streamer_id, audience_comment, streamer_response]`; the
  addressee-recognition samples have a list of 10 audience comments.
  `basic_profile.json` has the streamers' traits "replaced by anonymous
  numbers"; `text_profile.json` the text profiles.
- Usernames and PII: the samples have no username field (`streamer_id` is a
  label); the text can still hold a name in a thank-you ("感谢XX的火箭"),
  mentions, numbers and, because the speech is ASR, a streamer's own words
  with their names. `flow/realdata.py` drops gift thank-yous and welcomes,
  mentions, URLs, e-mails and 7+ digit numbers, emote-only and spam lines and
  anything on its keyword list, and keeps only 8-120 character replies.
- Pickles: `.pk` files are read through a restricted unpickler that builds
  lists, tuples, dicts, strings and numbers only. Open files from the source.

### OshiChats v2 (pykeio) -- second hand, unverified

- Source: https://huggingface.co/datasets/pykeio/oshichats-v2 (not
  readable from here; the summary below comes from a web search result, not
  from the card).
- Reported: licence CC BY 4.0 (tag `cc-by-4.0`); the card requires attribution
  to **pyke.io** for derivatives, **including trained models**; about 56
  million English chat messages from VTuber live streams before 18 November
  2023; a single JSONL file of about 30 GB. Its predecessor, OshiChats v1,
  was reported as CC BY-NC-SA 4.0 (non-commercial, share-alike, and
  gated behind a contact form): do not mix the two up.
- What CC BY 4.0 would let us do: use, adapt and share, commercially, with
  attribution; no non-commercial or share-alike clause. The attribution
  sentence we would put in the data folder, the docs and the model card:
  "Contains data from OshiChats v2 by pyke.io
  (https://huggingface.co/datasets/pykeio/oshichats-v2), CC BY 4.0."
  (`flow.realdata.SOURCES["oshichats"]`; confirm the exact wording on the
  card.)
- Caution that no licence tag settles: the messages are viewers' words, and
  the uploader may not hold the rights to them to license them. Treat the
  lines as references (the simulated viewer sees four of them to copy the way
  people type, and one is used verbatim now and then), never as targets.
- Usernames and PII: unknown (the schema was not read); a chat export
  normally has an author name, a channel id and a timestamp per message. The
  loader reads the message text only (`message`, `text`, `content`, `body`,
  `msg`, `comment`), never an author, channel or id field, removes "@mentions"
  and a leading "name:" prefix, and drops URLs, e-mails and long digit runs.
  Lines are not checked against a list of names: the pilot's lines are to be
  read by a person (`scripts/style_ref.py --show 50`).
- Status: **do not use until confirmed** on the card (licence line, the
  attribution wording, the schema).

### Differentiable/processed_chats -- second hand, unverified

- Source: https://huggingface.co/datasets/Differentiable/processed_chats (not
  readable from here).
- Reported: licence bigscience-openrail-m, English, tags twitch and chat,
  10K-100K rows, one file `processed.json` of about 10 MB, uploaded in March
  2023. Whether streamers consented to their transcripts being included
  (the task's note: "transcripts only with streamer consent") was **not
  found**; the search result said nothing on it.
- What bigscience-openrail-m would mean (from general knowledge of that
  licence family, **not from the card or the licence text**): a Responsible AI
  licence (Attachment
  A, behavioural use restrictions) that passes down to derivatives and, in
  the BigScience form, to a model trained on it: the restrictions are to be
  carried into the model's own licence. Commercial use is allowed by the
  licence itself; the restrictions are about use (no illegal use, no
  deception about generated content, no discrimination, and so on). The text
  of Attachment A was not read here.
- Usernames and PII: unknown; the same loader rules apply.
- Status: **do not use until confirmed** (card, consent statement, schema).
  The data is small and Twitch-only; the other two sources are the better
  start.

### Recommendation

1. Start with neither of the two Hugging Face sets and LiveChat unused
   until a licence is in hand: that costs nothing but time, because the
   calibration needs only a few thousand lines.
2. Read the OshiChats v2 card on the CPU server (it is the one most likely to
   be open: CC BY 4.0). If the licence line and the attribution sentence are
   as reported, record them here, then run the pilot with it as
   `--real-viewer` and `scripts/style_ref.py` on the same file; keep the
   attribution in the data folder.
3. Ask LiveChat's authors for a statement on the data (the code is MIT, the
   data has none); until then it is read-only calibration on the CPU
   server, nothing stored.
4. Whatever the source: the lines are references for the viewer and rhythm
   seeds for the teacher (labelled "don't copy content, names or facts"),
   never SFT targets; each used line is in the record's `real` field for
   audit; the cleaning (and its limits) is in the docstring of
   `flow/realdata.py`; no file of any of these datasets goes into the
   repository.

`--real-licence-ok` is the explicit step: `flow.dataset` refuses a file whose
source is one of the three above without it (a `custom` source is the
user's responsibility).
