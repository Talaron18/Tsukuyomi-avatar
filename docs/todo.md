# To do

## r4: a teacher with substance (2026-11)

Built (`beats3` in `flow/pose.py`, `--reminders` / `--notes` / `--game-share` /
`--history-turns` / `--real-viewer` / `--style-seeds` in `flow.dataset`,
`flow/style.py`, `flow/datanotes.py`, `flow/realdata.py`,
`scripts/style_ref.py`; the section "beats3 (r4 data)" in
`docs/embodiment.md`), tested with fake models only: **no DeepSeek call has
been made with any of it.** The order, on the data machine (`DEEPSEEK_API_KEY` set; VOICEVOX
for the voices that need it):

1. **Read the licences first** (docs/research.md, section 5). The session that
   wrote it could not reach huggingface.co: the OshiChats v2 and
   processed_chats cards were not read, and LiveChat's data has no stated
   licence. Open each card on the server (`curl -sL
   https://huggingface.co/datasets/pykeio/oshichats-v2/raw/main/README.md |
   head -80`), write the licence line, attribution sentence and schema into
   section 5, and only then download. Nothing is "go" until then; the pilot
   below runs without any real data (step 3a) and adds it later (3b).

2. **Download** (CPU server; outside the repo; never commit these):

       mkdir -p ~/realdata && cd ~/realdata
       # OshiChats v2: CC BY 4.0 as reported, ~30 GB JSONL; take the head of it
       # (confirm the licence first, attribution: pyke.io)
       # (huggingface-cli download pykeio/oshichats-v2 --repo-type dataset
       #  --local-dir oshichats-v2, or curl the first GB of the file)
       head -n 2000000 oshichats-v2/*.jsonl > oshichats.head.jsonl
       # LiveChat: the subset is in the repo (MIT code, data terms unstated):
       git clone --depth 1 https://github.com/gaojingsheng/LiveChat
       # the full processed data are the Google Drive files named in
       # LiveChat/Dataset/README.md (train_data.pk of RawDialogueData)
       # processed_chats (openrail-m as reported; consent unclear): skip for now

   Read 50 lines of each by eye before use: `python scripts/style_ref.py
   oshichats.head.jsonl --side viewer --show 50` and the same for LiveChat.

3. **The pilot** (about 60 sessions x 10 turns, ~$1-2; compare it with r3 before
   anything bigger):

       # a. no real data: the prompt, bibles, viewer modes, games, notes, reminders
       SESSIONS=60 TURNS=10 CONC=8 BUDGET_USD=3 OUT=data/datagen/r4p \
           AVATARS=Hiyori,lafei,Mao,seed-san PERSONAS=cheerful,sleepy,tsundere,teacher,idol \
           EXTRA="--teacher-prompt beats3 --notes auto --reminders auto \
                  --history-turns 5 --game-share 0.2" bash scripts/datagen.sh
       # b. the same with real chat (after step 1; +attribution in OUT/README.md)
       SESSIONS=60 TURNS=10 CONC=8 BUDGET_USD=3 OUT=data/datagen/r4pr \
           AVATARS=Hiyori,lafei,Mao,seed-san PERSONAS=cheerful,sleepy,tsundere,teacher,idol \
           EXTRA="--teacher-prompt beats3 --notes auto --reminders auto \
                  --history-turns 5 --real-viewer ~/realdata/oshichats.head.jsonl \
                  --style-seeds ~/realdata/LiveChat/Dataset/Subset/subset.json \
                  --real-licence-ok" bash scripts/datagen.sh

   (The source of each file is guessed from its path -- `oshichats`,
   `LiveChat` -- for the attribution in `OUT/README.md`; `--real-source`
   forces one for both. `--real-licence-ok` is yours to give, only after
   step 1.)
   Look at 20 replies by hand first (`python -m flow.dataexport stats
   data/datagen/r4p`), the `real` field of a few turns, and a game session
   end to end.

4. **Compare the style numbers** (820 turns each, so distinct-n is comparable):

       python scripts/style_ref.py ~/realdata/LiveChat/Dataset/Subset/subset.json \
           --compare data/datagen/r3 data/datagen/r4p data/datagen/r4pr \
           runs/eval_q8r3/recorded.jsonl --sample 820 --json style_r4.json
       python scripts/checks.py data/datagen/r3 data/datagen/r4p --by avatar

   Targets: words per reply 30-45 and chars (ja/zh) 55-80 (r3 teacher 25 /
   53 / 43); sentences per reply 2.5-3.5; the echo rate below 0.2 (r3 0.28);
   interjection openings below 0.2 (0.37 teacher, 0.48 student); the top
   opening share below 0.04; substance above 0.6 (0.41); and, with the same
   rows, near the real replies where the file has them (LiveChat pairs). The
   failure rows must not get worse: `beats` (beats per sentence >= 0.8),
   `look_sign`, `look_at_me`, `glance_at_mention`, `wave_obeyed`,
   `frame_before_wave`, `language`, `correction_obeyed`. Tune
   `SENTENCE_RULE["beats3"]`, the examples (`_replies3`) and the bibles, then
   regenerate the pilot; do not tune the metric definitions to fit.

5. **The full r4** with the pilot's settings: `SESSIONS=1600 TURNS=10 CONC=8
   BUDGET_USD=25 OUT=data/datagen/r4 AVATARS=Hiyori,lafei,Mao,huangjiafangzhou_3,
   八千代辉夜姬,seed-san` (the Azur Lane and fan-made rigs: train and test
   only, CLAUDE.md), the pilot's `EXTRA`. Then on the GPU machine
   (`~/runs`, `.venv`, `scp` the data from the server):

       python scripts/train_sft.py stats --data data/datagen/r4 --base Qwen/Qwen3-8B \
           --per turn --student-prompt short
       python scripts/train_sft.py train --data data/datagen/r4 --out ~/runs/sft/q8r4 \
           --base Qwen/Qwen3-8B --per turn --student-prompt short --avatar-weight lafei=2
       # merge, gguf, modelfile --name tsk-q8r4 --create as for q8r3;
       RUN=~/runs/sft/q8r4 STUDENT_MODEL=tsk-q8r4 RECORDED=0 TEACHER=0 \
           PROMPT_CONDS="short+rem" bash scripts/eval_sft.sh
       python scripts/checks.py ~/runs/sft/q8r4/eval runs/eval_q8r3 --by avatar

   `--per turn` (not the default session mode) so the notes and reminders sit
   on the latest turn only, as live shows them. Serve with `--body-prompt
   short --memory notes --reminders auto --history-turns 5`.

## r4p2: beats3 revision 2 (2026-11)

r4p (60 sessions x 10 turns, DeepSeek both sides) held the motion checks and
lifted substance 0.40 -> 0.79 and cut interjection openings 0.37 -> 0.17, but
its replies were one length (47 % 25-45 words, 42 % 45-70), literary, to a
viewer who typed paragraphs. Revision 2 (`docs/embodiment.md`, "beats3
revision 2") adds a per-turn length plan told to the teacher only, a spoken
register, a stronger glance rule, a different detail each time (`bible2`), a
viewer with chat-line limits, and flattened line breaks. Tested with fake
models only: **no DeepSeek call has been made with it.** On the CPU server:

       SESSIONS=60 TURNS=10 CONC=8 BUDGET_USD=3 OUT=data/datagen/r4p2 \
           AVATARS=Hiyori,lafei,Mao,seed-san PERSONAS=cheerful,sleepy,tsundere,teacher,idol \
           EXTRA="--teacher-prompt beats3 --beats3-rev 2 --length-mix 35,45,20 \
                  --notes auto --reminders auto --history-turns 5 \
                  --game-share 0.2" bash scripts/datagen.sh

(`--beats3-rev 2` and `--length-mix 35,45,20` are the defaults; `--beats3-rev
1` regenerates r4p byte for byte.) Then compare:

       python scripts/checks.py data/datagen/r4p data/datagen/r4p2 --by avatar
       python scripts/lengths.py data/datagen/r4p2

Read 20 replies and a `chat_burst`, a `callback` and a `place` session by hand
first. **Targets** (r4p2 against r4p):

- length mix of the non-quiet replies (`lengths.py`, the short / medium / long
  line) within 10 points of 35 / 45 / 20, and a hit rate per target worth
  reporting (no target under about 50 %: a low one means the band is wrong or
  the rule is ignored, not that the metric is);
- substance >= 0.7 (r4p 0.79), interjection openings <= 0.2 (0.17);
- `glance_at_mention` fail rate <= 20 % (r4p 39 %, 7/18; small n: also count
  the `place` turns across all of r4p2);
- the motion checks no worse than r4p: `beats` (beats per sentence >= 0.8; it
  was 0.74 and line breaks no longer count as sentence ends, so look at
  both), `look_sign`, `look_at_me`, `wave_obeyed`, `frame_before_wave`,
  `language`, `correction_obeyed`, reach 99.1 %;
- similes per reply (style rows) well under r4p's, by eye on the replies;
  viewer lines mostly 3-20 words, `place` in at most 2 turns a session, no
  invented promise in a `callback` turn, no `\n` in `reply_raw`;
- the sleepy sessions no longer all lean on one image (read them).

If the length plan is ignored, tighten `LENGTH_RULE2` or put the band in the
line; if a viewer filter drops too much (the `user` field versus `llm.user.raw`
in the records shows it), loosen `CHAT_WORDS`. Then r4p2 replaces r4p in
step 4 / 5 above (the same flags, `--beats3-rev 2`).

## q14r4 (after r4)

Plan and arithmetic: "A 14B student" in `docs/embodiment.md` (QLoRA r32 at 4.5k
tokens is ~18 GiB, ~15.5 with `--io-dtype bf16`, which is automatic for 14B;
Q4_K_M ~9 GB, KV at 16k ~2.5 GiB; merge on the CPU, ~30 GB of RAM). On the GPU
machine, after q8r4 shows the data is worth the 2x training time:

    python scripts/train_sft.py stats --data data/datagen/r4 --base Qwen/Qwen3-14B \
        --per turn --student-prompt short          # prints the memory lines
    python scripts/train_sft.py train --data data/datagen/r4 --out ~/runs/sft/q14r4 \
        --base Qwen/Qwen3-14B --per turn --student-prompt short --max-len 6144 \
        --max-steps 20                              # peak memory and tokens/s
    python scripts/train_sft.py train --data data/datagen/r4 --out ~/runs/sft/q14r4 \
        --base Qwen/Qwen3-14B --per turn --student-prompt short --max-len 6144 --restart
    python scripts/train_sft.py merge --run ~/runs/sft/q14r4        # --device auto: the CPU
    python scripts/train_sft.py gguf --run ~/runs/sft/q14r4 --llama-cpp ~/llama.cpp
    python scripts/train_sft.py modelfile --run ~/runs/sft/q14r4 --name tsk-q14r4 \
        --num-ctx 16384 --create
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve   # optional: KV halved
    RUN=~/runs/sft/q14r4 STUDENT_MODEL=tsk-q14r4 RECORDED=0 TEACHER=0 \
        PROMPT_CONDS="short+rem" bash scripts/eval_sft.sh
    python -m flow.live ... --backend ollama --model tsk-q14r4 --ollama-ctx 16384 \
        --body-prompt short --memory notes --reminders auto --history-turns 5

Check: peak memory of the 20-step probe against the table; first-text latency
next to the TTS and the renderer (the 8B's is 3.1 s median in the q8r3 eval:
the 14B decodes ~1.8x slower); the style rows and the failure rows against
q8r4. If the probe peaks above ~21 GiB: `--max-len 4096`, `--accum 8`, or the
`short` prompt only.


## DPO from the failure checkers and a VLM judge (2026-11)

Code is in (`flow/judge.py`, `flow/dpo.py`, `scripts/dpo_pairs.py`; the
section "VLM judge and DPO pairs" in `docs/embodiment.md`); none of it has run
on the GPU machine (the Live2D renders need GL and a model, the judge a VLM).
Order:

- [ ] **Look at a sheet.** `xvfb-run -a python -m flow.judge render --data
  data/datagen/r3 --id <session>/tNN --out sheet.png` for a look request, a
  wave and a dance on Hiyori, lafei, Mao: do the stills show what the markers
  say (bust and full views, hand raised, head turned)?
- [ ] **Judge calibration.** `scripts/dpo_pairs.py --calibrate 80` with the
  judge model chosen (Qwen3-VL-8B on vLLM first, then a 72B-class on
  SiliconFlow with `--max-calls`): per kind win / tie / loss, especially
  `look_flip` (the mirror convention) and `hand_drop`. Trust it for pairs only
  where it wins clearly; note the numbers in `docs/embodiment.md`.
- [ ] **Pairs from q8r3 samples.** K = 4 on the held-out sessions, plus
  `--synthetic` and `--teacher-chosen`; read `dpo.meta.jsonl` and the
  summary (drops, contradictions) and eyeball 20 pairs by hand.
- [ ] **DPO trainer** (written; not run on the GPU): `flow/train/dpo.py`,
  `scripts/train_dpo.py`; native, no TRL. Reference = the SFT merged model with
  the new LoRA off, reference log-probs precomputed, loss + RPO NLL, pairs from
  `dpo.jsonl`; merge / gguf / modelfile are SFT's (section "DPO training" in
  `docs/embodiment.md`). Build the pairs on the TRAINING sessions
  (`--side train`): the SFT run's held-out sessions are the eval and are left
  out of training. First GPU run: a `--max-steps 10` probe (step-0 check ~0,
  memory, speed), then the full run.
- [ ] **New pair kinds for q8r3's failures** (written): `wave_frame`,
  `opening_repeat` (`--opening-from runs/eval_q8r3/student.jsonl`), `look_flip`
  with the wider look detector (そっち / 君の / 你的 ...), `--kinds`,
  `--kind-n`; make them on the GPU machine and look at 20 by hand.
- [x] **look_at_me** (written, 2026-11): pair kind (`--kinds look_at_me`) and
  checker (`flow/checks.py`, in `scripts/checks.py`'s table) for "look at me /
  the camera" answered with a sideways head.yaw; make pairs on the GPU machine
  from data/datagen/r3 and look at 20 by hand.
- [x] **Live robustness** (written, 2026-11): first-token / stall watchdog,
  backend warm-up, chat batching, context-budget trimming (`--ollama-ctx`);
  not yet run on a real Ollama.
- [x] **correction / pose_copy** (written, 2026-11): pair kinds `correction`
  (real lines plus synthetic injected corrections, see embodiment.md) and
  `pose_copy`, checkers `correction_obeyed` and `pose_copy`
  (`scripts/checks.py`). Still to do on the GPU machine: the checkers'
  rates for the teacher shards vs `runs/eval_q8r3/student.jsonl`, the counts
  of source turns in data/datagen/r3, and 20 pairs of each kind by hand.
- [ ] **Eval with the checkers.** The tuned student on the held-out sessions
  (`scripts/eval_sft.sh`) and `scripts/checks.py` against q8r3: look_sign,
  body_request_obeyed, wave_obeyed, opening, template_repeat, language,
  beats; then the judge's means on the same replies, then Ollama as live.
- [ ] **VRM judging** (unsupported: NA): render a posed frame headless from
  `flow.vrmbody` states through `flow.vrmrender.Page` if the r2 seed-san
  needs it.

## Repair layer (2026-11)

Design: `docs/research.md` section 4; built and described in
`docs/embodiment.md` ("Repair layer"). Not yet run on the GPU machine:

- [ ] **Live with `--repair auto`.** A 40-turn session on Mao and on lafei
  with `--record`: read the `repair:` lines and the `{"kind": "repair"}`
  records, check that no fix lands on a reply that was fine, and that
  `scripts/checks.py runs/live/X/turns.jsonl` agrees with the log.
- [ ] **Train DPO with `repair_pairs.jsonl`.** Collect the pairs of several
  live sessions (`runs/live/*/repair_pairs.jsonl`), look at 20 by hand, then
  `scripts/train_dpo.py train --sft-run runs/sft/q8r3 --data
  runs/live/A/repair_pairs.jsonl,runs/live/B/repair_pairs.jsonl[,dpo.jsonl:2]`;
  re-evaluate with `scripts/checks.py` (look_sign, look_at_me,
  frame_before_wave, frame_for_dance, wave_obeyed).

## Rule reminders and the prompt A/B (2026-11)

`flow/rules.py` (`--reminders`, default off) and `evalsft --prompt-cond` are
built and documented in `docs/embodiment.md` ("Rule graph and reminders",
"Prompt A/B"); nothing has run on the GPU machine. In order:

- [ ] **Run the A/B on the GPU machine.** `PROMPT_CONDS="full full+rem
  short+rem"` on q8r3 (Ollama as live serves it, then vLLM `--conc 8`) and on
  the short-trained student (r3s) for `short+rem`, then `python scripts/checks.py
  runs/sft/q8r3/eval --by avatar`. The commands are in the section. The
  question: does `full+rem` beat `full` on look_sign / look_at_me /
  glance_at_mention / wave_obeyed / frame_before_wave, and does `short+rem`
  keep up (and the style rows: beats, opening, pose_copy)? If `full+rem` wins,
  switch `--reminders` to `auto` in live; if `short+rem` holds, r4 uses the
  short core prompt.
- [ ] **r4 data with the core prompt + reminders.** If the A/B favours
  `short+rem`: regenerate the r4 teacher data so the student learns to rely on
  the reminders (context distillation) instead of the long rules: the `short`
  student prompt as the system prompt and `rules.block()` on the last line of
  each user turn in the teacher's prompt (the live renderer, with the memory
  notes block of the long games above), then export with the same
  `--student-prompt short` and check the rows above on the teacher's own
  replies first.
- [ ] **Train DPO with `repair_pairs.jsonl`** (see "Repair layer" above), after
  the A/B: the pairs are on-policy for whatever prompt the live session ran
  with, so collect them with the prompt/reminders setting the student will be
  served with.
- [ ] **More nodes if the A/B shows rules the checks flag but no node
  covers** (`template_repeat`, `pose_copy`, `opening`): a node needs a
  detector that fires on the turn (the checks read the reply, a reminder needs
  the line), so they are not nodes yet.

## vLLM for student evals (2026-11)

vLLM (v0.33.0 checked) helps with evaluating and comparing students, not
with training or live. Ollama stays the live backend.

- [x] **A `vllm` backend.** `flow/director.py`: an `OPENAI_COMPAT` entry
  `vllm` (base `http://localhost:8000/v1`, a dummy key variable), in the
  `--backend` choices, and the request body carries the Modelfile's sampling
  (`flow/train/chat.py` `SAMPLING`: temperature 0.7, top_p 0.8, top_k 20)
  so the eval samples as the served student does.
- [x] **Sessions in parallel in the eval.** `flow/train/evalsft.py` runs the
  held-out sessions one after another; `--conc N` runs N sessions at once
  (turns within a session stay in order), `CONC` in `scripts/eval_sft.sh`.
  Without it vLLM's batching and prefix caching (the ~3k-token system prompt
  computed once) buy nothing.
- [ ] **Students as LoRA adapters.** On the GPU machine, in a venv of its own
  (the wheels are CUDA 13.0 only: the Windows driver must support it; WSL is
  the supported way on Windows):

      vllm serve Qwen/Qwen3-8B --enable-lora --max-lora-rank 32 \
          --lora-modules q8r2=<run>/adapter q8r3=<run>/adapter q8r3s=<run>/adapter \
          --chat-template <the flow.train.chat jinja> --gpu-memory-utilization <tuned>

  No merge or GGUF for an eval, and several students on one server for a
  side-by-side. The project's chat template must be passed, or the server
  uses Qwen3's own. Then document it in `docs/embodiment.md` (written, as
  "vLLM for student evals"; not yet run on the GPU machine).
- **DPO sampling.** The same server batch-samples several replies per
  held-out turn for the preference pairs (on-policy rejected answers picked
  by `flow/checks.py` and the VLM judge): `scripts/dpo_pairs.py --backend
  vllm --model q8r3 --k 4 --conc 8` (written; see the DPO section above).
- **vLLM is not what live serves.** Its numbers are the bf16 base plus the
  adapter; live serves the merged Q4_K_M in Ollama, so the chosen model is
  re-evaluated through Ollama (`scripts/eval_sft.sh` with the default
  backend).

Not now, and why:

- **Live on vLLM.** It reserves most of the GPU by default (renders and TTS
  share the 3090 Ti); its 4-bit options are worse fits than Ollama's Q4_K_M
  (~5 GB): GGUF is "highly experimental" and a plugin, bitsandbytes a plugin,
  AWQ/GPTQ need the merged model quantized first.
- **Hidden states from vLLM** (`extract_hidden_states`, a speculative-decoding
  method). One `.safetensors` file per request, written when it finishes:
  not streamed during generation, so no answer to serving hidden states live
  (`docs/research.md`). Offline it always saves the prompt tokens too, ~32 MB
  a turn for one layer at ~4k tokens, where `flow/train/hidden.py` reuses the
  system prompt's cache. Its docs do not say whether it works with LoRA.
  Look again if vLLM streams intermediate layers.
- **Structured outputs for the markers.** Parse warnings are 1.4%; a grammar
  for speech with markers in it costs more than it saves.

## Memory (2026-11)

`flow/memory.py` (documented in `docs/embodiment.md`, "Memory") is built and
off by default. Not yet:

- **Notes in the r4 teacher data.** The student was never trained with the
  `(stream notes: ...)` line, so it can only read it as context. Generate
  long multi-turn games (number guessing, riddles, corrected words, names,
  promises) where the teacher sees the notes block on the last user line
  (the live format, from the same `flow.memory` renderer) and the history
  window is shorter than the game, so the right reply needs the notes; the
  student then learns to rely on them. Use the `memory` records of live
  `turns.jsonl` (before / after / diff) as the first examples of what the
  consolidator writes.
- **A local consolidator.** An optional Ollama backend (`--memory-backend
  ollama`) so a session needs no API key. It must be a second model or share
  the student's server without evicting it (the 3090 Ti also renders); check
  the JSON validity rate against DeepSeek's on recorded sessions first.
- **Measure it.** Score the live sessions with and without `--memory` on the
  number game and the corrected-word cases (`flow.checks`
  `correction_obeyed`), and look at whether she reads the notes aloud.

## History, trims and glances (2026-11)

Built and described in `docs/embodiment.md` ("Live history: markers, chunked
trims, trailing poses, glances"). Not yet:

- **Measure the cache.** Compare first-text latency per turn with
  `--trim-chunk 1` and 4 on a 40-turn live session (Ollama
  `prompt_eval_count` / `prompt_eval_duration` show how much of the prompt
  was recomputed); choose the default chunk from it.
- **Tune `--history-markers`.** 2 is a guess; check `pose_copy` and
  `template_repeat` (`scripts/checks.py`) on a long session at 0, 2 and 4.
- **Generate the `place` topic.** Regenerate r4 teacher data with the rev-3
  beats2 prompt and the `place` topic, check `glance_at_mention` on the
  teacher's replies first (the pairs need a teacher that glances often
  enough to learn from), then add the `glance_at_mention` DPO pairs.
- **Glance for other mentions.** Someone arriving (a door, "behind you" with
  a name), a sound, a thing thrown: the same reflex with other triggers.
- **Trailing pose by design.** If the model learns to end on a pose on
  purpose (a final look), drop-or-merge becomes "keep": a marker that is
  wanted at the end should be an explicit `[hold]`-like beat.

