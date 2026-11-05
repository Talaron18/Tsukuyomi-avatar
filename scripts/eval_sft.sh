#!/usr/bin/env bash
# The tuned director against its teacher (flow.train.evalsft): the
# held-out sessions of a training run (RUN/split.json), user line by user
# line, through the student (as it will be served: Ollama by default),
# the teacher live (DeepSeek, the same prompts) and the teacher's recorded
# replies; each reply parsed and played on the avatar (poses solved, moves
# built), scored: parse-warning rate, refusals, markers per turn, unplayed
# markers, reach / MISS, first-token latency and tokens/s. Then the table,
# and with AB=1 a blind A/B page of rendered first turns (the
# scripts/body_vs_tags.sh layout, flow.legacy.abtest).
#
# Usage, from the repo root on the server (Ollama serving the student,
# DEEPSEEK_API_KEY for the live teacher):
#   RUN=runs/sft/q8 DATA=data/datagen/r1 STUDENT_MODEL=tsk-q8 \
#       bash scripts/eval_sft.sh
#   STUDENT_BACKEND=hf STUDENT_MODEL=Qwen/Qwen3-8B \
#       HF_ADAPTER=runs/sft/q8/adapter bash scripts/eval_sft.sh
#   STUDENT_BACKEND=vllm STUDENT_MODEL=q8r3 CONC=4 bash scripts/eval_sft.sh
#       (a vllm server with the adapter as a LoRA module: docs/embodiment.md)
#   PROMPT_CONDS="full full+rem short+rem" STUDENT_BACKEND=vllm \
#       STUDENT_MODEL=q8r3 CONC=8 RECORDED=0 TEACHER=0 bash scripts/eval_sft.sh
#       (the prompt A/B: the student on the same held-out turns three ways,
#        student.jsonl, student.full+rem.jsonl, student.short+rem.jsonl;
#        then: python scripts/checks.py $RUN/eval)
#
# Environment knobs (defaults in brackets):
#   RUN              [runs/sft/q8] flow.train.sft's --out (split.json)
#   DATA             [the data folder in RUN/data.json] flow.dataset --out
#   OUT              [$RUN/eval]
#   STUDENT_BACKEND  [ollama] ollama | hf | vllm
#   STUDENT_MODEL    [tsk] the Ollama name (or the hf base / merged folder;
#                    vllm: the LoRA module name of vllm serve)
#   VLLM_URL         [] vllm: the server's base URL [http://localhost:8000/v1]
#   CONC             [1] held-out sessions at once, for the student and the
#                    live teacher (a batching server: vllm, or Ollama with
#                    OLLAMA_NUM_PARALLEL); the recorded run is local
#   HF_ADAPTER       [] --backend hf: the adapter on STUDENT_MODEL
#   STUDENT_PROMPT   [teacher] short = the student was trained with
#                    flow.train.sft --student-prompt short: the held-out
#                    turns are sent with the short prompt (student only)
#   PROMPT_CONDS     [full] the student's prompt conditions, space or comma
#                    separated, one run each (flow.train.evalsft run
#                    --prompt-cond): full = the data's prompt (student.jsonl);
#                    full+rem = and flow.rules' reminders on each user turn
#                    (student.full+rem.jsonl); short+rem = the short student
#                    prompt and the reminders (student.short+rem.jsonl)
#   HF_4BIT          1 = the hf base in 4 bit [0]
#   TEACHER_BACKEND  [deepseek]; TEACHER_MODEL [its default]
#   TEACHER          1 = run the live teacher [1 when its key is set]
#   RECORDED         1 = score the recorded teacher replies (free) [1]
#   BASE_MODEL       an Ollama name for the untuned base too, e.g.
#                    qwen3:8b (sent num_ctx 8192) []
#   SESSIONS         [all] the first N held-out sessions; TURNS [all]
#   AB               1 = the A/B pack (renders: GL / Chromium) [0]
#   AB_N             [6] first turns per avatar; TTS [auto] for the renders
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

RUN=${RUN:-runs/sft/q8}
if [[ -z "${DATA:-}" && -f "$RUN/data.json" ]]; then
    DATA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['data'])" "$RUN/data.json")
fi
DATA=${DATA:?set DATA to the flow.dataset folder}
OUT=${OUT:-$RUN/eval}
STUDENT_BACKEND=${STUDENT_BACKEND:-ollama}
STUDENT_MODEL=${STUDENT_MODEL:-tsk}
HF_ADAPTER=${HF_ADAPTER:-}
STUDENT_PROMPT=${STUDENT_PROMPT:-teacher}
PROMPT_CONDS=${PROMPT_CONDS:-full}
HF_4BIT=${HF_4BIT:-0}
CONC=${CONC:-1}
TEACHER_BACKEND=${TEACHER_BACKEND:-deepseek}
TEACHER_MODEL=${TEACHER_MODEL:-}
RECORDED=${RECORDED:-1}
BASE_MODEL=${BASE_MODEL:-}
AB=${AB:-0}
AB_N=${AB_N:-6}
TTS=${TTS:-auto}
if [[ -z "${TEACHER:-}" ]]; then
    TEACHER=0
    [[ $TEACHER_BACKEND == deepseek && -n "${DEEPSEEK_API_KEY:-}" ]] && TEACHER=1
    [[ $TEACHER_BACKEND == siliconflow && -n "${SILICONFLOW_API_KEY:-}" ]] && TEACHER=1
fi

common=(--data "$DATA" --split "$RUN/split.json" --out "$OUT")
[[ -n "${SESSIONS:-}" ]] && common+=(--sessions "$SESSIONS")
[[ -n "${TURNS:-}" ]] && common+=(--turns "$TURNS")
mkdir -p "$OUT"

student=(--name student --backend "$STUDENT_BACKEND" --model "$STUDENT_MODEL")
[[ -n "$HF_ADAPTER" ]] && student+=(--hf-adapter "$HF_ADAPTER")
[[ $STUDENT_PROMPT != teacher ]] && student+=(--student-prompt "$STUDENT_PROMPT")
[[ $HF_4BIT == 1 ]] && student+=(--hf-4bit)
[[ $CONC != 1 ]] && student+=(--conc "$CONC")
if [[ $STUDENT_BACKEND == vllm ]]; then
    [[ -n "${VLLM_URL:-}" ]] && student+=(--api-base "$VLLM_URL")
    # The sampling of the run's template family (train.json), else Qwen3's.
    if [[ -f "$RUN/train.json" ]]; then
        fam=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('family',''))" "$RUN/train.json")
        [[ -n "$fam" ]] && student+=(--vllm-family "$fam")
    fi
fi
for cond in ${PROMPT_CONDS//,/ }; do
    extra=()
    log="$OUT/student.log"
    if [[ $cond != full ]]; then
        extra=(--prompt-cond "$cond")
        log="$OUT/student.$cond.log"
    fi
    python -m flow.train.evalsft run "${common[@]}" "${student[@]}" \
        ${extra[@]+"${extra[@]}"} 2>&1 | tee "$log"
done

if [[ $RECORDED == 1 ]]; then
    python -m flow.train.evalsft run "${common[@]}" --name recorded \
        --recorded 2>&1 | tee "$OUT/recorded.log"
fi
if [[ $TEACHER == 1 ]]; then
    teacher=(--name teacher --backend "$TEACHER_BACKEND")
    [[ -n "$TEACHER_MODEL" ]] && teacher+=(--model "$TEACHER_MODEL")
    [[ $CONC != 1 ]] && teacher+=(--conc "$CONC")
    python -m flow.train.evalsft run "${common[@]}" "${teacher[@]}" \
        2>&1 | tee "$OUT/teacher.log"
fi
if [[ -n "$BASE_MODEL" ]]; then
    python -m flow.train.evalsft run "${common[@]}" --name base \
        --backend ollama --model "$BASE_MODEL" --ollama-ctx 8192 \
        2>&1 | tee "$OUT/base.log"
fi

python -m flow.train.evalsft compare "$OUT" | tee "$OUT/compare.txt"

if [[ $AB == 1 ]]; then
    names=student
    [[ $TEACHER == 1 ]] && names="teacher,$names"
    [[ $TEACHER != 1 && $RECORDED == 1 ]] && names="recorded,$names"
    python -m flow.train.evalsft ab --data "$DATA" --out "$OUT" \
        --names "$names" --n "$AB_N" --tts "$TTS"
fi
