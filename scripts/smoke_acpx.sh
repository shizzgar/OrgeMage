#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACPX_BIN="${ACPX_BIN:-/home/shizz/.nvm/versions/node/v22.22.1/bin/acpx}"
ORGEMAGE_BIN="${ORGEMAGE_BIN:-$ROOT_DIR/.venv/bin/orgemage}"
ORGEMAGE_DB="${ORGEMAGE_DB:-/tmp/orgemage-smoke.db}"
ORGEMAGE_MODEL="${ORGEMAGE_MODEL:-qwen::qwen3-coder-plus}"
ACPX_TIMEOUT="${ACPX_TIMEOUT:-180}"
SESSION_NAME="${SESSION_NAME:-smoke-$(date +%s)}"

if [[ ! -x "$ACPX_BIN" ]]; then
  echo "acpx not found or not executable: $ACPX_BIN" >&2
  exit 1
fi

if [[ ! -x "$ORGEMAGE_BIN" ]]; then
  echo "orgemage not found or not executable: $ORGEMAGE_BIN" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "rg is required for smoke assertions" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for smoke assertions" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap ':' EXIT

agent_cmd="$ORGEMAGE_BIN --db $ORGEMAGE_DB acp --stdio"
common_args=(
  --verbose
  --format json
  --timeout "$ACPX_TIMEOUT"
  --approve-reads
  --non-interactive-permissions deny
  --model "$ORGEMAGE_MODEL"
  --agent "$agent_cmd"
)

new_out="$tmpdir/session-new.jsonl"
prompt_one_out="$tmpdir/prompt-identity.jsonl"
prompt_two_out="$tmpdir/prompt-analysis.jsonl"

"$ACPX_BIN" "${common_args[@]}" sessions new --name "$SESSION_NAME" >"$new_out"
"$ACPX_BIN" "${common_args[@]}" prompt -s "$SESSION_NAME" "Ты кто?" >"$prompt_one_out"
"$ACPX_BIN" "${common_args[@]}" prompt -s "$SESSION_NAME" "Без изменения файлов и без вмешательства в OS: проанализируй проект в текущей директории и кратко ответь по пунктам. 1) Какова цель проекта? 2) Какие 3 модуля здесь ключевые? 3) Какой один риск в ACP/orchestrator path виден по коду или тестам? 4) Какой smoke-test через acpx ты бы рекомендовал?" >"$prompt_two_out"

rg -q "\"currentModelId\":\"$ORGEMAGE_MODEL\"" "$new_out" "$prompt_one_out" "$prompt_two_out"
python3 - "$ORGEMAGE_MODEL" "$prompt_one_out" "$prompt_two_out" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
prompt_one = Path(sys.argv[2])
prompt_two = Path(sys.argv[3])


def load_json_lines(path: Path):
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def contains_stop_reason(records, expected: str) -> bool:
    for record in records:
        result = record.get("result")
        if isinstance(result, dict) and result.get("stopReason") == expected:
            return True
    return False


def has_protocol_error(records) -> bool:
    return any("error" in record for record in records)


def iter_updates(records):
    for record in records:
        params = record.get("params")
        if not isinstance(params, dict):
            continue
        update = params.get("update")
        if isinstance(update, dict):
            yield update


prompt_one_records = load_json_lines(prompt_one)
prompt_two_records = load_json_lines(prompt_two)

if not contains_stop_reason(prompt_one_records, "end_turn"):
    raise SystemExit("identity prompt did not finish with end_turn")
if not contains_stop_reason(prompt_two_records, "end_turn"):
    raise SystemExit("analysis prompt did not finish with end_turn")
if has_protocol_error(prompt_two_records):
    raise SystemExit("analysis prompt produced ACP error payload")

planning_sources = []
for update in iter_updates(prompt_two_records):
    meta = update.get("_meta")
    if isinstance(meta, dict):
        planning = meta.get("planningProvenance")
        if isinstance(planning, dict):
            source = planning.get("source")
            if isinstance(source, str):
                planning_sources.append(source)

if "coordinator" not in planning_sources:
    raise SystemExit("analysis prompt did not report coordinator planning provenance")
if "local_fallback" in planning_sources:
    raise SystemExit("analysis prompt unexpectedly used local_fallback")
PY

echo "Smoke passed"
echo "Session: $SESSION_NAME"
echo "Model:   $ORGEMAGE_MODEL"
echo "Logs:    $tmpdir"
