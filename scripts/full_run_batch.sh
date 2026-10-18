#!/bin/bash
# 事務所の場面で A → B → C → D を、条件（摩擦、机の前で止まる位置）を変えて N 回通す（CPU、4 つずつ）
# 使い方：[FR_ARGS="full_run.py のほかの引数"] scripts/full_run_batch.sh <方策> <回数> <結果の jsonl> [D の方策]
cd "$(dirname "$0")/.."
walk=$1; n=$2; out=$3; pw=${4:-$1}
seq 1 $n | xargs -P 4 -I{} bash -c "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/full_run.py --walk $walk --place_walk $pw $FR_ARGS --seed {} --json $out > /dev/null 2>&1"
python3 - "$out" <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
n = len(rs)
for k in ("A", "B", "C", "D", "通し"):
    print(f"{k}: {sum(r[k] for r in rs)} / {n}")
print("箱と壁・ドアの距離の最小（cm）", sorted(r.get("wall_clear_cm") for r in rs))
print("机の前で止まったずれ（cm）", sorted(r["stop_err_cm"] for r in rs))
PY
