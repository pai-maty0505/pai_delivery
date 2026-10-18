#!/bin/bash
# 学習を裏で動かす。物理計算の警告（Playground の G1 の設定で毎回出る）はログから除く
# 使い方：scripts/run_train.sh <name> <train_payload.py の引数...>
cd "$(dirname "$0")/.."
name=$1; shift
# JAX が先に取る GPU のメモリを 55% に（元は 75%）。物理の計算（Warp）は JAX の外のメモリを使い、ケースを持つ環境（B）では
# 残り 25% では足りず、学習の途中の評価でメモリ不足で止まった
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
.venv_mjx/bin/python -u scripts/train_payload.py --name "$name" "$@" 2>&1 \
  | grep --line-buffered -v "iterations limit\|warn_overflow\|nefc overflow" > "logs/train_$name.out"
