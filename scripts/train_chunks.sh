#!/bin/bash
# B を区切って学習する（物理の計算 Warp のメモリが学習を続けるうちに増え、6500 万〜1 億歩でメモリ不足で止まるため）。
# 使い方：scripts/train_chunks.sh <名前> <始める重み> <区切りの数> <1 区切りの歩数> [train_payload.py のほかの引数...]
# 学習の途中の評価のたびに Warp のメモリが残っていき、約 10 回の評価でメモリ不足になる。--evals 4 などで評価を減らす
# 区切りごとに <名前>_1, <名前>_2, ... に保存し、最後の区切りを <名前> にもコピーする。そのあと試験と動画
cd "$(dirname "$0")/.."
name=$1; init=$2; n=$3; steps=$4; shift 4
prev=$init
for k in $(seq 1 $n); do
  scripts/run_train.sh ${name}_$k --env carry --estimator models/estimator.json --init $prev --steps $steps "$@"
  if [ ! -f models/${name}_$k/params.pkl ]; then echo "区切り $k の学習の結果がない" >> logs/train_$name.chunks; break; fi
  grep " 歩 \|終わり\|Failed" logs/train_${name}_$k.out | tail -2 >> logs/train_$name.chunks
  prev=models/${name}_$k/params.pkl
done
mkdir -p models/$name && cp $prev models/$name/params.pkl && cp models/${name}_1/config.json models/$name/config.json
F="iterations limit\|warn_overflow\|nefc overflow\|^W0\|^I0\|^Module\|^Warp\|CUDA Toolkit\|Devices:\|\"cpu\"\|\"cuda:0\"\|Kernel cache\|\.cache/warp\|To dis\|for all)"
.venv_mjx/bin/python -u scripts/eval_carry.py --name $name --video logs/$name.mp4 2>&1 | grep -v "$F" > logs/eval_$name.out
echo B の試験が終わった >> logs/eval_$name.out
