#!/bin/bash
# carry24 の続き：2000 万歩ずつでは、学習の最後の評価のとき GPU のメモリが足りず止まり、各回の後半 1000 万歩が残らなかった
# （1000 万歩の評価で残した方策は使える）。carry24_3（3000 万歩分）から、1000 万歩 × 4 回（評価は始めと終わりだけ）で続ける
cd "$(dirname "$0")/.."
scripts/train_chunks.sh carry24b models/carry24_3/params.pkl 4 10e6 --grip tray --crouch --crouch_ff --crouch_lean 8 \
  --crouch_file place_traj_tray_quick_h0.10.npz --place_touch_file place_touch_h0.10_b2.npz --office --drift_all \
  --stop_drift_weight -10 --crouch_weight -40 --place_yaw_weight -5 --feet_vel_weight -8 --mode_p 0.2,0.15,0.05,0.05,0.55 \
  --estimator models/estimator_tray.json --evals 2 "$@"
echo carry24b（事務所の場面）の学習が終わった >> logs/queue_carry24b.out
