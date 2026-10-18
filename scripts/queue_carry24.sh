#!/bin/bash
# carry24：事務所の場面で学習する（2026-10-18）。机・椅子・壁・棚は最初からその場所にあり、当たり判定・摩擦・接触の固さは
# 事務所の場面（full_run.py）と同じ。置く練習は C の道順の途中から、事務所と同じ歩き方の指令（道順・減速・その場で向きを変える・
# 最後の前進）で机の前まで歩いて止まり、2 秒待って、事務所と同じ表・同じ触れ方の決め方で置く。止まってからの流れは置き終わるまで減点。
# GPU のメモリが学習を続けるうちに増えるので、2000 万歩 × 6 回に分ける
cd "$(dirname "$0")/.."
scripts/train_chunks.sh carry24 models/carry21_3/params.pkl 6 20e6 --grip tray --crouch --crouch_ff --crouch_lean 8 \
  --crouch_file place_traj_tray_quick_h0.10.npz --place_touch_file place_touch_h0.10_b2.npz --office --drift_all \
  --stop_drift_weight -10 --crouch_weight -40 --place_yaw_weight -5 --feet_vel_weight -8 --mode_p 0.2,0.15,0.05,0.05,0.55 \
  --estimator models/estimator_tray.json --evals 3 "$@"
echo carry24（事務所の場面）の学習が終わった >> logs/queue_carry24.out
