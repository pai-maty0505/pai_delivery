# pai_delivery

人型ロボット **Unitree G1** が事務所の椅子から立ち上がってドアを開け、廊下の配達員から **MacBook Air の箱（33 × 25 × 5 cm、1.23 kg）** を手で受け取り、机まで運んで置く。物理シミュレータ **MuJoCo** の中だけで行う。

このリポジトリには、最終の形を動かすのと、最後の方策を学習し直すのに要るものだけを置いた。

---

## 1. 目的

### 1.1 課題

人型ロボットが「歩く」と「腕で物を扱う」を組み合わせ、人とやり取りしながら作業する動画（1〜3 分）を作る。作業は、事務所で次の 4 つを続けて行うこと。

| 区間 | 内容 |
|---|---|
| A 座った状態→ドアを開ける | 椅子から立ち上がり、ドアまで歩き、両手でドアを押し開ける |
| B 受け取る | ドアの外で、配達員が差し出して下ろした箱を両腕で受け取る |
| C 机まで運ぶ | 戸口から下がって部屋へ戻り、椅子の北を通って机の正面で止まる |
| D 机に置く | 腰を沈め、指先から机に下ろし、箱の奥の縁が着いたら手→肘の順に開いて置く |

### 1.2 守る決まり

- **物理だけで動かす**：箱は、手との摩擦と重力だけで持つ。固定ジョイント、物体を手に貼り付けること、テレポート、当たり判定を消すことは使わない。
  - 配達員が持っている間（受け渡しの前）だけは、箱を配達員の手と一緒に動かす。
  - ロボットの手のひらの 3 mm 上で放してからは、物理で計算する。
- **接触・衝突・摩擦は、その場面で触れるはずの所どうしに入れる**：人がドアを手で開けても足はドアに当たらない、荷物を受け取っても手どうしは触れない、と同じ考え方（7.2）。
- **環境はオフィスと廊下**：板張りの床と白い壁。物理の計算に使う床（灰色のタイル）は描かない。

### 1.3 研究としての問い

Unitree の学習済みの歩き方は、1.5 kg 以上の荷物を持つと倒れた。そこで、荷物を持って歩ける歩き方を強化学習で新しく作った。そのうえで、次の工夫を試した。

| 工夫 | 内容 | 使っているか |
|---|---|---|
| A | 受け取った箱の重さを、腕の 14 関節のトルクから推定し（1 次の式）、歩き方の方策の入力にする | 使っている（`estimate`・`carry10`・`carry24b_1`） |
| B | 学習の練習の中で、歩いている途中や立っているときに、荷物の重さが急に載る・なくなる（受け取り・降ろす・机に置く） | 使っている（学習のやり方） |
| C | 腕を IK のお手本から模倣学習する | 使っていない（腕は IK の動きをそのまま使う） |

A と B は組で効く（A は重さを感じる感覚、B は重さが変わる場面の練習）。

---

## 2. 開発環境

### 2.1 確かめた環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| GPU | NVIDIA GeForce RTX 3060（12 GB）。学習に要る。実行（試験・動画）は CPU で動く |
| CPU・メモリ | 8 コア、16 GB |

### 2.2 ライブラリ

| ライブラリ | 版 | 使い道 |
|---|---|---|
| MuJoCo | 3.16.0 | 物理シミュレータ。事務所の場面の実行（CPU） |
| MuJoCo MJX ＋ NVIDIA Warp | 3.16.0 ＋ 1.18.0 | GPU で 8192 体を同時に動かして学習する物理計算 |
| MuJoCo Playground | 0.2.0 | Unitree G1 のモデル（形は mujoco_menagerie、29 関節）と、歩き方の学習の土台 |
| JAX | 0.11.2 | GPU での計算 |
| Brax | 0.14.2 | 強化学習のアルゴリズム PPO |
| Flax・Optax | 0.12.10・0.2.8 | 方策のニューラルネットワークと最適化 |
| NumPy・SciPy | 2.5.4・1.18.1 | 数値計算、腕の逆運動学 |
| Pillow | 12.3.0 | 箱の「MAC 13」の文字 |
| ffmpeg | — | 動画の書き出し |
| Open JTalk | — | 動画の声 |

版は `requirements.txt` に固定してある。

### 2.3 準備

Python の環境を作る：

```bash
python3 -m venv .venv_mjx
```

```bash
.venv_mjx/bin/pip install -r requirements.txt
```

MuJoCo Playground は、初めて使うときに mujoco_menagerie 全体をダウンロードしようとする。G1 だけを取ってきて、リンクを置く：

```bash
mkdir -p third_party && cd third_party && git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git && cd mujoco_menagerie && git sparse-checkout set unitree_g1
```

```bash
ln -s $PWD/third_party/mujoco_menagerie .venv_mjx/lib/python3.12/site-packages/mujoco_playground/external_deps/mujoco_menagerie
```

動画の ffmpeg と、声の Open JTalk（辞書 `/var/lib/mecab/dic/open-jtalk/naist-jdic`、声 `/usr/share/hts-voice/nitech-jp-atr503-m001/`）を入れる：

```bash
sudo apt install ffmpeg open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001
```

---

## 3. 動画作成方法

動画は、3.1 で作り、3.2 で見る。

| | 3.1 A〜D を通して動画を作る | 3.2 動画を見る |
|---|---|---|
| すること | シミュレーションを動かし、その様子を動画のファイルに書き出す。声とノックの音を入れ、93 秒に切る | 3.1 でできた動画のファイルを、ブラウザやプレーヤーで再生する |
| ロボットを動かすか | 動かす（学習済みの方策 `models/` を使う。CPU で 1 回およそ 5〜10 分） | 動かさない（できた動画を見るだけ） |
| 使うもの | Python の環境（`.venv_mjx`）、ffmpeg、Open JTalk | 動画を再生できるソフト（Chrome・Firefox・ffplay） |
| できるもの | `logs/full_run.mp4` などの動画のファイル | なし |

### 3.1 A〜D を通して動画を作る

シミュレーションの中でロボットを動かし、A〜D を通す様子を動画のファイルにする。条件（摩擦・止まる位置）は 2 番目を使う。

まず、A〜D を通し、動画を書き出す（声なし）：

```bash
mkdir -p logs && JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/full_run.py --walk carry10 --place_walk carry24b_1 --lateral 1 --seed 2 --video logs/full_run.mp4 --events logs/full_run_events.json
```

声（「はい」「荷物が届いています」「ありがとう。」）とノックの音を入れる：

```bash
python3 scripts/make_audio.py --events logs/full_run_events.json --video logs/full_run.mp4 --out logs/full_run_voice.mp4
```

置き終わって手を上げたところ（93 秒）までに切る（その後の体を起こす場面と、最後に真上から箱を写す場面を除く）：

```bash
ffmpeg -y -i logs/full_run_voice.mp4 -t 93 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -movflags +faststart logs/full_run_voice_93s.mp4
```

| ファイル | 中身 |
|---|---|
| `logs/full_run.mp4` | 動画（声なし、約 101 秒） |
| `logs/full_run_events.json` | 声と音を入れる時刻 |
| `logs/full_run_voice.mp4` | 声とノックの音つき |
| `logs/full_run_voice_93s.mp4` | 93 秒に切ったもの |

### 3.2 動画を見る

3.1 でできた動画のファイルを再生する。ロボットは動かさない。

動画は普通の形式（H.264 の映像と AAC の音声、1280 × 720）。Ubuntu の「ビデオ」（Totem）は、OpenGL が使えない環境では「OpenGL サポートを初期化できませんでした」と出て開けない。そのときは、ブラウザか ffplay で開く。

Google Chrome で開く：

```bash
google-chrome logs/full_run_voice_93s.mp4
```

Firefox で開く：

```bash
firefox logs/full_run_voice_93s.mp4
```

ffplay で開く（音も出る）：

```bash
ffplay -autoexit logs/full_run_voice_93s.mp4
```

ファイルマネージャーからなら、ファイルを右クリックして「別のアプリケーションで開く」から Chrome か Firefox を選ぶ。

### 3.3 `full_run.py` の主な引数（3.1 で使う）

| 引数 | 意味 |
|---|---|
| `--walk` | B・C を動かす方策（`carry10`） |
| `--place_walk` | D を動かす方策（`carry24b_1`）。机の前で止まっている間に切り替える |
| `--seed` | 条件の番号（なしなら決まった条件） |
| `--lateral 1` | 机の前へ進むとき、横のずれも直す |
| `--video` | 動画を書き出す |
| `--events` | 声と音を入れる時刻を書き出す |
| `--stills` | 受け渡しと D（0.5 秒ごとに真横・真上から）を近くから写した画像を書き出すフォルダ |
| `--json` | 結果を 1 行の JSON で書き足す |
| `--no_label` | 箱に「MAC 13」を描かない（速く動く） |

---

## 4. 学習方法

### 4.1 学習のやり方

- 強化学習（PPO、Brax）で、脚 12 ＋腰 3 の関節を動かす方策を学習する。8192 体のロボットを GPU で同時に練習させる。
- ネットワークは方策・価値とも 512・256・128 の多層パーセプトロン。
- 方策の入力は、次のものです。
  - 腰の速さ・角速度・重力の向き
  - 歩き方の指令
  - 脚と腰の関節の角度と速さ、前の出力
  - 歩く周期
  - **腕のトルクから推定した箱の重さ**（工夫 A）
  - しゃがむ量の指令
- 練習の場面は、次のものを割合を決めてランダムに選ぶ（工夫 B）。
  - 箱を持って歩く
  - 歩きながら受け取る
  - 降ろす
  - 手ぶら
  - 止まって机に置く
- 学習を続けると、GPU の物理計算（Warp）のメモリが増えて止まる。そのため、`scripts/train_chunks.sh` で学習を区切って続ける。

### 4.2 方策ごとの学習

学習を裏で動かす（ログは `logs/train_<名前>.out`）：

| 方策 | 学習のコマンド | 出発点 |
|---|---|---|
| `standup`（A 立ち上がる） | `scripts/run_train.sh standup --env standup --steps 100e6` | なし。お手本 `models/standup_ref.npz`（`scripts/standup_ref.py`）からのずれを学習 |
| `estimate`（A 歩く） | `scripts/run_train.sh estimate --obs estimate --estimator models/estimator.json`（1.5 億歩） | なし。`models/estimator.json` は、本当の重さを入力にした方策 `oracle`（`--obs oracle`）で歩かせたデータから `scripts/calib_estimator.py --name oracle` で作った |
| `carry24b_1`（D 置く） | `scripts/queue_carry24.sh` の後に `scripts/queue_carry24b.sh`（その 1 回目が carry24b_1） | `models/carry21_3` |
| `carry10`（B・C）・`carry21_3` | 長い学習の系譜（carry7 → carry9x → carry10、… → carry19_1 → carry21_3）。各方策の `config.json` の `init` と `env` に、出発点と学習の設定が残っている | — |

`carry24` の学習（`scripts/queue_carry24.sh`）の中身：

- **場所**：事務所の部屋そのもので学習する（机・椅子・壁・棚・ドアが最初からその場所）。当たり判定・摩擦・接触の固さは、実行（`full_run.py`）と同じ。
- **置く練習**：C の道順の途中から始め、事務所と同じ歩き方の指令で机の前まで歩いて止まり、2 秒待って置く。
- **置く腕の動き**：実行と同じ表（`models/place_touch_h0.10_b2.npz`）と、同じ「机に触れたか」の決め方を使う。

### 4.3 表を作り直す

しゃがむ姿勢の表（股・膝・足首の角度）：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/crouch_offsets.py
```

机に触れた時刻ごとの置く腕の動きの表（上体の傾きは、実際に沈んだとき後ろへ約 2° なので −2）：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv_mjx/bin/python scripts/crouch_place_touch.py 0.10 -2 0.08 models/place_touch_h0.10_b2.npz
```

### 4.4 学習した方策を試す

GPU で 100 体ずつ、学習の環境の中の場面（運ぶ・受け取る・降ろす・止まって降ろす・置く）を試す：

```bash
.venv_mjx/bin/python scripts/eval_carry.py --name carry24b_1
```

事務所の場面で試すときは、5.2 の `full_run_batch.sh` の最後の引数（D の方策）を学習した方策の名前にする。

---

## 5. 結果

### 5.1 結果

条件をばらつかせて A〜D を 8 回通した（確かめ方は 5.2）。

| A | B | C | D | 通し |
|---|---|---|---|---|
| 8 | 8 | 6 | 7 | 5 / 8 |

成功の条件：

- A：ドアが 45° 以上開き、倒れない。
- B：箱が持つ位置から 3 cm 以内。
- C：箱を落とさず、止まる位置のずれが 6 cm 未満。
- D：箱の底が机の高さ、傾き 3° 未満、四隅が机の内側、足の動き 2 cm 未満。

### 5.2 結果の確かめ方（条件を変えて 8 回通す）

動画を作る手順（3 章）とは別の試験で、いつ行ってもよい。動画は作らずに、床の摩擦 0.5〜1.0、箱と手の摩擦 0.8〜1.2、箱と机の摩擦 0.3〜0.6、机の前で止まる位置（前後 ±2 cm・左右 ±3 cm）を変えて、8 回通す。4 つずつ同時に動かす。結果は `logs/test.jsonl`（1 回 1 行）：

```bash
mkdir -p logs && FR_ARGS="--lateral 1 --no_label" scripts/full_run_batch.sh carry10 8 logs/test.jsonl carry24b_1
```

---

## 6. 区間ごとに動かすもの

| 区間 | 脚・腰（強化学習の方策） | 腕 |
|---|---|---|
| A 立ち上がる | `standup` | お手本の姿勢 |
| A 歩く・ドアを押す | `estimate`（手ぶら） | 腕を振る・両手を前に出して押す |
| B 受け取る・C 運ぶ | `carry10` | 両前腕と指で受け皿を作る持ち方 |
| D 置く | `carry24b_1` | 机に触れた時刻ごとの腕の動きの表（IK で作成） |

---

## 7. 物理の決まり

### 7.1 守っていること

- 箱は、手との摩擦と重力だけで持つ。
- 机・椅子・壁などは固定（ぶつからない前提）。
- 物理の床（灰色のタイル）は計算には使うが、描かない。見えるのはオフィスと廊下の床だけ。

### 7.2 接触を入れる組み合わせ

| 場面 | 接触する | 接触しない |
|---|---|---|
| ドアを開ける | ロボットの手 ↔ ドア | ロボットの足・脚・胴・腕 ↔ ドア |
| 荷物を受け取る | 箱 ↔ ロボットの手の受け皿（指の面・胸）、箱 ↔ 配達員の手のひら・押す指 | 配達員の手・腕 ↔ ロボットの手・腕、配達員の体 ↔ 箱 |
| ドアを閉める | 配達員のドアを閉める手 ↔ ドア | 配達員の体・腕 ↔ ドア |
| 床 | 足の裏・箱・倒れたときの体 ↔ 床 | 足首・すね ↔ 床 |
| 椅子 | 骨盤・もも・すね ↔ 椅子 | 骨盤や腰の見た目の形 ↔ 椅子 |
| 机・壁・棚・モニターなど | 箱、ロボットの体・腕 | — |
| 配達員とロボット | 配達員の体・脚 ↔ ロボットの体・脚 | 手・腕どうし |

ロボットの腕・頭・腰・膝など、元のモデルに当たり判定のなかった所は、見た目の形（メッシュ）をそのまま当たり判定に使う（`scripts/full_run.py` の `add_all_contacts`）。

---

## 8. 構成

```
scripts/
  full_run.py            事務所の場面で A〜D を通す（試験・動画）。接触の組み合わせ（add_all_contacts）もここ
  full_run_batch.sh      条件を変えて N 回通し、成功の数を数える
  deploy.py              ロボット（A の立ち上がる・歩く・ドアを押す）
  office_scene.py        事務所と廊下（壁・ドア・机・椅子・棚など）
  make_audio.py          動画に声（Open JTalk）とノックの音を入れる
  payload_env.py         荷物を持って歩く学習の環境（工夫 A・B）。G1 のモデルを組み立てる
  carry_env.py           箱を持って歩く・受け取る・置く学習の環境（事務所の場面での学習も）
  standup_env.py         椅子から立ち上がる学習の環境
  standup_ref.py         立ち上がるお手本（models/standup_ref.npz）を作る
  place_test.py          方策の読み込み、置く場面の部品
  policy_io.py           方策の読み込み
  train_payload.py       学習（PPO）。--env payload / standup / carry
  run_train.sh           学習を裏で動かす
  train_chunks.sh        学習を区切って続ける（GPU のメモリ不足を避ける）
  eval_carry.py          学習した carry の方策を GPU で試す
  calib_estimator.py     重さの推定の式（models/estimator.json）を作る
  calib_carry_estimator.py  箱の持ち方での重さの推定の式（models/estimator_tray.json）を作る
  crouch_offsets.py      しゃがむ姿勢の表（models/crouch_offsets.npy）を作る
  crouch_place_quick.py  置く腕の動き（models/place_traj_tray_quick_h0.10.npz）を作る
  crouch_place_touch.py  机に触れた時刻ごとの置く腕の動きの表（models/place_touch_h0.10_b2.npz）を作る
  queue_carry24.sh       最後の学習（carry24：事務所の場面で、carry21_3 から）
  queue_carry24b.sh      その続き（carry24b_1 はこの 1 回目）
models/
  standup/  estimate/  carry10/  carry24b_1/   実行で使う方策（params.pkl・config.json）
  carry21_3/                                    最後の学習の出発点
  standup_ref.npz  crouch_offsets.npy  place_traj_tray_tilt_crouch.npz  place_traj_tray_quick_h0.10.npz
  place_touch_h0.10_b2.npz  estimator.json  estimator_tray.json
docs/report_2026-10-18.docx   作業報告書
requirements.txt              Python のライブラリ（版を固定）
```

コードには、最終の形では使わない試しのオプションが残っている。

- 机の前で止まった後の調整（`full_run.py --adjust`・`--yaw_shift`）
- A の続きから B〜D を練習する学習（`train_payload.py --office_flow`）

これらに要るデータ（`shift_offsets.npy`、`flow/flow_states.jsonl`）は、このリポジトリには入れていない。使わなければ、動きには影響しない。
