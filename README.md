# pai_humanoid_delivery

人型ロボット **Unitree G1** が事務所の椅子から立ち上がってドアを開け、廊下の配達員から **MacBook Air の箱（33 × 25 × 5 cm、1.23 kg）** を手で受け取り、机まで運んで置く。物理シミュレータ **MuJoCo** の中だけで行う。

このリポジトリには、最終の形を動かすのと、最後の方策を学習し直すのに要るものだけを置いた。学習方法・結果・物理の決まりなどは、説明資料（`docs/説明資料_PAI最終課題.docx`）にある。

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
- **接触・衝突・摩擦は、その場面で触れるはずの所どうしに入れる**：人がドアを手で開けても足はドアに当たらない、荷物を受け取っても手どうしは触れない、と同じ考え方（組み合わせの表は説明資料の 7.2）。
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

動画は、3.1 の手順で作り、3.2 の方法で再生する。

| | 3.1 A〜D を通して動画を作る手順 | 3.2 3.1 で作成した動画を再生する方法 |
|---|---|---|
| すること | シミュレーションを動かし、その様子を動画のファイルに書き出す。声とノックの音を入れ、93 秒に切る | 3.1 でできた動画のファイルを、ブラウザやプレーヤーで再生する |
| ロボットを動かすか | 動かす（学習済みの方策 `models/` を使う。CPU で 1 回およそ 5〜10 分） | 動かさない（できた動画を見るだけ） |
| 使うもの | Python の環境（`.venv_mjx`）、ffmpeg、Open JTalk | 動画を再生できるソフト（Chrome・Firefox・ffplay） |
| できるもの | `logs/full_run.mp4` などの動画のファイル | なし |

### 3.1 A〜D を通して動画を作る手順

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

### 3.2 3.1 で作成した動画を再生する方法

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
