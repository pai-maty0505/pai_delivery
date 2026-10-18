#!/usr/bin/env python3
"""
動画に声と音を入れる。

  ノックの音（木の戸を 3 回たたく音、計算で作る）→ ロボット「はい」→ 配達員「荷物が届いています」→ ロボット「ありがとう。」
  声は Open JTalk（男性の声 1 種類）。ロボットは少し高く速め、配達員は普通の高さ。
  入れる時刻は scripts/full_run.py --events が書き出す JSON（シミュレーションの時刻 ＝ 動画の時刻）。

    python3 scripts/make_audio.py --events logs/full_run_events.json --video logs/full_run.mp4 --out logs/full_run_voice.mp4
"""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 48000
DIC = "/var/lib/mecab/dic/open-jtalk/naist-jdic"
VOICE = "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"


def tts(text, fm=0.0, r=1.0):
    """Open JTalk で声を作り、48 kHz の配列で返す。fm：声の高さ（半音）、r：速さ。"""
    with tempfile.TemporaryDirectory() as td:
        txt, wav = Path(td) / "t.txt", Path(td) / "t.wav"
        txt.write_text(text, encoding="utf-8")
        subprocess.run(["open_jtalk", "-x", DIC, "-m", VOICE, "-ow", str(wav), "-fm", str(fm), "-r", str(r), str(txt)], check=True)
        with wave.open(str(wav)) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    t_new = np.arange(int(len(x) * SR / sr)) / SR
    return np.interp(t_new, np.arange(len(x)) / sr, x)


def knock(n=3, gap=0.28):
    """木の戸をたたく音：低い減衰振動と短い雑音を n 回。"""
    rng = np.random.default_rng(0)
    one_t = np.arange(int(0.18 * SR)) / SR
    env = np.exp(-one_t * 38)
    one = (0.55 * np.sin(2 * np.pi * 180 * one_t) + 0.3 * np.sin(2 * np.pi * 410 * one_t) + 0.15 * np.sin(2 * np.pi * 920 * one_t)) * env
    one += 0.25 * rng.standard_normal(len(one_t)) * np.exp(-one_t * 120)
    out = np.zeros(int((gap * (n - 1) + 0.2) * SR))
    for k in range(n):
        i = int(k * gap * SR)
        out[i:i + len(one)] += one
    return 0.8 * out / np.abs(out).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ev = json.loads(Path(args.events).read_text())
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.video],
                               capture_output=True, text=True).stdout)
    track = np.zeros(int((dur + 1) * SR))

    def put(x, t, gain=1.0):
        i = int(t * SR)
        j = min(len(track), i + len(x))
        track[i:j] += gain * x[:j - i]

    put(knock(), ev["knock"])
    put(tts("はい", fm=2.0, r=1.05), ev["hai"], 0.9)
    put(tts("荷物が届いています", fm=-1.0, r=1.0), ev["courier"], 0.9)
    put(tts("ありがとう。", fm=2.0, r=1.05), ev["thanks"], 0.9)
    track = np.clip(track, -1, 1)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((track * 32767).astype(np.int16).tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    print(args.out, {k: round(v, 1) for k, v in ev.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
