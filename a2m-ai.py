#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a2m-ai.py — Audio → MIDI mit KI (Viertelnoten, 1 Oktave)
========================================================

Kurzidee
--------
Dieses Skript erkennt aus monophonem WAV-Audiomaterial (z. B. Flöte) pro
Viertel genau EINE Klasse: 12 Tonhöhen einer Oktave (base_midi .. base_midi+11)
oder REST (Stille). Aus den Vorhersagen wird ein MIDI-File erzeugt.

Pipeline (Subcommands)
----------------------
1) mkdata  : synthetische Trainingsdaten erzeugen (Log-Mel-Patches + Labels)
2) train   : kleines CNN auf diesen Patches trainieren
3) infer   : WAV + BPM → MIDI (eine Note pro Viertel; REST wird ausgelassen)
4) inspect : Datensatzgröße & Batches/Epoche anzeigen

Wichtige Eigenschaften / Annahmen
---------------------------------
• Monophon, eine Oktave + REST → 13 Klassen insgesamt.
• Viertelraster: Dauer eines Viertels = 60/BPM Sekunden.
• Exaktes Grid bei der Inferenz:
  - Start immer bei t=0.0 s (kein Auto-Offset).
  - Es werden genau bars*4 Viertel erzeugt (wenn `--bars` gesetzt).
    Ohne `--bars` wird die Audiolänge auf die nächste volle Taktzahl
    gerundet. Fehlende Samples am Ende werden mit Stille gepolstert.
• (Neu) Barweises Glätten (Default an): pro Takt werden inkonsistente
  Einzelviertel begradigt (≥3/4 gleiche Tonklasse → ganzer Takt ein Ton).
• Keine Merge-Logik über Taktgrenzen hinweg: Das Modell gibt pro Viertel
  eine Klasse. Lange Noten erscheinen als mehrere Viertel derselben Tonhöhe.
• Robustheit:
  - Log-Mel-Berechnung polstert zu kurze Segmente → kein "n_fft too large".
  - Time-Stretch ist versionssicher (Fallback via Phase-Vocoder).
• Training:
  - Regelmäßiger Status-Output (Batch-Heartbeats, Epoche-Zeit).
  - Checkpoints nach jeder Epoche (`…epXX.pt`) + bestes Modell (`model.pt`).
  - Kleinere Defaults für `mkdata` (schnelleres Setup).

---------------------------
WICHTIGE BEGRIFFE (einfach):
---------------------------

• CNN (Convolutional Neural Network)
  = Ein neuronales Netz, das mit kleinen Filtern über ein „Bild“ fährt
    und Muster erkennt. Unser „Bild“ ist ein Zeit-Frequenz-Bild des Tons.

• Log-Mel-Patch
  = Ein kleiner Ausschnitt eines Spektrogramms pro Viertel (z. B. 64
    Frequenzbänder × 96 Zeit-"Pixel"). „Mel“ ist eine Skala ähnlich
    unserem Gehör, „Log“ = Lautstärke als dB.

• Augmentations
  = Zufällige Mini-Veränderungen beim Training (z. B. Lautstärke,
    leichtes Rauschen, minimaler Zeit-Stretch), damit das Modell
    robust wird und nicht nur einen einzigen Synth-Sound „kennt“.

• Blip
  = Ein sehr kurzer, ungewollter Ton am Viertelanfang/-ende (z. B.
    durch Legato/Overlap). Unsere Labels kommen aus der Mitte des
    Viertels – so ignorieren wir diese Blips im Ziel und vermeiden
    doppelte Noten.

---------------------------
AUFGABE / LABELS:
---------------------------

• Wir beschränken uns auf eine Oktave: 12 Töne + REST (=Stille)
  → zusammen 13 Klassen.

• Jede Audio-Scheibe ist genau ein Viertel lang (60/BPM Sekunden).
  Für jede dieser Scheiben berechnen wir ein Log-Mel-Patch und lassen
  das CNN eine Klasse vorhersagen (welcher Ton oder REST).

• Label-Regel: Nimm die Note, die die Mitte des Viertels überdeckt.
  Das verhindert doppelte Labels, selbst wenn am Rand Blips/Legato sind.

Beispiel-Aufrufe
----------------
# 1) Kleines Set erzeugen
python a2m-ai.py mkdata --out data_quarters.npz --bpm 130 --base_midi 60

# 2) Größe checken
python a2m-ai.py inspect --data data_quarters.npz --batch 128

# 3) Training (unbuffered für Live-Logs)
set PYTHONUNBUFFERED=1 && python -u a2m-ai.py train \
  --data data_quarters.npz --out model.pt --epochs 20

# 4) Inferenz (Start=0.0, volle Takte)
python a2m-ai.py infer \
  --wav floete.wav --out floete.mid --model model.pt \
  --bpm 130 --base_midi 60           # Takte automatisch gerundet

# 4b) Exakt N Takte (Padding/Truncate aufs Grid)
python a2m-ai.py infer \
  --wav floete.wav --out floete.mid --model model.pt \
  --bpm 130 --base_midi 60 --bars 4
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import soundfile as sf
import librosa
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, random_split

# ------------------------- Print/Timer-Helpers -------------------------

def p(*args, **kw):
    kw.setdefault("flush", True)
    print(*args, **kw)

class Timer:
    def __init__(self, label): self.label=label; self.t0=time.perf_counter()
    def done(self): p(f"[TIMER] {self.label}: {time.perf_counter()-self.t0:.2f}s")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ------------------------- Utils & Audio-Features -------------------------

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def logmel(y, sr, n_mels=64, n_fft=1024, hop=256, fmin=50, fmax=8000):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="constant")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    return librosa.power_to_db(S, ref=np.max)

def pad_or_crop_time(M, T_target):
    n_mels, T = M.shape
    if T == T_target: return M
    if T > T_target:
        start = (T - T_target)//2
        return M[:, start:start+T_target]
    pad_total = T_target - T
    left = pad_total//2
    right = pad_total - left
    return np.pad(M, ((0,0),(left,right)), mode="edge")

def time_stretch_safe(y, rate, sr=22050, hop=256):
    if abs(rate - 1.0) < 1e-3:
        return y
    try:
        return librosa.effects.time_stretch(y, rate=rate)
    except TypeError:
        D = librosa.stft(y, n_fft=1024, hop_length=hop, window="hann")
        D_st = librosa.phase_vocoder(D, rate=rate, hop_length=hop)
        y_out = librosa.istft(D_st, hop_length=hop, length=int(round(len(y)/rate)))
        return y_out.astype(np.float32)

def augment_wave(y, sr):
    y = y * (10**(np.random.uniform(-6,6)/20))
    if np.random.rand() < 0.5 and len(y) > 2048:
        S = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        tilt = 1.0 + 0.00003*np.random.uniform(-1,1)*freqs
        y = np.fft.irfft(S*tilt, n=len(y)).astype(np.float32)
    noise = np.random.normal(0, 1e-4*np.maximum(1.0, np.std(y)), size=len(y)).astype(np.float32)
    y = (y + noise).astype(np.float32)
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.97, 1.03)
        y = time_stretch_safe(y, rate, sr=sr, hop=256)
    return y

def simple_flute_synth(note_hz, dur_sec, sr=22050, amp=0.2):
    t = np.linspace(0, dur_sec, int(sr*dur_sec), endpoint=False)
    env = np.minimum(1.0, t/(0.02+1e-6)) * np.exp(-t/(dur_sec*2.5))
    y = (1.0*np.sin(2*np.pi*note_hz*t)
         + 0.3*np.sin(2*np.pi*2*note_hz*t)
         + 0.15*np.sin(2*np.pi*3*note_hz*t))
    return (amp * y * env).astype(np.float32)

# ------------------------- Exaktes Grid-Slicing (Start=0, volle Takte) -------------------------

def grid_slices(y, sr, bpm, bars=None):
    step = 60.0 / bpm
    target_len = int(round(step * sr))

    if bars is None:
        n_quarters = int(round((len(y) / sr) / step))
        n_bars = max(1, int(round(n_quarters / 4.0)))
    else:
        n_bars = max(1, int(bars))

    n_quarters = n_bars * 4

    slices = []
    for k in range(n_quarters):
        ts = k * step
        s = int(round(ts * sr))
        e = s + target_len
        if s >= len(y):
            seg = np.zeros(target_len, dtype=np.float32)
        else:
            raw = y[s:min(e, len(y))]
            seg = raw if len(raw) == target_len else np.pad(raw, (0, target_len - len(raw)), mode="constant")
        slices.append((ts, ts + step, seg))
    return slices, step, 0.0  # offset immer 0

# ------------------------- Barweises Glätten (einfach & robust) -------------------------

def smooth_bar_quarters(preds, enable=True):
    """
    Wenn in einem 4er-Block (Takt) ≥3/4 Viertel dieselbe Tonklasse (≠ REST=12) haben,
    setze den ganzen Block auf diese Tonklasse. Fix für "Note startet erst auf 2".
    """
    if not enable: return preds
    out = preds[:]
    for i in range(0, len(preds), 4):
        block = preds[i:i+4]
        if len(block) < 4: break
        labs = [l for _,_,l in block]
        non = [l for l in labs if l != 12]
        if not non:  # kompletter REST-Takt
            continue
        mode, cnt = Counter(non).most_common(1)[0]
        if cnt >= 3:
            for j in range(4):
                if labs[j] != mode:
                    ts, te, _ = out[i+j]
                    out[i+j] = (ts, te, mode)
    return out

# ------------------------- Datensatz-Erzeugung (mkdata) -------------------------

def mkdata(out_npz, n_sequences=500, bpm=130, base_midi=60, sr=22050,
           bars=6, rest_prob=0.1, seed=42):
    set_seed(seed)
    t_total = Timer("mkdata total")
    step = 60.0/bpm
    X, y = [], []
    for idx in range(n_sequences):
        seq_len = bars*4
        labels = [12 if np.random.rand() < rest_prob else np.random.randint(0,12)
                  for _ in range(seq_len)]
        audio = []
        for lab in labels:
            if lab == 12:
                audio.append(np.zeros(int(sr*step), dtype=np.float32))
            else:
                hz = librosa.midi_to_hz(base_midi + lab)
                audio.append(simple_flute_synth(hz, step, sr=sr))
        y_seq = np.concatenate(audio)
        y_seq = augment_wave(y_seq, sr)
        target_len = int(round(seq_len * step * sr))
        if len(y_seq) < target_len:
            y_seq = np.pad(y_seq, (0, target_len - len(y_seq)), mode="constant")
        elif len(y_seq) > target_len:
            y_seq = y_seq[:target_len]
        for i, lab in enumerate(labels):
            s = int(i*step*sr); e = s + int(round(step*sr))
            M = logmel(y_seq[s:e], sr)
            M = pad_or_crop_time(M, 96)
            X.append(M[None, ...]); y.append(lab)
        if (idx+1) % 100 == 0 or (idx+1) == n_sequences:
            p(f"[mkdata] {idx+1}/{n_sequences} sequences")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(out_npz, X=X, y=y, meta=dict(bpm=bpm, base_midi=base_midi, sr=sr))
    p(f"[mkdata] wrote {out_npz}: X={X.shape}, y={y.shape}, bpm={bpm}, base_midi={base_midi}")
    t_total.done()

# ------------------------- Modell -------------------------

class SmallCNN(nn.Module):
    def __init__(self, n_classes=13):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,32,32,48)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,64,16,24)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B,128,8,12)
        x = x.mean(dim=[2,3])                           # (B,128)
        return self.head(x)                             # (B,13)

# ------------------------- Training (Logs & Checkpoints) -------------------------

def train_model(data_npz, out_pt, epochs=15, batch=128, lr=1e-3,
                val_split=0.1, seed=42, patience=0, save_every=1,
                max_samples=0, resume=None):
    set_seed(seed)
    d = np.load(data_npz, allow_pickle=True)
    X, y = d["X"], d["y"]
    if max_samples and max_samples < len(X):
        idx = np.random.permutation(len(X))[:max_samples]
        X, y = X[idx], y[idx]
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    n_val = int(len(ds)*val_split)
    n_tr  = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_tr, n_val])
    dl_tr  = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    if resume and os.path.exists(resume):
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state)
        p(f"[train] resumed weights from {resume}")

    p(f"[train] device={device} | samples: train={len(ds_tr)}, val={len(ds_val)} | "
      f"batch={batch}, epochs={epochs}")

    best_acc = -1.0
    no_improve = 0

    try:
        for ep in range(1, epochs+1):
            t_ep = Timer(f"epoch {ep}")
            # TRAIN
            model.train()
            tr_loss=tr_acc=0.0; n_tr_samp=0
            it = tqdm(dl_tr, total=len(dl_tr), ncols=80, desc=f"ep{ep:02d} train", leave=False) if _HAS_TQDM else dl_tr
            for i,(Xb,yb) in enumerate(it, 1):
                Xb=Xb.to(device); yb=yb.to(device)
                opt.zero_grad()
                logits = model(Xb)
                loss = crit(logits, yb)
                loss.backward(); opt.step()
                tr_loss += loss.item()*Xb.size(0)
                tr_acc  += (logits.argmax(1)==yb).float().sum().item()
                n_tr_samp += Xb.size(0)
                if not _HAS_TQDM and (i % 50 == 0 or i == len(dl_tr)):
                    p(f"  [train] ep{ep:02d} batch {i}/{len(dl_tr)}")
            tr_loss/=max(1,n_tr_samp); tr_acc/=max(1,n_tr_samp)

            # VAL
            model.eval(); va_loss=va_acc=0.0; n_va_samp=0
            with torch.no_grad():
                for Xb,yb in dl_val:
                    Xb=Xb.to(device); yb=yb.to(device)
                    logits = model(Xb)
                    loss = crit(logits, yb)
                    va_loss += loss.item()*Xb.size(0)
                    va_acc  += (logits.argmax(1)==yb).float().sum().item()
                    n_va_samp += Xb.size(0)
            va_loss/=max(1,n_va_samp); va_acc/=max(1,n_va_samp)

            t_ep.done()
            p(f"[train] ep{ep:02d}  tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.3f} val_acc={va_acc:.3f}")

            # Epochen-Checkpoint
            ck = out_pt.replace(".pt", f".ep{ep:02d}.pt")
            #torch.save(model.state_dict(), ck)
            #p(f"[train] checkpoint saved: {ck}")

            # Bestes Modell
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), out_pt)
                p(f"[train] BEST updated -> {out_pt} (val_acc={best_acc:.3f})")
                no_improve = 0
            else:
                no_improve += 1
            if patience and no_improve >= patience:
                p(f"[train] early stop (no improve {no_improve}/{patience})")
                break

    except KeyboardInterrupt:
        ck = out_pt.replace(".pt", ".interrupt.pt")
        torch.save(model.state_dict(), ck)
        p(f"[train] interrupted. Saved current weights -> {ck}")

# ------------------------- Inference: WAV -> MIDI (Start=0, volle Takte) -------------------------

def infer_to_midi(wav, out_mid, model_pt, bpm, base_midi=60, sr=22050, bars=None, bar_smooth=True):
    """
    WAV -> MIDI mit exaktem Viertel-Grid:
      - Start immer bei 0.0 s
      - Genau bars*4 Viertel (wenn bars angegeben), sonst Auto-Rundung auf volle Takte
      - fehlendes Audio am Ende wird mit Stille gepolstert
      - optional: barweises Glätten (>=3/4 gleiche Tonklasse => ganzer Takt)
    """
    t_all = Timer("inference total")

    # 1) Audio + Slices
    t0=Timer("load+grid-slice")
    y, sr_ = librosa.load(wav, sr=sr, mono=True)
    slices, step, off = grid_slices(y, sr, bpm, bars=bars)
    t0.done()

    # 2) Modell
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state); model.eval()

    # 3) Vorhersagen
    t1=Timer("features+forward")
    preds=[]
    for (ts,te,seg) in slices:
        M = logmel(seg, sr)
        M = pad_or_crop_time(M,96)
        X = torch.from_numpy(M[None,None,...].astype(np.float32)).to(device)
        with torch.no_grad():
            lab = int(model(X).argmax(1).cpu().item())
        preds.append((ts,te,lab))
    t1.done()

    # 4) Bar-smoothing (fix für „Note startet erst auf 2“)
    preds = smooth_bar_quarters(preds, enable=bar_smooth)

    # 5) MIDI
    t2=Timer("write MIDI")
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=73)
    for (ts,te,lab) in preds:
        if lab==12: continue
        pitch = base_midi + lab
        inst.notes.append(pretty_midi.Note(velocity=96, pitch=int(pitch),
                                           start=float(ts), end=float(te)))
    pm.instruments.append(inst)
    pm.write(out_mid)
    t2.done()
    p(f"[infer] wrote {out_mid} | quarters={len(preds)} | bars={len(preds)//4} | step={step:.3f}s")
    t_all.done()

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Beat-synchroner Viertelnoten-Klassifikator (1 Oktave + REST).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("mkdata", help="synthetische Viertel-Daten erzeugen (kleineres Default-Set)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--n_sequences", type=int, default=500)
    sp.add_argument("--bpm", type=float, default=130.0)
    sp.add_argument("--base_midi", type=int, default=60)
    sp.add_argument("--sr", type=int, default=22050)
    sp.add_argument("--bars", type=int, default=6)
    sp.add_argument("--rest_prob", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("inspect", help="Datensatz-Info anzeigen")
    sp.add_argument("--data", required=True)
    sp.add_argument("--batch", type=int, default=128)

    sp = sub.add_parser("train", help="CNN trainieren (mit Status & Epoche-Checkpoints)")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--batch", type=int, default=128)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--val_split", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--patience", type=int, default=0)
    sp.add_argument("--save_every", type=int, default=1)
    sp.add_argument("--max_samples", type=int, default=0)
    sp.add_argument("--resume", default=None)

    sp = sub.add_parser("infer", help="WAV -> MIDI (exaktes Grid, optional Bar-Smoothing)")
    sp.add_argument("--wav", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--bpm", type=float, required=True)
    sp.add_argument("--base_midi", type=int, default=60)
    sp.add_argument("--sr", type=int, default=22050)
    sp.add_argument("--bars", type=int, default=None)
    sp.add_argument("--no_bar_smooth", action="store_true", help="barweises Glätten deaktivieren")

    args = ap.parse_args()

    if args.cmd == "mkdata":
        mkdata(args.out, args.n_sequences, args.bpm, args.base_midi, args.sr,
               args.bars, args.rest_prob, args.seed)

    elif args.cmd == "inspect":
        d = np.load(args.data, allow_pickle=True)
        X, y = d["X"], d["y"]
        n = X.shape[0]; b = args.batch
        p(f"[inspect] file={args.data}")
        p(f"[inspect] samples={n}, patch={tuple(X.shape[1:])}, dtype={X.dtype}")
        p(f"[inspect] batches/epoch @ batch={b}: ~{(n + b - 1)//b}")

    elif args.cmd == "train":
        train_model(args.data, args.out, args.epochs, args.batch, args.lr,
                    args.val_split, args.seed, args.patience, args.save_every,
                    args.max_samples, args.resume)

    elif args.cmd == "infer":
        infer_to_midi(args.wav, args.out, args.model, args.bpm,
                      args.base_midi, args.sr, args.bars,
                      bar_smooth=(not args.no_bar_smooth))

if __name__=="__main__":
    main()
