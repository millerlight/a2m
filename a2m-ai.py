#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a2m-ai.py
================

Dieses Skript trainiert ein kleines KI-Modell, das aus WAV-Audio
(Flöte oder ähnliches, monophon) **Viertelnoten einer Oktave** erkennt
und als MIDI ausgibt. Es hat drei Unterbefehle:

  - mkdata  : synthetische Trainingsdaten erzeugen (Features + Labels)
  - train   : ein kleines CNN trainieren
  - infer   : WAV + BPM -> MIDI (eine Note pro Viertel)

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
    **robust** wird und nicht nur einen einzigen Synth-Sound „kennt“.

• Blip
  = Ein sehr kurzer, ungewollter Ton am Viertelanfang/-ende (z. B.
    durch Legato/Overlap). Unsere Labels kommen aus der **Mitte des
    Viertels** – so ignorieren wir diese Blips im Ziel und vermeiden
    doppelte Noten.

---------------------------
AUFGABE / LABELS:
---------------------------

• Wir beschränken uns auf **eine Oktave**: 12 Töne + REST (=Stille)
  → zusammen **13 Klassen**.

• Jede Audio-Scheibe ist **genau ein Viertel** lang (60/BPM Sekunden).
  Für jede dieser Scheiben berechnen wir ein Log-Mel-Patch und lassen
  das CNN **eine Klasse** vorhersagen (welcher Ton oder REST).

• Label-Regel: Nimm die **Note, die die Mitte** des Viertels überdeckt.
  Das verhindert doppelte Labels, selbst wenn am Rand Blips/Legato sind.

Aufruf:
A. Daten erzeugen:
python a2m-ai.py mkdata --out data_quarters.npz --bpm 130 --base_midi 60 --n_sequences 3000

B. Training:
python a2m-ai.py train --data data_quarters.npz --out model.pt --epochs 20

Training unbuffered, damit schneller eine Meldung kommt:
$env:PYTHONUNBUFFERED=1; python -u a2m-ai.py train --data data_quarters.npz --out model.pt --epochs 20


C. Inference (WAV->Midi)
python a2m-ai.py infer --wav floete.wav --out floete.mid --model model.pt --bpm 130 --base_midi 60




Abhängigkeiten:
  pip install torch librosa numpy soundfile pretty_midi
"""

import argparse, json, math, os, random, sys
import numpy as np
import soundfile as sf
import librosa
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import time
from contextlib import contextmanager

@contextmanager
def timed(msg: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {msg}: {dt:.2f}s")



# ---------------------------- Grund-Utils ----------------------------

def set_seed(s=42):
    """Für reproduzierbare Ergebnisse (Zufall festnageln)."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def logmel(y, sr, n_mels=64, n_fft=1024, hop=256, fmin=50, fmax=8000):
    """
    Erzeuge ein Log-Mel-Spektrogramm.
    Das ist unser „Bild“, das ins CNN geht.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)
    return S  # (n_mels, T)

def pad_or_crop_time(M, T_target):
    """
    Bring die Zeitbreite auf eine feste Länge (z. B. 96 Frames),
    damit alle Patches gleich groß sind.
    """
    n_mels, T = M.shape
    if T == T_target:
        return M
    if T > T_target:
        start = (T - T_target)//2
        return M[:, start:start+T_target]
    pad_total = T_target - T
    left = pad_total//2
    right = pad_total - left
    return np.pad(M, ((0,0),(left,right)), mode='edge')

def simple_flute_synth(note_hz, dur_sec, sr=22050, amp=0.2):
    """
    Kleiner „Haus-Synth“ (nur für mkdata):
    Ein paar Sinus-Obertöne + Hüllkurve -> flötenähnlich.
    Reicht aus, um Trainingsbeispiele zu generieren.
    """
    t = np.linspace(0, dur_sec, int(sr*dur_sec), endpoint=False)
    env = np.minimum(1.0, t/(0.02+1e-6)) * np.exp(-t/(dur_sec*2.5))
    y = (1.0*np.sin(2*np.pi*note_hz*t)
         + 0.3*np.sin(2*np.pi*2*note_hz*t)
         + 0.15*np.sin(2*np.pi*3*note_hz*t))
    y = y * env
    return (amp * y).astype(np.float32)

def time_stretch_safe(y, rate, sr=22050, hop=256):
    """Versionstolerantes Time-Stretch ohne Pitch-Shift."""
    if abs(rate - 1.0) < 1e-3:
        return y
    try:
        # neuere librosa-Versionen
        return librosa.effects.time_stretch(y, rate=rate)
    except TypeError:
        # Fallback: Phase-Vocoder
        D = librosa.stft(y, n_fft=1024, hop_length=hop, window="hann")
        D_st = librosa.phase_vocoder(D, rate=rate, hop_length=hop)
        y_out = librosa.istft(D_st, hop_length=hop, length=int(round(len(y)/rate)))
        return y_out.astype(np.float32)


def augment_wave(y, sr):
    """
    Einfache Augmentations (robuster machen):
    - Lautstärke
    - ganz leichtes EQ-Gefälle
    - minimales Rauschen
    - kleiner Zeit-Stretch
    """
    # Lautstärke ±6 dB
    g_db = np.random.uniform(-6, 6); y = y * (10**(g_db/20))
    # leichter Tilt-EQ
    if np.random.rand() < 0.5:
        S = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        tilt = 1.0 + 0.00003*np.random.uniform(-1,1)*freqs
        y = np.fft.irfft(S*tilt, n=len(y)).astype(np.float32)
    # minimales Rauschen
    noise = np.random.normal(0, 1e-4*np.maximum(1.0, np.std(y)), size=len(y)).astype(np.float32)
    y = (y + noise).astype(np.float32)
    # kleiner Zeit-Stretch
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.97, 1.03)
        # y = librosa.effects.time_stretch(y, rate)
        y = time_stretch_safe(y, rate, sr=sr, hop=256)

    return y

def spectral_flux(y, sr, hop=256):
    """
    Spektraler Flux (grob: „wie stark ändert sich das Spektrum?“).
    Dient dazu, einen globalen Start-Offset zum BPM-Raster zu schätzen.
    """
    S = librosa.stft(y, n_fft=1024, hop_length=hop, window="hann")
    mag = np.abs(S)
    d = np.diff(mag, axis=1)
    flux = np.clip(d, 0, None).sum(axis=0)
    flux = np.concatenate([[0], flux])
    times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop)
    return times, flux

def estimate_global_offset(y, sr, bpm, search_frac=0.5, hop=256):
    """
    Schätzt, wie weit das Audio zeitlich vom Viertel-Raster versetzt ist.
    Wir testen kleine Offsets und nehmen den mit maximalem „Flux auf Grid“.
    """
    if bpm <= 0: return 0.0
    step = 60.0/bpm
    t, flux = spectral_flux(y, sr, hop)
    offs = np.linspace(-step*search_frac, +step*search_frac, 41)
    scores = []
    for o in offs:
        grid = np.arange(max(0, -o), t[-1], step) + o
        idx = np.clip(np.round(grid / (hop/sr)).astype(int), 0, len(flux)-1)
        scores.append(np.mean(flux[idx]) if len(idx)>0 else 0.0)
    best = offs[int(np.argmax(scores))]
    return float(best)


# ----------------------- Datensatz (mkdata & Laden) -----------------------

def mkdata(out_npz, n_sequences=2000, bpm=130, base_midi=60, sr=22050,
           bars=8, rest_prob=0.1, seed=42):
    """
    Erzeuge synthetische Trainingsdaten:
    - Wir bauen zufällige Folgen aus 12 Tönen + REST (pro Viertel eine Klasse)
    - Synthetisieren Audio (einfacher Flöten-Synth)
    - Schneiden in Viertel und berechnen Log-Mel-Patches
    Ergebnis: .npz mit X (N,1,64,96) und y (N,) Label 0..12
    """
    set_seed(seed)
    step = 60.0/bpm
    X = []
    y = []
    for _ in range(n_sequences):
        seq_len = bars*4  # 4 Viertel pro Takt
        labels = []
        for i in range(seq_len):
            if np.random.rand() < rest_prob:
                labels.append(12)  # REST
            else:
                labels.append(np.random.randint(0,12))  # 12 Töne der Oktave
        # gesamte Sequenz synthetisieren
        audio = []
        for lab in labels:
            if lab == 12:
                audio.append(np.zeros(int(sr*step), dtype=np.float32))
            else:
                hz = librosa.midi_to_hz(base_midi + lab)
                audio.append(simple_flute_synth(hz, step, sr=sr))
        y_seq = np.concatenate(audio)
        y_seq = augment_wave(y_seq, sr)

        # Ziel-Länge (genau N_Viertel * step * sr)
        target_len = int(round(seq_len * step * sr))
        cur_len = len(y_seq)
        if cur_len < target_len:
            y_seq = np.pad(y_seq, (0, target_len - cur_len), mode="constant")
        elif cur_len > target_len:
            y_seq = y_seq[:target_len]

        # in Viertel schneiden -> Feature/Label pro Viertel
        for i, lab in enumerate(labels):
            start = int(i*step*sr)
            end   = int((i+1)*step*sr)
            seg = y_seq[start:end]
            M = logmel(seg, sr)          # (64, T)
            M = pad_or_crop_time(M, 96)  # (64, 96)
            X.append(M[None, ...])       # (1, 64, 96)
            y.append(lab)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(out_npz, X=X, y=y, meta=dict(bpm=bpm, base_midi=base_midi, sr=sr))
    print(f"Wrote {out_npz}: X={X.shape}, y={y.shape}, bpm={bpm}, base_midi={base_midi}")

class QuarterNPZ(Dataset):
    """Einfacher Dataset-Wrapper für die erzeugte NPZ-Datei."""
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"]; self.y = d["y"]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


# ---------------------------- Model (CNN) ----------------------------

class SmallCNN(nn.Module):
    """
    Kleines CNN:
      Eingabe:  (B, 1, 64, 96)  = Batch, 1 Kanal, 64 Mel-Bänder, 96 Zeit-Frames
      Ausgabe:  (B, 13)         = 12 Töne + REST
    """
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

    def forward(self, x):  # (B,1,64,96)
        # Faltungen + Pooling verdichten das „Bild“ und extrahieren Muster/Obertöne.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (B,32,32,48)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (B,64,16,24)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (B,128,8,12)
        x = x.mean(dim=[2,3])                           # Global Average Pooling -> (B,128)
        return self.head(x)                             # Klassen-Scores (Logits)


# ---------------------------- Training ----------------------------

def train_model(data_npz, out_pt, epochs=15, batch=128, lr=1e-3, val_split=0.1, seed=42):
    """
    Trainiert das CNN mit Cross-Entropy (Standard für Klassifikation).
    Speichert die besten Gewichte als model.pt.
    """
    set_seed(seed)
    ds = QuarterNPZ(data_npz)

    # Train/Val-Split (einfach zufällig)
    n = len(ds)
    n_val = int(n*val_split)
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    ds_val = torch.utils.data.Subset(ds, val_idx)
    ds_tr  = torch.utils.data.Subset(ds, tr_idx)
    dl_tr  = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best = (0.0, None)
    for ep in range(1, epochs+1):
        # ---- Training ----
        model.train()
        tr_loss=tr_acc=0.0; n_tr=0
        for X,y in dl_tr:
            X=X.to(device); y=y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()*X.size(0)
            tr_acc  += (logits.argmax(1)==y).float().sum().item()
            n_tr    += X.size(0)
        tr_loss/=n_tr; tr_acc/=n_tr

        # ---- Validation (keine Gradienten) ----
        model.eval()
        va_loss=va_acc=0.0; n_va=0
        with torch.no_grad():
            for X,y in dl_val:
                X=X.to(device); y=y.to(device)
                logits = model(X)
                loss = crit(logits, y)
                va_loss += loss.item()*X.size(0)
                va_acc  += (logits.argmax(1)==y).float().sum().item()
                n_va    += X.size(0)
        va_loss/=n_va; va_acc/=n_va

        print(f"ep{ep:02d}  tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} | val_loss={va_loss:.3f} val_acc={va_acc:.3f}")
        if va_acc>best[0]:
            best=(va_acc, model.state_dict())

    # Beste Gewichte speichern
    if best[1] is not None:
        torch.save(best[1], out_pt)
        print(f"Saved best model to {out_pt} (val_acc={best[0]:.3f})")
    else:
        torch.save(model.state_dict(), out_pt)
        print(f"Saved last model to {out_pt}")


# ---------------------------- Inference -> MIDI ----------------------------

def slice_quarters(y, sr, bpm, offset_sec=0.0):
    """
    Schneidet das Audio in Viertel-Scheiben (Länge = 60/BPM).
    offset_sec: globaler Zeitversatz; 'auto' -> automatisch schätzen.
    """
    step = 60.0/bpm
    if offset_sec == "auto":
        off = estimate_global_offset(y, sr, bpm)
    else:
        off = float(offset_sec)

    # beginne am ersten Rasterpunkt >= 0
    t0 = off
    while t0 < 0:
        t0 += step

    slices=[]
    t = t0
    while int((t+step)*sr) <= len(y):
        s = int(t*sr); e = int((t+step)*sr)
        slices.append((t, t+step, y[s:e]))
        t += step
    return slices, step, off

def infer_to_midi(wav, out_mid, model_pt, bpm, base_midi=60, sr=22050, offset="auto"):
    """
    Lädt Audio und Modell, klassifiziert pro Viertel eine Klasse
    (12 Töne + REST) und schreibt eine MIDI-Spur (eine Note pro Viertel).
    """
    # 1) Audio laden & in Viertel schneiden
    y, sr_ = librosa.load(wav, sr=sr, mono=True)
    slices, step, off = slice_quarters(y, sr, bpm, offset)

    # 2) Modell laden
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) Pro Viertel Feature berechnen und Klasse vorhersagen
    preds=[]
    for (ts,te,seg) in slices:
        M = logmel(seg, sr)        # (64, T)
        M = pad_or_crop_time(M,96) # (64, 96)
        X = torch.from_numpy(M[None,None,...].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(X)
            lab = int(logits.argmax(1).cpu().item())
        preds.append((ts,te,lab))

    # 4) MIDI erzeugen (REST=12 -> keine Note)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=73)  # Standard: Flöte
    for (ts,te,lab) in preds:
        if lab==12:
            continue
        pitch = base_midi + lab
        inst.notes.append(pretty_midi.Note(velocity=96, pitch=int(pitch),
                                           start=float(ts), end=float(te)))
    pm.instruments.append(inst)
    pm.write(out_mid)
    print(f"Wrote {out_mid} with {sum(1 for _,_,l in preds if l!=12)} notes | offset={off:.3f}s, step={step:.3f}s")


# ---------------------------- CLI (mkdata/train/infer) ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Beat-synchroner Viertelnoten-Klassifikator (1 Oktave + REST).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Daten erzeugen
    sp = sub.add_parser("mkdata", help="synthetische Viertel-Daten erzeugen")
    sp.add_argument("--out", required=True)
    sp.add_argument("--n_sequences", type=int, default=2000, help="Wie viele zufällige Sequenzen (8 Takte) erzeugen")
    sp.add_argument("--bpm", type=float, default=130.0)
    sp.add_argument("--base_midi", type=int, default=60, help="Grundton der Oktave (z. B. 60=C4)")
    sp.add_argument("--sr", type=int, default=22050)
    sp.add_argument("--bars", type=int, default=8, help="Takte pro Sequenz")
    sp.add_argument("--rest_prob", type=float, default=0.1, help="Anteil REST-Viertel")
    sp.add_argument("--seed", type=int, default=42)

    # Trainieren
    sp = sub.add_parser("train", help="CNN trainieren")
    sp.add_argument("--data", required=True, help="NPZ von mkdata oder eigenem Builder")
    sp.add_argument("--out", required=True, help="Ausgabedatei für Gewichte, z. B. model.pt")
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--batch", type=int, default=128)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--val_split", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)

    # Inference
    sp = sub.add_parser("infer", help="WAV -> MIDI")
    sp.add_argument("--wav", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--bpm", type=float, required=True)
    sp.add_argument("--base_midi", type=int, default=60)
    sp.add_argument("--sr", type=int, default=22050)
    sp.add_argument("--offset", default="auto", help="'auto' oder Sekunden-Offset (z. B. 0.05)")

    args = ap.parse_args()
    if args.cmd == "mkdata":
        with timed("mkdata total"):
            mkdata(args.out, args.n_sequences, args.bpm, args.base_midi, args.sr,
                   args.bars, args.rest_prob, args.seed)

    elif args.cmd == "train":
        with timed("training total"):
            train_model(args.data, args.out, args.epochs, args.batch, args.lr, args.val_split, args.seed)

    elif args.cmd == "infer":
        with timed("inference total"):
            infer_to_midi(args.wav, args.out, args.model, args.bpm, args.base_midi, args.sr, args.offset)


if __name__=="__main__":
    main()
