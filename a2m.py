#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio -> MIDI (monophon) mit pYIN, Barline-Blip-Filter, Alignment, Quantisierung & Monophonie.

- pYIN (f0) -> Rohnoten (mit dur_raw)
- FRÜH: Blip-Filter für Mini-Overlaps am Taktanfang (Barline)
- Alignment (globaler Shift) + musikalische Quantisierung
- NACH Quantisierung: strikte Monophonie & Duplikat-Merge
- Velocity aus RMS


Mini-Cheat-Sheet (typischer Aufruf)
python a2m.py floete.wav floete.mid --fmin C4 --fmax C7 --transpose 12 --bpm 130 --quant 1 --sig_num 4 --sig_den 4 --bar_blip_ms 220 --bar_eps_ms 35 --gap_ms 0


Basics (I/O)
input – Pfad zur Audio-Datei (wav/mp3/flac …).
output – Pfad zur Ziel-MIDI-Datei (.mid).

Pitch-Erkennung (pYIN)
--sr (int, default 22050) – Samplerate fürs Laden/Resampling.
--hop (int, 256) – Hop-Length (Zeitauflösung). Kleiner = feinere Onsets, mehr CPU.
--win (int, 2048) – Fensterlänge für STFT/pYIN/RMS.
--fmin (str/Hz, "C3") – untere Tonhöhe (z. B. C4 oder 261.63). Eng fassen verbessert Erkennung.
--fmax (str/Hz, "C6") – obere Tonhöhe (z. B. C7).
--min_frames (int, 3) – minimale Notenlänge in Frames (filtert Mikro-Noten).

MIDI-Ausgabe
--instrument (int, 73) – GM-Programmnummer (73 = Flöte).
--transpose (int, 0) – Transponieren in Halbtönen (z. B. 12 = +1 Oktave).

Tempo & Raster
--bpm (float, 0.0) – Projekttempo; 0 = ohne Raster/Quantisierung.
--quant (int, 0) – Unterteilung pro Beat: 1=Viertel, 2=Achtel, 4=16tel …; 0=aus.
--quant_mode (nearest|floorceil, default nearest)
– Start auf nächstliegenden Gridpunkt; Dauer auf Vielfache (bei floorceil immer nach oben runden).
--sig_num (int, 4) – Zähler der Taktart (z. B. 4 in 4/4).
--sig_den (int, 4) – Nenner der Taktart (z. B. 4 in 4/4).
--no_align (Flag) – globales Onset-Alignment deaktivieren (sonst werden Starts aufs Grid geschoben).

Cleanup / Anti-Legato-Blips
--bar_blip_ms (float, 200) – vor dem Quantisieren: kurze Noten am Taktanfang (≤ Wert) entfernen.
--bar_eps_ms (float, 30) – Toleranz um die Barline; wie nah an „Taktanfang“ erkannt wird.
--overlap_eps_ms (float, 2) – Toleranz beim Erkennen/Kappen von Überlappungen nach dem Quantisieren.

Feinschliff

--gap_ms (float, 0) – kurze unvoiced-Lücken im f₀ bis zu X ms füllen (bei Legato meist 0 lassen).
--tail_ms (float, 10) – Noten minimal verlängern (+X ms) für natürlichere Sustains.



"""

import argparse
import numpy as np
import librosa
import pretty_midi


# --------- Helfer ---------

def parse_note_or_hz(s: str) -> float:
    try:
        return float(librosa.note_to_hz(s))
    except Exception:
        return float(s)

def velocity_from_rms(rms_segment: np.ndarray,
                      db_min: float = -50.0,
                      db_max: float = -10.0,
                      vmin: int = 30,
                      vmax: int = 110) -> int:
    if rms_segment.size == 0:
        return 64
    r = float(np.mean(rms_segment))
    r = max(r, 1e-12)
    r_db = librosa.amplitude_to_db(np.array([r]), ref=1.0)[0]
    r_db = float(np.clip(r_db, db_min, db_max))
    alpha = (r_db - db_min) / (db_max - db_min + 1e-12)
    vel = int(round(vmin + alpha * (vmax - vmin)))
    return int(np.clip(vel, 1, 127))

def estimate_global_onset_shift(note_starts_sec: np.ndarray, bpm: float) -> float:
    if bpm <= 0 or len(note_starts_sec) == 0:
        return 0.0
    beats = note_starts_sec * bpm / 60.0
    resid = beats - np.round(beats)
    shift_beats = np.median(resid)
    return float(-shift_beats * 60.0 / bpm)

def quantize_time(value_sec: float, step_sec: float, mode: str = "nearest") -> float:
    if step_sec <= 0:
        return value_sec
    x = value_sec / step_sec
    if mode == "floor":
        q = np.floor(x)
    elif mode == "ceil":
        q = np.ceil(x)
    else:
        q = np.round(x)
    return float(q * step_sec)

def bar_length_seconds(bpm: float, sig_num: int, sig_den: int) -> float:
    # 1 Beat = 60/bpm s; Bar = sig_num * (4/sig_den) Beats
    return (60.0 / bpm) * sig_num * (4.0 / sig_den)

def near_barline(t: float, bpm: float, sig_num: int, sig_den: int, eps_sec: float) -> bool:
    if bpm <= 0:
        return False
    bar = bar_length_seconds(bpm, sig_num, sig_den)
    if bar <= 0:
        return False
    pos = t / bar
    dist = abs(pos - round(pos)) * bar
    return dist <= eps_sec


# --------- Kernschritte ---------

def extract_raw_notes(times, f0_hz, rms, hop_length, sr, min_frames):
    """Rohnoten: Dicts mit s,e,p,v,dur_raw (Sekunden)."""
    midi_seq = librosa.hz_to_midi(f0_hz)   # NaN bleibt NaN
    frame_dur = hop_length / float(sr)
    notes = []

    def flush(start_idx, end_idx, pitch_int):
        if end_idx - start_idx < min_frames:
            return
        s = float(times[start_idx])
        e = float(times[end_idx - 1] + frame_dur)
        vel = velocity_from_rms(rms[start_idx:end_idx])
        notes.append({"s": s, "e": e, "p": int(pitch_int), "v": int(vel), "dur_raw": e - s})

    active_pitch = None
    start_idx = None

    for i, val in enumerate(midi_seq):
        if np.isnan(val):
            if active_pitch is not None:
                flush(start_idx, i, active_pitch)
                active_pitch = None
                start_idx = None
            continue
        curr = int(round(val))
        if active_pitch is None:
            active_pitch = curr
            start_idx = i
        elif curr != active_pitch:
            flush(start_idx, i, active_pitch)
            active_pitch = curr
            start_idx = i

    if active_pitch is not None:
        flush(start_idx, len(midi_seq), active_pitch)

    return sorted(notes, key=lambda n: (n["s"], n["e"]))

def pre_cleanup_bar_blips(notes, bpm, sig_num, sig_den,
                          max_raw_ms=200.0, bar_eps_ms=30.0):
    """Entfernt KURZE Noten am Taktanfang (Barline), bevor sie quantisiert werden."""
    if not notes or bpm <= 0:
        return notes
    eps = bar_eps_ms / 1000.0
    max_raw = max_raw_ms / 1000.0
    out = []
    for i, n in enumerate(notes):
        s, e, p, v, dr = n["s"], n["e"], n["p"], n["v"], n["dur_raw"]
        if dr <= max_raw and near_barline(s, bpm, sig_num, sig_den, eps):
            # existiert eine vorausgehende Note, die bis an die Barline reicht / leicht überlappt?
            prev = None
            for j in range(i - 1, -1, -1):
                if notes[j]["s"] < s:
                    prev = notes[j]
                    break
            if prev is not None and prev["e"] >= s - eps:
                # klarer Legato-Überhang -> DROP
                continue
        out.append(n)
    return out

def postprocess_quantize_align(notes, bpm, quant, transpose, tail_ms,
                               align_onsets=True, start_at_zero=True, quant_mode="nearest"):
    if not notes:
        return [], 0.0
    step = 60.0 / bpm / quant if (bpm > 0 and quant > 0) else 0.0
    starts = np.array([n["s"] for n in notes], dtype=float)
    global_shift = estimate_global_onset_shift(starts, bpm) if align_onsets and bpm > 0 else 0.0

    out = []
    for n in notes:
        s, e, p, v, dr = n["s"], n["e"], n["p"], n["v"], n["dur_raw"]
        s += global_shift; e += global_shift
        e += max(0.0, tail_ms) / 1000.0

        if step > 0:
            s_q = quantize_time(s, step, mode="nearest")
            dur = max(step, e - s)
            k = max(1, int(np.round(dur / step))) if quant_mode == "nearest" else int(np.ceil(dur / step))
            e_q = s_q + k * step
            s, e = s_q, e_q

        s = max(0.0, s); e = max(s + 0.01, e)
        out.append({"s": s, "e": e, "p": p + int(transpose), "v": v, "dur_raw": dr})

    if start_at_zero and out:
        t0 = min(n["s"] for n in out)
        if t0 > 0:
            for n in out:
                n["s"] -= t0; n["e"] -= t0
    return sorted(out, key=lambda n: (n["s"], n["e"])), global_shift

def enforce_monophony(notes, overlap_eps_ms=2.0):
    """strikte Monophonie + Duplikat-Merge nach Quantisierung."""
    if not notes:
        return notes
    eps = overlap_eps_ms / 1000.0
    out = []
    for n in notes:
        if not out:
            out.append(n); continue
        s,e,p,v = n["s"], n["e"], n["p"], n["v"]
        S,E,P,V = out[-1]["s"], out[-1]["e"], out[-1]["p"], out[-1]["v"]

        # a) exakt gleicher Start & Pitch -> nimm die LÄNGERE
        if abs(s - S) <= eps and p == P:
            out[-1] = n if (e - s) > (E - S) else out[-1]
            continue

        if s < E - eps:  # Überlappung
            if p == P:
                # gleicher Pitch -> mergen
                out[-1]["e"] = max(E, e)
            else:
                # neue Note -> alte exakt bis Start kürzen
                out[-1]["e"] = max(S, s)
                out.append(n)
        else:
            # lückenlos & gleicher Pitch -> mergen
            if abs(s - E) <= eps and p == P:
                out[-1]["e"] = max(E, e)
            else:
                out.append(n)
    return out

def build_midi_from_notes(notes, bpm, program):
    pm = pretty_midi.PrettyMIDI(initial_tempo=(bpm if bpm > 0 else 120.0))
    inst = pretty_midi.Instrument(program=int(program))
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(n["v"]), pitch=int(n["p"]),
            start=float(n["s"]), end=float(n["e"])
        ))
    pm.instruments.append(inst)
    return pm


# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Audio→MIDI (monophon) mit Barline-Blip-Filter & Monophonie.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--fmin", type=str, default="C3")
    ap.add_argument("--fmax", type=str, default="C6")
    ap.add_argument("--instrument", type=int, default=73)
    ap.add_argument("--min_frames", type=int, default=3)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--bpm", type=float, default=0.0)
    ap.add_argument("--quant", type=int, default=0, help="1=Viertel, 2=Achtel, 4=Sechzehntel ...; 0=aus")
    ap.add_argument("--quant_mode", choices=["nearest","floorceil"], default="nearest")
    ap.add_argument("--sig_num", type=int, default=4, help="Zähler der Taktart (z. B. 4 in 4/4)")
    ap.add_argument("--sig_den", type=int, default=4, help="Nenner der Taktart (z. B. 4 in 4/4)")
    ap.add_argument("--gap_ms", type=float, default=0.0, help="kurze NaN-Lücken füllen (0=aus)")
    ap.add_argument("--tail_ms", type=float, default=10.0)
    ap.add_argument("--bar_blip_ms", type=float, default=200.0, help="max. Rohdauer für Barline-Blip")
    ap.add_argument("--bar_eps_ms", type=float, default=30.0, help="Toleranz um die Barline")
    ap.add_argument("--overlap_eps_ms", type=float, default=2.0)
    ap.add_argument("--no_align", action="store_true")
    args = ap.parse_args()

    # Audio laden
    y, sr = librosa.load(args.input, sr=args.sr, mono=True)
    fmin_hz = parse_note_or_hz(args.fmin)
    fmax_hz = parse_note_or_hz(args.fmax)

    f0, vf, vp = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz, sr=sr,
        frame_length=args.win, hop_length=args.hop,
        center=True, fill_na=np.nan
    )
    times = librosa.times_like(f0, sr=sr, hop_length=args.hop)
    rms = librosa.feature.rms(y=y, frame_length=args.win, hop_length=args.hop, center=True)[0]

    L = min(len(times), len(f0), len(rms))
    times, f0, rms = times[:L], f0[:L], rms[:L]

    # (optional) Gap closing – bei Legato meist 0 lassen
    if args.gap_ms > 0:
        max_gap_frames = int(round((args.gap_ms / 1000.0) * sr / args.hop))
        x = f0.copy()
        n = len(x); i = 0
        while i < n:
            if np.isnan(x[i]):
                j = i
                while j < n and np.isnan(x[j]): j += 1
                if (j - i) <= max_gap_frames:
                    left = x[i-1] if i-1 >= 0 and not np.isnan(x[i-1]) else None
                    right = x[j] if j < n and not np.isnan(x[j]) else None
                    if left is not None and right is not None:
                        x[i:j] = np.linspace(left, right, j-i+2)[1:-1]
                    elif left is not None:
                        x[i:j] = left
                    elif right is not None:
                        x[i:j] = right
                i = j
            else:
                i += 1
        f0 = x

    # 1) Rohnoten
    raw_notes = extract_raw_notes(times, f0, rms, args.hop, sr, args.min_frames)

    # 2) Barline-Blips vor Quantisierung entfernen
    raw_notes = pre_cleanup_bar_blips(raw_notes,
                                      bpm=args.bpm, sig_num=args.sig_num, sig_den=args.sig_den,
                                      max_raw_ms=args.bar_blip_ms, bar_eps_ms=args.bar_eps_ms)

    # 3) Alignment & Quantisierung
    notes_q, gshift = postprocess_quantize_align(
        raw_notes, bpm=args.bpm, quant=args.quant, transpose=args.transpose,
        tail_ms=args.tail_ms, align_onsets=(not args.no_align),
        start_at_zero=True, quant_mode=args.quant_mode
    )

    # 4) Strikte Monophonie
    notes_final = enforce_monophony(notes_q, overlap_eps_ms=args.overlap_eps_ms)

    pm = build_midi_from_notes(notes_final, bpm=args.bpm, program=args.instrument)
    pm.write(args.output)

    print(f"OK: '{args.output}' geschrieben. Noten: {len(notes_final)}")
    if args.bpm > 0 and args.quant > 0:
        step = 60.0 / args.bpm / args.quant
        print(f"   BPM={args.bpm}, Grid={step*1000:.1f} ms, globaler Shift={gshift*1000:.1f} ms")
    else:
        print("   Ohne Grid/Quantisierung exportiert.")

if __name__ == "__main__":
    main()
