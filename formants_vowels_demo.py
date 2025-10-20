#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formants_vowels_demo.py

Uso (grabar 3 s y analizar):
  python formants_vowels_demo.py --record --seconds 3 --sr 16000 --out prefix_a

Uso (analizar un WAV existente):
  python formants_vowels_demo.py --wav tu_audio.wav --out analisis_audio

Genera:
  <out>.wav        (si se grabó)
  <out>_figure.png (espectrograma + PSD + LPC)
  <out>_formants.csv (F1..F3 en Hz)
"""

from pathlib import Path
import argparse, sys, numpy as np, matplotlib.pyplot as plt
import sounddevice as sd, soundfile as sf
import librosa, librosa.display
import scipy.signal as sig
import pandas as pd

# ---------- utilidades ----------
def rec(seconds: float, sr: int, device=None) -> np.ndarray:
    print(f"🎙️ Grabando {seconds:.1f} s a {sr} Hz …")
    sd.default.samplerate = sr
    sd.default.channels = 1
    x = sd.rec(int(seconds*sr), dtype="float32", device=device)
    sd.wait()
    return x.squeeze()

def highpass(x, sr, fc=60.0):
    # elimina DC/rumble (suave) para clase
    b, a = sig.butter(2, fc/(sr/2), btype="highpass")
    return sig.lfilter(b, a, x)

def preemphasis(x, k=0.97):
    return sig.lfilter([1, -k], [1], x)

def welch_psd(x, sr, nper=1024, nover=512):
    f, Pxx = sig.welch(x, fs=sr, nperseg=nper, noverlap=nover, window="hann")
    return f, Pxx

def lpc_envelope(x, sr, order=16, n_fft=4096):
    # LPC con librosa; devuelve envolvente densa (malla de frecuencias)
    a = librosa.lpc(x.astype(float), order=order)  # a[0]≈1
    # respuesta en frecuencia de 1/A(z)
    w, h = sig.freqz(b=[1.0], a=a, worN=n_fft, fs=sr)
    env = np.abs(h) + 1e-12
    return w, env, a

def find_formants_from_envelope(freqs, env, k=3, fmax=4000, prominence_db=6.0):
    """Busca picos en la envolvente LPC hasta fmax. Devuelve hasta k formantes (Hz)."""
    mask = freqs <= fmax
    f = freqs[mask]; y = 20*np.log10(env[mask])
    # realce local para que alce los picos (suavizado leve)
    y_s = sig.savgol_filter(y, 15 if len(y) >= 15 else max(5, len(y)//2*2+1), 3, mode="interp")
    peaks, _ = sig.find_peaks(y_s, prominence=prominence_db)
    f_peaks = f[peaks]
    f_peaks = np.sort(f_peaks)[:k]
    # completa con NaN si faltan
    out = list(f_peaks.astype(float))
    while len(out) < k: out.append(np.nan)
    return out, f, y_s

# ---------- plot ----------
def plot_all(x, sr, out_png, f1=None, f2=None, f3=None, n_fft_spec=1024, hop=256):
    fig, axes = plt.subplots(3, 1, figsize=(10,10), constrained_layout=True)

    # 1) Onda
    t = np.arange(len(x))/sr
    axes[0].plot(t, x, lw=0.9)
    axes[0].set_title("Onda temporal")
    axes[0].set_xlabel("Tiempo [s]"); axes[0].set_ylabel("Amplitud")

    # 2) Espectrograma
    S = np.abs(librosa.stft(x, n_fft=n_fft_spec, hop_length=hop, window="hann"))**2
    Sdb = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(Sdb, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma")
    axes[1].set_title("Espectrograma (STFT)")
    fig.colorbar(img, ax=axes[1], format="%+0.0f dB")
    for f in [f1, f2, f3]:
        if f and np.isfinite(f):
            axes[1].axhline(f, color="c", ls="--", lw=2)

    # 3) PSD + LPC
    f_psd, Pxx = welch_psd(x, sr)
    axes[2].semilogy(f_psd, Pxx, label="PSD (Welch)")
    # LPC envelope
    w, env, _ = lpc_envelope(x, sr, order=16)
    axes[2].plot(w, env/np.max(env)*np.max(Pxx), label="Envolvente LPC (esc. a PSD)")
    axes[2].set_xlim(0, 4000)
    axes[2].set_xlabel("Frecuencia [Hz]"); axes[2].set_ylabel("Densidad / Amplitud")
    axes[2].set_title("PSD y envolvente LPC (0–4 kHz)")
    for f in [f1, f2, f3]:
        if f and np.isfinite(f):
            axes[2].axvline(f, color="r", ls="--", lw=1)
    axes[2].legend()

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", action="store_true", help="Grabar desde micrófono")
    src.add_argument("--wav", type=str, help="Analizar un WAV existente")
    ap.add_argument("--seconds", type=float, default=3.0, help="Segundos a grabar (si --record)")
    ap.add_argument("--sr", type=int, default=16000, help="Frecuencia de muestreo")
    ap.add_argument("--out", type=str, default="formant_demo", help="Prefijo de salida")
    ap.add_argument("--no_preemph", action="store_true", help="Desactiva pre-énfasis")
    ap.add_argument("--hp", type=float, default=60.0, help="High-pass inicial [Hz]")
    ap.add_argument("--lpc-order", type=int, default=16, help="Orden LPC (16 es bueno para 16 kHz)")
    ap.add_argument("--fmax", type=float, default=4000, help="Máx. frecuencia para buscar F1–F3")
    args = ap.parse_args()

    out = Path(args.out)

    # ---- 1) audio
    if args.record:
        x = rec(args.seconds, args.sr)
        sf.write(out.with_suffix(".wav"), x, args.sr)
        print(f"💾 Guardado {out.with_suffix('.wav')}")
    else:
        x, sr_file = sf.read(args.wav)
        if x.ndim > 1: x = x.mean(axis=1)
        if sr_file != args.sr:
            print(f"⚠️ Re-sample {sr_file}→{args.sr}")
            x = librosa.resample(x.astype(float), sr_file, args.sr)
    sr = args.sr
    x = highpass(x, sr, fc=args.hp)
    if not args.no_preemph:
        x = preemphasis(x, k=0.97)
    x = x.astype(float)
    x = x/np.max(np.abs(x)+1e-12)  # normaliza suave

    # ---- 2) LPC + formantes
    w, env, _ = lpc_envelope(x, sr, order=args.lpc_order)
    (F1, F2, F3), f_env, env_db = find_formants_from_envelope(w, env, k=3, fmax=args.fmax)

    # ---- 3) figuras + CSV
    out_png = out.with_name(out.name + "_figure.png")
    plot_all(x, sr, out_png, F1, F2, F3)
    pd.DataFrame([{"F1_Hz":F1, "F2_Hz":F2, "F3_Hz":F3, "sr":sr,
                   "lpc_order":args.lpc_order, "preemph": (not args.no_preemph)}]).to_csv(
        out.with_name(out.name + "_formants.csv"), index=False)

    print(f"F1≈{F1:.0f} Hz, F2≈{F2:.0f} Hz, F3≈{F3:.0f} Hz")
    print(f"Figura: {out_png}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)