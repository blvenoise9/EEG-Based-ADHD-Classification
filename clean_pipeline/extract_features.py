"""
Leak-free feature extraction for the EEG-ADHD dataset.

Fixes relative to the original notebooks:
  * Every epoch row carries a `subject` id (needed for grouped CV / subject-level metrics).
  * Files are processed in sorted order -> deterministic, reproducible.
  * ICA actually removes ocular components (Fp1/Fp2 used as EOG proxies); the
    original code fitted ICA but excluded 0 components for every subject.
  * Epochs are exactly 2 s (256 samples), not 257.
  * No duplicate columns (old `Alpha_chN` == `alpha_power_chN`, `slow_chN` == `slow_power_chN`).
  * Higuchi FD uses the standard normalisation (old code returned FD - 1).
  * Band powers are log10-transformed (standard for EEG; helps SVM/KNN).
  * "Slow-wave" power is defined as 1-8 Hz (delta+theta). The old "slow" band
    (12.5 Hz - Nyquist) is kept under its honest name `fast12p5`.

All preprocessing is per-subject and label-free, so it cannot leak across subjects.

Usage:  python3 clean_pipeline/extract_features.py
Output: clean_pipeline/output/features.csv, clean_pipeline/output/ica_log.csv
"""
import glob
import os
import sys
import warnings

import mne
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from mne.preprocessing import ICA
from scipy import signal

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "clean_pipeline", "output")
SFREQ = 128
# Channel order as documented for this dataset (Nasrabadi et al., IEEE DataPort).
CH_NAMES = ["Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3",
            "F4", "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45),
         "slow": (1, 8),            # delta + theta (slow-wave activity)
         "fast12p5": (12.5, 64)}    # what the old notebooks called "slow"


def higuchi_fd(x, kmax=10):
    """Vectorised Higuchi fractal dimension. x: (n_channels, N) -> (n_channels,)"""
    n_ch, N = x.shape
    ks = np.arange(1, kmax + 1)
    logL = np.empty((n_ch, kmax))
    for j, k in enumerate(ks):
        Lm = []
        for m in range(k):
            n_max = (N - 1 - m) // k
            if n_max < 1:
                continue
            seg = x[:, m: m + n_max * k + 1: k]
            length = np.abs(np.diff(seg, axis=1)).sum(axis=1)
            Lm.append(length * (N - 1) / (n_max * k) / k)
        logL[:, j] = np.log(np.mean(Lm, axis=0))
    slope = np.polyfit(np.log(1.0 / ks), logL.T, 1)[0]
    return slope


def process_subject(path):
    subject = os.path.splitext(os.path.basename(path))[0]
    group = os.path.basename(os.path.dirname(path))
    label = 1 if group.startswith("ADHD") else 0
    data = scipy.io.loadmat(path)[subject].astype(float)  # (n_times, 19)

    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data.T, info)
    raw.set_montage("standard_1020")
    raw.filter(1.0, 50.0)

    ica = ICA(n_components=15, random_state=42, max_iter=800)
    ica.fit(raw)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"])
    ica.exclude = eog_idx
    raw_clean = ica.apply(raw.copy())

    events = mne.make_fixed_length_events(raw_clean, duration=2.0, overlap=1.0)
    epochs = mne.Epochs(raw_clean, events, tmin=0, tmax=2.0 - 1.0 / SFREQ,
                        baseline=None, preload=True)
    X = epochs.get_data()  # (n_epochs, 19, 256)

    freqs, psd = signal.welch(X, fs=SFREQ, nperseg=X.shape[-1], axis=-1)
    rows = {}
    bandpow = {}
    for b, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bandpow[b] = psd[..., mask].mean(axis=-1)  # (n_epochs, 19)
        for c, ch in enumerate(CH_NAMES):
            rows[f"{b}_{ch}"] = np.log10(bandpow[b][:, c])
    tbr = bandpow["theta"] / bandpow["beta"]
    for c, ch in enumerate(CH_NAMES):
        rows[f"tbr_{ch}"] = np.log10(tbr[:, c])
    hfd = np.stack([higuchi_fd(ep) for ep in X])  # (n_epochs, 19)
    for c, ch in enumerate(CH_NAMES):
        rows[f"hfd_{ch}"] = hfd[:, c]

    df = pd.DataFrame(rows)
    df.insert(0, "epoch", np.arange(len(df)))
    df.insert(0, "label", label)
    df.insert(0, "subject", subject)
    log = {"subject": subject, "label": label, "n_epochs": len(df),
           "ica_excluded": len(eog_idx), "duration_s": data.shape[0] / SFREQ}
    return df, log


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "Dataset", "*", "*.mat")))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    assert len(ids) == len(set(ids)), "duplicate subject ids across folders"
    print(f"{len(files)} subjects found", flush=True)

    results = Parallel(n_jobs=-1, verbose=5)(delayed(process_subject)(f) for f in files)
    feats = pd.concat([r[0] for r in results], ignore_index=True)
    log = pd.DataFrame([r[1] for r in results])
    feats.to_csv(os.path.join(OUT, "features.csv"), index=False)
    log.to_csv(os.path.join(OUT, "ica_log.csv"), index=False)
    print(feats.shape, "\n", log.groupby("label")[["n_epochs", "ica_excluded"]].describe().T)


if __name__ == "__main__":
    sys.exit(main())
