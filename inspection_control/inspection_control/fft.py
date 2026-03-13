# Frequency analysis for pitch_error_raw in your CSV (FFT + PSD/Welch)
# Works for ROS2-style nanosecond timestamps.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, get_window

#CSV_PATH = "/mnt/data/_1771456135.csv"   # <-- change if needed
SIGNAL_COL = "pitch_error_raw"
TIME_COL = "timestamp"                  # your file has this

# ----------------------------
# Load + basic cleanup
# ----------------------------
df = pd.read_csv("/home/macs/inspection-docker/data/OrientationControlData/_1771894373/_1771894373.csv")

if SIGNAL_COL not in df.columns:
    raise ValueError(f"Missing column '{SIGNAL_COL}'. Available: {list(df.columns)}")
if TIME_COL not in df.columns:
    raise ValueError(f"Missing time column '{TIME_COL}'. Available: {list(df.columns)}")

# Keep only valid rows
df = df[[TIME_COL, SIGNAL_COL]].dropna()

t_raw = df[TIME_COL].to_numpy(dtype=np.float64)
x_raw = df[SIGNAL_COL].to_numpy(dtype=np.float64)

# Sort by time (just in case)
idx = np.argsort(t_raw)
t_raw = t_raw[idx]
x_raw = x_raw[idx]

# Remove duplicate timestamps (can break dt)
dupe_mask = np.diff(t_raw, prepend=t_raw[0]-1) > 0
t_raw = t_raw[dupe_mask]
x_raw = x_raw[dupe_mask]

# ----------------------------
# Convert timestamp -> seconds
# ----------------------------
# ROS2 timestamps are typically nanoseconds. Convert to seconds.
t = (t_raw - t_raw[0]) * 1e-9

# Estimate sample rate
dt = np.diff(t)
dt = dt[np.isfinite(dt) & (dt > 0)]
if len(dt) < 10:
    raise ValueError("Not enough valid time steps to estimate sampling rate.")

dt_med = float(np.median(dt))
dt_mean = float(np.mean(dt))
print(f"Estimated dt (mean):   {dt_mean:.6f} s")
print(f"Estimated dt (median): {dt_med:.6f} s")
fs = 1.0 / dt_med

print(f"Estimated dt (median): {dt_med:.6f} s")
print(f"Estimated fs:          {fs:.3f} Hz")
print(f"Samples:              {len(x_raw)}")

# ----------------------------
# Detrend (recommended for noise analysis)
# ----------------------------
x = detrend(x_raw)  # removes mean + linear drift

# ----------------------------
# FFT magnitude spectrum
# ----------------------------
N = len(x)
# Hann window reduces spectral leakage
w = get_window("hann", N)
xw = x * w
# Coherent gain to roughly preserve amplitude
cg = np.sum(w) / N

X = np.fft.rfft(xw)
f_fft = np.fft.rfftfreq(N, d=dt_med)

mag = (np.abs(X) / N) / max(cg, 1e-12)   # magnitude spectrum (approx amplitude)

plt.figure()
plt.plot(f_fft, mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (approx.)")
plt.title(f"FFT Magnitude Spectrum - {SIGNAL_COL}")
plt.grid(True)

# Optional: zoom to useful band (0..fs/2)
plt.xlim(0, fs/2)
plt.show()

# ----------------------------
# PSD using Welch (best for noise)
# ----------------------------
# Choose segment length: power-of-2, not too large
nperseg = min(2048, N)
if nperseg < 256:
    nperseg = min(256, N)

f_psd, Pxx = welch(
    x,
    fs=fs,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg//2,
    detrend=False,
    scaling="density"
)

plt.figure()
plt.semilogy(f_psd, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (units^2/Hz)")
plt.title(f"PSD (Welch) - {SIGNAL_COL} | nperseg={nperseg}")
plt.grid(True)
plt.xlim(0, fs/2)
plt.show()

# ----------------------------
# Helpful summary metrics
# ----------------------------
var_time = float(np.var(x))                      # time-domain variance
df_bin = float(f_psd[1] - f_psd[0]) if len(f_psd) > 1 else np.nan
var_from_psd = float(np.trapz(Pxx, f_psd))       # should roughly match var_time

# Dominant frequency peak (ignore DC bin)
peak_i = int(np.argmax(Pxx[1:]) + 1) if len(Pxx) > 2 else 0
peak_f = float(f_psd[peak_i]) if len(Pxx) else np.nan
peak_val = float(Pxx[peak_i]) if len(Pxx) else np.nan

print("\n--- Summary ---")
print(f"Time-domain variance (detrended): {var_time:.6e}")
print(f"Variance from integrating PSD:     {var_from_psd:.6e}")
print(f"Dominant PSD peak frequency:       {peak_f:.3f} Hz")
print(f"Dominant PSD peak value:           {peak_val:.6e}")

# Optional: band power helper (e.g., noise above 5 Hz)
def band_power(f, Pxx, f1, f2):
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(Pxx[mask], f[mask]))

print("\nBand powers (variance contribution):")
print(f"  0–1 Hz:    {band_power(f_psd, Pxx, 0.0, 1.0):.6e}")
print(f"  1–5 Hz:    {band_power(f_psd, Pxx, 1.0, 5.0):.6e}")
print(f"  5–{fs/2:.1f} Hz: {band_power(f_psd, Pxx, 5.0, fs/2):.6e}")
