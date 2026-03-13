# ===================== FFT + PSD + Gaussianity + Whiteness (with LOW-PASS trend removal) =====================
# Input: CSV with columns: timestamp (ROS2 nanoseconds) and pitch_error_raw
# Output: trend plot, residual plots, FFT, Welch PSD, Gaussian tests, ACF-based whiteness check

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch, detrend
from scipy import stats

# ----------------------------
# USER SETTINGS
# ----------------------------
CSV_PATH   = "/home/macs/inspection-docker/data/OrientationControlData/_1773345647/_1773345647.csv"   # <-- change if needed
TIME_COL   = "timestamp"
SIGNAL_COL = "yaw_error_raw"

# Low-pass cutoff used to estimate the "motion/trend" component (Hz)
# Typical starting points: 0.2–0.8 Hz (slow motion), 0.8–2 Hz (faster motion)
FC_TREND_HZ = 0.5

# Welch PSD settings
NPERSEG = 1024  # will be clipped automatically if signal is shorter

# ACF/whiteness settings
MAX_LAG = 50

# ----------------------------
# Helpers
# ----------------------------
def estimate_fs_from_ros_ns(t_ns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Convert ROS2 nanoseconds timestamps to seconds and estimate fs using median dt."""
    t_sec = (t_ns - t_ns[0]) * 1e-9
    dt_arr = np.diff(t_sec)
    dt_med = float(np.median(dt_arr[np.isfinite(dt_arr) & (dt_arr > 0)]))
    fs = 1.0 / dt_med
    return t_sec, dt_med, fs

def lowpass_trend(x: np.ndarray, fs: float, fc_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass filter (Butterworth) to estimate trend."""
    if fc_hz <= 0:
        raise ValueError("FC_TREND_HZ must be > 0.")
    nyq = 0.5 * fs
    wn = fc_hz / nyq
    if wn >= 1.0:
        raise ValueError(f"FC_TREND_HZ ({fc_hz}) is too high for fs={fs:.3f} Hz. Must be < Nyquist={nyq:.3f} Hz.")
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def autocorr_normed(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation up to max_lag-1 (lag 0..max_lag-1)."""
    x0 = x - np.mean(x)
    c = np.correlate(x0, x0, mode="full")
    c = c[c.size // 2:]           # keep non-negative lags
    c = c / c[0]                  # normalize so acf[0] = 1
    return c[:max_lag]

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(CSV_PATH)

if TIME_COL not in df.columns:
    raise ValueError(f"Missing '{TIME_COL}'. Available columns: {list(df.columns)}")
if SIGNAL_COL not in df.columns:
    raise ValueError(f"Missing '{SIGNAL_COL}'. Available columns: {list(df.columns)}")

df = df[[TIME_COL, SIGNAL_COL]].dropna()

t_ns = df[TIME_COL].to_numpy(dtype=np.float64)
x_raw = df[SIGNAL_COL].to_numpy(dtype=np.float64)

t, dt, fs = estimate_fs_from_ros_ns(t_ns)
print(f"Estimated dt (median): {dt:.6f} s")
print(f"Estimated fs: {fs:.3f} Hz")
print(f"Trend LPF cutoff FC_TREND_HZ: {FC_TREND_HZ:.3f} Hz")

# ----------------------------
# 1) Low-pass trend removal
# ----------------------------
trend = lowpass_trend(x_raw, fs=fs, fc_hz=FC_TREND_HZ, order=4)
residual = x_raw - trend

# Remove any remaining constant offset from residual (so tests focus on noise)
x = detrend(residual, type="constant")  # de-mean residual

# Visual check: raw vs trend vs residual
plt.figure()
plt.plot(t, x_raw, label="raw pitch_error_raw")
plt.plot(t, trend, label=f"trend (LPF {FC_TREND_HZ} Hz)")
plt.plot(t, residual, label="residual = raw - trend", alpha=0.9)
plt.xlabel("Time (s)")
plt.ylabel("pitch_error_raw")
plt.title("Trend removal (low-pass estimate of motion)")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 2) FFT of residual
# ----------------------------
N = len(x)
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(N, d=dt)
fft_mag = np.abs(X) / N

plt.figure()
plt.plot(freqs, fft_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT magnitude of residual (trend removed)")
plt.grid(True)
plt.show()

# ----------------------------
# 3) PSD (Welch) of residual
# ----------------------------
nperseg_eff = min(NPERSEG, N)  # prevent errors if short signal
f_w, Pxx = welch(x, fs=fs, nperseg=nperseg_eff)

plt.figure()
plt.semilogy(f_w, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (power/Hz)")
plt.title("Welch PSD of residual (trend removed)")
plt.grid(True)
plt.show()

# ----------------------------
# 4) Gaussianity checks on residual
# ----------------------------
mu = float(np.mean(x))
sigma = float(np.std(x, ddof=0))

print("\n=== Gaussianity (on residual) ===")
print(f"Mean:     {mu:.6g}")
print(f"Std dev:  {sigma:.6g}")
print(f"Skewness: {stats.skew(x):.6g}")
print(f"Kurtosis (excess): {stats.kurtosis(x):.6g}")

# Shapiro-Wilk (note: can be slow for very large N; SciPy may warn)
try:
    sh_stat, sh_p = stats.shapiro(x)
    print(f"Shapiro-Wilk p-value: {sh_p:.6g}")
except Exception as e:
    print(f"Shapiro-Wilk failed: {e}")

# Jarque-Bera
jb_stat, jb_p = stats.jarque_bera(x)
print(f"Jarque-Bera p-value:  {jb_p:.6g}")

# Histogram + fitted normal pdf
plt.figure()
plt.hist(x, bins=50, density=True, alpha=0.7)
xx = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
plt.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sigma), linewidth=2)
plt.xlabel("Residual value")
plt.ylabel("Density")
plt.title("Residual histogram + fitted Gaussian PDF")
plt.grid(True)
plt.show()

# Q-Q plot
plt.figure()
stats.probplot(x, dist="norm", plot=plt)
plt.title("Q-Q plot (residual vs normal)")
plt.grid(True)
plt.show()

# ----------------------------
# 5) Whiteness check (ACF + simple confidence bounds)
# ----------------------------
acf = autocorr_normed(x, MAX_LAG)

plt.figure()
plt.plot(range(MAX_LAG), acf)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of residual (trend removed)")
plt.grid(True)
plt.show()

# Approx 95% bounds for white noise ACF (rule of thumb)
bound = 2.0 / np.sqrt(N)
print("\n=== Whiteness (on residual) ===")
print(f"Approx. 95% ACF bounds: ±{bound:.6g}")

# Count significant lags (excluding lag 0)
sig_lags = np.where(np.abs(acf[1:]) > bound)[0] + 1
print(f"Significant ACF lags beyond bounds (excluding lag 0): {sig_lags.tolist()}")

if len(sig_lags) == 0:
    print("Result: Residual looks approximately WHITE (no significant autocorrelation).")
else:
    print("Result: Residual is likely COLORED / not white (some significant autocorrelation).")

# ----------------------------
# Done
# ----------------------------
print("\nFinished: trend removed via low-pass, then FFT/PSD/Gaussian/whiteness computed on residual.")