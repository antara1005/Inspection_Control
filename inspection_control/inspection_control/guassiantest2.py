# ===================== Gaussian + Whiteness check for yaw_error_raw (from your CSV) =====================
# pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

CSV_PATH = "/home/macs/inspection-docker/data/OrientationControlData/_1780957749/_1780957749.csv"
COL = "yaw_error_raw"

# -------------------- Global Font Sizes --------------------
TITLE_SIZE  = 20
LABEL_SIZE  = 18
TICK_SIZE   = 15
LEGEND_SIZE = 15

# -------------------- 1) Load CSV + extract yaw_error_raw --------------------
df = pd.read_csv(CSV_PATH)

if COL not in df.columns:
    raise ValueError(f"Column '{COL}' not found. Available columns:\n{list(df.columns)}")

yaw = df[COL].dropna().to_numpy(dtype=float)

print("Columns in CSV:")
print(df.columns.tolist())

print("\nBasic Statistics for yaw_error_raw:")
print("N:", yaw.size)
print("Mean:", np.mean(yaw))
print("Std Dev:", np.std(yaw))
print("Min:", np.min(yaw))
print("Max:", np.max(yaw))

# -------------------- 2) Plot raw time series --------------------
plt.figure(figsize=(10,5))
plt.plot(yaw, linewidth=1.8)

plt.title("Yaw Error Raw - Time Series", fontsize=TITLE_SIZE)
plt.xlabel("Sample Index", fontsize=LABEL_SIZE)
plt.ylabel("Yaw Error (raw)", fontsize=LABEL_SIZE)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- 3) Gaussianity checks --------------------
# Moving-average detrending first, then demeaning.

window_size = min(10, max(3, len(yaw)//2))

yaw_trend = pd.Series(yaw).rolling(
    window=window_size,
    center=True,
    min_periods=1
).mean().values

yaw_detrended = yaw - yaw_trend

# Demean after detrending
yaw_noise = yaw_detrended - np.mean(yaw_detrended)

mu0 = np.mean(yaw_noise)
print(f"\nAfter moving-average detrending + demeaning: mean = {mu0:.6g}")

sigma = np.std(yaw_noise)
print(f"Standard deviation (sigma) of detrended+demean data: {sigma:.6g}")

# -------------------- Plot detrended noise --------------------
plt.figure(figsize=(10,5))
plt.plot(yaw_noise, linewidth=1.8)

plt.title("Yaw Error After Detrending + Demeaning", fontsize=TITLE_SIZE)
plt.xlabel("Sample Index", fontsize=LABEL_SIZE)
plt.ylabel("Yaw Error Noise Estimate", fontsize=LABEL_SIZE)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Histogram + Gaussian PDF --------------------
plt.figure(figsize=(8,5))

plt.hist(yaw_noise, bins=50, density=True, alpha=0.7)

x = np.linspace(yaw_noise.min(), yaw_noise.max(), 1000)
plt.plot(
    x,
    stats.norm.pdf(x, loc=mu0, scale=sigma),
    linewidth=2,
    label='Gaussian Fit'
)

plt.title("Histogram with Gaussian Fit", fontsize=TITLE_SIZE)
plt.xlabel("Yaw Error Noise Estimate", fontsize=LABEL_SIZE)
plt.ylabel("Probability Density", fontsize=LABEL_SIZE)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.legend(fontsize=LEGEND_SIZE)

plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Q-Q plot --------------------
plt.figure(figsize=(7,7))

stats.probplot(yaw_noise, dist="norm", plot=plt)

plt.title("Q-Q Plot of Detrended yaw_error_raw", fontsize=TITLE_SIZE)

plt.xlabel("Theoretical Quantiles", fontsize=LABEL_SIZE)
plt.ylabel("Ordered Values", fontsize=LABEL_SIZE)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Normality tests --------------------
print("\nNormality Test Results (p-values):")

n_shapiro = min(5000, yaw_noise.size)
shapiro_stat, shapiro_p = stats.shapiro(yaw_noise[:n_shapiro])

ks_stat, ks_p = stats.kstest(yaw_noise, "norm", args=(0.0, sigma))

dag_stat, dag_p = stats.normaltest(yaw_noise)

print(f"  Shapiro-Wilk (N={n_shapiro})   p = {shapiro_p:.6g}, W = {shapiro_stat:.6g}")
print(f"  Kolmogorov-Smirnov            p = {ks_p:.6g}, D = {ks_stat:.6g}")
print(f"  D'Agostino K^2                p = {dag_p:.6g}, K^2 = {dag_stat:.6g}")

print("\nInterpretation:")
print("  If p-value > 0.05  -> cannot reject Gaussian assumption")
print("  If p-value < 0.05  -> likely NOT Gaussian")

# Extra shape stats
skew = stats.skew(yaw_noise)
kurt_excess = stats.kurtosis(yaw_noise)

print("\nShape statistics:")
print(f"  Skewness        = {skew:.6g}   (Gaussian ~ 0)")
print(f"  Excess kurtosis = {kurt_excess:.6g}   (Gaussian ~ 0)")

# -------------------- 4) Whiteness check: autocorrelation --------------------
autocorr_full = np.correlate(yaw_noise, yaw_noise, mode="full")
autocorr = autocorr_full[autocorr_full.size // 2 :]
autocorr /= autocorr[0]

max_lag = min(200, autocorr.size - 1)

plt.figure(figsize=(10,5))

plt.plot(
    np.arange(max_lag + 1),
    autocorr[: max_lag + 1],
    linewidth=2
)

plt.title(
    "Autocorrelation of Detrended + Demeaned yaw_error_raw",
    fontsize=TITLE_SIZE
)

plt.xlabel("Lag (samples)", fontsize=LABEL_SIZE)
plt.ylabel("Normalized autocorrelation", fontsize=LABEL_SIZE)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- 5) Suggested Kalman measurement noise R --------------------
R = sigma**2
print(f"\nSuggested measurement noise variance R ≈ sigma^2 = {R:.6g}")