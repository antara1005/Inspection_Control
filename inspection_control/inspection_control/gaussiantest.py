# ===================== Gaussian + Whiteness check for yaw_error_raw (from your CSV) =====================
# pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

CSV_PATH = "/home/macs/inspection-docker/data/OrientationControlData/_1771524065/_1771524065.csv"   # <-- change if your file is elsewhere
COL = "yaw_error_raw"  # <-- change if your column has a different name (check the CSV header)

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
plt.figure()
plt.plot(yaw)
plt.title("yaw Error Raw - Time Series")
plt.xlabel("Sample Index")
plt.ylabel("yaw Error (raw)")
plt.grid(True)
plt.show()

# -------------------- 3) Gaussianity checks --------------------
# Demean for normality checks (Gaussian noise is usually modeled zero-mean)
yaw_demean = yaw - np.mean(yaw)
mu0 = np.mean(yaw_demean)          # ~0
print(f"\nAfter demeaning: mean = {mu0:.6g} (should be close to 0)")
sigma = np.std(yaw_demean)
print(f"Standard deviation (sigma) of demeaned data: {sigma:.6g}")
# 3a) Histogram + Gaussian PDF overlay (CORRECTLY aligned)
plt.figure()
plt.hist(yaw_demean, bins=50, density=True, alpha=0.7)
x = np.linspace(yaw_demean.min(), yaw_demean.max(), 1000)
plt.plot(x, stats.norm.pdf(x, loc=mu0, scale=sigma), linewidth=2)
plt.title("Histogram of (yaw_error_raw - mean) with Gaussian Fit")
plt.xlabel("yaw Error (demeaned)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# 3b) Q-Q plot
plt.figure()
stats.probplot(yaw_demean, dist="norm", plot=plt)
plt.title("Q-Q Plot (Normal Distribution)")
plt.grid(True)
plt.show()

# 3c) Normality tests
print("\nNormality Test Results (p-values):")

# Shapiro-Wilk: scipy recommends not huge N; take up to 5000 samples
n_shapiro = min(5000, yaw_demean.size)
shapiro_stat, shapiro_p = stats.shapiro(yaw_demean[:n_shapiro])

# KS test: compare to N(0, sigma^2) since we demeaned
ks_stat, ks_p = stats.kstest(yaw_demean, "norm", args=(0.0, sigma))

# D’Agostino’s K^2 test (works well for larger N)
dag_stat, dag_p = stats.normaltest(yaw_demean)

print(f"  Shapiro-Wilk (N={n_shapiro})   p = {shapiro_p:.6g}, W = {shapiro_stat:.6g}")
print(f"  Kolmogorov-Smirnov            p = {ks_p:.6g}, D = {ks_stat:.6g}")
print(f"  D'Agostino K^2                p = {dag_p:.6g}, K^2 = {dag_stat:.6g}")

print("\nInterpretation:")
print("  If p-value > 0.05  -> cannot reject Gaussian assumption")
print("  If p-value < 0.05  -> likely NOT Gaussian")

# Extra shape stats
skew = stats.skew(yaw_demean)
kurt_excess = stats.kurtosis(yaw_demean)   # excess kurtosis (0 for Gaussian)
print("\nShape statistics:")
print(f"  Skewness        = {skew:.6g}   (Gaussian ~ 0)")
print(f"  Excess kurtosis = {kurt_excess:.6g}   (Gaussian ~ 0)")

# -------------------- 4) Whiteness check: autocorrelation --------------------
# Normalized autocorrelation for first 200 lags
autocorr_full = np.correlate(yaw_demean, yaw_demean, mode="full")
autocorr = autocorr_full[autocorr_full.size // 2 :]   # keep non-negative lags
autocorr /= autocorr[0]                               # normalize

max_lag = min(200, autocorr.size - 1)
plt.figure()
plt.plot(np.arange(max_lag + 1), autocorr[: max_lag + 1])
plt.title("Autocorrelation of (yaw_error_raw - mean) [First 200 lags]")
plt.xlabel("Lag (samples)")
plt.ylabel("Normalized autocorrelation")
plt.grid(True)
plt.show()

print("\nWhiteness intuition:")
print("  If autocorrelation drops close to 0 quickly after lag 0 -> closer to white noise")

# -------------------- 5) Suggested Kalman measurement noise R --------------------
# If yaw_error_raw is your measurement y = true + v, and v ~ N(0, R),
# then R ≈ variance of (demeaned) measurement noise:
R = sigma**2
print(f"\nSuggested measurement noise variance R ≈ sigma^2 = {R:.6g}")
