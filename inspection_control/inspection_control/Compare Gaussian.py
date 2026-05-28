# ===================== Compare Gaussianity of TWO CSV files =====================
# Plots:
#   1) Overlaid histograms
#   2) Theoretical Gaussian PDFs for both datasets
#
# pip install numpy pandas matplotlib scipy pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# -------------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------------
CSV1 = "/home/macs/inspection-docker/data/OrientationControlData/_1776197156/ransacbladenoisy.csv"
CSV2 = "/home/macs/inspection-docker/data/OrientationControlData/_1776196458/pcabladenoisy.csv"

COL = "pitch_error_raw"   # change if needed

LABEL1 = "Hybrid RANSAC-PCA"
LABEL2 = "PCA"

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df1 = pd.read_csv(CSV1)
df2 = pd.read_csv(CSV2)

if COL not in df1.columns:
    raise ValueError(f"{COL} not found in CSV1")

if COL not in df2.columns:
    raise ValueError(f"{COL} not found in CSV2")

y1 = df1[COL].dropna().to_numpy(dtype=float)
y2 = df2[COL].dropna().to_numpy(dtype=float)

# -------------------------------------------------------------------
# DEMEAN
# -------------------------------------------------------------------
y1 = y1 - np.mean(y1)
y2 = y2 - np.mean(y2)

mu1, sigma1 = np.mean(y1), np.std(y1)
mu2, sigma2 = np.mean(y2), np.std(y2)

print("\n================ Dataset Statistics ================\n")

print(f"{LABEL1}:")
print(f"Mean  = {mu1}")
print(f"Std   = {sigma1}")
print(f"Skew  = {stats.skew(y1)}")
print(f"Kurt  = {stats.kurtosis(y1)}")

print()

print(f"{LABEL2}:")
print(f"Mean  = {mu2}")
print(f"Std   = {sigma2}")
print(f"Skew  = {stats.skew(y2)}")
print(f"Kurt  = {stats.kurtosis(y2)}")

# -------------------------------------------------------------------
# COMMON X RANGE
# -------------------------------------------------------------------
xmin = min(y1.min(), y2.min())
xmax = max(y1.max(), y2.max())

x = np.linspace(xmin, xmax, 1000)

# -------------------------------------------------------------------
# OVERLAID HISTOGRAM + GAUSSIAN PDFs
# -------------------------------------------------------------------
plt.figure(figsize=(8,5))

# Histograms
plt.hist(
    y1,
    bins=50,
    density=True,
    alpha=0.5,
    label=f"{LABEL1} Histogram"
)

plt.hist(
    y2,
    bins=50,
    density=True,
    alpha=0.5,
    label=f"{LABEL2} Histogram"
)

# Theoretical Gaussian PDFs
pdf1 = stats.norm.pdf(x, loc=mu1, scale=sigma1)
pdf2 = stats.norm.pdf(x, loc=mu2, scale=sigma2)

plt.plot(
    x,
    pdf1,
    linewidth=2.5,
    label=f"{LABEL1} Gaussian Fit"
)

plt.plot(
    x,
    pdf2,
    linewidth=2.5,
    linestyle='--',
    label=f"{LABEL2} Gaussian Fit"
)

# Labels
plt.title(f"Histogram + Gaussian Fit Comparison ({COL})")
plt.xlabel("Demeaned Error")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)

plt.show()

# -------------------------------------------------------------------
# OPTIONAL: Q-Q PLOTS SIDE BY SIDE
# -------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10,5))

stats.probplot(y1, dist="norm", plot=axs[0])
axs[0].set_title(f"Q-Q Plot ({LABEL1})")
axs[0].grid(True)

stats.probplot(y2, dist="norm", plot=axs[1])
axs[1].set_title(f"Q-Q Plot ({LABEL2})")
axs[1].grid(True)

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------
# AUTO-ZOOMED HISTOGRAM + GAUSSIAN PDF COMPARISON
# -------------------------------------------------------------------

# Robust zoom range using percentiles
low = min(np.percentile(y1, 0.5), np.percentile(y2, 0.5))
high = max(np.percentile(y1, 99.5), np.percentile(y2, 99.5))

# Slight padding
pad = 0.1 * (high - low)

xmin_zoom = low - pad
xmax_zoom = high + pad

# Dense x-grid ONLY over zoomed region
x_zoom = np.linspace(xmin_zoom, xmax_zoom, 1000)

plt.figure(figsize=(9,5))

# Histograms
plt.hist(
    y1,
    bins=60,
    density=True,
    alpha=0.5,
    label=f"{LABEL1} Histogram",
    range=(xmin_zoom, xmax_zoom)
)

plt.hist(
    y2,
    bins=60,
    density=True,
    alpha=0.5,
    label=f"{LABEL2} Histogram",
    range=(xmin_zoom, xmax_zoom)
)

# Gaussian fits
pdf1 = stats.norm.pdf(x_zoom, loc=mu1, scale=sigma1)
pdf2 = stats.norm.pdf(x_zoom, loc=mu2, scale=sigma2)

plt.plot(
    x_zoom,
    pdf1,
    linewidth=3,
    label=f"{LABEL1} Gaussian Fit"
)

plt.plot(
    x_zoom,
    pdf2,
    linewidth=3,
    linestyle='--',
    label=f"{LABEL2} Gaussian Fit"
)

# Zoomed axis
plt.xlim([xmin_zoom, xmax_zoom])

# Optional: automatically scale y-axis tightly
ymax = max(pdf1.max(), pdf2.max())
plt.ylim([0, ymax * 1.15])

# Labels
plt.title(f" Histogram + Gaussian Fit Comparison ({COL})")
plt.xlabel("Demeaned Error")
plt.ylabel("Probability Density")

plt.legend()
plt.grid(True)

plt.show()