# =============================================================================
# Section 2.1: PCA
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load returns
raw = pd.read_excel("Data/Data.xlsx")
raw.columns = ["PERMNO", "Date", "Ticker", "Price", "ReturnRaw"]
raw["Date"] = pd.to_datetime(raw["Date"])
prices_wide = raw.pivot(index="Date", columns="Ticker", values="Price").dropna()
ret_wide    = raw.pivot(index="Date", columns="Ticker", values="ReturnRaw").loc[prices_wide.index].dropna()
returns     = np.log(1 + ret_wide)

# PCA on the correlation matrix
# Assets span very different volatility regimes (HYG std ~0.5%, ASML std ~2.3%).
# PCA on the covariance matrix would be dominated by the highest-variance equities.
# Standardising to unit variance (= working on the correlation matrix) ensures the
# analysis captures co-movement structure independent of scale.
# pandas .std() uses ddof=1 (sample std), so the eigenvalues of the resulting
# covariance matrix equal those of the sample correlation matrix and sum to n_assets=10.
X       = ((returns - returns.mean()) / returns.std()).values
pca     = PCA()
F_all   = pca.fit_transform(X)
eigvals  = pca.explained_variance_
var_expl = pca.explained_variance_ratio_

print(f"Eigenvalues           : {np.round(eigvals, 3)}")
print(f"Variance (each)       : {np.round(var_expl, 3)}")
print(f"Variance (cumulative) : {np.round(np.cumsum(var_expl), 3)}")

# Number of factors
# The Kaiser criterion (eigenvalue > 1) selects k=2 (PC1=5.05, PC2=1.46).
# PC3 (eigenvalue=0.95) is just below the threshold but captures a distinct,
# economically interpretable factor (see loadings below).  Including it raises
# cumulative variance from 65.1% to 74.6% and lifts XOM's R2 from 0.42 to 0.78,
# confirming it contains genuine co-movement structure.  We therefore retain k=3.
k = 3

# Scree plot with cumulative variance
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(range(1, 11), eigvals, color="steelblue", alpha=0.75, label="Eigenvalue")
ax1.axhline(1, color="red", lw=1, ls="--", label="Kaiser threshold (lambda=1)")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Eigenvalue")
ax2 = ax1.twinx()
ax2.plot(range(1, 11), np.cumsum(var_expl) * 100, "ko-", ms=4, lw=1,
         label="Cumulative variance (%)")
ax2.set_ylabel("Cumulative variance explained (%)")
ax2.set_ylim(0, 105)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
ax1.set_title("Scree Plot")
plt.tight_layout()
plt.savefig("Output/pca_scree.png", dpi=150)
plt.show()

# Factor loadings:
loadings = pd.DataFrame(
    pca.components_[:k].T * np.sqrt(eigvals[:k]),
    index=returns.columns,
    columns=[f"PC{j+1}" for j in range(k)]
)

# PC1 (50.5%): global equity/risk factor.  All equity-like assets load strongly
# (SPY 0.96, EEM 0.84, HYG 0.81, MSFT 0.80, AAPL 0.77, JPM 0.75, ASML 0.74, XOM 0.61).
# TLT loads negatively (-0.24), reflecting the conventional flight-to-quality
# equity-bond relationship.  GLD is near-zero (0.07), consistent with gold's
# idiosyncratic safe-haven role that is largely uncorrelated with equity market moves.
#
# PC2 (14.6%): safe-haven / duration factor.  TLT (0.79) and GLD (0.75) load
# strongly, capturing assets that appreciate in risk-off regimes or when real yields
# fall.  JPM loads negatively (-0.35): bank profitability improves with higher rates
# (steeper yield curve widens net interest margins) while long Treasuries suffer,
# creating the observed opposition.
#
# PC3 (9.5%): energy/commodity vs. growth-tech factor.  XOM (0.60) and GLD (0.45)
# load positively; MSFT (-0.36) and AAPL (-0.34) load negatively.  This captures the
# divergence between commodity-sensitive assets (which benefit from high oil prices
# and inflation) and growth/tech stocks (which suffer when rates rise alongside
# commodity booms).  Including PC3 raises XOM's R2 from 0.42 to 0.78.
print("\nFactor Loadings:")
print(loadings.round(3))

# Loadings heatmap
fig, ax = plt.subplots(figsize=(9, 3.5))
im = ax.imshow(loadings.T.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(returns.columns)))
ax.set_xticklabels(returns.columns, fontsize=10)
ax.set_yticks(range(k))
ax.set_yticklabels([f"PC{j+1}" for j in range(k)], fontsize=10)
plt.colorbar(im, ax=ax, label="Loading (correlation)")
for i in range(k):
    for j, val in enumerate(loadings.iloc[:, i]):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > 0.55 else "black")
ax.set_title("PCA Factor Loadings")
plt.tight_layout()
plt.savefig("Output/pca_loadings.png", dpi=150)
plt.show()

# R2 per asset:
r2 = (loadings ** 2).sum(axis=1).rename("R2")
print(f"\nR2 per asset (k={k} factors):")
print(r2.round(3))

# Factor scores over time
scores = pd.DataFrame(F_all[:, :k], index=returns.index,
                      columns=[f"PC{j+1}" for j in range(k)])
fig, axes = plt.subplots(k, 1, figsize=(12, 2.2 * k), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(scores.index, scores.iloc[:, j], lw=0.6, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(f"PC{j+1}", fontsize=9)
axes[-1].set_xlabel("Date")
fig.suptitle("PCA Factor Scores Over Time")
plt.tight_layout()
plt.savefig("Output/pca_scores.png", dpi=150)
plt.show()
