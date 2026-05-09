# =============================================================================
# Section 2.2: Factor Analysis (Fama-French 5 + Momentum)
# =============================================================================
# Factor data is obtained from WRDS, ff5+momentum until Dec 2024, ff5+momentum1 until march 2026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# 1.  Load asset returns (same pipeline as setup.py / pca.py)
# ---------------------------------------------------------------------------
raw = pd.read_excel("Data/Data.xlsx")
raw.columns = ["PERMNO", "Date", "Ticker", "Price", "ReturnRaw"]
raw["Date"] = pd.to_datetime(raw["Date"])
prices_wide = raw.pivot(index="Date", columns="Ticker", values="Price").dropna()
ret_wide    = raw.pivot(index="Date", columns="Ticker", values="ReturnRaw") \
                 .loc[prices_wide.index].dropna()
returns     = np.log(1 + ret_wide)      # log returns (additive, Gaussian-friendly)

# ---------------------------------------------------------------------------
# 2.  Load and clean Fama-French 5 + Momentum factors
# ---------------------------------------------------------------------------
ff_raw = pd.read_excel("Data/ff5+momentum1.xlsx")

# Rename verbose column headers to standard short names
ff_raw.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
ff_raw["Date"] = pd.to_datetime(ff_raw["Date"])
ff_raw = ff_raw.set_index("Date")

# Values are already in decimal form (e.g. 0.0005 = 0.05% per day).
# No rescaling needed.

# ---------------------------------------------------------------------------
# 3.  Merge on the common date range
# ---------------------------------------------------------------------------
# Inner join: keeps only trading days present in BOTH datasets.
# Returns cover Jan 2015 – early 2026; FF covers Jan 2015 – Dec 2024.
# After merging, the effective sample ends 2024-12-31.
merged = returns.join(ff_raw, how="inner")

print(f"Merged period : {merged.index[0].date()} to {merged.index[-1].date()}")
print(f"Observations  : {len(merged)}")
print(f"Missing values: {merged.isnull().sum().sum()}")

factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
assets  = list(returns.columns)

# ---------------------------------------------------------------------------
# 4.  Compute excess returns  (r_i - RF)
# ---------------------------------------------------------------------------
# RF is the daily risk-free rate (one-month T-bill, daily).
# Subtracting it yields the compensation for bearing systematic risk.
excess_returns = merged[assets].subtract(merged["RF"], axis=0)

# ---------------------------------------------------------------------------
# 5.  Multivariate OLS regression for each asset
# ---------------------------------------------------------------------------
# Model: r_{i,t} - RF_t = alpha_i + beta_Mkt * (Mkt-RF)_t
#                        + beta_SMB * SMB_t + beta_HML * HML_t
#                        + beta_RMW * RMW_t + beta_CMA * CMA_t
#                        + beta_MOM * MOM_t + epsilon_{i,t}
#
# We include a constant (alpha) so that an asset's idiosyncratic average
# excess return is not absorbed into the factor betas.

X = sm.add_constant(merged[factors])   # shape (T, 7): const + 6 factors

results   = {}   # {ticker: RegressionResults}
betas     = {}   # {ticker: pd.Series of factor loadings}
alphas    = {}   # {ticker: float}
r2_values = {}   # {ticker: float}

for asset in assets:
    y   = excess_returns[asset]
    res = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")   # HC3 for heteroskedasticity
    results[asset]   = res
    betas[asset]     = res.params[factors]     # drop the constant
    alphas[asset]    = res.params["const"]
    r2_values[asset] = res.rsquared

# Collect into DataFrames
betas_df = pd.DataFrame(betas, index=factors).T    # (10 assets × 6 factors)
r2_series = pd.Series(r2_values, name="R²")

print("\n--- Factor Betas ---")
print(betas_df.round(3))

print("\n--- Alphas (annualised, %) ---")
ann_alphas = pd.Series(alphas) * 252 * 100
print(ann_alphas.round(2))

print("\n--- R2 per asset ---")
print(r2_series.round(3))

# ---------------------------------------------------------------------------
# 6.  Print individual regression summaries
# ---------------------------------------------------------------------------
print("\n" + "="*70)
for asset in assets:
    print(f"\n{'='*70}")
    print(f"Asset: {asset}")
    print(results[asset].summary2(title=f"OLS – {asset} excess returns"))

# ---------------------------------------------------------------------------
# 7.  Visualisation A – Factor Loading Heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))

# Symmetric colour scale centred on zero; ±2 covers all typical beta values.
vmax = max(abs(betas_df.values.max()), abs(betas_df.values.min()))
vmax = np.ceil(vmax * 10) / 10      # round up to nearest 0.1

im = ax.imshow(
    betas_df.values,            # rows = assets, cols = factors
    cmap="RdBu_r",
    vmin=-vmax, vmax=vmax,
    aspect="auto"
)
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Beta (factor loading)", fontsize=9)

ax.set_xticks(range(len(factors)))
ax.set_xticklabels(factors, fontsize=10)
ax.set_yticks(range(len(assets)))
ax.set_yticklabels(assets, fontsize=10)
ax.set_title("Fama-French 5 + Momentum: Factor Loadings (Betas)", fontsize=12, pad=10)

# Annotate each cell
for i, asset in enumerate(assets):
    for j, fac in enumerate(factors):
        val = betas_df.loc[asset, fac]
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center", fontsize=8,
            color="white" if abs(val) > 0.6 * vmax else "black"
        )

plt.tight_layout()
plt.savefig("Output/fa_loadings_heatmap.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8.  Visualisation B – R² Bar Chart (FF vs PCA comparison)
# ---------------------------------------------------------------------------
# PCA R² (k=3 factors) from Section 2.1 — computed directly from returns

X_pca        = ((returns - returns.mean()) / returns.std()).values
pca_model    = PCA()
pca_model.fit(X_pca)
eigvals_pca  = pca_model.explained_variance_
k_pca        = 3
pca_loadings = pd.DataFrame(
    pca_model.components_[:k_pca].T * np.sqrt(eigvals_pca[:k_pca]),
    index=returns.columns,
)
pca_r2_series = (pca_loadings ** 2).sum(axis=1).reindex(assets)

fig, ax = plt.subplots(figsize=(10, 4.5))
x       = np.arange(len(assets))
width   = 0.38

bars_ff  = ax.bar(x - width/2, r2_series.reindex(assets),
                  width, label="FF5+MOM (Section 2.2)",
                  color="steelblue", alpha=0.85)
bars_pca = ax.bar(x + width/2, pca_r2_series,
                  width, label="PCA k=3 (Section 2.1)",
                  color="darkorange", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(assets, fontsize=10)
ax.set_ylabel("$R^2$", fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_title("$R^2$ by Asset: FF5+Momentum vs PCA Factors", fontsize=12)
ax.legend(fontsize=9)
ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)

# Label each bar
for bar in bars_ff:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars_pca:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig("Output/fa_r2_comparison.png", dpi=150)
plt.show()
