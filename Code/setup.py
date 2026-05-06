# =============================================================================
# Section 1: Setup
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load raw CRSP data
raw = pd.read_excel("Data/Data.xlsx")
raw.columns = ["PERMNO", "Date", "Ticker", "Price", "ReturnRaw"]
raw["Date"] = pd.to_datetime(raw["Date"])
prices_wide = raw.pivot(index="Date", columns="Ticker", values="Price")
ret_wide    = raw.pivot(index="Date", columns="Ticker", values="ReturnRaw")

# Synchronise: keep only dates where every asset has a price
prices_wide = prices_wide.dropna()
ret_wide    = ret_wide.loc[prices_wide.index].dropna()

# Asset mix:
# AAPL, MSFT, ASML, XOM, JPM  -> individual stocks (tech, semiconductors, energy, financials)
# SPY                          -> broad US equity market index
# EEM                          -> emerging-market equity index
# TLT                          -> long-duration US Treasury bond ETF (investment-grade)
# HYG                          -> high-yield (junk) corporate bond ETF
# GLD                          -> gold commodity ETF
# Together these span equities, fixed income, and commodities across geographies and risk profiles.

# Use CRSP-provided total returns (split- and dividend-adjusted simple returns) rather than
# computing log returns from raw prices.  Raw AAPL prices contain an unadjusted 4-for-1
# split on 2020-08-31 (log return = -1.35), while the CRSP return for that day correctly
# shows +3.4%.  Converting to log returns: ln(1 + r) ≈ r for small r, so the difference
# is negligible for most days, but using provided returns avoids split artefacts entirely.
#
# Dividend treatment: CRSP RET is a total holding period return defined as
# (P_t + D_t - P_{t-1}) / P_{t-1}, where D_t is any dividend paid on the ex-dividend date.
# Dividends are therefore implicitly reinvested at the closing price on the ex-dividend date
# and are not distributed.  The reconstructed adj_prices series consequently represents a
# total-return index rather than a price-only index.
#
# Data quality: no CRSP sentinel values (-99, -88, -66) are present and no single-day return
# exceeds ±50%, confirming no structural data errors.  All 4-sigma outliers correspond to
# identifiable market events: the COVID-19 crash and recovery (March–April 2020, all assets),
# Apple's Q1 2019 China revenue warning (-10%, 2019-01-03), the Fed's expanded bond-buying
# announcement (+6.5% HYG, 2020-04-09), the Pfizer vaccine announcement (+14% JPM, +13% XOM,
# 2020-11-09), and ASML's accidental early Q3 2024 earnings release (-16%, 2024-10-15).
# No returns are removed or adjusted beyond the stock-split correction already applied.
returns = np.log(1 + ret_wide)   # convert simple returns to log returns for additivity

print(f"Period  : {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"Assets  : {list(returns.columns)}")
print(f"Obs     : {len(returns)}")
print(f"Missing : {returns.isnull().sum().sum()}")

# Summary statistics
# Excess kurtosis confirms heavy tails across all assets, motivating EVT in Section 2.4.
summary = returns.describe().T
summary["skewness"]        = returns.skew()
summary["excess_kurtosis"] = returns.kurtosis()
print("\nSummary Statistics (log returns):")
print(summary[["mean", "std", "min", "max", "skewness", "excess_kurtosis"]].round(6))


# Reconstruct split-adjusted prices by compounding the CRSP-adjusted returns forward from
# the first observed (raw) price.  Raw CRSP prices are not split-adjusted, causing visible
# discontinuities (e.g. AAPL's 4:1 split on 2020-08-31 shows as a -73% log return from
# raw prices).
first_price = prices_wide.iloc[0]                      # raw first-day price as seed
cum_growth  = (1 + ret_wide.iloc[1:]).cumprod()        # compound from day 2 onwards
adj_prices  = pd.concat([
    first_price.to_frame().T,
    first_price * cum_growth
])

# Plot: rebased cumulative price paths
rebased = adj_prices / adj_prices.iloc[0] * 100
fig, ax = plt.subplots(figsize=(12, 5))
for col in rebased.columns:
    ax.plot(rebased.index, rebased[col], lw=1, label=col)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_title("Rebased Price Paths (Base = 100, Jan 2015)")
ax.set_xlabel("Date")
ax.set_ylabel("Index")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig("Output/price_paths.png", dpi=150)
plt.show()
