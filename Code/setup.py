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

returns = np.log(1 + ret_wide)

print(f"Period  : {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"Assets  : {list(returns.columns)}")
print(f"Obs     : {len(returns)}")
print(f"Missing : {returns.isnull().sum().sum()}")

summary = returns.describe().T
summary["skewness"]        = returns.skew()
summary["excess_kurtosis"] = returns.kurtosis()
print("\nSummary Statistics (log returns):")
print(summary[["mean", "std", "min", "max", "skewness", "excess_kurtosis"]].round(6))

first_price = prices_wide.iloc[0]
cum_growth  = (1 + ret_wide.iloc[1:]).cumprod()  
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

returns.to_excel("Data/cleaned_data.xlsx")
