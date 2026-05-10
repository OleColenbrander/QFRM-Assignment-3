# =============================================================================
# Section 2.4: EVT
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

returns = pd.read_excel("Data/cleaned_data.xlsx", index_col=0, parse_dates=True)

# 1. Heaviest tail
kurt = returns.kurtosis().sort_values(ascending=False)
asset = kurt.index[0]

print("\nExcess kurtosis:")
print(kurt.round(3))
print("\nSelected asset:", asset)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for ax, col in zip(axes.ravel(), returns.columns):
    r = returns[col].dropna()
    df, loc, scale = stats.t.fit(r)
    stats.probplot(r, dist=stats.t, sparams=(df, loc, scale), plot=ax)
    ax.set_title(f"{col}, kurt={returns[col].kurtosis():.2f}", fontsize=9)

plt.tight_layout()
plt.savefig("Output/evt_student_t_qq.png", dpi=150)
plt.show()

# 2. POT/GPD on losses
loss = -returns[asset].dropna()
n = len(loss)

u = loss.quantile(0.95)
excess = loss[loss > u] - u
Nu = len(excess)

xi, loc, beta = stats.genpareto.fit(excess, floc=0)

print("\nPOT/GPD:")
print("u:", round(u, 5))
print("Nu:", Nu)
print("xi:", round(xi, 4))
print("beta:", round(beta, 4))

thresholds = loss.quantile(np.linspace(0.80, 0.98, 30))
mean_excess = [(loss[loss > x] - x).mean() for x in thresholds]

plt.figure(figsize=(6, 4))
plt.plot(thresholds, mean_excess, marker="o")
plt.axvline(u, color="red", linestyle="--")
plt.title(f"Mean excess plot: {asset}")
plt.xlabel("Threshold")
plt.ylabel("Mean excess")
plt.tight_layout()
plt.savefig("Output/evt_mean_excess.png", dpi=150)
plt.show()

plt.figure(figsize=(5, 5))
stats.probplot(excess, dist=stats.genpareto, sparams=(xi, 0, beta), plot=plt)
plt.title(f"GPD QQ plot: {asset}")
plt.tight_layout()
plt.savefig("Output/evt_gpd_qq.png", dpi=150)
plt.show()

# 3. VaR and ES
def pot_var_es(alpha, u, xi, beta, Nu, n):
    p_u = Nu / n

    if abs(xi) < 1e-6:
        var = u - beta * np.log((1 - alpha) / p_u)
        es = var + beta
    else:
        var = u + beta / xi * ((p_u / (1 - alpha)) ** xi - 1)
        es = np.nan if xi >= 1 else (var + beta - xi * u) / (1 - xi)

    return var, es

levels = [0.95, 0.99]

print("\nVaR and ES:")
for a in levels:
    var_evt, es_evt = pot_var_es(a, u, xi, beta, Nu, n)

    var_hist = loss.quantile(a)
    es_hist = loss[loss > var_hist].mean()

    print(f"\nLevel {a:.0%}")
    print(f"EVT        VaR={100*var_evt:.3f}%   ES={100*es_evt:.3f}%")
    print(f"Historical VaR={100*var_hist:.3f}%   ES={100*es_hist:.3f}%")

# 4. GARCH-t comparison
r_pct = returns[asset].dropna() * 100

model = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT")
fit = model.fit(disp="off")

mu = fit.params["mu"]
nu = fit.params["nu"]
sigma = np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])

print("\nGARCH-t:")
for a in levels:
    z = stats.t.ppf(1 - a, df=nu)
    q = z * np.sqrt((nu - 2) / nu)

    var_garch = -(mu + sigma * q) / 100

    pdf_z = stats.t.pdf(z, df=nu)
    es_z = -((nu + z**2) / ((nu - 1) * (1 - a))) * pdf_z
    es_z *= np.sqrt((nu - 2) / nu)

    es_garch = -(mu + sigma * es_z) / 100

    print(f"Level {a:.0%}: VaR={100*var_garch:.3f}%   ES={100*es_garch:.3f}%")
