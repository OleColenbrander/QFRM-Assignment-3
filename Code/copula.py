# =============================================================================
# Section 2.3: Copula Analysis
# Pairs: AAPL-MSFT | SPY-TLT | SPY-GLD
# Copulas: Gaussian, Student-t, Clayton, Gumbel
#
# Clayton and Gumbel fitting/sampling/PDF via the `copulas` package (DataCebo).
# Gaussian and Student-t remain scipy-based (not in copulas.bivariate).
# Install: pip install copulas
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import (norm, multivariate_normal,
                          t as tdist, multivariate_t,
                          kendalltau, spearmanr, pearsonr)
from copulas.bivariate import Clayton as ClaytonCop
from copulas.bivariate import Gumbel  as GumbelCop

np.random.seed(42)

# =============================================================================
# 1.  Load returns
# =============================================================================
raw = pd.read_excel("Data/Data.xlsx")
raw.columns = ["PERMNO", "Date", "Ticker", "Price", "ReturnRaw"]
raw["Date"] = pd.to_datetime(raw["Date"])
prices_wide = raw.pivot(index="Date", columns="Ticker", values="Price").dropna()
ret_wide    = raw.pivot(index="Date", columns="Ticker", values="ReturnRaw") \
                 .loc[prices_wide.index].dropna()
returns     = np.log(1 + ret_wide)

# =============================================================================
# 2.  Pseudo-observations
# =============================================================================
def pseudo_obs(series):
    """Rank-based uniform transform; denominator n+1 keeps values in (0,1)."""
    return series.rank() / (len(series) + 1)

# =============================================================================
# 3.  Gaussian and Student-t copula log-likelihoods
#     (copulas package bivariate module only has Archimedean families)
# =============================================================================
def ll_gaussian(rho, u, v):
    if abs(rho) >= 1.0:
        return -np.inf
    x, y = norm.ppf(u), norm.ppf(v)
    bvn  = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return float(np.sum(bvn.logpdf(np.c_[x, y]) - norm.logpdf(x) - norm.logpdf(y)))

def ll_t(params, u, v):
    rho, nu = float(params[0]), float(params[1])
    if abs(rho) >= 1.0 or nu <= 2.0:
        return -np.inf
    x, y = tdist.ppf(u, df=nu), tdist.ppf(v, df=nu)
    bvt  = multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu)
    return float(np.sum(bvt.logpdf(np.c_[x, y])
                        - tdist.logpdf(x, df=nu) - tdist.logpdf(y, df=nu)))

# =============================================================================
# 4.  Fit all four copulas → AIC / BIC
# =============================================================================
_THETA_MIN = {ClaytonCop: 1e-4, GumbelCop: 1.0}

def _fit_archimedean(CopClass, u, v):
    """Fit an Archimedean copula via the copulas package; return (cop, ll).

    Falls back to the minimum valid theta when the moment estimator produces
    an out-of-bounds value (e.g. Clayton with negatively dependent data).
    """
    data = np.column_stack([u, v])
    cop  = CopClass()
    try:
        cop.fit(data)
    except ValueError:
        cop.theta = _THETA_MIN[CopClass]
    ll = float(np.sum(np.log(np.maximum(cop.pdf(data), 1e-300))))
    return cop, ll

def _ll_from_cop(cop, data):
    return float(np.sum(np.log(np.maximum(cop.pdf(data), 1e-300))))

def fit_all(u, v):
    n   = len(u)
    out = {}

    # Gaussian  (scipy MLE — package has no bivariate Gaussian)
    res   = optimize.minimize_scalar(
        lambda r: -ll_gaussian(r, u, v),
        bounds=(-0.9999, 0.9999), method="bounded")
    rho_g = float(res.x)
    out["Gaussian"] = {"params": {"rho": rho_g},
                       "ll": ll_gaussian(rho_g, u, v), "k": 1}

    # Student-t  (scipy MLE)
    res      = optimize.minimize(
        lambda p: -ll_t(p, u, v), x0=[rho_g, 6.0],
        bounds=[(-0.9999, 0.9999), (2.01, 200.0)], method="L-BFGS-B")
    rho_t, nu_t = float(res.x[0]), float(res.x[1])
    out["Student-t"] = {"params": {"rho": rho_t, "nu": nu_t},
                        "ll": ll_t([rho_t, nu_t], u, v), "k": 2}

    # Clayton  (copulas package MLE)
    cop_c, ll_c = _fit_archimedean(ClaytonCop, u, v)
    out["Clayton"] = {"params": {"theta": cop_c.theta},
                      "ll": ll_c, "k": 1, "_cop": cop_c}

    # Gumbel  (copulas package MLE)
    cop_g, ll_g = _fit_archimedean(GumbelCop, u, v)
    out["Gumbel"] = {"params": {"theta": cop_g.theta},
                     "ll": ll_g, "k": 1, "_cop": cop_g}

    for d in out.values():
        d["AIC"] = 2 * d["k"] - 2 * d["ll"]
        d["BIC"] = np.log(n) * d["k"] - 2 * d["ll"]
    return out

def select_best(fit_dict):
    return min(fit_dict, key=lambda k: fit_dict[k]["AIC"])

# =============================================================================
# 5.  Standard errors via numerical Hessian
#     Clayton/Gumbel SEs use the package's log_likelihood with varying theta.
# =============================================================================
def _numerical_hessian(f, x0, eps=1e-5):
    x0 = np.atleast_1d(np.float64(x0))
    n  = len(x0)
    H  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
    return H

def standard_errors(name, params, u, v):
    data = np.column_stack([u, v])
    if name == "Gaussian":
        neg_ll = lambda p: -ll_gaussian(p[0], u, v)
        x0 = np.array([params["rho"]])
    elif name == "Student-t":
        neg_ll = lambda p: -ll_t(p, u, v)
        x0 = np.array([params["rho"], params["nu"]])
    elif name == "Clayton":
        def neg_ll(p):
            c = ClaytonCop(); c.theta = float(p[0])
            return -_ll_from_cop(c, data)
        x0 = np.array([params["theta"]])
    else:  # Gumbel
        def neg_ll(p):
            c = GumbelCop(); c.theta = float(p[0])
            return -_ll_from_cop(c, data)
        x0 = np.array([params["theta"]])

    H = _numerical_hessian(neg_ll, x0)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(len(x0), np.nan)
    return se

# =============================================================================
# 6.  Dependence measures
# =============================================================================
def _samples_from_cop(cop, n):
    """Extract (u_s, v_s) arrays from a copulas package sample call."""
    s = cop.sample(n)
    if isinstance(s, pd.DataFrame):
        u_s, v_s = s.iloc[:, 0].values, s.iloc[:, 1].values
    else:
        u_s, v_s = s[:, 0], s[:, 1]
    return np.clip(u_s, 1e-6, 1-1e-6), np.clip(v_s, 1e-6, 1-1e-6)

def _cross_tail(u_s, v_s, q=0.10):
    mask_low  = u_s <= q
    mask_high = u_s >= 1 - q
    lam_lu = ((mask_low)  & (v_s >= 1 - q)).sum() / max(mask_low.sum(),  1)
    lam_ul = ((mask_high) & (v_s <= q     )).sum() / max(mask_high.sum(), 1)
    return float(lam_lu), float(lam_ul)

def implied_measures(name, params, n_mc=100_000, q=0.10, fitted_cop=None):
    """Model-implied dependence measures; uses copulas package for Archimedean sampling."""
    if name == "Gaussian":
        rho      = params["rho"]
        tau      = 2.0 / np.pi * np.arcsin(rho)
        spearman = 6.0 / np.pi * np.arcsin(rho / 2.0)
        pearson  = spearman
        lam_l, lam_u = 0.0, 0.0
        xy  = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).rvs(n_mc)
        u_s = norm.cdf(xy[:, 0]);  v_s = norm.cdf(xy[:, 1])

    elif name == "Student-t":
        rho, nu  = params["rho"], params["nu"]
        tau      = 2.0 / np.pi * np.arcsin(rho)
        xy       = multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu).rvs(n_mc)
        u_s = tdist.cdf(xy[:, 0], df=nu);  v_s = tdist.cdf(xy[:, 1], df=nu)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        lam      = 2.0 * tdist.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
        lam_l, lam_u = lam, lam

    elif name == "Clayton":
        theta = params["theta"]
        tau   = theta / (theta + 2.0)
        cop   = fitted_cop if fitted_cop is not None else ClaytonCop()
        if fitted_cop is None:
            cop.theta = theta
        u_s, v_s = _samples_from_cop(cop, n_mc)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        lam_l    = 2.0**(-1.0 / theta);  lam_u = 0.0

    else:  # Gumbel
        theta = params["theta"]
        tau   = 1.0 - 1.0 / theta
        cop   = fitted_cop if fitted_cop is not None else GumbelCop()
        if fitted_cop is None:
            cop.theta = theta
        u_s, v_s = _samples_from_cop(cop, n_mc)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        lam_u    = 2.0 - 2.0**(1.0 / theta);  lam_l = 0.0

    lam_lu, lam_ul = _cross_tail(u_s, v_s, q=q)
    return {"tau": tau, "spearman": spearman, "pearson": pearson,
            "lam_l": lam_l, "lam_u": lam_u, "lam_lu": lam_lu, "lam_ul": lam_ul}

def empirical_measures(u, v, q=0.10):
    tau      = kendalltau(u, v).statistic
    spearman = spearmanr(u, v).statistic
    pearson  = pearsonr(u, v).statistic
    n_low    = (u <= q).sum();  n_high = (u >= 1 - q).sum()
    lam_l    = ((u <= q)     & (v <= q    )).sum() / max(n_low,  1)
    lam_u    = ((u >= 1 - q) & (v >= 1 - q)).sum() / max(n_high, 1)
    lam_lu   = ((u <= q)     & (v >= 1 - q)).sum() / max(n_low,  1)
    lam_ul   = ((u >= 1 - q) & (v <= q    )).sum() / max(n_high, 1)
    return {"pearson": pearson, "tau": tau, "spearman": spearman,
            "lam_l": lam_l, "lam_u": lam_u, "lam_lu": lam_lu, "lam_ul": lam_ul}

# =============================================================================
# 7.  Copula density grid for contour plots
#     Clayton/Gumbel use cop.pdf(); Gaussian/Student-t use scipy.
# =============================================================================
def _pdf_from_cop(cop, uf, vf, g):
    """Evaluate copulas-package PDF on a flattened grid; return (g, g) array."""
    grid = np.column_stack([uf, vf])
    pdf  = cop.pdf(grid)
    return np.asarray(pdf).reshape(g, g)

def copula_density_grid(name, params, g=120, fitted_cop=None):
    eps = 2e-3
    pts = np.linspace(eps, 1 - eps, g)
    U, V = np.meshgrid(pts, pts)
    uf, vf = U.ravel(), V.ravel()

    if name == "Gaussian":
        rho   = params["rho"]
        x, y  = norm.ppf(uf), norm.ppf(vf)
        bvn   = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        density = np.exp(bvn.logpdf(np.c_[x, y]) - norm.logpdf(x) - norm.logpdf(y)).reshape(g, g)

    elif name == "Student-t":
        rho, nu = params["rho"], params["nu"]
        x, y    = tdist.ppf(uf, df=nu), tdist.ppf(vf, df=nu)
        bvt     = multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu)
        density = np.exp(bvt.logpdf(np.c_[x, y])
                         - tdist.logpdf(x, df=nu) - tdist.logpdf(y, df=nu)).reshape(g, g)

    elif name == "Clayton":
        cop = fitted_cop if fitted_cop is not None else ClaytonCop()
        if fitted_cop is None:
            cop.theta = params["theta"]
        density = _pdf_from_cop(cop, uf, vf, g)

    else:  # Gumbel
        cop = fitted_cop if fitted_cop is not None else GumbelCop()
        if fitted_cop is None:
            cop.theta = params["theta"]
        density = _pdf_from_cop(cop, uf, vf, g)

    cap = np.nanpercentile(density, 99)
    return U, V, np.clip(density, 0, cap)

# =============================================================================
# 8.  Main loop over the three pairs
# =============================================================================
PAIRS = [
    ("AAPL", "MSFT", "AAPL vs MSFT  (Tech, Same Class)"),
    ("SPY",  "TLT",  "SPY vs TLT  (Equity vs Bond)"),
    ("SPY",  "GLD",  "SPY vs GLD  (Equity vs Commodity)"),
]

all_summary = []

for (a1, a2, title) in PAIRS:
    print(f"\n{'='*65}")
    print(f"  Pair: {title}")
    print(f"{'='*65}")

    u = pseudo_obs(returns[a1]).values
    v = pseudo_obs(returns[a2]).values

    fits     = fit_all(u, v)
    best     = select_best(fits)
    best_cop = fits[best].get("_cop")   # fitted copulas-package object (or None)

    print("\n  AIC / BIC across copulas:")
    print(f"  {'Copula':<12} {'LogLik':>10}  {'AIC':>10}  {'BIC':>10}  Params")
    for cname, d in fits.items():
        param_str = "  ".join(f"{k}={v_:.4f}" for k, v_ in d["params"].items())
        marker    = "  <-- best" if cname == best else ""
        print(f"  {cname:<12} {d['ll']:>10.2f}  {d['AIC']:>10.2f}  "
              f"{d['BIC']:>10.2f}  {param_str}{marker}")

    best_params = fits[best]["params"]
    se = standard_errors(best, best_params, u, v)
    print(f"\n  Best copula: {best}")
    for (pname, pval), pse in zip(best_params.items(), se):
        print(f"    {pname} = {pval:.5f}  (SE = {pse:.5f})")

    emp = empirical_measures(u, v, q=0.10)
    imp = implied_measures(best, best_params, fitted_cop=best_cop)

    print("\n  Dependence measures (empirical vs model-implied):")
    print(f"  {'Measure':<26} {'Empirical':>12}  {'Model-Implied':>14}")
    print(f"  {'-'*54}")
    measures = [("Pearson corr",            "pearson"),
                ("Kendall tau",             "tau"),
                ("Spearman rho",            "spearman"),
                ("Lower tail dep",          "lam_l"),
                ("Upper tail dep",          "lam_u"),
                ("Cross-tail (U lo, V hi)", "lam_lu"),
                ("Cross-tail (U hi, V lo)", "lam_ul")]
    for label, key in measures:
        print(f"  {label:<26} {emp[key]:>12.4f}  {imp[key]:>14.4f}")

    all_summary.append({
        "Pair": f"{a1}-{a2}", "Best": best, "AIC": round(fits[best]["AIC"], 1),
        **{f"emp_{k}": round(emp[k], 4)
           for k in ["pearson","tau","spearman","lam_l","lam_u","lam_lu","lam_ul"]},
        **{f"imp_{k}": round(imp[k], 4)
           for k in ["pearson","tau","spearman","lam_l","lam_u","lam_lu","lam_ul"]},
    })

    # -- plots --
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.scatter(u, v, s=2, alpha=0.25, color="steelblue", rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"$U_{{{a1}}}$  (pseudo-obs)", fontsize=10)
    ax.set_ylabel(f"$U_{{{a2}}}$  (pseudo-obs)", fontsize=10)
    ax.set_title("Empirical pseudo-observations", fontsize=10)

    ax = axes[1]
    U_grid, V_grid, dens = copula_density_grid(best, best_params, g=120, fitted_cop=best_cop)
    levels = np.linspace(0, dens.max(), 20)
    cf = ax.contourf(U_grid, V_grid, dens, levels=levels, cmap="Blues")
    ax.contour(U_grid, V_grid, dens, levels=levels[1::3],
               colors="navy", linewidths=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Copula density")
    ax.set_xlabel(f"$U_{{{a1}}}$", fontsize=10)
    ax.set_ylabel(f"$U_{{{a2}}}$", fontsize=10)
    param_str = "  ".join(f"{k}={v_:.3f}" for k, v_ in best_params.items())
    ax.set_title(f"Best fit: {best}  [{param_str}]", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"Output/copula_{a1.lower()}_{a2.lower()}.png", dpi=150)
    plt.show()

# =============================================================================
# 9.  Summary table figure
# =============================================================================
summary_df = pd.DataFrame(all_summary)
print("\n\nSummary across all pairs:")
print(summary_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 2.2))
ax.axis("off")
cols = ["Pair", "Best", "AIC",
        "emp_pearson", "imp_pearson", "emp_tau",    "imp_tau",
        "emp_lam_l",   "imp_lam_l",   "emp_lam_u",  "imp_lam_u",
        "emp_lam_lu",  "imp_lam_lu",  "emp_lam_ul", "imp_lam_ul"]
col_labels = ["Pair", "Best copula", "AIC",
              "rho emp", "rho model", "tau emp", "tau model",
              "lam_L emp", "lam_L model", "lam_U emp", "lam_U model",
              "lam_LU emp", "lam_LU model", "lam_UL emp", "lam_UL model"]
tbl = ax.table(cellText=summary_df[cols].values, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)
plt.tight_layout()
plt.savefig("Output/copula_summary_table.png", dpi=150, bbox_inches="tight")
plt.show()
