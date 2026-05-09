# =============================================================================
# Section 2.3: Copula Analysis
# Pairs: AAPL-MSFT | SPY-TLT | SPY-GLD
# Copulas: Gaussian, Student-t, Clayton, Gumbel
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from scipy.stats import (norm, multivariate_normal,
                          t as tdist, multivariate_t,
                          kendalltau, spearmanr, pearsonr)

np.random.seed(42)

# =============================================================================
# 1.  Load returns  (same pipeline as pca.py / factor_analysis.py)
# =============================================================================
raw = pd.read_excel("Data/Data.xlsx")
raw.columns = ["PERMNO", "Date", "Ticker", "Price", "ReturnRaw"]
raw["Date"] = pd.to_datetime(raw["Date"])
prices_wide = raw.pivot(index="Date", columns="Ticker", values="Price").dropna()
ret_wide    = raw.pivot(index="Date", columns="Ticker", values="ReturnRaw") \
                 .loc[prices_wide.index].dropna()
returns     = np.log(1 + ret_wide)

# =============================================================================
# 2.  Pseudo-observations  (empirical CDF transform, Hazen formula)
# =============================================================================
def pseudo_obs(series):
    """Map a return series to U[0,1] via rank / (n+1).

    Using n+1 in the denominator ensures pseudo-obs are strictly inside (0,1),
    which is required because copula densities diverge at the boundary.
    """
    return series.rank() / (len(series) + 1)

# =============================================================================
# 3.  Log-likelihood functions for four bivariate copulas
# =============================================================================

def ll_gaussian(rho, u, v):
    """Bivariate Gaussian copula log-likelihood."""
    if abs(rho) >= 1.0:
        return -np.inf
    x, y = norm.ppf(u), norm.ppf(v)
    bvn   = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return float(np.sum(bvn.logpdf(np.c_[x, y])
                        - norm.logpdf(x) - norm.logpdf(y)))


def ll_t(params, u, v):
    """Bivariate Student-t copula log-likelihood (rho, nu)."""
    rho, nu = float(params[0]), float(params[1])
    if abs(rho) >= 1.0 or nu <= 2.0:
        return -np.inf
    x, y  = tdist.ppf(u, df=nu), tdist.ppf(v, df=nu)
    bvt   = multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu)
    return float(np.sum(bvt.logpdf(np.c_[x, y])
                        - tdist.logpdf(x, df=nu) - tdist.logpdf(y, df=nu)))


def ll_clayton(theta, u, v):
    """Bivariate Clayton copula log-likelihood (theta > 0, lower tail dep)."""
    theta = float(theta)
    if theta <= 0:
        return -np.inf
    A = u**(-theta) + v**(-theta) - 1.0
    if np.any(A <= 0):
        return -np.inf
    log_c = (np.log(theta + 1)
             - (theta + 1) * (np.log(u) + np.log(v))
             - (2.0 + 1.0 / theta) * np.log(A))
    return float(np.sum(log_c))


def ll_gumbel(theta, u, v):
    """Bivariate Gumbel copula log-likelihood (theta >= 1, upper tail dep).

    Density (Nelsen 2006, p.116):
        c(u,v) = C(u,v) * A^(theta-1) * B^(theta-1) / (u*v)
                 * S^(1/theta - 2) * (S^(1/theta) + theta - 1)
    where A=-log(u), B=-log(v), S = A^theta + B^theta.
    """
    theta = float(theta)
    if theta < 1.0:
        return -np.inf
    A = -np.log(u)
    B = -np.log(v)
    S = A**theta + B**theta
    if np.any(S <= 0):
        return -np.inf
    C_val = np.exp(-S**(1.0 / theta))
    log_c = (np.log(C_val)
             + (theta - 1) * (np.log(A) + np.log(B))
             + (1.0 / theta - 2.0) * np.log(S)
             + np.log(S**(1.0 / theta) + theta - 1.0)
             - np.log(u) - np.log(v))
    return float(np.sum(log_c))

# =============================================================================
# 4.  Fit all four copulas and rank by AIC / BIC
# =============================================================================

def fit_all(u, v):
    """Return dict of copula results keyed by name."""
    n   = len(u)
    out = {}

    # Gaussian  (1 parameter)
    res = optimize.minimize_scalar(
        lambda r: -ll_gaussian(r, u, v),
        bounds=(-0.9999, 0.9999), method="bounded")
    rho_g = float(res.x)
    ll_g  = ll_gaussian(rho_g, u, v)
    out["Gaussian"] = {"params": {"rho": rho_g}, "ll": ll_g, "k": 1}

    # Student-t  (2 parameters: rho, nu)
    res = optimize.minimize(
        lambda p: -ll_t(p, u, v),
        x0=[rho_g, 6.0],
        bounds=[(-0.9999, 0.9999), (2.01, 200.0)],
        method="L-BFGS-B")
    rho_t, nu_t = float(res.x[0]), float(res.x[1])
    ll_t_ = ll_t([rho_t, nu_t], u, v)
    out["Student-t"] = {"params": {"rho": rho_t, "nu": nu_t}, "ll": ll_t_, "k": 2}

    # Clayton  (1 parameter: theta > 0)
    res = optimize.minimize_scalar(
        lambda t: -ll_clayton(t, u, v),
        bounds=(1e-4, 50.0), method="bounded")
    th_c = float(res.x)
    ll_c = ll_clayton(th_c, u, v)
    out["Clayton"] = {"params": {"theta": th_c}, "ll": ll_c, "k": 1}

    # Gumbel  (1 parameter: theta >= 1)
    res = optimize.minimize_scalar(
        lambda t: -ll_gumbel(t, u, v),
        bounds=(1.0, 50.0), method="bounded")
    th_g = float(res.x)
    ll_g2 = ll_gumbel(th_g, u, v)
    out["Gumbel"] = {"params": {"theta": th_g}, "ll": ll_g2, "k": 1}

    # AIC and BIC
    for d in out.values():
        d["AIC"] = 2 * d["k"] - 2 * d["ll"]
        d["BIC"] = np.log(n) * d["k"] - 2 * d["ll"]

    return out


def select_best(fit_dict):
    """Return name of copula with the lowest AIC."""
    return min(fit_dict, key=lambda k: fit_dict[k]["AIC"])

# =============================================================================
# 5.  Standard errors via numerical Hessian of the neg-log-likelihood
# =============================================================================

def _numerical_hessian(f, x0, eps=1e-5):
    """Finite-difference Hessian of scalar f at vector x0."""
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
    """MLE standard errors from the inverse Hessian."""
    if name == "Gaussian":
        neg_ll = lambda p: -ll_gaussian(p[0], u, v)
        x0 = np.array([params["rho"]])
    elif name == "Student-t":
        neg_ll = lambda p: -ll_t(p, u, v)
        x0 = np.array([params["rho"], params["nu"]])
    elif name == "Clayton":
        neg_ll = lambda p: -ll_clayton(p[0], u, v)
        x0 = np.array([params["theta"]])
    else:  # Gumbel
        neg_ll = lambda p: -ll_gumbel(p[0], u, v)
        x0 = np.array([params["theta"]])

    H = _numerical_hessian(neg_ll, x0)
    try:
        cov = np.linalg.inv(H)
        se  = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(x0), np.nan)
    return se

# =============================================================================
# 6.  Copula-implied dependence measures  (analytical + MC simulation)
# =============================================================================

def _simulate_positive_stable(alpha, size, rng):
    """Draw from a totally-skewed positive stable(alpha, 1) distribution.

    Used in the Marshall-Olkin algorithm to simulate Gumbel copulas.
    Chambers-Mallows-Stuck parametrisation: E[exp(-t*S)] = exp(-t^alpha).
    """
    if alpha >= 1.0 - 1e-9:         # independence limit (theta=1)
        return np.ones(size)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W   = rng.exponential(1.0, size)
    num = np.sin(alpha * (phi + np.pi / 2))
    den = np.cos(phi)
    S   = (num / den**(1.0 / alpha)) * (np.cos(phi - alpha * (phi + np.pi / 2)) / W)**((1 - alpha) / alpha)
    return np.maximum(S, 1e-10)


def _sim_clayton(theta, n, rng):
    """Simulate n observations from Clayton copula via conditional inverse."""
    U = rng.uniform(0, 1, n)
    W = rng.uniform(0, 1, n)
    # Closed-form inversion of the conditional CDF C_{2|1}(v|u; theta)
    V = (W**(-theta / (1.0 + theta)) - 1.0 + U**(-theta))**(-1.0 / theta)
    return U, np.clip(V, 1e-6, 1 - 1e-6)


def _sim_gumbel(theta, n, rng):
    """Simulate n observations from Gumbel copula (Marshall-Olkin method)."""
    alpha = 1.0 / theta
    S     = _simulate_positive_stable(alpha, n, rng)
    E1    = rng.exponential(1.0, n)
    E2    = rng.exponential(1.0, n)
    U     = np.exp(-(E1 / S)**alpha)
    V     = np.exp(-(E2 / S)**alpha)
    return np.clip(U, 1e-6, 1 - 1e-6), np.clip(V, 1e-6, 1 - 1e-6)


def _cross_tail(u_s, v_s, q=0.10):
    """Cross-tail conditional probabilities from simulated (u_s, v_s).

    lam_lu = P(V >= 1-q | U <= q)  — asset 1 crashes while asset 2 rallies.
    lam_ul = P(V <= q   | U >= 1-q) — asset 1 rallies while asset 2 crashes.
    """
    mask_low  = u_s <= q
    mask_high = u_s >= 1 - q
    lam_lu = ((mask_low)  & (v_s >= 1 - q)).sum() / max(mask_low.sum(),  1)
    lam_ul = ((mask_high) & (v_s <= q     )).sum() / max(mask_high.sum(), 1)
    return float(lam_lu), float(lam_ul)


def implied_measures(name, params, n_mc=100_000, seed=42, q=0.10):
    """Model-implied dependence measures via analytical formulae + MC.

    Always simulates n_mc draws from the copula so that cross-tail
    coefficients (lam_lu, lam_ul) are available for every copula family.
    """
    rng = np.random.default_rng(seed)

    if name == "Gaussian":
        rho      = params["rho"]
        tau      = 2.0 / np.pi * np.arcsin(rho)
        spearman = 6.0 / np.pi * np.arcsin(rho / 2.0)   # closed form
        pearson  = spearman                               # uniform margins
        lam_l, lam_u = 0.0, 0.0
        # Simulate for cross-tail
        xy  = multivariate_normal(mean=[0, 0],
                                   cov=[[1, rho], [rho, 1]]).rvs(n_mc)
        u_s = norm.cdf(xy[:, 0])
        v_s = norm.cdf(xy[:, 1])

    elif name == "Student-t":
        rho, nu  = params["rho"], params["nu"]
        tau      = 2.0 / np.pi * np.arcsin(rho)          # same as Gaussian
        xy       = multivariate_t(loc=[0, 0],
                                  shape=[[1, rho], [rho, 1]], df=nu).rvs(n_mc)
        u_s = tdist.cdf(xy[:, 0], df=nu)
        v_s = tdist.cdf(xy[:, 1], df=nu)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        # Symmetric tail dependence (Joe 1997)
        lam      = 2.0 * tdist.cdf(
            -np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
        lam_l, lam_u = lam, lam

    elif name == "Clayton":
        theta    = params["theta"]
        tau      = theta / (theta + 2.0)                  # closed form
        u_s, v_s = _sim_clayton(theta, n_mc, rng)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        lam_l    = 2.0**(-1.0 / theta)
        lam_u    = 0.0

    else:  # Gumbel
        theta    = params["theta"]
        tau      = 1.0 - 1.0 / theta                     # closed form
        u_s, v_s = _sim_gumbel(theta, n_mc, rng)
        spearman = spearmanr(u_s, v_s).statistic
        pearson  = pearsonr(u_s, v_s).statistic
        lam_u    = 2.0 - 2.0**(1.0 / theta)
        lam_l    = 0.0

    lam_lu, lam_ul = _cross_tail(u_s, v_s, q=q)

    return {"tau": tau, "spearman": spearman, "pearson": pearson,
            "lam_l": lam_l, "lam_u": lam_u,
            "lam_lu": lam_lu, "lam_ul": lam_ul}


def empirical_measures(u, v, q=0.10):
    """Empirical Pearson, Kendall, Spearman, same-tail and cross-tail dependence."""
    tau      = kendalltau(u, v).statistic
    spearman = spearmanr(u, v).statistic
    pearson  = pearsonr(u, v).statistic

    n_low  = (u <= q).sum()
    n_high = (u >= 1 - q).sum()

    # Same-tail: both low / both high
    lam_l = ((u <= q)     & (v <= q    )).sum() / max(n_low,  1)
    lam_u = ((u >= 1 - q) & (v >= 1 - q)).sum() / max(n_high, 1)

    # Cross-tail: one low while the other is high
    lam_lu = ((u <= q)     & (v >= 1 - q)).sum() / max(n_low,  1)  # U low, V high
    lam_ul = ((u >= 1 - q) & (v <= q    )).sum() / max(n_high, 1)  # U high, V low

    return {"pearson": pearson, "tau": tau, "spearman": spearman,
            "lam_l": lam_l, "lam_u": lam_u,
            "lam_lu": lam_lu, "lam_ul": lam_ul}

# =============================================================================
# 7.  Copula density on a grid  (for contour plots)
# =============================================================================

def copula_density_grid(name, params, g=120):
    """Evaluate the copula density on a g x g grid in (eps, 1-eps)^2."""
    eps = 2e-3
    pts = np.linspace(eps, 1 - eps, g)
    U, V = np.meshgrid(pts, pts)
    uf, vf = U.ravel(), V.ravel()

    if name == "Gaussian":
        rho   = params["rho"]
        x, y  = norm.ppf(uf), norm.ppf(vf)
        bvn   = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        log_c = bvn.logpdf(np.c_[x, y]) - norm.logpdf(x) - norm.logpdf(y)

    elif name == "Student-t":
        rho, nu = params["rho"], params["nu"]
        x, y    = tdist.ppf(uf, df=nu), tdist.ppf(vf, df=nu)
        bvt     = multivariate_t(loc=[0, 0],
                                  shape=[[1, rho], [rho, 1]], df=nu)
        log_c   = (bvt.logpdf(np.c_[x, y])
                   - tdist.logpdf(x, df=nu) - tdist.logpdf(y, df=nu))

    elif name == "Clayton":
        theta = params["theta"]
        A     = uf**(-theta) + vf**(-theta) - 1.0
        A     = np.maximum(A, 1e-300)
        log_c = (np.log(theta + 1)
                 - (theta + 1) * (np.log(uf) + np.log(vf))
                 - (2.0 + 1.0 / theta) * np.log(A))

    else:  # Gumbel
        theta = params["theta"]
        A     = -np.log(uf)
        B     = -np.log(vf)
        S     = A**theta + B**theta
        C_val = np.exp(-S**(1.0 / theta))
        log_c = (np.log(C_val)
                 + (theta - 1) * (np.log(A) + np.log(B))
                 + (1.0 / theta - 2.0) * np.log(S)
                 + np.log(S**(1.0 / theta) + theta - 1.0)
                 - np.log(uf) - np.log(vf))

    density = np.exp(log_c).reshape(g, g)
    # Cap at 99th percentile to prevent boundary spikes from swamping the plot
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

all_summary = []   # collect rows for the final table

for (a1, a2, title) in PAIRS:
    print(f"\n{'='*65}")
    print(f"  Pair: {title}")
    print(f"{'='*65}")

    # --- pseudo-observations ------------------------------------------------
    u = pseudo_obs(returns[a1]).values
    v = pseudo_obs(returns[a2]).values

    # --- fit ----------------------------------------------------------------
    fits   = fit_all(u, v)
    best   = select_best(fits)

    print("\n  AIC / BIC across copulas:")
    print(f"  {'Copula':<12} {'LogLik':>10}  {'AIC':>10}  {'BIC':>10}  "
          f"{'Params'}")
    for cname, d in fits.items():
        param_str = "  ".join(f"{k}={v_:.4f}" for k, v_ in d["params"].items())
        marker = "  <-- best" if cname == best else ""
        print(f"  {cname:<12} {d['ll']:>10.2f}  {d['AIC']:>10.2f}  "
              f"{d['BIC']:>10.2f}  {param_str}{marker}")

    # --- standard errors for the best copula --------------------------------
    best_params = fits[best]["params"]
    se = standard_errors(best, best_params, u, v)
    print(f"\n  Best copula: {best}")
    for (pname, pval), pse in zip(best_params.items(), se):
        print(f"    {pname} = {pval:.5f}  (SE = {pse:.5f})")

    # --- empirical dependence measures --------------------------------------
    emp = empirical_measures(u, v, q=0.10)

    # --- model-implied dependence measures ----------------------------------
    imp = implied_measures(best, best_params)

    print("\n  Dependence measures (empirical vs model-implied):")
    print(f"  {'Measure':<22} {'Empirical':>12}  {'Model-Implied':>14}")
    print(f"  {'-'*50}")
    measures = [("Pearson corr",        "pearson"),
                ("Kendall tau",         "tau"),
                ("Spearman rho",        "spearman"),
                ("Lower tail dep",      "lam_l"),
                ("Upper tail dep",      "lam_u"),
                ("Cross-tail (U lo, V hi)", "lam_lu"),
                ("Cross-tail (U hi, V lo)", "lam_ul")]
    for label, key in measures:
        print(f"  {label:<22} {emp[key]:>12.4f}  {imp[key]:>14.4f}")

    # store for summary table
    all_summary.append({
        "Pair": f"{a1}-{a2}",
        "Best": best,
        "AIC": round(fits[best]["AIC"], 1),
        **{f"emp_{k}": round(emp[k], 4)
           for k in ["pearson","tau","spearman","lam_l","lam_u","lam_lu","lam_ul"]},
        **{f"imp_{k}": round(imp[k], 4)
           for k in ["pearson","tau","spearman","lam_l","lam_u","lam_lu","lam_ul"]},
    })

    # =========================================================================
    # Figure: scatter of pseudo-obs (left) + copula density contour (right)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # -- left: scatter of pseudo-observations --
    ax = axes[0]
    ax.scatter(u, v, s=2, alpha=0.25, color="steelblue", rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"$U_{{{a1}}}$  (pseudo-obs)", fontsize=10)
    ax.set_ylabel(f"$U_{{{a2}}}$  (pseudo-obs)", fontsize=10)
    ax.set_title("Empirical pseudo-observations", fontsize=10)

    # -- right: contour of best-fit copula density --
    ax = axes[1]
    U_grid, V_grid, dens = copula_density_grid(best, best_params, g=120)
    levels = np.linspace(0, dens.max(), 20)
    cf = ax.contourf(U_grid, V_grid, dens, levels=levels, cmap="Blues")
    ax.contour(U_grid, V_grid, dens, levels=levels[1::3], colors="navy",
               linewidths=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Copula density")
    ax.set_xlabel(f"$U_{{{a1}}}$", fontsize=10)
    ax.set_ylabel(f"$U_{{{a2}}}$", fontsize=10)

    # Annotate with fitted parameters
    param_str = "  ".join(f"{k}={v_:.3f}" for k, v_ in best_params.items())
    ax.set_title(f"Best fit: {best}  [{param_str}]", fontsize=10)

    plt.tight_layout()
    tag = f"{a1.lower()}_{a2.lower()}"
    plt.savefig(f"Output/copula_{tag}.png", dpi=150)
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
        "emp_pearson", "imp_pearson",
        "emp_tau",     "imp_tau",
        "emp_lam_l",   "imp_lam_l",
        "emp_lam_u",   "imp_lam_u",
        "emp_lam_lu",  "imp_lam_lu",
        "emp_lam_ul",  "imp_lam_ul"]
col_labels = ["Pair", "Best copula", "AIC",
              "rho emp", "rho model",
              "tau emp", "tau model",
              "lam_L emp","lam_L model",
              "lam_U emp","lam_U model",
              "lam_LU emp","lam_LU model",
              "lam_UL emp","lam_UL model"]
tbl_data = summary_df[cols].values
tbl = ax.table(cellText=tbl_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)
plt.tight_layout()
plt.savefig("Output/copula_summary_table.png", dpi=150, bbox_inches="tight")
plt.show()
