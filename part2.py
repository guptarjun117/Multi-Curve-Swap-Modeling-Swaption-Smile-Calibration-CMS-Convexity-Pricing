"""
Part II - Swaption Calibration: Displaced-Diffusion and SABR Models

Requires part1.py in the same directory.
"""

import sys, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# Import bootstrapped curves from Part I
sys.path.insert(0, os.path.dirname(__file__))
from part1 import (load_libor, load_ois_sofr, load_term_sofr,
                   bootstrap_ois, make_interp)

EXCEL = r"Swap and Swaption Markets.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Load swaption data
# ─────────────────────────────────────────────────────────────────────────────

def load_swaptions():
    """
    Returns a DataFrame with columns:
      Expiry (float, years), Tenor (float, years),
      strikes (list of floats, as ATM offsets in decimal),
      ivs (list of floats, as fractions, e.g. 0.225 for 22.5%)
    """
    df = pd.read_excel(EXCEL, sheet_name="Swaption", header=None)
    # Row 2 has headers: Expiry, Tenor, -200bps, ..., +200bps
    # Row 3+ has data
    offsets_bps = [-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]
    offsets_dec = [o / 10000 for o in offsets_bps]  # in decimal

    records = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        exp = row[0]
        ten = row[1]
        ivs_pct = [row[2+j] for j in range(11)]  # in percent
        if pd.isna(exp) or pd.isna(ten):
            continue

        def parse_yr(x):
            x = str(x).strip().upper()
            return float(x.replace("Y", ""))

        records.append({
            "expiry": parse_yr(exp),
            "tenor":  parse_yr(ten),
            "offsets": offsets_dec,
            "ivs":     [v / 100.0 for v in ivs_pct]  # convert % → decimal
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Black-76 formulas
# ─────────────────────────────────────────────────────────────────────────────

def black_price(F, K, sigma, T, is_call=True):
    """Black-76 price. is_call=True → payer swaption; False → receiver."""
    if K <= 0 or F <= 0 or sigma <= 0 or T <= 0:
        return max(0.0, (F - K) if is_call else (K - F))
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def black_iv(price, F, K, T, is_call=True, tol=1e-10, max_iter=200):
    """Implied vol from Black-76 price. Returns NaN if no solution."""
    intrinsic = max(0.0, (F - K) if is_call else (K - F))
    if price < intrinsic - 1e-10:
        return np.nan
    if price < intrinsic + 1e-12:
        return 1e-8  # essentially at intrinsic

    def f(sigma):
        return black_price(F, K, sigma, T, is_call) - price

    try:
        lo, hi = 1e-6, 20.0
        if f(lo) > 0:
            return lo
        if f(hi) < 0:
            return hi
        return brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Forward swap rate and annuity (OIS-discounted, semi-annual)
# ─────────────────────────────────────────────────────────────────────────────

def annuity_and_fsr(T_exp, tenor, ois_fn):
    """
    Compute semi-annual annuity A and forward swap rate S.
    A = 0.5 * Σ Do(T_exp + k*0.5) for k=1,...,2*tenor
    S = (Do(T_exp) - Do(T_exp + tenor)) / A
    """
    dates = np.arange(T_exp + 0.5, T_exp + tenor + 1e-9, 0.5)
    A = 0.5 * sum(float(ois_fn(t)) for t in dates)
    Do_start = float(ois_fn(T_exp))
    Do_end   = float(ois_fn(T_exp + tenor))
    S = (Do_start - Do_end) / A
    return A, S


# ─────────────────────────────────────────────────────────────────────────────
# Displaced-Diffusion Model
# ─────────────────────────────────────────────────────────────────────────────

def dd_iv(K, S, beta, sigma_dd, T):
    """
    Implied Black vol for displaced-diffusion model.
    Model: (S+β) is log-normal with vol σ_DD.
    Price = BlackCall(S+β, K+β, σ_DD, T)  → invert to Black vol at (S, K).
    """
    if K + beta <= 0 or S + beta <= 0:
        return np.nan
    price_dd = black_price(S + beta, K + beta, sigma_dd, T, is_call=True)
    # Invert to Black vol on (S, K)
    # For put: Black put-call parity: payer - receiver = A*(S-K), so IV is same
    if K <= 0 or S <= 0:
        return np.nan
    return black_iv(price_dd, S, K, T, is_call=True)


def sigma_dd_from_atm(S, beta, atm_iv, T):
    """
    Given beta, find sigma_DD such that the displaced-diffusion ATM price
    equals the market ATM price (exact inversion via brentq).
    """
    atm_price_mkt = black_price(S, S, atm_iv, T, is_call=True)
    F_shift = S + beta

    def eq(sig):
        return black_price(F_shift, F_shift, sig, T, is_call=True) - atm_price_mkt

    try:
        return brentq(eq, 1e-6, 10.0, xtol=1e-10, maxiter=100)
    except Exception:
        return atm_iv * S / F_shift  # approximate fallback


def calibrate_dd_smile(S, offsets, mkt_ivs, T):
    """
    Calibrate (beta, sigma_DD) with ATM constraint.
    sigma_DD is pinned by ATM market price for each candidate beta.
    Optimize beta (1-D) to minimise price SSE across all 11 strikes.
    Returns (sigma_dd, beta).
    """
    K_arr  = np.array([S + o for o in offsets])
    iv_arr = np.array(mkt_ivs)
    atm_iv = iv_arr[5]  # index 5 = ATM (offset=0)

    # Precompute market prices
    mkt_prices = np.array([black_price(S, K, iv, T, is_call=True)
                           for K, iv in zip(K_arr, iv_arr)])

    from scipy.optimize import minimize_scalar

    def price_sse(log_beta):
        beta_val = np.exp(log_beta)
        sig_val  = sigma_dd_from_atm(S, beta_val, atm_iv, T)
        err = 0.0
        for i, K in enumerate(K_arr):
            if K + beta_val <= 0 or K <= 0:
                continue
            p_dd = black_price(S + beta_val, K + beta_val, sig_val, T, is_call=True)
            err += (p_dd - mkt_prices[i]) ** 2
        return err

    # beta in [~0.001%, 15%] in log-space: (-9.2, -1.9)
    lb_lo, lb_hi = -9.2, -1.9
    try:
        res = minimize_scalar(price_sse,
                              bounds=(lb_lo, lb_hi),
                              method="bounded",
                              options={"xatol": 1e-8, "maxiter": 1000})
        best_beta = float(np.exp(np.clip(res.x, lb_lo, lb_hi)))
    except Exception:
        best_beta = 0.005

    sigma_out = sigma_dd_from_atm(S, best_beta, atm_iv, T)
    return sigma_out, best_beta  # (sigma_dd, beta)


# ─────────────────────────────────────────────────────────────────────────────
# SABR Model (Hagan 2002)
# ─────────────────────────────────────────────────────────────────────────────

def sabr_vol(K, F, T, alpha, beta, rho, nu):
    """
    Hagan (2002) SABR lognormal vol approximation.
    β is fixed at 0.75 in Part II.2.
    """
    if F <= 0 or K <= 0 or alpha <= 0 or T <= 0:
        return np.nan

    # Small-strike / negative-strike guard
    eps = 1e-8

    # ATM approximation when K ≈ F
    if abs(K - F) < eps * F:
        Fb = F ** (1 - beta)
        vol = alpha / Fb * (
            1 + ((1 - beta)**2 / 24 * alpha**2 / Fb**2
                 + rho * beta * nu * alpha / (4 * F**((1 - beta)))
                 + (2 - 3*rho**2) / 24 * nu**2) * T
        )
        return vol

    FK   = F * K
    log_FK = np.log(F / K)

    # z and χ(z)
    z = (nu / alpha) * FK**((1 - beta) / 2) * log_FK
    if abs(z) < eps:
        z_over_chi = 1.0
    else:
        chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        if abs(chi) < eps:
            z_over_chi = 1.0
        else:
            z_over_chi = z / chi

    # Leading term
    denom_A = FK**((1 - beta) / 2) * (
        1 + (1 - beta)**2 / 24 * log_FK**2
          + (1 - beta)**4 / 1920 * log_FK**4
    )
    leading = alpha / denom_A * z_over_chi

    # Correction term
    Fb = FK**((1 - beta) / 2)
    correction = 1 + (
        (1 - beta)**2 / 24 * alpha**2 / FK**(1 - beta)
        + rho * beta * nu * alpha / (4 * Fb)
        + (2 - 3*rho**2) / 24 * nu**2
    ) * T

    vol = leading * correction
    return float(vol) if np.isfinite(vol) else np.nan


def calibrate_sabr_smile(S, offsets, mkt_ivs, T, beta=0.75):
    """
    Calibrate SABR (alpha, rho, nu) with fixed beta.
    Minimizes sum of squared vol differences.
    Returns (alpha, rho, nu).
    """
    K_arr  = np.array([S + o for o in offsets])
    iv_arr = np.array(mkt_ivs)
    atm_iv = iv_arr[5]

    # Initial alpha from ATM: sigma_ATM ~ alpha / S^(1-beta)
    alpha0 = atm_iv * S ** (1 - beta)

    def objective(params):
        a, r, n = params
        if a <= 1e-8 or n <= 1e-8 or abs(r) >= 0.9999:
            return 1e8
        err = 0.0
        for K, iv_mkt in zip(K_arr, iv_arr):
            if K <= 0:
                continue
            iv_mod = sabr_vol(K, S, T, a, beta, r, n)
            if iv_mod is None or np.isnan(iv_mod) or iv_mod <= 0:
                err += 1.0
            else:
                err += (iv_mod - iv_mkt) ** 2
        return err

    best_obj = np.inf
    best_p   = [alpha0, 0.0, 0.3]

    for rho_init in [-0.3, 0.0, 0.3]:
        for nu_init in [0.3, 0.6]:
            x0 = [alpha0, rho_init, nu_init]
            try:
                res = minimize(objective, x0,
                               method="Nelder-Mead",
                               options={"xatol": 1e-8, "fatol": 1e-10,
                                        "maxiter": 3000})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_p   = res.x
            except Exception:
                pass

    # Final polish
    try:
        res2 = minimize(objective, best_p,
                        method="L-BFGS-B",
                        bounds=[(1e-6, 5.0), (-0.9999, 0.9999), (1e-6, 5.0)],
                        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000})
        if res2.fun < best_obj:
            best_p = res2.x
    except Exception:
        pass

    return float(best_p[0]), float(best_p[1]), float(best_p[2])


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Rebuild OIS curve ────────────────────────────────────────────────────
    ois_data       = load_ois_sofr()
    ois_T, ois_DF  = bootstrap_ois(ois_data)
    ois_fn         = make_interp(ois_T[1:], ois_DF[1:])

    # Extend OIS curve via log-linear extrapolation beyond 50Y (for 30×30)
    log_ois = interp1d(ois_T, np.log(ois_DF),
                       kind="linear", fill_value="extrapolate")
    def ois_fn_ext(t):
        if t <= 0:
            return 1.0
        return float(np.exp(log_ois(t)))

    # ── Load swaption data ───────────────────────────────────────────────────
    swn_data = load_swaptions()

    # ── Calibration results storage ──────────────────────────────────────────
    expiries = [1.0, 5.0, 10.0]
    tenors   = [1.0, 2.0, 3.0, 5.0, 10.0]

    # Tables indexed by (expiry, tenor)
    S_grid  = {}     # ATM forward swap rates
    A_grid  = {}     # annuities

    dd_sigma_grid = {}   # DD model: σ_DD
    dd_beta_grid  = {}   # DD model: β

    sabr_alpha_grid = {}  # SABR: α
    sabr_rho_grid   = {}  # SABR: ρ
    sabr_nu_grid    = {}  # SABR: ν

    print("=== ATM Forward Swap Rates ===")
    for rec in swn_data:
        Te = rec["expiry"]
        n  = rec["tenor"]
        A, S = annuity_and_fsr(Te, n, ois_fn_ext)
        S_grid[(Te, n)] = S
        A_grid[(Te, n)] = A
        print(f"  {Te:.0f}×{n:.0f}  S={S*100:.4f}%  A={A:.4f}")

    print("\n=== Calibrating Displaced-Diffusion Model ===")
    for rec in swn_data:
        Te      = rec["expiry"]
        n       = rec["tenor"]
        S       = S_grid[(Te, n)]
        offsets = rec["offsets"]
        ivs     = rec["ivs"]

        sigma_dd, beta = calibrate_dd_smile(S, offsets, ivs, Te)
        dd_sigma_grid[(Te, n)] = sigma_dd
        dd_beta_grid[(Te, n)]  = beta

        # Compute RMSE
        K_arr = [S + o for o in offsets]
        iv_mod = [dd_iv(K, S, beta, sigma_dd, Te) for K in K_arr]
        rmse = np.sqrt(np.nanmean([(m - v)**2
                                    for m, v in zip(iv_mod, ivs)
                                    if m is not None and not np.isnan(m)]))
        print(f"  {Te:.0f}x{n:.0f}  beta={beta*100:.2f}%  sigma_DD={sigma_dd*100:.2f}%  "
              f"RMSE={rmse*10000:.2f}bp")

    print("\n=== Calibrating SABR Model (β=0.75) ===")
    beta_fixed = 0.75
    for rec in swn_data:
        Te      = rec["expiry"]
        n       = rec["tenor"]
        S       = S_grid[(Te, n)]
        offsets = rec["offsets"]
        ivs     = rec["ivs"]

        alpha, rho, nu = calibrate_sabr_smile(S, offsets, ivs, Te, beta=beta_fixed)
        sabr_alpha_grid[(Te, n)] = alpha
        sabr_rho_grid[(Te, n)]   = rho
        sabr_nu_grid[(Te, n)]    = nu

        # Compute RMSE
        K_arr  = [S + o for o in offsets]
        iv_mod = [sabr_vol(K, S, Te, alpha, beta_fixed, rho, nu) for K in K_arr]
        rmse   = np.sqrt(np.nanmean([(m - v)**2
                                      for m, v in zip(iv_mod, ivs)
                                      if m is not None and not np.isnan(m)]))
        print(f"  {Te:.0f}x{n:.0f}  alpha={alpha*100:.4f}%  rho={rho:.4f}  nu={nu:.4f}  "
              f"RMSE={rmse*10000:.2f}bp")

    # ── Print parameter tables ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("DISPLACED-DIFFUSION: sigma_DD table (%)")
    print(f"{'':>8}", end="")
    for n in tenors:
        print(f"  {n:.0f}Y", end="")
    print()
    for Te in expiries:
        print(f"{Te:.0f}Y exp  ", end="")
        for n in tenors:
            v = dd_sigma_grid.get((Te, n), np.nan)
            print(f"  {v*100:4.1f}", end="")
        print()

    print("\nDISPLACED-DIFFUSION: beta table (%)")
    print(f"{'':>8}", end="")
    for n in tenors:
        print(f"  {n:.0f}Y", end="")
    print()
    for Te in expiries:
        print(f"{Te:.0f}Y exp  ", end="")
        for n in tenors:
            v = dd_beta_grid.get((Te, n), np.nan)
            print(f"  {v*100:4.2f}", end="")
        print()

    print("\nSABR: alpha table (%)")
    print(f"{'':>8}", end="")
    for n in tenors:
        print(f"  {n:.0f}Y", end="")
    print()
    for Te in expiries:
        print(f"{Te:.0f}Y exp  ", end="")
        for n in tenors:
            v = sabr_alpha_grid.get((Te, n), np.nan)
            print(f"  {v*100:5.3f}", end="")
        print()

    print("\nSABR: rho table")
    print(f"{'':>8}", end="")
    for n in tenors:
        print(f"    {n:.0f}Y", end="")
    print()
    for Te in expiries:
        print(f"{Te:.0f}Y exp  ", end="")
        for n in tenors:
            v = sabr_rho_grid.get((Te, n), np.nan)
            print(f"  {v:+.4f}", end="")
        print()

    print("\nSABR: nu table")
    print(f"{'':>8}", end="")
    for n in tenors:
        print(f"    {n:.0f}Y", end="")
    print()
    for Te in expiries:
        print(f"{Te:.0f}Y exp  ", end="")
        for n in tenors:
            v = sabr_nu_grid.get((Te, n), np.nan)
            print(f"  {v:.4f}", end="")
        print()

    # ── Part II.3: Price swaptions across K ∈ [1%, 10%] ─────────────────────
    K_plot = np.linspace(0.01, 0.10, 100)

    pricing_cases = [
        {"label": "Payer 1×1",    "expiry": 1.0,  "tenor": 1.0,  "is_call": True},
        {"label": "Payer 10×10",  "expiry": 10.0, "tenor": 10.0, "is_call": True},
        {"label": "Receiver 30×30", "expiry": 30.0, "tenor": 30.0, "is_call": False},
    ]

    # For 30×30: use flat extrapolation from 10×10 SABR and DD params
    # and forward swap rate computed from extended OIS curve
    A_30_30, S_30_30 = annuity_and_fsr(30.0, 30.0, ois_fn_ext)
    S_grid[(30.0, 30.0)]  = S_30_30
    dd_sigma_grid[(30.0, 30.0)] = dd_sigma_grid[(10.0, 10.0)]
    dd_beta_grid[(30.0, 30.0)]  = dd_beta_grid[(10.0, 10.0)]
    sabr_alpha_grid[(30.0, 30.0)] = sabr_alpha_grid[(10.0, 10.0)]
    sabr_rho_grid[(30.0, 30.0)]   = sabr_rho_grid[(10.0, 10.0)]
    sabr_nu_grid[(30.0, 30.0)]    = sabr_nu_grid[(10.0, 10.0)]

    print(f"\n=== 30×30 Forward Swap Rate ===")
    print(f"  S(30,60) = {S_30_30*100:.4f}%  "
          f"(params: flat extrapolation from 10×10)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, case in zip(axes, pricing_cases):
        Te = case["expiry"]
        n  = case["tenor"]
        S  = S_grid[(Te, n)]

        sigma_dd = dd_sigma_grid[(Te, n)]
        beta_dd  = dd_beta_grid[(Te, n)]
        alpha    = sabr_alpha_grid[(Te, n)]
        rho      = sabr_rho_grid[(Te, n)]
        nu       = sabr_nu_grid[(Te, n)]

        # Displaced-diffusion implied vols
        iv_dd   = [dd_iv(K, S, beta_dd, sigma_dd, Te) for K in K_plot]

        # SABR implied vols
        iv_sabr = [sabr_vol(K, S, Te, alpha, beta_fixed, rho, nu)
                   for K in K_plot]

        ax.plot(K_plot * 100, [v * 100 if v and not np.isnan(v) else np.nan
                               for v in iv_dd],
                lw=2, label="Displaced-Diffusion", color="steelblue")
        ax.plot(K_plot * 100, [v * 100 if v and not np.isnan(v) else np.nan
                               for v in iv_sabr],
                lw=2, ls="--", label="SABR", color="darkorange")
        ax.axvline(S * 100, color="gray", ls=":", lw=1, label=f"ATM ({S*100:.2f}%)")

        # Plot market data points (only where strikes fall in [1%,10%])
        rec = next((r for r in swn_data
                    if r["expiry"] == Te and r["tenor"] == n), None)
        if rec:
            K_mkt = [S + o for o in rec["offsets"]]
            iv_mkt = rec["ivs"]
            K_mkt_filt = [(k, v) for k, v in zip(K_mkt, iv_mkt)
                          if 0.01 <= k <= 0.10]
            if K_mkt_filt:
                xs, ys = zip(*K_mkt_filt)
                ax.scatter([x * 100 for x in xs], [y * 100 for y in ys],
                           zorder=5, color="black", s=30, label="Market")

        ax.set_xlim(1, 10)
        ax.set_xlabel("Strike K (%)")
        ax.set_ylabel("Implied Lognormal Vol (%)")
        ax.set_title(f"{case['label']}  (S={S*100:.2f}%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Implied Vol Smiles: DD vs SABR Models", fontsize=13)
    plt.tight_layout()
    plt.savefig("part2_smile_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: part2_smile_plots.png")

    # ── Market vs model fit plots for all 15 smiles ──────────────────────────
    fig2, axes2 = plt.subplots(3, 5, figsize=(20, 12))
    for idx, rec in enumerate(swn_data):
        Te = rec["expiry"]
        n  = rec["tenor"]
        S  = S_grid[(Te, n)]
        ax = axes2[expiries.index(Te)][tenors.index(n)]

        K_arr  = np.array([S + o for o in rec["offsets"]])
        iv_mkt = np.array(rec["ivs"])

        sigma_dd = dd_sigma_grid[(Te, n)]
        beta_dd  = dd_beta_grid[(Te, n)]
        alpha    = sabr_alpha_grid[(Te, n)]
        rho      = sabr_rho_grid[(Te, n)]
        nu       = sabr_nu_grid[(Te, n)]

        iv_dd_pts   = [dd_iv(K, S, beta_dd, sigma_dd, Te) for K in K_arr]
        iv_sabr_pts = [sabr_vol(K, S, Te, alpha, beta_fixed, rho, nu)
                       for K in K_arr]

        # Fine curve
        K_fine = np.linspace(max(0.001, K_arr[0]-0.005), K_arr[-1]+0.005, 100)
        iv_dd_fine   = [dd_iv(K, S, beta_dd, sigma_dd, Te) for K in K_fine]
        iv_sabr_fine = [sabr_vol(K, S, Te, alpha, beta_fixed, rho, nu) for K in K_fine]

        ax.scatter(K_arr*100, iv_mkt*100, color="black", s=25, zorder=5, label="Market")
        ax.plot([k*100 for k in K_fine],
                [v*100 if v and not np.isnan(v) else np.nan for v in iv_dd_fine],
                lw=1.5, color="steelblue", label="DD")
        ax.plot([k*100 for k in K_fine],
                [v*100 if v and not np.isnan(v) else np.nan for v in iv_sabr_fine],
                lw=1.5, ls="--", color="darkorange", label="SABR")

        ax.set_title(f"{Te:.0f}Y × {n:.0f}Y", fontsize=9)
        ax.set_xlabel("K (%)", fontsize=7)
        ax.set_ylabel("IV (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if Te == 1.0 and n == 1.0:
            ax.legend(fontsize=6)

    plt.suptitle("Market vs DD & SABR Calibration (all 15 smiles)", fontsize=13)
    plt.tight_layout()
    plt.savefig("part2_calibration_fits.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: part2_calibration_fits.png")

    return (dd_sigma_grid, dd_beta_grid,
            sabr_alpha_grid, sabr_rho_grid, sabr_nu_grid,
            S_grid, A_grid, ois_fn_ext)


if __name__ == "__main__":
    main()
