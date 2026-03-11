"""
Part III - Constant Maturity Swaps (CMS)

Uses SABR model calibrated in Part II for convexity correction via static replication (Hagan replication formula).

Requires part1.py and part2.py in the same directory.
"""

import sys, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from part1 import (load_ois_sofr, bootstrap_ois, make_interp)
from part2 import (load_swaptions, annuity_and_fsr,
                   black_price, sabr_vol, calibrate_sabr_smile)

EXCEL = r"Swap and Swaption Markets.xlsx"


# ===========================================================================
# Rebuild curves and SABR parameters (mirrors Part II main())
# ===========================================================================

def build_curves_and_sabr():
    """Bootstrap OIS and calibrate all 15 SABR smiles."""
    ois_data      = load_ois_sofr()
    ois_T, ois_DF = bootstrap_ois(ois_data)
    ois_fn_base   = make_interp(ois_T[1:], ois_DF[1:])

    # Log-linear extrapolation for T > 50Y
    log_ois = interp1d(ois_T, np.log(ois_DF),
                       kind="linear", fill_value="extrapolate")
    def ois_fn(t):
        if t <= 0:
            return 1.0
        return float(np.exp(log_ois(t)))

    swn_data = load_swaptions()
    beta_fixed = 0.75

    sabr_params = {}  # (Te, n) -> (alpha, rho, nu)
    S_grid      = {}  # (Te, n) -> ATM fwd swap rate
    A_grid      = {}  # (Te, n) -> annuity

    print("Rebuilding SABR calibration for Part III ...")
    for rec in swn_data:
        Te, n = rec["expiry"], rec["tenor"]
        A, S  = annuity_and_fsr(Te, n, ois_fn)
        S_grid[(Te, n)] = S
        A_grid[(Te, n)] = A
        alpha, rho, nu = calibrate_sabr_smile(S, rec["offsets"], rec["ivs"],
                                              Te, beta=beta_fixed)
        sabr_params[(Te, n)] = (alpha, rho, nu)
        print(f"  {Te:.0f}x{n:.0f}: alpha={alpha*100:.3f}%  rho={rho:.4f}  nu={nu:.4f}")

    return ois_fn, sabr_params, S_grid, A_grid


# ===========================================================================
# Interpolate SABR parameters for arbitrary (T_exp, tenor)
# ===========================================================================

def get_sabr_params(T_exp, tenor, sabr_params):
    """
    Retrieve or bilinearly interpolate SABR parameters for (T_exp, tenor).
    Interpolates linearly in expiry (log-linear in alpha) and linearly in tenor.
    """
    exp_grid = sorted(set(k[0] for k in sabr_params.keys()))
    ten_grid = sorted(set(k[1] for k in sabr_params.keys()))

    T_clamped = min(max(T_exp, exp_grid[0]), exp_grid[-1])
    n_clamped = min(max(tenor, ten_grid[0]), ten_grid[-1])

    def interp_at_T(T_near):
        """Interpolate SABR params across tenors at a fixed expiry."""
        if n_clamped <= ten_grid[0]:
            return sabr_params[(T_near, ten_grid[0])]
        if n_clamped >= ten_grid[-1]:
            return sabr_params[(T_near, ten_grid[-1])]
        for i in range(len(ten_grid) - 1):
            n_lo, n_hi = ten_grid[i], ten_grid[i+1]
            if n_lo <= n_clamped <= n_hi:
                w = (n_clamped - n_lo) / (n_hi - n_lo)
                a_lo, r_lo, v_lo = sabr_params[(T_near, n_lo)]
                a_hi, r_hi, v_hi = sabr_params[(T_near, n_hi)]
                alpha = np.exp((1-w)*np.log(a_lo) + w*np.log(a_hi))
                rho   = (1-w)*r_lo + w*r_hi
                nu    = (1-w)*v_lo + w*v_hi
                return alpha, rho, nu
        return sabr_params[(T_near, ten_grid[-1])]

    # Clamp to expiry grid
    if T_clamped <= exp_grid[0]:
        return interp_at_T(exp_grid[0])
    if T_clamped >= exp_grid[-1]:
        return interp_at_T(exp_grid[-1])

    # Linear interpolation across expiries
    for i in range(len(exp_grid) - 1):
        T_lo, T_hi = exp_grid[i], exp_grid[i+1]
        if T_lo <= T_clamped <= T_hi:
            wT = (T_clamped - T_lo) / (T_hi - T_lo)
            a_lo, r_lo, v_lo = interp_at_T(T_lo)
            a_hi, r_hi, v_hi = interp_at_T(T_hi)
            alpha = np.exp((1-wT)*np.log(a_lo) + wT*np.log(a_hi))
            rho   = (1-wT)*r_lo + wT*r_hi
            nu    = (1-wT)*v_lo + wT*v_hi
            return alpha, rho, nu

    return interp_at_T(exp_grid[-1])


# ===========================================================================
# SABR swaption price (payer or receiver)
# ===========================================================================

def swaption_price_sabr(K, S, T_exp, tenor, ois_fn, sabr_params,
                        is_payer=True, beta=0.75):
    """Price a swaption using SABR model. Returns price per unit notional."""
    A, _ = annuity_and_fsr(T_exp, tenor, ois_fn)
    alpha, rho, nu = get_sabr_params(T_exp, tenor, sabr_params)
    if K <= 0 or S <= 0:
        return max(0.0, A * ((S - K) if is_payer else (K - S)))
    vol = sabr_vol(K, S, T_exp, alpha, beta, rho, nu)
    if vol is None or np.isnan(vol) or vol <= 0:
        return max(0.0, A * ((S - K) if is_payer else (K - S)))
    raw = black_price(S, K, vol, T_exp, is_call=is_payer)
    return A * raw


# ===========================================================================
# CMS rate via static replication (Hagan 2002 approach)
# ===========================================================================

def annuity_fn(S, T_exp, tenor, ois_fn):
    """
    Annuity A(S) as a function of the underlying swap rate S.
    Under standard CMS replication, A(S) is approximated by the market annuity.
    We use the fixed-point (market) annuity as a first approximation.
    A(T_exp, tenor) = sum of OIS DFs at semi-annual dates from T_exp to T_exp+tenor.
    This is independent of S (we use market OIS DFs).
    """
    dates = np.arange(T_exp + 0.5, T_exp + tenor + 1e-9, 0.5)
    return 0.5 * sum(ois_fn(t) for t in dates)


def cms_rate_replication(T_pay, tenor, ois_fn, sabr_params,
                         n_points=200, beta=0.75, K_min=0.0001, K_max=0.25):
    """
    CMS rate for fixing and payment at T_pay (in-arrears convention),
    based on an underlying swap of 'tenor' years.

    Uses static replication under the annuity measure (Hagan 2002).

    Change-of-numeraire from annuity measure to T_pay-forward measure:
        CMS rate = E^{T_pay}[S_T]
                 = (A0/Do(0,T_pay)) * E^A[S_T / A(S_T)]

    With linear annuity approximation A(K) = A0 + A1*(K-S0)  (A1 < 0):
        f(K) = K/A(K),  f''(K) = -2*alp*d / (alp + d*K)^3  > 0

    By static replication under E^A:
        CMS rate = (A0/Do_pay) * [S0/A0 + int_0^S0 f''(K)*rcvr(K)dK
                                         + int_S0^inf f''(K)*pay(K)dK]

    where rcvr(K), pay(K) are normalized swaption prices (per unit A0).
    """
    # Fixing in-arrears: T_fix = T_pay = T_exp
    T_exp = T_pay
    A0, S0 = annuity_and_fsr(T_exp, tenor, ois_fn)
    Do_pay = float(ois_fn(T_pay))

    # ── Annuity sensitivity (linear TSR model) ───────────────────────────────
    # A(K) ≈ A0 + A1*(K - S0),  A1 = dA/dS ≈ -Dur_ann * A0  (A1 < 0)
    dates   = np.arange(T_exp + 0.5, T_exp + tenor + 1e-9, 0.5)
    Dur_ann = sum(k * 0.5 * float(ois_fn(T_exp + k * 0.5))
                  for k in range(1, len(dates) + 1)) / A0
    A1  = -Dur_ann * A0        # < 0
    alp = A0 - A1 * S0         # A(0) in the linear model  (> 0)
    d   = A1                   # < 0

    # ── Constant f''(S0) — standard linear TSR approximation ─────────────────
    # f(K) = K/A(K),  f''(K) = -2*alp*d / (alp + d*K)^3
    # At K=S0: denom = alp + d*S0 = A0  =>  f''(S0) = -2*alp*d / A0^3  (> 0)
    # Using a constant f'' avoids the singularity at K_sing = alp/(-d)
    # and is the standard approximation in the linear TSR model.
    f_pp_const = -2.0 * alp * d / A0 ** 3   # > 0  (alp > 0, d < 0)

    K_lo = max(K_min, 1e-4)
    K_hi = min(K_max, S0 * 8)  # no singularity concern with constant f''

    alpha, rho, nu = get_sabr_params(T_exp, tenor, sabr_params)

    def rcvr_norm(K):
        """Receiver swaption price per unit A0 (normalized)."""
        if K <= 0 or S0 <= 0:
            return max(0.0, K - S0)
        vol = sabr_vol(K, S0, T_exp, alpha, 0.75, rho, nu)
        if vol is None or np.isnan(vol) or vol <= 0:
            return max(0.0, K - S0)
        return black_price(S0, K, vol, T_exp, is_call=False)

    def pay_norm(K):
        """Payer swaption price per unit A0 (normalized)."""
        if K <= 0 or S0 <= 0:
            return max(0.0, S0 - K)
        vol = sabr_vol(K, S0, T_exp, alpha, 0.75, rho, nu)
        if vol is None or np.isnan(vol) or vol <= 0:
            return max(0.0, S0 - K)
        return black_price(S0, K, vol, T_exp, is_call=True)

    # Numerical integration  (factor out the constant f'')
    K_rcvr = np.linspace(K_lo, S0, n_points)
    int_rcvr = np.trapezoid([rcvr_norm(K) for K in K_rcvr], K_rcvr)

    K_payer = np.linspace(S0, K_hi, n_points)
    int_payer = np.trapezoid([pay_norm(K) for K in K_payer], K_payer)

    # CMS rate = (A0/Do_pay) * [S0/A0 + f''_const * integral]
    cms_rate = (A0 / Do_pay) * (S0 / A0 + f_pp_const * (int_rcvr + int_payer))
    convexity_correction = cms_rate - S0

    return cms_rate, S0, convexity_correction


# ===========================================================================
# PV of CMS leg
# ===========================================================================

def pv_cms_leg(payment_dates, dt, tenor, ois_fn, sabr_params):
    """
    Compute PV of a CMS leg.
    payment_dates: array of payment/fixing dates
    dt: payment fraction (0.5 for semi-annual, 0.25 for quarterly)
    tenor: underlying CMS swap tenor in years
    """
    pv = 0.0
    cms_rates = []
    for t in payment_dates:
        rate, S0, cc = cms_rate_replication(t, tenor, ois_fn, sabr_params)
        Do_t = ois_fn(t)
        pv += rate * dt * Do_t
        cms_rates.append((t, rate, S0, cc))
    return pv, cms_rates


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Build OIS curve and SABR parameters
    ois_fn, sabr_params, S_grid, A_grid = build_curves_and_sabr()

    print("\n" + "="*65)
    print("PART III: CONSTANT MATURITY SWAPS")
    print("="*65)

    # ── III.1a: CMS10y semi-annual, 5 years ─────────────────────────────────
    print("\n--- CMS10y leg: semi-annual, 5 years ---")
    pay_dates_10y5y = np.arange(0.5, 5.0 + 1e-9, 0.5)
    pv_10y5y, cms_10y5y = pv_cms_leg(pay_dates_10y5y, 0.5, 10.0,
                                      ois_fn, sabr_params)

    print(f"  {'Pay Date':>10}  {'CMS10y Rate':>12}  {'Fwd Swap Rate':>14}  "
          f"{'Conv Corr (bp)':>15}  {'Do(0,t)':>10}")
    for t, rate, S0, cc in cms_10y5y:
        Do_t = ois_fn(t)
        print(f"  {t:10.2f}  {rate*100:12.4f}%  {S0*100:14.4f}%  "
              f"{cc*10000:15.3f}bp  {Do_t:10.6f}")
    print(f"\n  >>> PV of CMS10y semi-annual leg (5Y) = {pv_10y5y:.6f}")

    # ── III.1b: CMS2y quarterly, 10 years ───────────────────────────────────
    print("\n--- CMS2y leg: quarterly, 10 years ---")
    pay_dates_2y10y = np.arange(0.25, 10.0 + 1e-9, 0.25)
    pv_2y10y, cms_2y10y = pv_cms_leg(pay_dates_2y10y, 0.25, 2.0,
                                      ois_fn, sabr_params)

    print(f"  {'Pay Date':>10}  {'CMS2y Rate':>12}  {'Fwd Swap Rate':>14}  "
          f"{'Conv Corr (bp)':>15}  {'Do(0,t)':>10}")
    for t, rate, S0, cc in cms_2y10y:
        Do_t = ois_fn(t)
        print(f"  {t:10.2f}  {rate*100:12.4f}%  {S0*100:14.4f}%  "
              f"{cc*10000:15.3f}bp  {Do_t:10.6f}")
    print(f"\n  >>> PV of CMS2y quarterly leg (10Y) = {pv_2y10y:.6f}")

    # ── III.2: Convexity correction comparison ───────────────────────────────
    print("\n" + "="*65)
    print("CONVEXITY CORRECTION: Forward Swap Rate vs CMS Rate")
    print("="*65)

    pairs = [
        (1.0, 1.0), (1.0, 10.0),
        (5.0, 1.0), (5.0, 10.0),
        (10.0, 1.0), (10.0, 10.0),
    ]

    print(f"\n  {'Pair':>8}  {'Fwd Swap Rate':>14}  {'CMS Rate':>10}  "
          f"{'Conv Corr (bp)':>16}  {'Annuity':>9}")

    cc_results = {}
    for (Te, n) in pairs:
        cms, S0, cc = cms_rate_replication(Te, n, ois_fn, sabr_params)
        A0, _ = annuity_and_fsr(Te, n, ois_fn)
        cc_results[(Te, n)] = (S0, cms, cc)
        print(f"  {Te:.0f}x{n:.0f}{' ':>4}  "
              f"{S0*100:14.4f}%  {cms*100:10.4f}%  "
              f"{cc*10000:16.4f}bp  {A0:9.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────

    # Plot 1: CMS10y rates over payment dates
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    t_10y5y = [x[0] for x in cms_10y5y]
    r_10y5y = [x[1]*100 for x in cms_10y5y]
    s_10y5y = [x[2]*100 for x in cms_10y5y]
    ax.step(t_10y5y, r_10y5y, where="post", lw=2, color="steelblue",
            label="CMS10y rate (with convexity)")
    ax.step(t_10y5y, s_10y5y, where="post", lw=2, ls="--", color="darkorange",
            label="Forward swap rate S(t, t+10)")
    ax.set_xlabel("Payment Date (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"CMS10y Semi-Annual Leg (5Y)  |  PV = {pv_10y5y:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    t_2y10y = [x[0] for x in cms_2y10y]
    r_2y10y = [x[1]*100 for x in cms_2y10y]
    s_2y10y = [x[2]*100 for x in cms_2y10y]
    ax.step(t_2y10y, r_2y10y, where="post", lw=1.5, color="steelblue",
            label="CMS2y rate (with convexity)")
    ax.step(t_2y10y, s_2y10y, where="post", lw=1.5, ls="--", color="darkorange",
            label="Forward swap rate S(t, t+2)")
    ax.set_xlabel("Payment Date (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"CMS2y Quarterly Leg (10Y)  |  PV = {pv_2y10y:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part3_cms_legs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: part3_cms_legs.png")

    # Plot 2: Convexity correction bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels    = [f"{Te:.0f}x{n:.0f}" for (Te, n) in pairs]
    cc_bps    = [cc_results[(Te,n)][2] * 10000 for (Te,n) in pairs]
    colors    = ["steelblue"]*2 + ["darkorange"]*2 + ["seagreen"]*2
    bars = ax.bar(labels, cc_bps, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f bp", padding=2, fontsize=9)
    ax.set_xlabel("(Expiry x Tenor)")
    ax.set_ylabel("Convexity Correction (bps)")
    ax.set_title("CMS Convexity Correction: Forward Swap Rate vs CMS Rate")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.8)

    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(color="steelblue",  label="1Y expiry"),
               Patch(color="darkorange", label="5Y expiry"),
               Patch(color="seagreen",   label="10Y expiry")]
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.savefig("part3_convexity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: part3_convexity.png")

if __name__ == "__main__":
    main()
