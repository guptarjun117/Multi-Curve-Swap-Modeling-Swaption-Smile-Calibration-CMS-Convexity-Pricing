"""
Part I - Bootstrapping Swap Curves

Curves bootstrapped:
1. OIS (SOFR) discount curve   Do(0,T)          [T ∈ 0..50]
2. LIBOR single-curve           D_single(0,T)    [T ∈ 0..30]
3. LIBOR multi-curve            D_multi(0,T)     [T ∈ 0..30]
4. 3m forward Term SOFR rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

EXCEL = r"Swap and Swaption Markets.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_libor():
    df = pd.read_excel(EXCEL, sheet_name="LIBOR (legacy)", header=None)
    rows = df.iloc[1:, :3].copy()
    rows.columns = ["Tenor", "Product", "Rate"]
    rows = rows.reset_index(drop=True)
    def parse_tenor(t):
        t = str(t).strip().lower()
        if t.endswith("m"):   return float(t[:-1]) / 12
        elif t.endswith("y"): return float(t[:-1])
        raise ValueError(t)
    rows["T"] = rows["Tenor"].apply(parse_tenor)
    rows["Rate"] = rows["Rate"].astype(float)
    return rows          # columns: Tenor, Product, Rate, T


def load_ois_sofr():
    df = pd.read_excel(EXCEL, sheet_name="OIS (SOFR)", header=None)
    rows = df.iloc[1:, :2].copy()
    rows.columns = ["Term", "Rate"]
    rows = rows.reset_index(drop=True)
    def parse_term(t):
        t = str(t).strip().upper()
        if t.endswith("WK"):  return float(t[:-2]) / 52
        elif t.endswith("MO"): return float(t[:-2]) / 12
        elif t.endswith("YR"): return float(t[:-2])
        raise ValueError(t)
    rows["T"] = rows["Term"].apply(parse_term)
    rows["Rate"] = rows["Rate"].astype(float) / 100.0   # given in percent
    return rows


def load_term_sofr():
    df = pd.read_excel(EXCEL, sheet_name="OIS (Term SOFR)", header=None)
    rows = df.iloc[1:, :2].copy()
    rows.columns = ["Tenor", "Rate"]
    rows = rows.reset_index(drop=True)
    def parse_tenor(t):
        t = str(t).strip().upper()
        if t.endswith("Y"): return float(t[:-1])
        raise ValueError(t)
    rows["T"] = rows["Tenor"].apply(parse_tenor)
    rows["Rate"] = rows["Rate"].astype(float)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Linear interpolation on discount factors (D(0,0)=1 always prepended)
# ─────────────────────────────────────────────────────────────────────────────

def make_interp(T_arr, DF_arr):
    """Linear interpolator for a discount-factor curve. D(0)=1 prepended."""
    t = np.concatenate([[0.0], np.asarray(T_arr)])
    d = np.concatenate([[1.0], np.asarray(DF_arr)])
    return interp1d(t, d, kind="linear", fill_value="extrapolate")


# ─────────────────────────────────────────────────────────────────────────────
# 1. OIS (SOFR) Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ois(data):
    """
    Convention: 30/360, annual fixed coupon, O/N float (zero-coupon equivalent).

    For T ≤ 1Y  (zero-coupon):
        Do(0,T) = 1 / (1 + r*T)

    For T > 1Y  (annual fixed coupon at 1, 2, …, floor(T) and T):
        r*Do(1) + r*Do(2) + … + r*Do(floor(T)) + (1 + r*frac)*Do(T) = 1
        where frac = T − floor(T), and if T is an integer, frac = 1
        (coupon at year T has fraction 1 for a full year, or frac for a stub).

        Rearranged:
        Do(T) = (1 − r * Σ_{i=1}^{floor(T)} Do(i)) / (1 + r*frac)
        (For integer T: floor(T)=T but sum excludes T; see below.)

    Actually, re-derive cleanly:
        Payment dates: 1, 2, …, floor(T), and T (if non-integer).
        If T is integer: last coupon at T with fraction 1; sum in bootstrap
            excludes T (it is the unknown):
            Do(T) = (1 − r * Σ_{i=1}^{T−1} Do(i)) / (1 + r)

        If T is non-integer (e.g., 1.5Y):
            Annual coupon at i=1,…,floor(T) already known.
            Final stub coupon at T with fraction = T − floor(T).
            Do(T) = (1 − r * Σ_{i=1}^{floor(T)} Do(i)) / (1 + r*frac)
    """
    T_pil = [0.0]
    D_pil = [1.0]

    for T, r in zip(data["T"].values, data["Rate"].values):
        if T <= 1.0 + 1e-9:
            D = 1.0 / (1.0 + r * T)
        else:
            interp = make_interp(T_pil[1:], D_pil[1:])
            T_int = int(T)        # floor
            frac  = T - T_int     # 0 if integer

            if frac < 1e-9:
                # Integer maturity: sum over i = 1 … T-1
                s = sum(float(interp(i)) for i in range(1, T_int))
                D = (1.0 - r * s) / (1.0 + r)
            else:
                # Non-integer maturity (e.g. 18M = 1.5Y):
                # coupons at 1, …, floor(T), stub at T
                s = sum(float(interp(i)) for i in range(1, T_int + 1))
                D = (1.0 - r * s) / (1.0 + r * frac)

        T_pil.append(T)
        D_pil.append(D)

    return np.array(T_pil), np.array(D_pil)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIBOR Single-Curve Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_libor_single(data):
    """
    Convention: 30/360, semi-annual coupons (Δt = 0.5).
    6m cash rate → D(0, 0.5) = 1/(1+r*0.5).
    IRS par swap with rate c and tenor T:
        c*0.5*Σ_{t=0.5,1,…,T} D(0,t) + D(0,T) = 1
        ⟹ D(0,T) = (1 − c*0.5 * Σ_{t < T} D(0,t)) / (1 + c*0.5)
    Linear interpolation on DFs for intermediate coupon dates.
    """
    dt = 0.5
    T_pil = [0.0]
    D_pil = [1.0]

    for T, r in zip(data["T"].values, data["Rate"].values):
        T = round(T, 8)
        if abs(T - 0.5) < 1e-9:
            D = 1.0 / (1.0 + r * 0.5)
        else:
            interp = make_interp(T_pil[1:], D_pil[1:])
            coupon_dates = np.arange(dt, T - 1e-9, dt)  # all dates before T
            s = sum(float(interp(t)) for t in coupon_dates)
            D = (1.0 - r * dt * s) / (1.0 + r * dt)

        T_pil.append(T)
        D_pil.append(D)

    return np.array(T_pil), np.array(D_pil)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIBOR Multi-Curve Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_libor_multi(libor_data, ois_T, ois_DF):
    """
    Multi-curve: OIS discounting + LIBOR projection.
    Between consecutive LIBOR pillars, assume flat (constant) forward rate.

    For IRS with tenor T, previous LIBOR pillar at T_prev:
      N = 2*(T − T_prev) semi-annual periods in the new block.
      Under flat-forward, each period has the same rate f_new.
      Block sum B = 0.5 * Σ Do(0,t) for t in new block payment dates.

      Par condition:
        c * A_OIS(T) = [known float PV from earlier blocks] + f_new * B

      Solving:  f_new = (c * A_OIS(T) − known_PV) / B
      New DF:   D_L(0,T) = D_L(0,T_prev) / (1 + f_new*0.5)^N

    Returns pillar arrays (T, D_L) and forward list [(t0,t1,f), ...].
    """
    ois_fn = make_interp(ois_T[1:], ois_DF[1:])

    dt = 0.5
    # LIBOR pillars
    T_pil = [0.0]
    D_pil = [1.0]

    # Forward blocks: [(T_start, T_end, f_flat, payment_dates), ...]
    blocks = []

    # 6m cash rate: one period [0, 0.5], forward = cash rate
    r6m   = libor_data["Rate"].values[0]
    pay_dates_0 = [0.5]
    B_0   = 0.5 * float(ois_fn(0.5))
    f0    = r6m          # F(0, 0.5) = 6m LIBOR cash rate
    T_pil.append(0.5)
    D_pil.append(D_pil[-1] / (1.0 + f0 * 0.5))
    blocks.append((0.0, 0.5, f0, pay_dates_0))

    for T, c in zip(libor_data["T"].values[1:], libor_data["Rate"].values[1:]):
        T = round(T, 8)
        T_prev = T_pil[-1]

        # OIS annuity over [0, T]: all semi-annual dates
        all_pay = np.arange(dt, T + 1e-9, dt)
        A_OIS   = 0.5 * sum(float(ois_fn(t)) for t in all_pay)

        # Known float PV from already-resolved blocks
        known_PV = 0.0
        for (ta, tb, f_b, pd_b) in blocks:
            block_sum = 0.5 * sum(float(ois_fn(t)) for t in pd_b)
            known_PV += f_b * block_sum

        # New block payment dates: (T_prev, T]
        new_pay = np.arange(T_prev + dt, T + 1e-9, dt)
        B_new   = 0.5 * sum(float(ois_fn(t)) for t in new_pay)
        N_new   = len(new_pay)          # number of new periods

        f_new = (c * A_OIS - known_PV) / B_new
        D_new = D_pil[-1] / (1.0 + f_new * dt) ** N_new

        T_pil.append(T)
        D_pil.append(D_new)
        blocks.append((T_prev, T, f_new, list(new_pay)))

    # Build forward list for reporting: (t0, t1, F) per semi-annual period
    fwd_list = []
    for (ta, tb, f_b, pd_b) in blocks:
        t0 = ta
        for t1 in pd_b:
            fwd_list.append((round(t0, 4), round(t1, 4), f_b))
            t0 = t1

    return np.array(T_pil), np.array(D_pil), fwd_list


# ─────────────────────────────────────────────────────────────────────────────
# 4. Term SOFR Bootstrap → 3m forward rates
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_term_sofr(ts_data, ois_T, ois_DF):
    """
    Term SOFR IRS: fixed annual (rate c), float quarterly (Term SOFR 3M).
    OIS discounting for both legs (collateralized).

    Strategy: between consecutive annual D_TS pillars, assume flat 3m forward.
    For each new annual pillar T_k:

      Par condition:
        c_k * Σ_{i=1}^{T_k} Do(0,i) = [known float PV] + f_new * B_new

      where B_new = 0.25 * Σ Do(0,t) for quarterly t in new year block.

      f_new = (c_k * A_fix − known_PV) / B_new
      D_TS(0,T_k) = D_TS(0,T_{k−1}) / (1 + f_new*0.25)^4

    Returns: D_TS pillar arrays, forward rate list, quarterly D_TS grid.
    """
    ois_fn = make_interp(ois_T[1:], ois_DF[1:])
    dt_fix = 1.0
    dt_flt = 0.25

    T_pil = [0.0]
    D_pil = [1.0]
    blocks = []   # (T_start, T_end, f_flat, payment_dates)

    for T, c in zip(ts_data["T"].values, ts_data["Rate"].values):
        T = int(round(T))   # tenors are integer years

        T_prev = T_pil[-1]

        # Fixed leg annuity (OIS discounting, annual payments)
        fix_dates = [float(i) for i in range(1, T + 1)]
        A_fix     = sum(float(ois_fn(t)) for t in fix_dates)

        # Known float PV from earlier blocks
        known_PV = 0.0
        for (ta, tb, f_b, pd_b) in blocks:
            B_b = dt_flt * sum(float(ois_fn(t)) for t in pd_b)
            known_PV += f_b * B_b

        # New block: quarterly dates from T_prev+0.25 to T (one year of quarters)
        new_pay = [round(T_prev + (k + 1) * dt_flt, 8) for k in range(4)]
        B_new   = dt_flt * sum(float(ois_fn(t)) for t in new_pay)

        f_new = (c * A_fix - known_PV) / B_new
        D_new = D_pil[-1] / (1.0 + f_new * dt_flt) ** 4

        T_pil.append(float(T))
        D_pil.append(D_new)
        blocks.append((float(T_prev), float(T), f_new, new_pay))

    # Build quarterly D_TS grid via log-linear interpolation between pillars
    ts_pil_T  = np.array(T_pil)
    ts_pil_DF = np.array(D_pil)

    # Generate all quarterly dates
    q_dates = np.arange(0.25, ts_pil_T[-1] + 1e-9, 0.25)
    log_df_interp = interp1d(ts_pil_T, np.log(ts_pil_DF),
                             kind="linear", fill_value="extrapolate")
    q_dfs = np.exp(log_df_interp(q_dates))

    # 3m forward rates
    fwd_T = q_dates[:-1]
    fwd_r = (q_dfs[:-1] / q_dfs[1:] - 1.0) / 0.25

    # Forward list for reporting
    fwd_list = []
    for (ta, tb, f_b, pd_b) in blocks:
        fwd_list.append((ta, tb, f_b))

    return ts_pil_T, ts_pil_DF, fwd_T, fwd_r


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    libor_data = load_libor()
    ois_data   = load_ois_sofr()
    ts_data    = load_term_sofr()

    # ── OIS ──────────────────────────────────────────────────────────────────
    ois_T, ois_DF = bootstrap_ois(ois_data)
    ois_fn = make_interp(ois_T[1:], ois_DF[1:])

    print("=== OIS Discount Factors (pillar dates) ===")
    for t, d in zip(ois_T, ois_DF):
        print(f"  T={t:7.4f}  Do={d:.8f}")

    # Verify monotone
    assert np.all(np.diff(ois_DF) < 0), "OIS DF not monotone!"

    # ── LIBOR Single-curve ───────────────────────────────────────────────────
    lib_T, lib_DF = bootstrap_libor_single(libor_data)
    lib_fn = make_interp(lib_T[1:], lib_DF[1:])

    print("\n=== LIBOR Single-Curve DF (pillar dates) ===")
    for t, d in zip(lib_T, lib_DF):
        print(f"  T={t:5.2f}  D_single={d:.8f}")

    # Sanity: re-implied par rates
    print("\n=== LIBOR Single-Curve Par-Rate Check ===")
    dt = 0.5
    for _, row in libor_data.iterrows():
        T, c_in = round(row["T"], 8), row["Rate"]
        if T <= 0.5:
            continue
        cpn = np.arange(dt, T + 1e-9, dt)
        A   = sum(float(lib_fn(t)) * dt for t in cpn)
        c_out = (1.0 - float(lib_fn(T))) / A
        print(f"  T={T:5.2f}  in={c_in*100:.3f}%  out={c_out*100:.3f}%  "
              f"err={abs(c_in-c_out)*1e4:.2f}bp")

    # ── LIBOR Multi-curve ────────────────────────────────────────────────────
    mul_T, mul_DF, fwd_list = bootstrap_libor_multi(libor_data, ois_T, ois_DF)
    mul_fn = make_interp(mul_T[1:], mul_DF[1:])

    print("\n=== LIBOR Multi-Curve DF (pillar dates) ===")
    for t, d in zip(mul_T, mul_DF):
        print(f"  T={t:5.2f}  D_multi={d:.8f}")

    print("\n=== LIBOR 6M Forward Rates (Multi-Curve, per block) ===")
    seen = set()
    for t0, t1, F in fwd_list:
        if (t0, t1) not in seen:
            print(f"  F({t0:.2f},{t1:.2f}) = {F*100:.4f}%")
            seen.add((t0, t1))

    # Verify multi-curve par-rate sanity
    print("\n=== LIBOR Multi-Curve Par-Rate Check ===")
    for _, row in libor_data.iterrows():
        T, c_in = round(row["T"], 8), row["Rate"]
        if T <= 0.5:
            continue
        cpn   = np.arange(dt, T + 1e-9, dt)
        A_OIS = sum(float(ois_fn(t)) * dt for t in cpn)
        # Float PV: use block forwards
        float_pv = 0.0
        for t0, t1, F in fwd_list:
            if t1 <= T + 1e-9:
                float_pv += F * dt * float(ois_fn(t1))
        c_out = float_pv / A_OIS
        print(f"  T={T:5.2f}  in={c_in*100:.3f}%  out={c_out*100:.3f}%  "
              f"err={abs(c_in-c_out)*1e4:.2f}bp")

    # ── Term SOFR ────────────────────────────────────────────────────────────
    ts_T, ts_DF, fwd3m_T, fwd3m_r = bootstrap_term_sofr(ts_data, ois_T, ois_DF)

    print("\n=== Term SOFR D_TS Pillars ===")
    for t, d in zip(ts_T, ts_DF):
        print(f"  T={t:5.2f}  D_TS={d:.8f}")

    print("\n=== 3M Term SOFR Forwards (first 10 entries) ===")
    for t, f in zip(fwd3m_T[:10], fwd3m_r[:10]):
        print(f"  t={t:.2f}  F_3m={f*100:.4f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    t30 = np.linspace(0.001, 30, 500)
    t50 = np.linspace(0.001, 50, 500)

    # Figure 1: LIBOR single vs multi-curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t30, [float(lib_fn(t)) for t in t30],
            lw=2, label="Single-curve D(0,T)")
    ax.plot(t30, [float(mul_fn(t)) for t in t30],
            lw=2, ls="--", label="Multi-curve D_L(0,T)")
    ax.scatter(lib_T[1:], lib_DF[1:], s=50, zorder=5)
    ax.scatter(mul_T[1:], mul_DF[1:], s=50, marker="^", zorder=5)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Discount Factor")
    ax.set_title("LIBOR Discount Curve: Single vs Multi-Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff_bps = [(float(lib_fn(t)) - float(mul_fn(t))) * 1e4 for t in t30]
    ax2.plot(t30, diff_bps, color="crimson", lw=2)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Maturity T (years)")
    ax2.set_ylabel("Difference (bps)")
    ax2.set_title("Single-Curve minus Multi-Curve D(0,T) [bps]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part1_libor_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Figure 2: OIS discount curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t50, [float(ois_fn(t)) for t in t50], lw=2, color="steelblue",
            label="OIS (SOFR) Do(0,T)")
    ax.scatter(ois_T[1:], ois_DF[1:], s=40, zorder=5, color="steelblue")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Discount Factor Do(0,T)")
    ax.set_title("OIS (SOFR) Discount Curve,  T ∈ [0, 50]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("part1_ois_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Figure 3: 3m forward Term SOFR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(fwd3m_T, fwd3m_r * 100, where="post", lw=2, color="darkorange",
            label="3M Forward Term SOFR")
    ax.set_xlabel("Fixing Date t (years)")
    ax.set_ylabel("Forward Rate (%)")
    ax.set_title("3M Forward Term SOFR Rates (bootstrapped)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("part1_term_sofr_fwd.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Summary table
    print("\n=== Summary: Key Tenors ===")
    print(f"{'T':>5}  {'D_single':>12}  {'D_multi':>12}  {'Do_OIS':>12}")
    for t in [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]:
        ds = float(lib_fn(t))
        dm = float(mul_fn(t))
        do = float(ois_fn(t))
        print(f"{t:5.1f}  {ds:12.6f}  {dm:12.6f}  {do:12.6f}")

    # Return curves for use in Parts II & III
    return ois_T, ois_DF, lib_T, lib_DF, mul_T, mul_DF, ts_T, ts_DF


if __name__ == "__main__":
    main()
