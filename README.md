# Multi-Curve-Swap-Modeling-Swaption-Smile-Calibration-CMS-Convexity-Pricing

---

## Executive Summary

### Part I — Bootstrapping Swap Curves (`part1.py`)
The data comes from a single Excel workbook with three sheets: LIBOR (legacy), OIS (SOFR), and OIS (Term SOFR).

OIS bootstrap handles two regimes. For T ≤ 1Y the instruments are zero-coupon (simple interest), so D(0,T) = 1/(1 + r·T) directly. For T > 1Y the instruments are annual fixed-coupon OIS swaps, so each new pillar's DF is solved by stripping out the previously-known coupon cash flows. One subtlety: the code handles non-integer pillars (e.g., 18M) separately from integer ones, using a stub fraction for the last coupon period. Results: Do(10Y) ≈ 0.681, Do(20Y) ≈ 0.429, Do(50Y) ≈ 0.152, implying roughly 3.8–4.0% risk-free rate.

LIBOR single-curve bootstrap uses semi-annual 30/360 conventions throughout. The 6M cash deposit seeds the first DF; every subsequent pillar is solved by the standard par-swap condition where fixed and floating PVs are equal, using the same curve for both discounting and projection. Linear interpolation on DFs handles intermediate coupon dates.

LIBOR multi-curve bootstrap is the more interesting one. It separates discounting (OIS DFs) from forward projection (LIBOR DFs). The algorithm works by assuming a flat forward rate between consecutive LIBOR pillars. For each new pillar, it: (1) computes the OIS-discounted fixed-leg annuity A_OIS(T) over the full swap tenor, (2) sums the already-known float PV from prior blocks, (3) solves for the single flat forward rate f_new in the new block, and (4) compounds back to get the LIBOR projection DF. The divergence between single and multi curves is negligible below 10Y (<2 bps) but grows substantially at long tenors — 31 bps at 20Y, before a crossover where the multi-curve is 40 bps higher at 30Y. The crossover arises because OIS discounting gives more weight to far-dated cashflows, requiring bootstrapped LIBOR forwards to be higher to reprice the same par rates.

Term SOFR bootstrap uses annual-frequency SOFR swap quotes with quarterly floating payments. Same flat-forward logic, but now the fixed leg pays annually and the float leg pays quarterly. It then interpolates log-linearly between annual pillars to build a quarterly DF grid, from which 3M forward rates are extracted. The resulting curve has a humped shape: starts ~4.59% (current SOFR), dips to ~3.55% at 3Y (market pricing Fed cuts), then recovers to ~4.1% at 9–15Y.

### Part II — Swaption Calibration (`part2.py`)
The swaption dataset covers 15 expiry×tenor pairs (1Y, 5Y, 10Y expiries × 1Y, 2Y, 3Y, 5Y, 10Y tenors), with 11 strikes per smile expressed as ATM offsets from −200 to +200 bps. All pricing is Black-76 using OIS-discounted annuities.

ATM forward swap rates are computed from the OIS curve: the forward annuity A(Te, n) = 0.5 × ΣDo(Te + k/2), and S(Te, n) = [Do(Te) − Do(Te+n)] / A. Forward rates increase with expiry, from 3.28% at 1×1 to 4.66% at 10×10, reflecting the upward-sloping OIS term structure. The OIS curve is extrapolated log-linearly beyond 50Y to handle the 30×30 case.

Displaced Diffusion (DD) shifts the forward rate by β, making (S+β) lognormal with vol σ_DD. The calibration uses an ATM constraint to pin σ_DD for any given β (via Brent's method solving for the β that minimises price-space SSE). The diagnostic result is that β saturates at its boundary (~14.96%) for virtually all 15 smiles. This is a structural failure signal — the model only has one shape parameter to jointly control skew direction and smile curvature, which are two independent degrees of freedom. Short-expiry smiles with strong vol-of-vol are particularly problematic: RMSE of 2135 bps for the 1×1 smile.

SABR with β fixed at 0.75 (as assigned) calibrates (α, ρ, ν) per smile. The optimizer uses Nelder-Mead with multiple initial conditions for robustness, followed by an L-BFGS-B polish. Key patterns in the calibrated parameters:

α (initial vol): peaks in the 3Y–5Y tenor band, ranging 8–12.5%. Increases mildly with expiry as the vol surface is richer.
ρ (skew): uniformly negative across all 15 cells (−0.21 to −0.61). This captures the classic interest rate negative skew — when rates fall, vol rises (leveraged receiver dynamic). More negative at shorter expiries and longer tenors.
ν (vol-of-vol): highest at short expiries (2.03 at 1×1), declining monotonically with both expiry and tenor. Consistent with mean-reversion of realized volatility over time.

SABR reduces the 1×1 RMSE from 2135 to 316 bps (6.8× improvement), and achieves near-perfect fits at longer expiries (10×10 RMSE ≈ 7 bps). The 30×30 case is handled by flat constant extrapolation from the 10×10 parameters.
The report includes a sharp observation about why SABR performance matters for Part III: the static replication integral for CMS convexity correction integrates swaption prices over all strikes from 0 to ∞. The low-strike receiver swaptions that DD fails to price are precisely the instruments that dominate the convexity integral, especially at long expiry–tenor combinations. DD underpricing of the wings would materially understate convexity corrections.

### Part III — CMS Valuation (`part3.py`)
The core formula is Hagan's static replication for the CMS rate under the T_pay-forward measure. The change of numeraire from the annuity measure introduces the function f(K) = K/A(K), and its second derivative f''(K) is the weighting kernel in the replication integral:

CMS rate = (A0/Do_pay) × $[S0/A0 + f''_const × (∫₀^S receiver(K)dK + ∫_S^∞ payer(K)dK)]$

The code uses a linear TSR (Terminal Swap Rate) approximation for the annuity: A(K) ≈ A0 + A1·(K − S0), where A1 is approximated as −Duration_annuity × A0. This gives a constant f'' rather than a K-dependent one, which avoids a singularity at K_sing = A0/|A1| and is the standard industry approximation. The swaption prices inside the integrals use bilinearly interpolated SABR parameters (log-linear in α, linear in ρ and ν across the calibration grid).
The integration is done numerically with `np.trapezoid` over 200 points per side.

CMS10Y semi-annual leg (5Y): 10 payment dates at 0.5, 1.0, ..., 5.0Y. Convexity correction grows from 13.7 bps at 0.5Y to 223.8 bps at 5Y. PV = 0.2408.
CMS2Y quarterly leg (10Y): 40 payment dates. Smaller per-period correction due to shorter tenor (smaller annuity sensitivity) but accumulates over 10 years. PV = 0.4759.

The convexity correction table is the most striking output. At 10×10, the CMS rate is 11.10% vs a forward swap rate of only 4.66% — a correction of 643 bps, meaning the CMS rate is more than double the forward rate. This is driven by the joint effect of: (i) long expiry giving the stochastic vol process more time to create dispersion, (ii) long underlying tenor meaning a larger annuity sensitivity dA/dS, which amplifies the convex link between the swap rate and PV, and (iii) the SABR smile wings at the 10×10 cell having meaningful curvature (ν = 0.46) that makes the integration blow up.

---

## Detailed Report

