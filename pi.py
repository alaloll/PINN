#!/usr/bin/env python3
# pi_fixed_prettyplots_final.py
# Stable implicit upwind HJB + stationary KFE for 2-income Huggett/HACT-like model
#
# Key points:
# - Generator built from final drift s(a) with correct row-sum ~ 0
# - State-constraint boundaries (no-flow)
# - SAFE guard for a_min feasibility: y_min + r*a_min > 0 (otherwise V blows up artificially)
# - Plotting: robust y-scale + inset near a_min, discrete bin-mass shown as scatter (no fake "density line")
# - Atom (Dirac mass) is plotted ONLY if it is actually non-negligible.

import numpy as np
from scipy.sparse import diags, bmat, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time

# ----------------------------
# PARAMETERS (MODEL)
# ----------------------------
rho = 0.05
gamma = 2.0

y1, y2 = 0.1, 0.2
lambda1, lambda2 = 0.02, 0.03

# Asset grid (your intended values)
A_MIN_INPUT = -3.2
A_MAX_INPUT = 50.0

# Interest rate to solve at (single run)
R_TEST = 0.03

# Grid size
N_test = 10000

# HJB iteration controls
maxit_hjb = 5000
tol_hjb = 1e-8
Delta = 1000.0
print_every = 20

# Numerical safety
VP_MIN = 1e-8
C_CAP = 1e6
ROW_SUM_TOL = 1e-8

# Plot controls
PLOT_MASS_Q = 0.995
PLOT_MARGIN_FRAC = 0.08
PLOT_MAX_SCATTER = 6000

# Atom detect: if mass at first node below this, treat as "no atom"
ATOM_TOL = 1e-6

# Feasibility floor for y_min + r*a_min (to avoid c->0 artifacts in plots/initialization)
A_MIN_C_FLOOR = 1e-8


# ----------------------------
# Utility
# ----------------------------
def u(c):
    c = np.clip(c, 1e-16, C_CAP)
    if abs(gamma - 1.0) < 1e-14:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def uprime_inv(vp):
    vp = np.maximum(vp, VP_MIN)
    c = vp ** (-1.0 / gamma)
    return np.clip(c, 1e-16, C_CAP)


# ----------------------------
# Generator builder
# ----------------------------
def build_A_from_drift(s, da):
    """
    Upwind generator for drift s(a):
      move left intensity  X = max(-s,0)/da
      move right intensity Z = max( s,0)/da
    boundary no-flow: X[0]=0, Z[-1]=0
    diag = -(X+Z), offdiag = X (lower), Z (upper)
    """
    N = s.size
    X = np.maximum(-s, 0.0) / da
    Z = np.maximum(s, 0.0) / da

    X[0] = 0.0
    Z[-1] = 0.0

    main = -(X + Z)
    lower = X[1:]
    upper = Z[:-1]

    A = diags([main, lower, upper], [0, -1, 1], shape=(N, N), format="csr")
    return A


# ----------------------------
# HJB solver
# ----------------------------
def solve_hjb(r, a, da, a_min, a_max):
    N = a.size
    y = np.stack([np.full(N, y1), np.full(N, y2)], axis=1)

    # Safer initial guess than c0 = y + r a (which can be near/<=0 at left tail):
    c0 = np.maximum(y + r * a[:, None], 1e-4)
    V = u(c0) / rho

    I_N = eye(N, format="csr")
    Aswitch = bmat([
        [-lambda1 * I_N,  lambda1 * I_N],
        [lambda2 * I_N, -lambda2 * I_N]
    ], format="csr")

    # fallback derivative when drift ~ 0
    coh = np.maximum(y + r * a[:, None], 1e-8)   # avoid negative/zero in power
    dV0 = coh ** (-gamma)

    for it in range(maxit_hjb):
        V_old = V.copy()

        dVf = np.empty_like(V)
        dVb = np.empty_like(V)

        dVf[:-1, :] = (V[1:, :] - V[:-1, :]) / da
        dVf[-1, :] = (np.maximum(y[-1, :] + r * a_max, 1e-8)) ** (-gamma)

        dVb[1:, :] = (V[1:, :] - V[:-1, :]) / da
        dVb[0, 0] = (np.maximum(y1 + r * a_min, 1e-8)) ** (-gamma)
        dVb[0, 1] = (np.maximum(y2 + r * a_min, 1e-8)) ** (-gamma)

        cf = uprime_inv(dVf)
        cb = uprime_inv(dVb)

        ssf = y + r * a[:, None] - cf
        ssb = y + r * a[:, None] - cb

        If = ssf > 0.0
        Ib = ssb < 0.0
        I0 = ~(If | Ib)

        dV_up = dVf * If + dVb * Ib + dV0 * I0

        c = uprime_inv(dV_up)
        s = y + r * a[:, None] - c

        # state constraints (no leaving grid)
        for j in range(2):
            if s[0, j] < 0.0:
                s[0, j] = 0.0
                c[0, j] = y[0, j] + r * a_min
            if s[-1, j] > 0.0:
                s[-1, j] = 0.0
                c[-1, j] = y[-1, j] + r * a_max

        uflow = u(c)

        A1 = build_A_from_drift(s[:, 0], da)
        A2 = build_A_from_drift(s[:, 1], da)
        A_block = bmat([[A1, None], [None, A2]], format="csr")
        A = A_block + Aswitch

        rowSum = np.max(np.abs(A.sum(axis=1).A1))

        B = ((rho + 1.0 / Delta) * eye(2 * N, format="csr")) - A
        rhs = np.concatenate([uflow[:, 0], uflow[:, 1]]) + \
            np.concatenate([V_old[:, 0], V_old[:, 1]]) / Delta

        V_new_stacked = spsolve(B.tocsr(), rhs)
        V_new = np.column_stack([V_new_stacked[:N], V_new_stacked[N:]])

        diff = np.max(np.abs(V_new - V_old))
        V = V_new

        if (it % print_every == 0) or (it == maxit_hjb - 1):
            print(f"HJB it {it:4d}, diff = {diff:.3e}, rowSum = {rowSum:.3e}")

        if not np.all(np.isfinite(V)):
            raise RuntimeError("HJB produced non-finite V (nan/inf).")

        if diff < tol_hjb:
            if rowSum > ROW_SUM_TOL:
                raise RuntimeError(
                    f"HJB converged in V but generator rowSum={rowSum:.3e} not ~0."
                )
            print(
                f"HJB converged in {it+1} iterations, diff={diff:.3e}, rowSum={rowSum:.3e}")
            break
    else:
        raise RuntimeError(
            f"HJB did not converge within {maxit_hjb} iterations (last diff={diff:.3e}).")

    # rebuild final policy and final A cleanly
    y = np.stack([np.full(N, y1), np.full(N, y2)], axis=1)
    coh = np.maximum(y + r * a[:, None], 1e-8)
    dV0 = coh ** (-gamma)

    dVf = np.empty_like(V)
    dVb = np.empty_like(V)
    dVf[:-1, :] = (V[1:, :] - V[:-1, :]) / da
    dVf[-1, :] = (np.maximum(y[-1, :] + r * a_max, 1e-8)) ** (-gamma)
    dVb[1:, :] = (V[1:, :] - V[:-1, :]) / da
    dVb[0, 0] = (np.maximum(y1 + r * a_min, 1e-8)) ** (-gamma)
    dVb[0, 1] = (np.maximum(y2 + r * a_min, 1e-8)) ** (-gamma)

    cf = uprime_inv(dVf)
    cb = uprime_inv(dVb)
    ssf = y + r * a[:, None] - cf
    ssb = y + r * a[:, None] - cb

    If = ssf > 0.0
    Ib = ssb < 0.0
    I0 = ~(If | Ib)
    dV_up = dVf * If + dVb * Ib + dV0 * I0

    c = uprime_inv(dV_up)
    s = y + r * a[:, None] - c

    for j in range(2):
        if s[0, j] < 0.0:
            s[0, j] = 0.0
            c[0, j] = y[0, j] + r * a_min
        if s[-1, j] > 0.0:
            s[-1, j] = 0.0
            c[-1, j] = y[-1, j] + r * a_max

    I_N = eye(N, format="csr")
    Aswitch = bmat([
        [-lambda1 * I_N,  lambda1 * I_N],
        [lambda2 * I_N, -lambda2 * I_N]
    ], format="csr")

    A1 = build_A_from_drift(s[:, 0], da)
    A2 = build_A_from_drift(s[:, 1], da)
    A_block = bmat([[A1, None], [None, A2]], format="csr")
    A_final = A_block + Aswitch

    rowSum_final = np.max(np.abs(A_final.sum(axis=1).A1))
    print(f"Final generator check: max |row-sum| = {rowSum_final:.3e}")

    return V[:, 0], V[:, 1], c[:, 0], c[:, 1], s[:, 0], s[:, 1], A_final


# ----------------------------
# Stationary KFE solver
# ----------------------------
def solve_kfe(A, da):
    """
    Solve A' g = 0 with normalization sum(g)*da = 1
    Replace first equation by normalization.
    """
    N2 = A.shape[0]
    AT = A.transpose().tocsr().tolil()

    b = np.zeros(N2)
    AT[0, :] = 1.0
    b[0] = 1.0

    gg = spsolve(AT.tocsr(), b)

    if not np.all(np.isfinite(gg)):
        raise RuntimeError("KFE returned non-finite g (nan/inf).")

    total = np.sum(gg) * da
    if (not np.isfinite(total)) or (total <= 0):
        raise RuntimeError(f"KFE normalization failed: total mass = {total}")

    gg /= total
    return gg[:N2 // 2], gg[N2 // 2:]


# ----------------------------
# Wrapper with a_min feasibility guard
# ----------------------------
def solve_model_for_r(r, N):
    y_min = min(y1, y2)

    # feasibility: y_min + r*a_min > 0  ->  a_min > -y_min/r
    a_min_req = (-y_min + A_MIN_C_FLOOR) / r
    a_min = max(A_MIN_INPUT, a_min_req)
    a_max = A_MAX_INPUT

    if a_min != A_MIN_INPUT:
        print("\n[IMPORTANT] a_min was infeasible for given (r, y_min).")
        print(f"  Input A_MIN_INPUT = {A_MIN_INPUT}")
        print(f"  Required a_min > -y_min/r ≈ {-y_min/r:.6f}")
        print(
            f"  Using a_min = {a_min:.6f} so that y_min + r*a_min ≈ {y_min + r*a_min:.3e} > 0\n")

    a = np.linspace(a_min, a_max, N)
    da = a[1] - a[0]
    print(
        f"=== Solving model for r = {r:.6f}, N = {N}, da = {da:.6e}, a_min = {a_min:.6f}, a_max = {a_max:.6f}")

    v1, v2, c1, c2, s1, s2, A = solve_hjb(r, a, da, a_min=a_min, a_max=a_max)
    g1, g2 = solve_kfe(A, da)

    Aagg = np.sum((g1 + g2) * a) * da
    Cagg = np.sum((g1 * c1 + g2 * c2)) * da
    print(
        f"Aggregate assets A = {Aagg:.8f}, Aggregate consumption C = {Cagg:.8f}")

    # extra sanity numbers
    print(f"Min cash-on-hand (y1+r*a) at a_min: {y1 + r*a_min:.6e}")
    print(f"Min cash-on-hand (y2+r*a) at a_min: {y2 + r*a_min:.6e}")

    return dict(a=a, da=da, v1=v1, v2=v2, c1=c1, c2=c2, s1=s1, s2=s2, g1=g1, g2=g2, A=Aagg, C=Cagg, a_min=a_min, a_max=a_max)


# ----------------------------
# Plot helpers
# ----------------------------
def _maybe_downsample_for_scatter(x, y, max_points=PLOT_MAX_SCATTER):
    n = x.size
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]


def _mass_zoom_xlim(a, mass_per_bin, q=PLOT_MASS_Q, margin_frac=PLOT_MARGIN_FRAC):
    cum = np.cumsum(mass_per_bin)
    total = cum[-1]
    idx = int(np.searchsorted(cum, q * total))
    idx = min(max(idx, 1), len(a) - 1)
    left = a[0]
    right = a[idx]
    span = max(1e-12, right - left)
    return left - margin_frac * span, right + margin_frac * span, idx, total


def _robust_ylim(y, lo=0.01, hi=0.99, pad_frac=0.08):
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    ql, qh = np.quantile(y, [lo, hi])
    if not np.isfinite(ql) or not np.isfinite(qh) or ql == qh:
        return None
    pad = pad_frac * (qh - ql)
    return ql - pad, qh + pad


def _beautify_axis_no_offset(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(True)
    ax.yaxis.set_major_formatter(fmt)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    start = time.time()

    out = solve_model_for_r(r=R_TEST, N=N_test)
    a = out["a"]
    da = out["da"]

    v1, v2 = out["v1"], out["v2"]
    c1, c2 = out["c1"], out["c2"]
    g1, g2 = out["g1"], out["g2"]

    g_tot = g1 + g2
    mass_per_bin = g_tot * da

    # atom detection
    m0_raw = mass_per_bin[0]
    m0 = float(max(m0_raw, 0.0))  # tiny negative can happen numerically
    has_atom = (m0 > ATOM_TOL)
    cont_mass = 1.0 - m0

    # x-zoom for distribution plots
    xL, xR, idx_q, total_mass_check = _mass_zoom_xlim(a, mass_per_bin)

    # choose what we call "continuous part"
    if has_atom:
        a_cont = a[1:]
        g_cont_raw = g_tot[1:]
        c_density = g1 * c1 + g2 * c2
        c_density_cont = c_density[1:]
        # conditional density given not at atom
        g_cont_cond = g_cont_raw / max(cont_mass, 1e-16)
        cont_note = f"atom detected: m0≈{m0:.6g}, cont mass≈{cont_mass:.6g}"
    else:
        a_cont = a
        g_cont_raw = g_tot
        c_density = g1 * c1 + g2 * c2
        c_density_cont = c_density
        g_cont_cond = g_cont_raw  # same
        cont_note = "no atom (numerical m0≈0) → conditional = raw"

    # ----------------------------
    # FIGURE: 6 panels
    # ----------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 1.05])

    axV = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axBin = fig.add_subplot(gs[1, 0])
    axRaw = fig.add_subplot(gs[1, 1])
    axCond = fig.add_subplot(gs[2, 0])
    axCdens = fig.add_subplot(gs[2, 1])

    # ---- Value functions (robust y-scale + inset near a_min)
    axV.plot(a, v1, label="v1")
    axV.plot(a, v2, label="v2")
    axV.set_title("Value functions (robust y-scale + inset near a_min)")
    axV.grid(True)
    axV.legend()

    yl = _robust_ylim(np.r_[v1, v2], lo=0.02, hi=0.98)
    if yl is not None:
        axV.set_ylim(*yl)

    # inset near a_min (show full un-clipped behavior in left tail)
    axVins = inset_axes(axV, width="45%", height="40%",
                        loc="lower right", borderpad=1.1)
    left_span = a[0] + 0.18 * (a[-1] - a[0])
    mask_left = a <= left_span
    axVins.plot(a[mask_left], v1[mask_left])
    axVins.plot(a[mask_left], v2[mask_left])
    axVins.grid(True, alpha=0.4)
    axVins.set_title("inset: left tail", fontsize=9)

    # ---- Consumption (with inset near a_min)
    axC.plot(a, c1, label="c1")
    axC.plot(a, c2, label="c2")
    axC.set_title("Consumption policies (with inset near a_min)")
    axC.grid(True)
    axC.legend()

    axCins = inset_axes(axC, width="45%", height="40%",
                        loc="upper left", borderpad=1.1)
    axCins.plot(a[mask_left], c1[mask_left])
    axCins.plot(a[mask_left], c2[mask_left])
    axCins.grid(True, alpha=0.4)
    axCins.set_title("inset: left tail", fontsize=9)

    # ---- Bin-mass (DISCRETE) as scatter + vlines
    xb, yb = _maybe_downsample_for_scatter(a, mass_per_bin)
    axBin.scatter(xb, yb, s=10, marker=".",
                  label="bin mass = (g1+g2)*da", rasterized=True)
    # add thin vlines for extra clarity of "discrete bins"
    xvl, yvl = _maybe_downsample_for_scatter(a, mass_per_bin, max_points=2500)
    axBin.vlines(xvl, 0.0, yvl, linewidth=0.3, alpha=0.6)

    # mass cutoff marker
    axBin.vlines(a[idx_q], 0.0, max(np.max(yb), 1e-16) * 1.02,
                 linestyles=":", label=f"{PLOT_MASS_Q*100:.2f}% mass cutoff")

    # atom marker only if meaningful
    if has_atom:
        axBin.vlines(a[0], 0.0, m0, linestyles="--",
                     label=f"atom at a_min: m0≈{m0:.4g}")
        axBin.scatter([a[0]], [m0], s=70, marker="o")

    axBin.set_xlim(xL, xR)
    axBin.set_ylim(bottom=0.0)
    axBin.set_title(
        "Stationary distribution as DISCRETE bin-mass (scatter)\n"
        f"sum mass≈{total_mass_check:.6f}, {cont_note}"
    )
    axBin.grid(True)
    axBin.legend(loc="best")

    # ---- Raw continuous density (whatever we define as continuous part)
    axRaw.plot(a_cont, g_cont_raw, label="raw density used for plots")
    axRaw.set_xlim(xL, xR)
    axRaw.set_ylim(bottom=0.0)
    axRaw.set_title("Density plot (raw) — not bin-mass")
    axRaw.grid(True)
    axRaw.legend(loc="best")
    _beautify_axis_no_offset(axRaw)

    # ---- Conditional density (only differs if atom exists)
    axCond.plot(a_cont, g_cont_cond,
                label="conditional density (if atom exists)")
    axCond.set_xlim(xL, xR)
    axCond.set_ylim(bottom=0.0)
    axCond.set_title(
        "Conditional density given not at atom (if no atom → same as raw)")
    axCond.grid(True)
    axCond.legend(loc="best")
    _beautify_axis_no_offset(axCond)

    # ---- Consumption density (separate panel, no twinx)
    axCdens.plot(a_cont, c_density_cont, label="c-density = g1*c1 + g2*c2")
    axCdens.set_xlim(xL, xR)
    axCdens.set_ylim(bottom=0.0)
    axCdens.set_title("Consumption density (same x-zoom)")
    axCdens.grid(True)
    axCdens.legend(loc="best")
    _beautify_axis_no_offset(axCdens)

    plt.tight_layout()
    plt.show()

    print("Elapsed (sec):", time.time() - start)
