#!/usr/bin/env python3
# pinn.py
# Minimal stable "PINN-like" solver that matches pi.py physics:
# - NN approximates V(a) on a grid
# - policy from upwind finite differences (viscosity/entropy selection)
# - HJB residual uses discrete generator action
# - stationary density solved from KFE: A' g = 0 with normalization (atom appears naturally)

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.sparse import diags, bmat, eye
from scipy.sparse.linalg import spsolve


# =========================
# Parameters (same as pi.py)
# =========================
rho = 0.05
gamma = 2.0

y1, y2 = 0.1, 0.2
lambda1, lambda2 = 0.02, 0.03

A_MIN_INPUT = -3.2
A_MAX_INPUT = 50.0
R_TEST = 0.03

N_GRID = 10000

# Numerical safety (same spirit as pi.py)
VP_MIN = 1e-8
C_CAP = 1e6
A_MIN_C_FLOOR = 1e-8

# Training hyperparams
SEED = 0
DTYPE = torch.float64           # important for stability near a_min
LR = 5e-4
STEPS = 8000
PRINT_EVERY = 200

# Regularization (keep tiny; prevents oscillatory garbage)
W_SMOOTH = 1e-8      # smoothness penalty (second differences)
W_CONCAVE = 1e-8     # concavity penalty (positive second diff)
W_NEG_DV = 1e-10     # penalize negative dV_up (shouldn't happen)

# Weighting of residual near left tail (helps where V is steep)
LEFT_WEIGHT_ALPHA = 8.0
LEFT_WEIGHT_SCALE_FRAC = 0.05   # scale as fraction of (a_max - a_min)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================
# Utility (torch)
# ================
def u_torch(c: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(c, 1e-16, C_CAP)
    if abs(gamma - 1.0) < 1e-14:
        return torch.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def uprime_inv_torch(vp: torch.Tensor) -> torch.Tensor:
    vp = torch.clamp(vp, min=VP_MIN)
    c = vp ** (-1.0 / gamma)
    return torch.clamp(c, 1e-16, C_CAP)


# ===============
# Utility (numpy)
# ===============
def u_np(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 1e-16, C_CAP)
    if abs(gamma - 1.0) < 1e-14:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def uprime_inv_np(vp: np.ndarray) -> np.ndarray:
    vp = np.maximum(vp, VP_MIN)
    c = vp ** (-1.0 / gamma)
    return np.clip(c, 1e-16, C_CAP)


# =============================
# Grid + feasibility guard
# =============================
def build_grid(r: float, N: int):
    y_min = min(y1, y2)
    a_min_req = (-y_min + A_MIN_C_FLOOR) / r
    a_min = max(A_MIN_INPUT, a_min_req)
    a_max = A_MAX_INPUT

    a = np.linspace(a_min, a_max, N)
    da = a[1] - a[0]
    return a_min, a_max, a, da


# =======================
# NN for delta V (small)
# =======================
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=128, depth=3, out_dim=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        # important: start near 0 correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ============================================
# Upwind policy + drift from V on the full grid
# (this matches pi.py logic)
# ============================================
def policy_from_V_torch(V: torch.Tensor, a: torch.Tensor, da: float, a_min: float, a_max: float, r: float):
    # V: (N,2), a: (N,)
    N = a.shape[0]

    # incomes stacked
    y = torch.stack([torch.full((N,), y1, dtype=V.dtype, device=V.device),
                     torch.full((N,), y2, dtype=V.dtype, device=V.device)], dim=1)

    coh = torch.clamp(y + r * a[:, None], min=1e-8)
    dV0 = coh ** (-gamma)

    # forward/backward diffs
    dVf = torch.empty_like(V)
    dVb = torch.empty_like(V)

    dVf[:-1, :] = (V[1:, :] - V[:-1, :]) / da
    dVf[-1, 0] = torch.clamp(y[-1, 0] + r * a_max, min=1e-8) ** (-gamma)
    dVf[-1, 1] = torch.clamp(y[-1, 1] + r * a_max, min=1e-8) ** (-gamma)

    dVb[1:, :] = (V[1:, :] - V[:-1, :]) / da
    dVb[0, 0] = torch.clamp(torch.tensor(
        y1 + r * a_min, dtype=V.dtype, device=V.device), min=1e-8) ** (-gamma)
    dVb[0, 1] = torch.clamp(torch.tensor(
        y2 + r * a_min, dtype=V.dtype, device=V.device), min=1e-8) ** (-gamma)

    cf = uprime_inv_torch(dVf)
    cb = uprime_inv_torch(dVb)

    ssf = coh - cf
    ssb = coh - cb

    If = ssf > 0.0
    Ib = ssb < 0.0
    I0 = ~(If | Ib)

    dV_up = dVf * If + dVb * Ib + dV0 * I0

    c = uprime_inv_torch(dV_up)
    s = coh - c

    # state constraints: no leaving grid
    # (exactly like pi.py loop, but vectorized)
    c2 = c.clone()
    s2 = s.clone()

    # left boundary: if s[0,j] < 0 => set s=0, c=coh
    left_mask = s2[0, :] < 0.0
    if left_mask.any():
        s2[0, left_mask] = 0.0
        c2[0, left_mask] = coh[0, left_mask]

    # right boundary: if s[-1,j] > 0 => set s=0, c=coh
    right_mask = s2[-1, :] > 0.0
    if right_mask.any():
        s2[-1, right_mask] = 0.0
        c2[-1, right_mask] = coh[-1, right_mask]

    return c2, s2, dV_up, coh


# ============================================
# Discrete generator action Av (no sparse build)
# for each state, plus switching terms
# ============================================
def generator_apply_torch(V: torch.Tensor, s: torch.Tensor, da: float):
    # V: (N,2), s: (N,2)
    V1, V2 = V[:, 0], V[:, 1]
    s1, s2 = s[:, 0], s[:, 1]

    def drift_apply(Vj: torch.Tensor, sj: torch.Tensor):
        X = torch.relu(-sj) / da
        Z = torch.relu(sj) / da
        X = X.clone()
        Z = Z.clone()
        X[0] = 0.0
        Z[-1] = 0.0

        Av = -(X + Z) * Vj
        Av = Av.clone()
        Av[1:] += X[1:] * Vj[:-1]
        Av[:-1] += Z[:-1] * Vj[1:]
        return Av

    Av1 = drift_apply(V1, s1)
    Av2 = drift_apply(V2, s2)

    # switching (same as pi.py blocks)
    Av1_total = Av1 + lambda1 * (V2 - V1)
    Av2_total = Av2 + lambda2 * (V1 - V2)

    return torch.stack([Av1_total, Av2_total], dim=1)


# =====================
# KFE solve (as pi.py)
# =====================
def build_A_from_drift_np(s: np.ndarray, da: float):
    N = s.size
    X = np.maximum(-s, 0.0) / da
    Z = np.maximum(s, 0.0) / da
    X[0] = 0.0
    Z[-1] = 0.0
    main = -(X + Z)
    lower = X[1:]
    upper = Z[:-1]
    return diags([main, lower, upper], [0, -1, 1], shape=(N, N), format="csr")


def solve_kfe_np(A, da: float):
    # Solve A' g = 0 with normalization sum(g)*da = 1
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


# =====================
# Evaluation / plotting
# =====================
@torch.no_grad()
def evaluate(net, a_np, da, a_min, a_max, r):
    # compute V,c,s on torch then to numpy
    a = torch.tensor(a_np, dtype=DTYPE, device=DEVICE)
    x = ((a - a_min) / (a_max - a_min)) * 2.0 - 1.0
    x = x[:, None]

    # baseline guess exactly like pi.py init: V0 = u(coh)/rho with coh>=1e-4
    y_np = np.stack([np.full_like(a_np, y1), np.full_like(a_np, y2)], axis=1)
    c0_np = np.maximum(y_np + r * a_np[:, None], 1e-4)
    V0_np = u_np(c0_np) / rho
    V0 = torch.tensor(V0_np, dtype=DTYPE, device=DEVICE)

    dV = net(x)
    V = V0 + dV

    c, s, dV_up, coh = policy_from_V_torch(V, a, da, a_min, a_max, r)

    V_np = V.cpu().numpy()
    c_np = c.cpu().numpy()
    s_np = s.cpu().numpy()

    # build sparse generator A like pi.py
    N = a_np.size
    I_N = eye(N, format="csr")
    Aswitch = bmat([
        [-lambda1 * I_N,  lambda1 * I_N],
        [lambda2 * I_N, -lambda2 * I_N],
    ], format="csr")

    A1 = build_A_from_drift_np(s_np[:, 0], da)
    A2 = build_A_from_drift_np(s_np[:, 1], da)
    A_block = bmat([[A1, None], [None, A2]], format="csr")
    A_final = A_block + Aswitch

    g1, g2 = solve_kfe_np(A_final, da)

    Aagg = np.sum((g1 + g2) * a_np) * da
    Cagg = np.sum(g1 * c_np[:, 0] + g2 * c_np[:, 1]) * da
    m0 = (g1[0] + g2[0]) * da

    # boundary drifts
    s_amin = (s_np[0, 0], s_np[0, 1])
    s_amax = (s_np[-1, 0], s_np[-1, 1])

    return {
        "V": V_np,
        "c": c_np,
        "s": s_np,
        "g1": g1,
        "g2": g2,
        "Aagg": float(Aagg),
        "Cagg": float(Cagg),
        "m0": float(m0),
        "s_amin": s_amin,
        "s_amax": s_amax,
    }


def make_plots(a, out, title_suffix=""):
    V = out["V"]
    c = out["c"]
    g1 = out["g1"]
    g2 = out["g2"]
    da = a[1] - a[0]

    mass_per_bin = (g1 + g2) * da

    plt.figure()
    plt.plot(a, V[:, 0], label="v1")
    plt.plot(a, V[:, 1], label="v2")
    plt.title(f"Value functions {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(a, c[:, 0], label="c1")
    plt.plot(a, c[:, 1], label="c2")
    plt.title(f"Consumption policies {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(a, g1 + g2, label="density g1+g2")
    # show atom as bin-mass at a_min (like pi.py concept)
    plt.scatter([a[0]], [mass_per_bin[0] / da],
                label="(bin) mass at a_min", s=25)
    plt.title(f"Stationary density {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.show()


# =====================
# Training
# =====================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    a_min, a_max, a_np, da = build_grid(R_TEST, N_GRID)

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"Params: rho={rho}, gamma={gamma}, y1={y1}, y2={y2}, lambda1={lambda1}, lambda2={lambda2}, r={R_TEST}, a_min={a_min}, a_max={a_max}")
    print(f"N_GRID={N_GRID}, da={da:.6e}")
    print(
        f"cash-on-hand at a_min: y1+r*a_min={y1+R_TEST*a_min:.6e}, y2+r*a_min={y2+R_TEST*a_min:.6e}")

    # torch grid
    a = torch.tensor(a_np, dtype=DTYPE, device=DEVICE)
    x = ((a - a_min) / (a_max - a_min)) * 2.0 - 1.0
    x = x[:, None]

    # baseline guess like pi.py
    y_np = np.stack([np.full_like(a_np, y1), np.full_like(a_np, y2)], axis=1)
    c0_np = np.maximum(y_np + R_TEST * a_np[:, None], 1e-4)
    V0_np = u_np(c0_np) / rho
    V0 = torch.tensor(V0_np, dtype=DTYPE, device=DEVICE)

    # residual weights (focus on left tail)
    left_scale = LEFT_WEIGHT_SCALE_FRAC * (a_max - a_min)
    w = 1.0 + LEFT_WEIGHT_ALPHA * torch.exp(-(a - a_min) / left_scale)
    w = w / w.mean()

    net = MLP(in_dim=1, hidden=128, depth=3, out_dim=2).to(DEVICE).to(DTYPE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)

        dV = net(x)
        V = V0 + dV

        c, s, dV_up, coh = policy_from_V_torch(V, a, da, a_min, a_max, R_TEST)
        Av = generator_apply_torch(V, s, da)  # includes switching

        # HJB residual: rho V - u(c) - A V = 0
        uflow = u_torch(c)
        res = rho * V - uflow - Av

        loss_main = torch.mean(w[:, None] * (res ** 2))

        # smoothness + concavity via second differences on grid (discrete)
        # second diff: V[i+1] - 2 V[i] + V[i-1]
        V_mid = V[1:-1, :]
        sec = V[2:, :] - 2.0 * V_mid + V[:-2, :]
        loss_smooth = torch.mean(sec ** 2)
        # penalize positive curvature
        loss_concave = torch.mean(torch.relu(sec) ** 2)

        # dV_up should be positive
        loss_neg_dv = torch.mean(torch.relu(-dV_up) ** 2)

        loss = loss_main + W_SMOOTH * loss_smooth + \
            W_CONCAVE * loss_concave + W_NEG_DV * loss_neg_dv
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()

        if (step % PRINT_EVERY == 0) or (step == 1) or (step == STEPS):
            with torch.no_grad():
                max_res = torch.max(torch.abs(res)).item()
                vmin = torch.min(V).item()
                vmax = torch.max(V).item()
                cmean = torch.mean(c).item()
                s_amin = (s[0, 0].item(), s[0, 1].item())
                s_amax = (s[-1, 0].item(), s[-1, 1].item())

            print(f"[{step:5d}] loss={loss.item():.3e} main={loss_main.item():.3e} max|res|={max_res:.3e} "
                  f"Vrange=[{vmin:.1f},{vmax:.1f}] mean_c={cmean:.4f} "
                  f"s(a_min)=({s_amin[0]:.3e},{s_amin[1]:.3e}) s(a_max)=({s_amax[0]:.3e},{s_amax[1]:.3e})")

    print(f"Training done. Elapsed {time.time()-t0:.1f}s")

    # final evaluation with KFE solve
    out = evaluate(net, a_np, da, a_min, a_max, R_TEST)

    print("\n[Eval]")
    print(f"atom at a_min (bin mass) m0 ≈ {out['m0']:.8f}")
    print(f"Aggregate assets A ≈ {out['Aagg']:.8f}")
    print(f"Aggregate consumption C ≈ {out['Cagg']:.8f}")
    print(f"s(a_min)={out['s_amin']}  s(a_max)={out['s_amax']}")

    make_plots(a_np, out, title_suffix="(NN + upwind FD, KFE solve)")


if __name__ == "__main__":
    main()
