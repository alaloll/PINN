# =========================
# FAST + FAIR: MLP-PINN vs Transformer-PINN (BLIND physics selection)
# - No jit_compile=True (avoids 10-30min XLA compile on fullbatch)
# - Fixed-shape minibatches for speed and stability
# - Separate "moment" batch for ce_pred to stabilize r^l
# - Same training procedure and time budget for both models
# - Oracle metrics only for logging / final report (NOT used for ckpt)
# =========================

import os, time, random
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import tensorflow as tf

# -------------------------
# CONFIG (edit here)
# -------------------------
SEED = 0

# Time budgets (wall-clock fairness)
MAX_MINUTES_PER_MODEL = 120  # <= set 60..180 as you want
MAX_STEPS_CAP = 300000       # safety cap (usually time stops earlier)
DISPLAY_EVERY = 500          # logging frequency in steps
PATIENCE_LOGS = 30           # early stop if no val improvement for this many logs

# Batches (fixed shapes, important for speed)
BATCH_PDE   = 4096          # PDE points for 2nd-derivs (main heavy part)
BATCH_MOM   = 16384         # moment batch for ce_pred (forward only, stabilizes r^l)
BATCH_BC_E  = 1024          # e=e_min Dirichlet points
BATCH_BC_Z  = 1024          # z=z_min/z_max Neumann points

# Validation batches (physics-only, blind)
VAL_BATCH_PDE  = 8192
VAL_BATCH_MOM  = 16384
VAL_BATCH_BC_E = 1024
VAL_BATCH_BC_Z = 1024

# Pool sizes (pre-sampled points for very fast gather)
POOL_DOMAIN = 200000
POOL_BC_E   = 20000
POOL_BC_Z   = 20000

# Heatmaps
HEAT_N = 300        # 400 is heavier; 300 is already clear
RESID_N = 200       # residual maps are expensive (need 2nd deriv), keep smaller

# Optim
LR0 = 5e-4
DECAY_STEPS = 6000
CLIPNORM = 1.0
RL_LR_MULT = 0.1     # rl lr = LR0 * RL_LR_MULT  (smaller helps prevent overshoot)

# Loss weights (same as original)
LW = tf.constant([1e6, 5e4, 1e3, 1e2, 1e3, 1e5], dtype=tf.float32)

# Transformer size (keep modest for speed; you can tune after baseline)
TR_D_MODEL = 32
TR_HEADS   = 4
TR_LAYERS  = 2
TR_D_FF    = 128
TR_SEQ_LEN = 3
TR_HEAD_H  = 64

# -------------------------
# Repro / TF setup
# -------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx("float32")

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# IMPORTANT: do NOT use jit_compile=True here.
# XLA auto-jit also can compile clusters; for "best time to first step" keep it OFF.
# If you later want to experiment, set before import TF:
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

# -------------------------
# Problem constants (same as your code)
# -------------------------
DATA_PATH = "data/"

k = 0.005
a = 0.3
phi = 10.0
w = 1.0
r = 0.03
sigma = 0.08
cf = 0.03
ce = 0.1

ind_g_eps = 0.015

z_min = 0.2
z_max = 10.0
z_mean = 5.0
e_min = 0.01
e_max = 1.2

stdev = (z_max - z_min) / 4.0
area = (z_max - z_min) * (e_max - e_min)

# -------------------------
# Load reference data (oracle only)
# -------------------------
e_true = np.loadtxt(open(f"{DATA_PATH}hjb_e.csv", "r"), delimiter=",", dtype=np.float32).flatten()
z_true = np.loadtxt(open(f"{DATA_PATH}hjb_z.csv", "r"), delimiter=",", dtype=np.float32).flatten()
v_true = np.loadtxt(open(f"{DATA_PATH}hjb_v.csv", "r"), delimiter=",", dtype=np.float32).flatten()[:, None]
g_true = np.loadtxt(open(f"{DATA_PATH}hjb_g.csv", "r"), delimiter=",", dtype=np.float32).flatten()[:, None]
X_true = np.vstack((e_true, z_true)).T

def func_solution_griddata(x_np):
    v_interp = griddata(X_true, v_true, x_np)
    g_interp = griddata(X_true, g_true, x_np)
    return np.hstack((v_interp, g_interp))

def v_l2_metric(y_true_np, y_pred_np):
    v_t = y_true_np[:, 0]
    v_p = y_pred_np[:, 0]
    mask = np.isfinite(v_t) & np.isfinite(v_p)
    v_t = v_t[mask]; v_p = v_p[mask]
    return np.linalg.norm(v_t - v_p) / (np.linalg.norm(v_t) + 1e-12)

def g_l2_metric(y_true_np, y_pred_np):
    g_t = y_true_np[:, 1]
    g_p = y_pred_np[:, 1]
    mask = np.isfinite(g_t) & np.isfinite(g_p)
    g_t = g_t[mask]; g_p = g_p[mask]
    g_t = np.maximum(g_t, 0)
    g_p = np.maximum(g_p, 0)
    g_t = g_t / (np.sum(g_t) + 1e-12)
    g_p = g_p / (np.sum(g_p) + 1e-12)
    return np.linalg.norm(g_t - g_p) / (np.linalg.norm(g_t) + 1e-12)

RL_TRUE_ORACLE = 0.043343

# -------------------------
# Sampling pools (fast gather, fixed shapes)
# -------------------------
def sample_domain_np(n):
    e = e_min + (e_max - e_min) * np.random.rand(n, 1).astype(np.float32)
    z = z_min + (z_max - z_min) * np.random.rand(n, 1).astype(np.float32)
    return np.hstack([e, z]).astype(np.float32)

def sample_bc_e_np(n):  # e = e_min, z uniform
    e = np.full((n,1), e_min, dtype=np.float32)
    z = z_min + (z_max - z_min) * np.random.rand(n,1).astype(np.float32)
    return np.hstack([e,z]).astype(np.float32)

def sample_bc_z_np(n):  # z in {z_min, z_max}, e uniform
    e = e_min + (e_max - e_min) * np.random.rand(n,1).astype(np.float32)
    side = (np.random.rand(n,1) > 0.5).astype(np.float32)
    z = z_min*(1.0-side) + z_max*side
    return np.hstack([e,z]).astype(np.float32)

pool_domain = tf.constant(sample_domain_np(POOL_DOMAIN))
pool_bc_e   = tf.constant(sample_bc_e_np(POOL_BC_E))
pool_bc_z   = tf.constant(sample_bc_z_np(POOL_BC_Z))

val_domain = tf.constant(sample_domain_np(POOL_DOMAIN//5))
val_bc_e   = tf.constant(sample_bc_e_np(POOL_BC_E//2))
val_bc_z   = tf.constant(sample_bc_z_np(POOL_BC_Z//2))

# oracle monitor set (NOT for training decisions)
ORACLE_N = 8192
oracle_x = sample_domain_np(ORACLE_N)
oracle_y = func_solution_griddata(oracle_x)
oracle_x_tf = tf.constant(oracle_x)

# -------------------------
# Math helpers
# -------------------------
@tf.function
def psi_tf(e, z):
    # e,z: (B,1)
    psi_unnorm = (1.0 / (stdev * tf.sqrt(tf.constant(2.0*pi, tf.float32))) *
                  tf.exp(-0.5 * tf.square((z - z_mean) / stdev)))
    return psi_unnorm * tf.constant(7.471114, tf.float32) * tf.cast(e <= 0.15, tf.float32)

@tf.function
def integrate_mean_tf(val):
    return tf.reduce_mean(val) * tf.constant(area, tf.float32)

@tf.function
def f_tf(m, n):
    return m * tf.pow(n, a)

# -------------------------
# Models
# -------------------------
class MLP_PINN(tf.keras.Model):
    def __init__(self, width=64, depth=6):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform(seed=SEED)
        self.hidden = []
        for i in range(depth):
            self.hidden.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=init))
        self.out = tf.keras.layers.Dense(2, activation=None, kernel_initializer=init)

    @tf.function
    def call(self, x):
        h = x
        for lyr in self.hidden:
            h = lyr(h)
        y = self.out(h)
        e = x[:,0:1]
        v = y[:,0:1]
        g_raw = y[:,1:2]
        ind_g = tf.cast(v > (e + ind_g_eps), tf.float32)
        g = tf.square(g_raw * ind_g)
        return tf.concat([v,g], axis=1)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation=tf.nn.tanh),
            tf.keras.layers.Dense(d_ff, activation=tf.nn.tanh),
            tf.keras.layers.Dense(d_model, activation=None),
        ])

    def call(self, x, training=False):
        y = self.ln1(x)
        y = self.attn(y, y, training=training)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y, training=training)
        return x + y

class Transformer_PINN(tf.keras.Model):
    def __init__(self, d_model=32, n_heads=4, d_ff=128, n_layers=2, seq_len=3, head_hidden=64):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform(seed=SEED)
        self.seq_len = seq_len
        self.emb = tf.keras.layers.Dense(d_model, activation=None, kernel_initializer=init)
        # IMPORTANT: index-based positional embedding (1,L,d_model)
        self.pos = self.add_weight("pos_emb", shape=(1, seq_len, d_model),
                                   initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=SEED),
                                   trainable=True)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(head_hidden, activation=tf.nn.tanh, kernel_initializer=init),
            tf.keras.layers.Dense(head_hidden, activation=tf.nn.tanh, kernel_initializer=init),
            tf.keras.layers.Dense(2, activation=None, kernel_initializer=init),
        ])

    @tf.function
    def call(self, x, training=False):
        # x: (B,2)
        B = tf.shape(x)[0]
        tokens = tf.repeat(x[:,None,:], repeats=self.seq_len, axis=1)  # (B,L,2)
        h = self.emb(tokens) + self.pos                              # (B,L,d_model)
        for blk in self.blocks:
            h = blk(h, training=training)
        cls = h[:,0,:]                                                # (B,d_model)
        y = self.head(cls, training=training)                         # (B,2)

        e = x[:,0:1]
        v = y[:,0:1]
        g_raw = y[:,1:2]
        ind_g = tf.cast(v > (e + ind_g_eps), tf.float32)
        g = tf.square(g_raw * ind_g)
        return tf.concat([v,g], axis=1)

# -------------------------
# Derivatives + residuals
# -------------------------
@tf.function
def model_and_derivs_2nd(model, x):
    # x: (B,2) -> returns v,g and needed derivs
    e = x[:,0:1]
    z = x[:,1:2]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([e,z])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([e,z])
            y = model(tf.concat([e,z], axis=1), training=True)
            v = y[:,0:1]
            g = y[:,1:2]
        dv_de = tape1.gradient(v, e)
        dv_dz = tape1.gradient(v, z)
        dg_de = tape1.gradient(g, e)
        dg_dz = tape1.gradient(g, z)
    dv_zz = tape2.gradient(dv_dz, z)
    dg_zz = tape2.gradient(dg_dz, z)
    del tape1
    del tape2
    return e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz

@tf.function
def model_and_dz_1st(model, x):
    # for BC on z: only need dv_dz and dg_dz (1st deriv)
    e = x[:,0:1]
    z = x[:,1:2]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([e,z])
        y = model(tf.concat([e,z], axis=1), training=True)
        v = y[:,0:1]
        g = y[:,1:2]
    dv_dz = tape.gradient(v, z)
    dg_dz = tape.gradient(g, z)
    del tape
    return e,z,v,g,dv_dz,dg_dz

@tf.function
def pde_residuals(e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz, rl):
    mu = -0.005*(z - 5.0)

    base1 = tf.maximum((rl - r) * z * a / w, 1e-12)
    term1 = tf.pow(base1, 1.0/(1.0-a))
    base2 = tf.maximum(phi * e / z, 1e-12)
    term2 = tf.pow(base2, 1.0/a)

    ind = tf.cast(term1 < term2, tf.float32)
    l_star = tf.minimum(term1, term2)

    pi_star = 2.0*(rl - r)*z*tf.pow(l_star, a) + e*r - 2.0*w*l_star - cf
    zeta = k * tf.maximum(0.0, phi*e - f_tf(z, l_star))
    v_u = e

    inside = (pi_star*(1.0+dv_de) + (1.0-dv_de)*ind*zeta + dv_dz*mu + 0.5*dv_zz*(sigma**2))
    hjb = r*v - tf.maximum(inside, r*v_u)

    psi_val = psi_tf(e,z)
    m_fixed = 0.1
    mu_z = -0.005
    mu_e = pi_star - ind*zeta

    l_star_e = (1.0 - ind) * tf.pow(tf.maximum(phi / z, 1e-12), 1.0/a) * (1.0/a) * tf.pow(tf.maximum(e,1e-12), (1.0/a)-1.0)
    pi_star_e = 2.0*(rl-r)*z*a*tf.pow(l_star, a-1.0)*l_star_e + r - 2.0*w*l_star_e

    zeta_e = k * tf.cast(phi*e > z*tf.pow(l_star,a), tf.float32) * (phi - z*a*tf.pow(l_star,a-1.0)*l_star_e)
    mu_ee = pi_star_e - zeta_e*ind

    ind_g = tf.cast((v - e) > ind_g_eps, tf.float32)

    kfe = (-mu_z*g - mu*dg_dz - mu_ee*g - mu_e*dg_de + 0.5*dg_zz*(sigma**2) + m_fixed*psi_val*ind_g) * ind_g

    return hjb, kfe

@tf.function
def sample_from_pool(pool, batch_size):
    n = tf.shape(pool)[0]
    idx = tf.random.uniform((batch_size,), minval=0, maxval=n, dtype=tf.int32)
    return tf.gather(pool, idx)

@tf.function
def compute_ce_pred(model, x_mom, rl):
    # ce_pred = integral v * psi over region (Monte-Carlo)
    e = x_mom[:,0:1]
    z = x_mom[:,1:2]
    y = model(x_mom, training=True)
    v = y[:,0:1]
    psi_val = psi_tf(e,z)
    ce_pred = integrate_mean_tf(v * psi_val)
    return ce_pred

@tf.function
def compute_losses(model, rl_var,
                   x_pde, x_mom, x_bc_e, x_bc_z):
    rl = rl_var / 100.0

    ce_pred = compute_ce_pred(model, x_mom, rl)

    # PDE
    e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz = model_and_derivs_2nd(model, x_pde)
    hjb, kfe = pde_residuals(e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz, rl)

    loss_hjb = tf.reduce_mean(tf.square(hjb))
    loss_kfe = tf.reduce_mean(tf.square(kfe))
    loss_free = tf.square(ce_pred - ce)  # scalar^2, same as mse of constant vector

    # BC e=e_min: v = 0.01
    y_e = model(x_bc_e, training=True)
    v_e = y_e[:,0:1]
    loss_bcDv = tf.reduce_mean(tf.square(v_e - 0.01))

    # BC z=z_min/z_max: dv/dz=0 and -mu*g + 0.5*sigma^2*dg/dz = 0
    e_b,z_b,v_b,g_b,dv_dz_b,dg_dz_b = model_and_dz_1st(model, x_bc_z)
    loss_bcNv = tf.reduce_mean(tf.square(dv_dz_b))
    mu_b = -0.005*(z_b - 5.0)
    bcNg_res = -mu_b*g_b + 0.5*(sigma**2)*dg_dz_b
    loss_bcNg = tf.reduce_mean(tf.square(bcNg_res))

    losses_vec = tf.stack([loss_hjb, loss_kfe, loss_free, loss_bcDv, loss_bcNv, loss_bcNg], axis=0)
    total = tf.reduce_sum(LW * losses_vec)

    return total, losses_vec, ce_pred

# -------------------------
# Training (blind ckpt by physics val_total)
# -------------------------
def train_model(model, name):
    print(f"\n=== TRAIN {name} ===")
    rl_train = tf.Variable(5.0, dtype=tf.float32, trainable=True)  # rl = rl_train/100

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=LR0,
        decay_steps=DECAY_STEPS,
        decay_rate=1.0,
        staircase=False,
    )
    # separate lr for rl (smaller)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # one dummy call to build weights
    _ = model(tf.constant(sample_domain_np(8)), training=True)

    @tf.function
    def train_step():
        x_pde = sample_from_pool(pool_domain, BATCH_PDE)
        x_mom = sample_from_pool(pool_domain, BATCH_MOM)
        x_be  = sample_from_pool(pool_bc_e,   BATCH_BC_E)
        x_bz  = sample_from_pool(pool_bc_z,   BATCH_BC_Z)

        with tf.GradientTape() as tape:
            total, losses_vec, ce_pred = compute_losses(model, rl_train, x_pde, x_mom, x_be, x_bz)

        vars_all = model.trainable_variables + [rl_train]
        grads = tape.gradient(total, vars_all)

        # split grads: apply smaller lr to rl by scaling its grad (cheap + stable)
        grads_net = grads[:-1]
        grad_rl   = grads[-1]

        # clip
        gnorm = tf.linalg.global_norm(grads_net)
        if CLIPNORM is not None:
            grads_net, _ = tf.clip_by_global_norm(grads_net, CLIPNORM)
            grad_rl = tf.clip_by_value(grad_rl, -CLIPNORM, CLIPNORM)

        # apply: net
        opt.apply_gradients(zip(grads_net, model.trainable_variables))
        # apply: rl (manual SGD-style step with scaled lr)
        lr_now = lr_schedule(opt.iterations)
        rl_train.assign_sub(RL_LR_MULT * lr_now * grad_rl)

        return total, losses_vec, ce_pred, gnorm

    @tf.function
    def val_step():
        x_pde = sample_from_pool(val_domain, VAL_BATCH_PDE)
        x_mom = sample_from_pool(val_domain, VAL_BATCH_MOM)
        x_be  = sample_from_pool(val_bc_e,   VAL_BATCH_BC_E)
        x_bz  = sample_from_pool(val_bc_z,   VAL_BATCH_BC_Z)
        total, losses_vec, ce_pred = compute_losses(model, rl_train, x_pde, x_mom, x_be, x_bz)
        return total, losses_vec, ce_pred

    hist = {
        "step": [], "tsec": [],
        "train_total": [], "val_total": [],
        "hjb": [], "kfe": [], "free": [], "bcDv": [], "bcNv": [], "bcNg": [],
        "rl": [], "ce": [],
        "oracle_v_l2": [], "oracle_g_l2": [],
    }

    best_val = float("inf")
    best_weights = None
    best_rl = None
    bad_logs = 0

    t0 = time.time()
    last_log = t0

    for step in range(1, MAX_STEPS_CAP+1):
        tr_total, tr_losses, tr_ce, tr_gnorm = train_step()

        now = time.time()
        elapsed = now - t0
        if elapsed > MAX_MINUTES_PER_MODEL * 60:
            break

        if (step % DISPLAY_EVERY == 0) or (step == 1):
            v_total, v_losses, v_ce = val_step()

            tr_total_f = float(tr_total.numpy())
            v_total_f  = float(v_total.numpy())
            rl_val = float((rl_train.numpy())/100.0)
            ce_val = float(v_ce.numpy())

            # oracle metrics (LOG ONLY)
            y_pred = model(oracle_x_tf, training=False).numpy()
            ov = v_l2_metric(oracle_y, y_pred)
            og = g_l2_metric(oracle_y, y_pred)

            hist["step"].append(step)
            hist["tsec"].append(elapsed)
            hist["train_total"].append(tr_total_f)
            hist["val_total"].append(v_total_f)

            lv = v_losses.numpy()
            hist["hjb"].append(float(lv[0])); hist["kfe"].append(float(lv[1])); hist["free"].append(float(lv[2]))
            hist["bcDv"].append(float(lv[3])); hist["bcNv"].append(float(lv[4])); hist["bcNg"].append(float(lv[5]))

            hist["rl"].append(rl_val)
            hist["ce"].append(ce_val)
            hist["oracle_v_l2"].append(float(ov))
            hist["oracle_g_l2"].append(float(og))

            dt = now - last_log
            last_log = now

            print(f"[{step:6d}] t={elapsed/60:6.1f}m dt={dt:5.2f}s "
                  f"train={tr_total_f:.3e} val={v_total_f:.3e} rl={rl_val:.6f} ce_pred={ce_val:.4f} "
                  f"| oracle v_l2={ov:.4e} g_l2={og:.4e}")

            # blind checkpoint selection
            if v_total_f < best_val:
                best_val = v_total_f
                best_weights = model.get_weights()
                best_rl = rl_train.numpy()
                bad_logs = 0
            else:
                bad_logs += 1
                if bad_logs >= PATIENCE_LOGS:
                    print(f"Early stop by blind val_total patience ({PATIENCE_LOGS} logs).")
                    break

    # restore best blind checkpoint
    if best_weights is not None:
        model.set_weights(best_weights)
        rl_train.assign(best_rl)

    elapsed_total = (time.time() - t0) / 60.0
    print(f"== DONE {name}: {elapsed_total:.1f} min | best_val_total={best_val:.3e} | rl={float(rl_train.numpy()/100):.6f}")

    return model, rl_train, hist, best_val

# -------------------------
# Train both models (same budget)
# -------------------------
mlp = MLP_PINN(width=64, depth=6)
mlp_model, mlp_rl, mlp_hist, mlp_best = train_model(mlp, "MLP_PINN")

tr = Transformer_PINN(d_model=TR_D_MODEL, n_heads=TR_HEADS, d_ff=TR_D_FF,
                      n_layers=TR_LAYERS, seq_len=TR_SEQ_LEN, head_hidden=TR_HEAD_H)
tr_model, tr_rl, tr_hist, tr_best = train_model(tr, "TRANSFORMER_PINN")

# -------------------------
# Final oracle evaluation (same eval set size)
# -------------------------
EVAL_N = 65536
x_eval = sample_domain_np(EVAL_N)
y_eval_true = func_solution_griddata(x_eval)
x_eval_tf = tf.constant(x_eval)

def eval_oracle(model, rl_var, tag):
    y_pred = model(x_eval_tf, training=False).numpy()
    vL2 = v_l2_metric(y_eval_true, y_pred)
    gL2 = g_l2_metric(y_eval_true, y_pred)
    rl = float(rl_var.numpy()/100.0)
    print(f"\n[{tag}] oracle v_l2={vL2:.6e} g_l2={gL2:.6e} rl={rl:.6f} (rl_true={RL_TRUE_ORACLE:.6f})")
    return vL2, gL2, rl

mlp_v, mlp_g, mlp_r = eval_oracle(mlp_model, mlp_rl, "MLP")
tr_v,  tr_g,  tr_r  = eval_oracle(tr_model,  tr_rl,  "TRANSFORMER")

# -------------------------
# Plot losses
# -------------------------
def plot_hist(hist, title):
    steps = np.array(hist["step"])
    trn = np.array(hist["train_total"])
    val = np.array(hist["val_total"])
    rl  = np.array(hist["rl"])
    plt.figure()
    plt.plot(steps, trn, label="train_total")
    plt.plot(steps, val, label="val_total (blind)")
    plt.yscale("log")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title(title + " loss (log)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(steps, rl)
    plt.xlabel("step"); plt.ylabel("r^l")
    plt.title(title + " r^l trajectory (trained)")
    plt.show()

plot_hist(mlp_hist, "MLP_PINN")
plot_hist(tr_hist,  "TRANSFORMER_PINN")

# -------------------------
# Heatmaps: v/g pred vs true
# -------------------------
def predict_grid(model, N):
    e_vals = np.linspace(e_min, e_max, N).astype(np.float32)
    z_vals = np.linspace(z_min, z_max, N).astype(np.float32)
    E, Z = np.meshgrid(e_vals, z_vals, indexing="xy")
    grid = np.stack([E.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float32)
    bs = 8192
    out = []
    for i in range(0, grid.shape[0], bs):
        out.append(model(tf.constant(grid[i:i+bs]), training=False).numpy())
    out = np.vstack(out)
    v_pred = out[:,0].reshape(N,N)
    g_pred = out[:,1].reshape(N,N)
    v_true_grid = griddata(X_true, v_true, grid).reshape(N,N)
    g_true_grid = np.maximum(griddata(X_true, g_true, grid).reshape(N,N), 0)
    return e_vals, z_vals, v_pred, g_pred, v_true_grid, g_true_grid, grid

def plot_heat(e_vals, z_vals, arr, title):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(e_vals, z_vals, arr, cmap="rainbow")
    ax.set_aspect(0.1)
    ax.set_xlabel("e"); ax.set_ylabel("z")
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.show()

def show_maps(model, tag):
    e_vals, z_vals, v_pred, g_pred, v_true_grid, g_true_grid, grid = predict_grid(model, HEAT_N)
    plot_heat(e_vals, z_vals, v_pred, f"{tag}: v_pred")
    plot_heat(e_vals, z_vals, v_true_grid, f"{tag}: v_true (ref)")
    plot_heat(e_vals, z_vals, g_pred, f"{tag}: g_pred")
    plot_heat(e_vals, z_vals, g_true_grid, f"{tag}: g_true (ref)")

show_maps(mlp_model, "MLP_PINN")
show_maps(tr_model,  "TRANSFORMER_PINN")

# -------------------------
# Residual heatmaps (HJB, KFE) on coarser grid
# -------------------------
@tf.function
def residuals_on_points(model, rl_var, x_pts):
    rl = rl_var/100.0
    e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz = model_and_derivs_2nd(model, x_pts)
    hjb, kfe = pde_residuals(e,z,v,g,dv_de,dv_dz,dv_zz,dg_de,dg_dz,dg_zz, rl)
    return hjb, kfe

def residual_grid(model, rl_var, N):
    e_vals = np.linspace(e_min, e_max, N).astype(np.float32)
    z_vals = np.linspace(z_min, z_max, N).astype(np.float32)
    E, Z = np.meshgrid(e_vals, z_vals, indexing="xy")
    grid = np.stack([E.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float32)

    bs = 2048  # smaller because 2nd deriv
    hjb_all = []
    kfe_all = []
    for i in range(0, grid.shape[0], bs):
        x_b = tf.constant(grid[i:i+bs])
        hjb, kfe = residuals_on_points(model, rl_var, x_b)
        hjb_all.append(tf.abs(hjb).numpy())
        kfe_all.append(tf.abs(kfe).numpy())
    hjb_all = np.vstack(hjb_all).reshape(N,N)
    kfe_all = np.vstack(kfe_all).reshape(N,N)
    return e_vals, z_vals, hjb_all, kfe_all

def show_residuals(model, rl_var, tag):
    e_vals, z_vals, hjb_abs, kfe_abs = residual_grid(model, rl_var, RESID_N)
    plot_heat(e_vals, z_vals, np.log10(hjb_abs + 1e-12), f"{tag}: log10|HJB residual|")
    plot_heat(e_vals, z_vals, np.log10(kfe_abs + 1e-12), f"{tag}: log10|KFE residual|")

show_residuals(mlp_model, mlp_rl, "MLP_PINN")
show_residuals(tr_model,  tr_rl,  "TRANSFORMER_PINN")

print("\nDONE.")
