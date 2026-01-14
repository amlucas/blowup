#!/usr/bin/env python3
import csv
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

def open_loss_csv(csv_path, fieldnames):
    csv_path = os.path.expanduser(csv_path)
    dirpath = os.path.dirname(csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    f = open(csv_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    f.flush()
    return f, w


# ----------------------------
# Grids: centers + faces in y
# ----------------------------

def make_grids(n1, n2, L1, L2, device, dtype):
    assert n1 % 2 == 1

    z1_c = torch.linspace(-L1/2, L1/2, n1, device=device, dtype=dtype)
    z2_c = torch.linspace(0.0,   L2/2, n2, device=device, dtype=dtype)
    dz1 = z1_c[1] - z1_c[0]
    dz2 = z2_c[1] - z2_c[0]

    # faces (midpoints + extrapolated)
    z1_f = torch.empty((n1+1,), device=device, dtype=dtype)
    z2_f = torch.empty((n2+1,), device=device, dtype=dtype)
    z1_f[1:-1] = 0.5 * (z1_c[1:] + z1_c[:-1])
    z2_f[1:-1] = 0.5 * (z2_c[1:] + z2_c[:-1])
    z1_f[0]  = z1_c[0]  - 0.5*dz1
    z1_f[-1] = z1_c[-1] + 0.5*dz1
    z2_f[0]  = z2_c[0]  - 0.5*dz2   # ghost below wall
    z2_f[-1] = z2_c[-1] + 0.5*dz2

    y1_c = torch.sinh(z1_c)
    y2_c = torch.sinh(z2_c)
    y1_f = torch.sinh(z1_f)
    y2_f = torch.sinh(z2_f)

    dy1_c = y1_f[1:] - y1_f[:-1]   # (n1,)
    dy2_c = y2_f[1:] - y2_f[:-1]   # (n2,)

    Y1c, Y2c = torch.meshgrid(y1_c, y2_c, indexing="ij")  # centers
    Y1x, Y2x = torch.meshgrid(y1_f, y2_c, indexing="ij")  # x-faces
    Y1y, Y2y = torch.meshgrid(y1_c, y2_f, indexing="ij")  # y-faces

    return {
        "z1_c": z1_c, "z2_c": z2_c, "dz1": dz1, "dz2": dz2,
        "z1_f": z1_f, "z2_f": z2_f,
        "y1_c": y1_c, "y2_c": y2_c,
        "y1_f": y1_f, "y2_f": y2_f,
        "dy1_c": dy1_c, "dy2_c": dy2_c,
        "Y1c": Y1c, "Y2c": Y2c,
        "Y1x": Y1x, "Y2x": Y2x,
        "Y1y": Y1y, "Y2y": Y2y,
    }


def make_masks(n1, n2, device):
    interior = torch.ones((n1, n2), device=device, dtype=torch.bool)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False

    far = torch.zeros((n1, n2), device=device, dtype=torch.bool)
    far[0, :] = True
    far[-1, :] = True
    far[:, -1] = True
    far[:, 0] = False

    wall_centers = torch.zeros((n1, n2), device=device, dtype=torch.bool)
    wall_centers[:, 0] = True

    return {"interior": interior, "far": far, "wall_centers": wall_centers}


# ----------------------------
# Fields with correct y1 parity
# ----------------------------

class Fields(torch.nn.Module):
    """
    Staggered MAC layout:
      U1x: x-faces (n1+1,n2)  odd in y1
      U2y: y-faces (n1,n2+1)  even in y1
      Phi: centers (n1,n2)    odd in y1
      Psi: centers (n1,n2)    even in y1
    """
    def __init__(self, n1, n2, lam0=1.9, device="cpu", dtype=torch.float64, learn_lambda=False, seed=0):
        super().__init__()
        assert n1 % 2 == 1
        self.n1, self.n2 = n1, n2
        self.i0 = n1 // 2

        # number of positive x-faces (strictly y1>0): since faces count is n1+1 even, half is (n1+1)/2
        self.nx_pos = (n1 + 1) // 2
        self.ic0f = self.nx_pos

        # number of positive centers including centerline
        self.n1h = n1 - self.i0

        # half parameters
        self._U1x_p = torch.nn.Parameter(torch.zeros((self.nx_pos, n2), device=device, dtype=dtype))
        self._U2y_h = torch.nn.Parameter(torch.zeros((self.n1h, n2+1), device=device, dtype=dtype))
        self._Phi_h = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._Psi_h = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))

        if learn_lambda:
            self.lam = torch.nn.Parameter(torch.tensor(float(lam0), device=device, dtype=dtype))
        else:
            self.lam = torch.tensor(float(lam0), device=device, dtype=dtype)

        # odd centerline at centers:
        self._Phi_h.data[0, :] = 0.0

    @staticmethod
    def _reflect_even_centers(pos):
        neg = torch.flip(pos[1:], dims=[0])
        return torch.cat([neg, pos], dim=0)

    @staticmethod
    def _reflect_odd_centers(pos):
        neg = torch.flip(pos[1:], dims=[0])
        center = torch.zeros_like(pos[:1])
        return torch.cat([-neg, center, pos[1:]], dim=0)

    @staticmethod
    def _reflect_even_faces(pos):
        neg = torch.flip(pos, dims=[0])
        return torch.cat([neg, pos], dim=0)

    @staticmethod
    def _reflect_odd_faces(pos):
        neg = torch.flip(pos, dims=[0])
        return torch.cat([-neg, pos], dim=0)

    @property
    def Phi(self):
        pos = self._Phi_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd_centers(pos)

    @property
    def Psi(self):
        return self._reflect_even_centers(self._Psi_h)

    @property
    def U2y(self):
        return self._reflect_even_centers(self._U2y_h)

    @property
    def U1x(self):
        return self._reflect_odd_faces(self._U1x_p)


# ----------------------------
# FV one-sided gradients on centers
# ----------------------------

def grad_bwd_y1_center(qc, dy1_c):
    out = torch.empty_like(qc)
    out[1:, :] = (qc[1:, :] - qc[:-1, :]) / (0.5*(dy1_c[1:, None] + dy1_c[:-1, None]))
    out[0, :]  = (qc[1, :] - qc[0, :]) / dy1_c[0]
    return out


def grad_bwd_y2_center(qc, dy2_c):
    out = torch.empty_like(qc)
    out[:, 1:] = (qc[:, 1:] - qc[:, :-1]) / (0.5*(dy2_c[None, 1:] + dy2_c[None, :-1]))
    out[:, 0]  = (qc[:, 1] - qc[:, 0]) / dy2_c[0]
    return out


# ----------------------------
# MAC divergence at centers (exact FV)
# ----------------------------

def div_U(U1x, U2y, dy1_c, dy2_c):
    d1 = (U1x[1:, :] - U1x[:-1, :]) / dy1_c[:, None]
    d2 = (U2y[:, 1:] - U2y[:, :-1]) / dy2_c[None, :]
    return d1 + d2


# ----------------------------
# Omega on centers with one-sided FV (no central)
# ----------------------------

def omega_center(U1x, U2y, dy1_c, dy2_c):
    U1c = 0.5 * (U1x[1:, :] + U1x[:-1, :])      # (n1,n2)
    U2c = 0.5 * (U2y[:, 1:] + U2y[:, :-1])      # (n1,n2)

    dU2_dy1 = grad_bwd_y1_center(U2c, dy1_c)
    dU1_dy2 = grad_bwd_y2_center(U1c, dy2_c)
    Om = dU2_dy1 - dU1_dy2
    return Om, U1c, U2c, dU2_dy1, dU1_dy2


# ----------------------------
# Upwind FV advection of center scalar
# ----------------------------

def center_to_xface_lr(qc):
    n1, n2 = qc.shape
    qL = torch.empty((n1+1, n2), device=qc.device, dtype=qc.dtype)
    qR = torch.empty_like(qL)
    qL[1:-1, :] = qc[:-1, :]
    qR[1:-1, :] = qc[1:, :]
    qL[0, :] = qc[0, :];   qR[0, :] = qc[0, :]
    qL[-1, :] = qc[-1, :]; qR[-1, :] = qc[-1, :]
    return qL, qR


def center_to_yface_bt(qc):
    n1, n2 = qc.shape
    qB = torch.empty((n1, n2+1), device=qc.device, dtype=qc.dtype)
    qT = torch.empty_like(qB)
    qB[:, 1:-1] = qc[:, :-1]
    qT[:, 1:-1] = qc[:, 1:]
    qB[:, 0] = qc[:, 0];   qT[:, 0] = qc[:, 0]
    qB[:, -1] = qc[:, -1]; qT[:, -1] = qc[:, -1]
    return qB, qT

def smooth_upwind(qL, qR, W, eps=1e-3):
    # s ~ +1 if W>>0, s ~ -1 if W<<0
    s = torch.tanh(W / eps)
    return 0.5 * ((1+s)*qL + (1-s)*qR)

def fv_advective_term(qc, W1x, W2y, dy1_c, dy2_c, divW_c):
    qL, qR = center_to_xface_lr(qc)
    qB, qT = center_to_yface_bt(qc)

    qx = smooth_upwind(qL, qR, W1x, eps=1e-3)
    qy = smooth_upwind(qB, qT, W2y, eps=1e-3)

    Fx = W1x * qx
    Fy = W2y * qy

    divF = (Fx[1:, :] - Fx[:-1, :]) / dy1_c[:, None] + (Fy[:, 1:] - Fy[:, :-1]) / dy2_c[None, :]
    return divF - qc * divW_c


# ----------------------------
# Residuals + loss
# ----------------------------

def residuals(fields: Fields, grids):
    dy1_c, dy2_c = grids["dy1_c"], grids["dy2_c"]

    U1x = fields.U1x
    U2y = fields.U2y
    Phi = fields.Phi
    Psi = fields.Psi
    lam = fields.lam

    DivU = div_U(U1x, U2y, dy1_c, dy2_c)
    Om, U1c, U2c, dU2_dy1, dU1_dy2 = omega_center(U1x, U2y, dy1_c, dy2_c)

    W1x = (1.0 + lam) * grids["Y1x"] + U1x
    W2y = (1.0 + lam) * grids["Y2y"] + U2y
    DivW = 2.0 * (1.0 + lam) + DivU

    adv_Om = fv_advective_term(Om,  W1x, W2y, dy1_c, dy2_c, DivW)
    adv_Ph = fv_advective_term(Phi, W1x, W2y, dy1_c, dy2_c, DivW)
    adv_Ps = fv_advective_term(Psi, W1x, W2y, dy1_c, dy2_c, DivW)

    # source terms
    dU1_dy1 = (U1x[1:, :] - U1x[:-1, :]) / dy1_c[:, None]
    dU2_dy2 = (U2y[:, 1:] - U2y[:, :-1]) / dy2_c[None, :]

    # compatibility (one-sided)
    dPsi_dy1 = grad_bwd_y1_center(Psi, dy1_c)
    dPhi_dy2 = grad_bwd_y2_center(Phi, dy2_c)
    R_comp = dPsi_dy1 - dPhi_dy2

    R_om  = Om + adv_Om - Phi
    R_phi = (2.0 + dU1_dy1) * Phi + adv_Ph + dU2_dy1 * Psi
    R_psi = (2.0 + dU2_dy2) * Psi + adv_Ps + dU1_dy2 * Phi
    R_div = DivU

    return {
        "R_om": R_om, "R_phi": R_phi, "R_psi": R_psi, "R_div": R_div, "R_comp": R_comp,
        "Omega": Om, "DivU": DivU,
        "U1c": U1c, "U2c": U2c,
    }


def mse_mask(A, mask):
    v = A[mask]
    return torch.mean(v*v)


def loss_total(fields: Fields, grids, masks, weights, enforce_reflection_wall=True):
    interior = masks["interior"]
    far = masks["far"]
    wall_centers = masks["wall_centers"]

    res = residuals(fields, grids)

    L_pde = (
        mse_mask(res["R_om"], interior) +
        mse_mask(res["R_phi"], interior) +
        mse_mask(res["R_psi"], interior) +
        mse_mask(res["R_div"], interior) +
        mse_mask(res["R_comp"], interior)
    )

    # gauge at (y1=0,y2=0)
    i0 = fields.i0
    j0 = 0
    dy1_c = grids["dy1_c"]
    Om = res["Omega"]
    dy_between = 0.5 * (dy1_c[i0] + dy1_c[i0+1])
    dOm_dy1_00 = (Om[i0+1, j0] - Om[i0, j0]) / dy_between
    g = dOm_dy1_00 + 1.0
    L_g = g*g

    # wall: U2y at wall face j=0 is 0
    U2y = fields.U2y
    L_wall = torch.mean(U2y[:, 0]*U2y[:, 0])

    if enforce_reflection_wall:
        Phi = fields.Phi
        Psi = fields.Psi

        # Psi=0 at wall centers
        L_wall = L_wall + mse_mask(Psi, wall_centers)

        # d_y2 Phi = 0 and d_y2 Omega = 0 at wall centers (forward)
        dy2_c = grids["dy2_c"]
        dy2_between = 0.5 * (dy2_c[0] + dy2_c[1])
        dPhi_dy2_wall = (Phi[:, 1] - Phi[:, 0]) / dy2_between
        dOm_dy2_wall  = (Om[:, 1]  - Om[:, 0])  / dy2_between
        L_wall = L_wall + torch.mean(dPhi_dy2_wall**2) + torch.mean(dOm_dy2_wall**2)

        # d_y2 U1x = 0 at wall for x-faces (forward)
        U1x = fields.U1x
        dU1x_dy2_wall = (U1x[:, 1] - U1x[:, 0]) / dy2_between
        L_wall = L_wall + torch.mean(dU1x_dy2_wall**2)

    # far: Phi=Psi=0 and flat grads of center velocities
    Phi = fields.Phi
    Psi = fields.Psi
    L_far = mse_mask(Phi, far) + mse_mask(Psi, far)

    U1c = res["U1c"]
    U2c = res["U2c"]
    dU1c_dy1 = grad_bwd_y1_center(U1c, dy1_c)
    dU1c_dy2 = grad_bwd_y2_center(U1c, grids["dy2_c"])
    dU2c_dy1 = grad_bwd_y1_center(U2c, dy1_c)
    dU2c_dy2 = grad_bwd_y2_center(U2c, grids["dy2_c"])
    L_far = L_far + mse_mask(dU1c_dy1, far) + mse_mask(dU1c_dy2, far) + mse_mask(dU2c_dy1, far) + mse_mask(dU2c_dy2, far)

    L = (
        weights["pde"] * L_pde +
        weights["gauge"] * L_g +
        weights["wall"] * L_wall +
        weights["far"] * L_far
    )

    stats = {
        "L": float(L.detach().cpu().item()),
        "L_pde": float(L_pde.detach().cpu().item()),
        "L_g": float(L_g.detach().cpu().item()),
        "L_wall": float(L_wall.detach().cpu().item()),
        "L_far": float(L_far.detach().cpu().item()),
        "dy1Om00": float(dOm_dy1_00.detach().cpu().item()),
        "lam": float(fields.lam.detach().cpu().item()),
    }
    return L, stats


# ----------------------------
# Init
# ----------------------------

@torch.no_grad()
def init_fields(fields: Fields, grids, R1=20.0, R2=20.0, d1=2.0, d2=2.0):
    Y1c, Y2c = grids["Y1c"], grids["Y2c"]
    absY1 = torch.abs(Y1c)
    chi1 = 0.5 * (1.0 - torch.tanh((absY1 - R1) / d1))
    chi2 = 0.5 * (1.0 - torch.tanh((Y2c  - R2) / d2))
    chi_c = chi1 * chi2

    # U1x init on faces
    chi_x = torch.empty_like(grids["Y1x"])
    chi_x[1:-1, :] = 0.5 * (chi_c[1:, :] + chi_c[:-1, :])
    chi_x[0, :] = chi_c[0, :]
    chi_x[-1, :] = chi_c[-1, :]
    U1x_full = -grids["Y1x"] * chi_x

    # U2y init on faces
    chi_y = torch.empty_like(grids["Y2y"])
    chi_y[:, 1:-1] = 0.5 * (chi_c[:, 1:] + chi_c[:, :-1])
    chi_y[:, 0] = chi_c[:, 0]
    chi_y[:, -1] = chi_c[:, -1]
    U2y_full = grids["Y2y"] * chi_y
    U2y_full[:, 0] = 0.0  # enforce wall

    # pack into half
    i0 = fields.i0
    ic0f = fields.ic0f
    fields._U1x_p.data[:] = U1x_full[ic0f:, :].clone()
    fields._U2y_h.data[:] = U2y_full[i0:, :].clone()
    fields._Phi_h.data.zero_()
    fields._Psi_h.data.zero_()
    fields._Phi_h.data[0, :] = 0.0


# ----------------------------
# IO helpers
# ----------------------------

@torch.no_grad()
def export_npz(path, fields, grids, it=None):
    path = os.path.expanduser(path)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    dy1_c = grids["dy1_c"]
    dy2_c = grids["dy2_c"]

    U1x = fields.U1x
    U2y = fields.U2y
    Om, U1c, U2c, _, _ = omega_center(U1x, U2y, dy1_c, dy2_c)

    payload = dict(
        lam=float(fields.lam.detach().cpu().item()),
        y1_c=grids["y1_c"].detach().cpu().numpy(),
        y2_c=grids["y2_c"].detach().cpu().numpy(),
        Omega=Om.detach().cpu().numpy(),
        U1=U1c.detach().cpu().numpy(),
        U2=U2c.detach().cpu().numpy(),
        Phi=fields.Phi.detach().cpu().numpy(),
        Psi=fields.Psi.detach().cpu().numpy(),
    )
    if it is not None:
        payload["it"] = int(it)

    np.savez_compressed(path, **payload)

# ----------------------------
# Main loop
# ----------------------------

# --- add/replace in your script: LBFGS phase after Adam ---

def solve_torch(
    n1=257, n2=129, L1=60.0, L2=60.0,
    lam0=1.9, learn_lambda=False,
    device=None, dtype=torch.float64,
    iters=20000, lr=2e-3, print_every=200,
    enforce_reflection_wall=True,
    seed=0,
    csv_path="loss.csv",
    log_every=1,
    out_dir="runs/boussinesq",
    save_every=10_000,
    # NEW:
    adam_iters=50_000,
    lbfgs_iters=50_000,
    lbfgs_lr=1.0,
    lbfgs_max_iter=20,
    lbfgs_history_size=50,
    lbfgs_tol_grad=1e-10,
    lbfgs_tol_change=1e-12,
    lbfgs_line_search="strong_wolfe",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    grids = make_grids(n1, n2, L1, L2, device=device, dtype=dtype)
    masks = make_masks(n1, n2, device=device)

    fields = Fields(n1, n2, lam0=lam0, device=device, dtype=dtype, learn_lambda=learn_lambda, seed=seed)
    init_fields(fields, grids)

    weights = {"pde": 100.0, "gauge": 1.0, "wall": 50.0, "far": 5.0}

    fieldnames = ["it", "phase", "time_s", "lr", "L", "L_pde", "L_g", "L_wall", "L_far", "dy1Om00", "lam"]
    t0 = time.time()
    csv_f, csv_w = open_loss_csv(csv_path, fieldnames)

    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def log_row(it, phase, lr_now, stats):
        row = {"it": it, "phase": phase, "time_s": time.time() - t0, "lr": lr_now, **stats}
        csv_w.writerow(row)
        csv_f.flush()

    def maybe_save(it):
        if save_every and (it % save_every == 0):
            export_npz(os.path.join(out_dir, "latest.npz"), fields, grids, it=it)
            export_npz(os.path.join(out_dir, f"ckpt_{it:06d}.npz"), fields, grids, it=it)

    it_global = 0

    try:
        # ----------------
        # Phase 1: Adam
        # ----------------
        opt = torch.optim.Adam(fields.parameters(), lr=lr)

        for k in range(1, adam_iters + 1):
            it_global += 1
            opt.zero_grad(set_to_none=True)
            L, stats = loss_total(fields, grids, masks, weights, enforce_reflection_wall=enforce_reflection_wall)
            L.backward()
            torch.nn.utils.clip_grad_norm_(fields.parameters(), max_norm=10.0)
            opt.step()

            lr_now = opt.param_groups[0]["lr"]

            if (it_global % log_every) == 0:
                log_row(it_global, "adam", lr_now, stats)

            if it_global % print_every == 0 or it_global == 1:
                print(
                    f"[adam  {it_global:06d}] "
                    f"L={stats['L']:.3e}  L_pde={stats['L_pde']:.3e}  "
                    f"L_g={stats['L_g']:.3e}  L_wall={stats['L_wall']:.3e}  L_far={stats['L_far']:.3e}  "
                    f"dy1Om00={stats['dy1Om00']:.6f}  lam={stats['lam']:.6f}  lr={lr_now:.2e}"
                )

            maybe_save(it_global)

        # Save right before LBFGS starts
        export_npz(os.path.join(out_dir, "pre_lbfgs.npz"), fields, grids, it=it_global)

        # ----------------
        # Phase 2: LBFGS
        # ----------------
        lbfgs = torch.optim.LBFGS(
            fields.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,          # inner iterations per .step()
            history_size=lbfgs_history_size,
            tolerance_grad=lbfgs_tol_grad,
            tolerance_change=lbfgs_tol_change,
            line_search_fn=lbfgs_line_search,
        )

        last = {"stats": None, "loss": None}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            L, stats = loss_total(fields, grids, masks, weights, enforce_reflection_wall=enforce_reflection_wall)
            L.backward()
            last["stats"] = stats
            last["loss"] = L
            return L

        for k in range(1, lbfgs_iters + 1):
            it_global += 1

            # One "outer" LBFGS step (will call closure multiple times internally)
            lbfgs.step(closure)

            stats = last["stats"]
            if stats is None:
                # extremely unlikely, but keep it safe
                _, stats = loss_total(fields, grids, masks, weights, enforce_reflection_wall=enforce_reflection_wall)

            lr_now = lbfgs.param_groups[0]["lr"]

            if (it_global % log_every) == 0:
                log_row(it_global, "lbfgs", lr_now, stats)

            if it_global % print_every == 0 or it_global == (adam_iters + 1):
                print(
                    f"[lbfgs {it_global:06d}] "
                    f"L={stats['L']:.3e}  L_pde={stats['L_pde']:.3e}  "
                    f"L_g={stats['L_g']:.3e}  L_wall={stats['L_wall']:.3e}  L_far={stats['L_far']:.3e}  "
                    f"dy1Om00={stats['dy1Om00']:.6f}  lam={stats['lam']:.6f}  lr={lr_now:.2e}"
                )

            maybe_save(it_global)

        export_npz(os.path.join(out_dir, "final.npz"), fields, grids, it=it_global)

    finally:
        csv_f.close()

    return fields, grids

# ----------------------------
# Viz
# ----------------------------

def visualize(fields: Fields,
              grids,
              y1_lim=(-20, 20),
              y2_lim=(0, 20),
              path=None):
    y1_c = grids["y1_c"].detach().cpu().numpy()
    y2_c = grids["y2_c"].detach().cpu().numpy()
    Y1c = grids["Y1c"].detach().cpu().numpy()
    Y2c = grids["Y2c"].detach().cpu().numpy()

    I = np.where((y1_c >= y1_lim[0]) & (y1_c <= y1_lim[1]))[0]
    J = np.where((y2_c >= y2_lim[0]) & (y2_c <= y2_lim[1]))[0]
    if I.size < 3 or J.size < 3:
        print("Visualization window too small; adjust y1_lim/y2_lim.")
        return

    def crop_center(T):
        A = T.detach().cpu().numpy()
        return A[np.ix_(I, J)]

    dy1_c = grids["dy1_c"]
    dy2_c = grids["dy2_c"]

    U1x = fields.U1x
    U2y = fields.U2y
    Om, U1c, U2c, _, _ = omega_center(U1x, U2y, dy1_c, dy2_c)

    Y1 = Y1c[np.ix_(I, J)]
    Y2 = Y2c[np.ix_(I, J)]

    Omc = crop_center(Om)
    Phc = crop_center(fields.Phi)
    Psc = crop_center(fields.Psi)
    U1c = crop_center(U1c)
    U2c = crop_center(U2c)

    fig = plt.figure(figsize=(14, 7))

    def surf(ax, Z, ttl):
        ax.plot_surface(Y1, Y2, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.set_title(ttl)
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.view_init(elev=18, azim=-60)

    ax1 = fig.add_subplot(2, 3, 1, projection="3d"); surf(ax1, Omc, r"$\Omega$")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d"); surf(ax2, Phc, r"$\Phi$")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d"); surf(ax3, Psc, r"$\Psi$")
    ax4 = fig.add_subplot(2, 3, 4, projection="3d"); surf(ax4, U1c, r"$U_1$")
    ax5 = fig.add_subplot(2, 3, 5, projection="3d"); surf(ax5, U2c, r"$U_2$")
    ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
    ax6.text(0.05, 0.6, f"lambda={float(fields.lam.detach().cpu().item()):.6f}", fontsize=16)

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def main():
    torch.set_default_dtype(torch.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    level = 7
    n1 = 2**(level + 1) + 1
    n2 = 2**level + 1

    print(f"n1, n2 = {n1}, {n2}")

    fields, grids = solve_torch(
        n1=n1, n2=n2,
        L1=60.0, L2=60.0,
        lam0=3.0,
        device=device,
        dtype=torch.float64,
        learn_lambda=False,
        enforce_reflection_wall=True,
        seed=0,
        csv_path="out_boussinesq/loss.csv",
        out_dir="out_boussinesq",
        save_every=10_000,
        log_every=1,
        print_every=200,
        adam_iters=20_000,
        lbfgs_iters=50_000,
        lr=2e-3,
        lbfgs_lr=1.0,
        lbfgs_max_iter=20,
    )
    print("Done.")

    visualize(fields, grids, path="out_boussinesq/view.png")


if __name__ == "__main__":
    main()
