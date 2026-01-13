#!/usr/bin/env python3
"""
PyTorch loss-based solve for 2D Boussinesq self-similar profile (vorticity formulation, no pressure)
Half-plane only: y2 >= 0, via z-mapping:
  y1 = sinh(z1),  z1 ∈ [-L1/2, L1/2]
  y2 = sinh(z2),  z2 ∈ [0,     L2/2]

Unknowns on grid:
  Omega, U1, U2, Phi, Psi (fields) + scalar lambda

Residuals:
  R_om   = Omega + W·∇Omega - Phi
  R_phi  = (2 + ∂y1 U1)*Phi + W·∇Phi + (∂y1 U2)*Psi
  R_psi  = (2 + ∂y2 U2)*Psi + W·∇Psi + (∂y2 U1)*Phi
  R_div  = ∂y1 U1 + ∂y2 U2
  R_vor  = Omega - (∂y1 U2 - ∂y2 U1)
  R_comp = ∂y1 Psi - ∂y2 Phi

where W = (1+lambda)*y + U.

Discretization:
  - PDE loss enforced only on INTERIOR nodes (exclude wall j=0 and far boundary i=0,i=n1-1,j=n2-1)
  - Boundary nodes contribute only BC loss.

Derivatives:
  ∂/∂y1 = (1/cosh(z1)) ∂/∂z1
  ∂/∂y2 = (1/cosh(z2)) ∂/∂z2

Advection term W·∇f:
  choose either "upwind" (recommended) or "central" (for debugging).

Optimizers:
  - LBFGS works well for stiff, deterministic problems.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Utilities: grids + masks
# ----------------------------

def make_grids(n1, n2, L1, L2, device, dtype):
    z1 = torch.linspace(-L1/2, L1/2, n1, device=device, dtype=dtype)
    z2 = torch.linspace(0.0,   L2/2, n2, device=device, dtype=dtype)
    h1 = z1[1] - z1[0]
    h2 = z2[1] - z2[0]
    Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")
    Y1 = torch.sinh(Z1)
    Y2 = torch.sinh(Z2)
    s1 = 1.0 / torch.cosh(z1)  # (n1,)
    s2 = 1.0 / torch.cosh(z2)  # (n2,)
    return z1, z2, h1, h2, Z1, Z2, Y1, Y2, s1, s2


def make_masks(n1, n2, device):
    # interior: exclude wall j=0 and outer boundaries i=0,i=n1-1,j=n2-1
    interior = torch.ones((n1, n2), device=device, dtype=torch.bool)
    interior[:, 0] = False
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, -1] = False

    wall = torch.zeros((n1, n2), device=device, dtype=torch.bool)
    wall[:, 0] = True

    far = torch.zeros((n1, n2), device=device, dtype=torch.bool)
    far[0, :] = True
    far[-1, :] = True
    far[:, -1] = True
    far[:, 0] = False  # don't include wall

    return interior, wall, far


# ----------------------------
# Derivative operators (tensor stencils)
# ----------------------------

def dz1_central(f, h1):
    # 2nd order centered interior, 2nd order one-sided at boundaries
    # f shape (n1,n2)
    n1 = f.shape[0]
    out = torch.empty_like(f)
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*h1)
    out[0, :]    = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*h1)
    out[-1, :]   = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*h1)
    return out


def dz2_central(f, h2):
    n2 = f.shape[1]
    out = torch.empty_like(f)
    out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*h2)
    out[:, 0]    = (-3*f[:, 0] + 4*f[:, 1] - f[:, 2]) / (2*h2)
    out[:, -1]   = (3*f[:, -1] - 4*f[:, -2] + f[:, -3]) / (2*h2)
    return out


def dy1_from_dz1(f, h1, s1):
    # s1 shape (n1,), broadcast across n2
    return dz1_central(f, h1) * s1[:, None]


def dy2_from_dz2(f, h2, s2):
    return dz2_central(f, h2) * s2[None, :]


def dy1_upwind(f, W1, h1, s1):
    """
    Upwind derivative in z1, then chain rule.
    Uses first-order upwind in the advective direction (stable).
    At boundaries uses one-sided consistent with upwind.
    """
    out = torch.empty_like(f)
    # backward/forward differences
    df_b = torch.empty_like(f)
    df_f = torch.empty_like(f)

    df_b[1:, :] = (f[1:, :] - f[:-1, :]) / h1
    df_b[0, :]  = (f[1, :] - f[0, :]) / h1  # forward at left boundary

    df_f[:-1, :] = (f[1:, :] - f[:-1, :]) / h1
    df_f[-1, :]  = (f[-1, :] - f[-2, :]) / h1  # backward at right boundary

    out = torch.where(W1 >= 0, df_b, df_f) * s1[:, None]
    return out


def dy2_upwind(f, W2, h2, s2):
    out = torch.empty_like(f)
    df_b = torch.empty_like(f)
    df_f = torch.empty_like(f)

    df_b[:, 1:] = (f[:, 1:] - f[:, :-1]) / h2
    df_b[:, 0]  = (f[:, 1] - f[:, 0]) / h2

    df_f[:, :-1] = (f[:, 1:] - f[:, :-1]) / h2
    df_f[:, -1]  = (f[:, -1] - f[:, -2]) / h2

    out = torch.where(W2 >= 0, df_b, df_f) * s2[None, :]
    return out


# ----------------------------
# Model parameters (fields)
# ----------------------------

class Fields(torch.nn.Module):
    """
    Enforce y1-parity exactly by parameterizing only the y1>=0 half
    and reflecting with +/- symmetry.

    Odd in y1:  U1, Phi, Omega
    Even in y1: U2, Psi
    """
    def __init__(self, n1, n2, init_eps=1e-3, lam0=1.9, device="cpu", dtype=torch.float64):
        super().__init__()
        assert n1 % 2 == 1, "Need odd n1 so that y1=0 is a gridline."
        self.n1, self.n2 = n1, n2
        self.i0 = n1 // 2
        self.n1h = n1 - self.i0  # includes centerline + positive side

        # Half-side learnable parameters (y1>=0 side, including centerline)
        self._Omega_h = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._U1_h    = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._U2_h    = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._Phi_h   = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._Psi_h   = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))

        # lambda: if you want it fixed, keep tensor; if you want it learned, make Parameter.
        self.lam = torch.tensor(float(lam0), device=device, dtype=dtype)

        torch.manual_seed(0)
        for p in [self._U1_h, self._U2_h, self._Phi_h, self._Psi_h]:
            p.data[:] = init_eps * torch.randn_like(p.data)

        # for odd fields, centerline should be zero -> enforce at init too
        self._Omega_h.data[0, :] = 0.0
        self._U1_h.data[0, :]    = 0.0
        self._Phi_h.data[0, :]   = 0.0

    @staticmethod
    def _reflect_even(pos):
        # pos: (n1h, n2) with pos[0] = centerline
        neg = torch.flip(pos[1:], dims=[0])               # (n1h-1, n2)
        return torch.cat([neg, pos], dim=0)               # (n1, n2)

    @staticmethod
    def _reflect_odd(pos):
        # pos: (n1h, n2) with pos[0] = centerline, must be 0 for odd symmetry
        neg = torch.flip(pos[1:], dims=[0])               # (n1h-1, n2)
        center = torch.zeros_like(pos[:1])
        return torch.cat([-neg, center, pos[1:]], dim=0)  # (n1, n2)

    @property
    def Omega(self):
        # odd in y1
        pos = self._Omega_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd(pos)

    @property
    def U1(self):
        # odd in y1
        pos = self._U1_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd(pos)

    @property
    def Phi(self):
        # odd in y1
        pos = self._Phi_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd(pos)

    @property
    def U2(self):
        # even in y1
        return self._reflect_even(self._U2_h)

    @property
    def Psi(self):
        # even in y1
        return self._reflect_even(self._Psi_h)

# ----------------------------
# Loss assembly
# ----------------------------

def pde_residuals(fields: Fields, Y1, Y2, h1, h2, s1, s2, adv_scheme="upwind"):
    Om, U1, U2, Ph, Ps = fields.Omega, fields.U1, fields.U2, fields.Phi, fields.Psi
    lam = fields.lam

    W1 = (1.0 + lam) * Y1 + U1
    W2 = (1.0 + lam) * Y2 + U2

    # non-advective derivatives: central is fine for div/curl/comp (identities)
    Om_y1_c = dy1_from_dz1(Om, h1, s1)
    Om_y2_c = dy2_from_dz2(Om, h2, s2)
    U1_y1   = dy1_from_dz1(U1, h1, s1)
    U1_y2   = dy2_from_dz2(U1, h2, s2)
    U2_y1   = dy1_from_dz1(U2, h1, s1)
    U2_y2   = dy2_from_dz2(U2, h2, s2)
    Ph_y1_c = dy1_from_dz1(Ph, h1, s1)
    Ph_y2_c = dy2_from_dz2(Ph, h2, s2)
    Ps_y1_c = dy1_from_dz1(Ps, h1, s1)
    Ps_y2_c = dy2_from_dz2(Ps, h2, s2)

    # advection derivatives: choose scheme
    if adv_scheme == "upwind":
        Om_y1 = dy1_upwind(Om, W1, h1, s1)
        Om_y2 = dy2_upwind(Om, W2, h2, s2)
        Ph_y1 = dy1_upwind(Ph, W1, h1, s1)
        Ph_y2 = dy2_upwind(Ph, W2, h2, s2)
        Ps_y1 = dy1_upwind(Ps, W1, h1, s1)
        Ps_y2 = dy2_upwind(Ps, W2, h2, s2)
    elif adv_scheme == "central":
        Om_y1, Om_y2 = Om_y1_c, Om_y2_c
        Ph_y1, Ph_y2 = Ph_y1_c, Ph_y2_c
        Ps_y1, Ps_y2 = Ps_y1_c, Ps_y2_c
    else:
        raise ValueError("adv_scheme must be 'upwind' or 'central'.")

    adv_Om = W1 * Om_y1 + W2 * Om_y2
    adv_Ph = W1 * Ph_y1 + W2 * Ph_y2
    adv_Ps = W1 * Ps_y1 + W2 * Ps_y2

    R_om   = Om + adv_Om - Ph
    R_phi  = (2.0 + U1_y1) * Ph + adv_Ph + (U2_y1) * Ps
    R_psi  = (2.0 + U2_y2) * Ps + adv_Ps + (U1_y2) * Ph
    R_div  = U1_y1 + U2_y2
    R_vor  = Om - (U2_y1 - U1_y2)
    R_comp = Ps_y1_c - Ph_y2_c

    return {
        "R_om": R_om,
        "R_phi": R_phi,
        "R_psi": R_psi,
        "R_div": R_div,
        "R_vor": R_vor,
        "R_comp": R_comp,
        "W1": W1,
        "W2": W2,
        "U1_y2": U1_y2,
        "Ph_y2": Ph_y2_c,
        "Om_y2": Om_y2_c,
    }


def loss_total(fields: Fields, grids, masks,
               weights,
               adv_scheme="upwind",
               enforce_reflection_wall=True):
    z1, z2, h1, h2, Z1, Z2, Y1, Y2, s1, s2 = grids
    interior, wall, far = masks

    # PDE residuals everywhere, but we will apply loss only on interior mask
    res = pde_residuals(fields, Y1, Y2, h1, h2, s1, s2, adv_scheme=adv_scheme)

    # PDE interior loss
    def mse_mask(A, mask):
        v = A[mask]
        return torch.mean(v*v)

    L_pde = (
        mse_mask(res["R_om"], interior) +
        mse_mask(res["R_phi"], interior) +
        mse_mask(res["R_psi"], interior) +
        mse_mask(res["R_div"], interior) +
        mse_mask(res["R_vor"], interior) +
        mse_mask(res["R_comp"], interior)
    )

    # Gauge: (dy1 Omega)(0,0) = -1  at z1=0 center, z2=0 wall
    i0 = fields.n1 // 2
    j0 = 0
    Om = fields.Omega
    dy1_Om = dy1_from_dz1(Om, h1, s1)
    g = dy1_Om[i0, j0] + 1.0
    L_g = g*g

    # BC losses
    Om, U1, U2, Ph, Ps = fields.Omega, fields.U1, fields.U2, fields.Phi, fields.Psi

    # Wall always: U2=0
    L_wall = mse_mask(U2, wall)

    if enforce_reflection_wall:
        # Psi=0, d_y2 U1 = 0, d_y2 Phi = 0, d_y2 Omega = 0 on wall
        U1_y2 = res["U1_y2"]
        Ph_y2 = res["Ph_y2"]
        Om_y2 = res["Om_y2"]
        L_wall = L_wall + mse_mask(Ps, wall) + mse_mask(U1_y2, wall) + mse_mask(Ph_y2, wall) + mse_mask(Om_y2, wall)

    # Far: Phi=0, Psi=0 and flatness U-gradients
    # compute grads (central)
    U1_y1 = dy1_from_dz1(U1, h1, s1)
    U1_y2 = dy2_from_dz2(U1, h2, s2)
    U2_y1 = dy1_from_dz1(U2, h1, s1)
    U2_y2 = dy2_from_dz2(U2, h2, s2)

    L_far = (
        mse_mask(Ph, far) +
        mse_mask(Ps, far) +
        mse_mask(U1_y1, far) +
        mse_mask(U1_y2, far) +
        mse_mask(U2_y1, far) +
        mse_mask(U2_y2, far)
    )

    # Total
    w_pde, w_g, w_wall, w_far = weights["pde"], weights["gauge"], weights["wall"], weights["far"]
    L = w_pde*L_pde + w_g*L_g + w_wall*L_wall + w_far*L_far

    stats = {
        "L": L.detach().item(),
        "L_pde": L_pde.detach().item(),
        "L_g": L_g.detach().item(),
        "L_wall": L_wall.detach().item(),
        "L_far": L_far.detach().item(),
        "dy1omega00": dy1_Om[i0, j0].detach().item(),
        "lam": fields.lam.detach().item(),
    }
    return L, stats


# ----------------------------
# Optimization driver
# ----------------------------

def solve_torch(
    n1=257, n2=129, L1=60.0, L2=60.0,
    adv_scheme="upwind",
    enforce_reflection_wall=True,
    device=None,
    dtype=torch.float64,
    max_lbfgs_steps=300,
    print_every=10,
    lam0=3.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    grids = make_grids(n1, n2, L1, L2, device=device, dtype=dtype)
    _, _, _, _, _, _, Y1, Y2, _, _ = grids
    masks = make_masks(n1, n2, device=device)

    fields = Fields(n1, n2, init_eps=1e-3, lam0=lam0, device=device, dtype=dtype)
    #seed_omega_from_y(fields, Y1, Y2)

    # weights (tune these)
    weights = {"pde": 1.0, "gauge": 100.0, "wall": 10.0, "far": 0.0}

    # LBFGS usually works best for these PDE residual minimizations
    opt = torch.optim.LBFGS(
        fields.parameters(),
        lr=1.0,
        max_iter=20,           # per .step() call
        max_eval=25,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    step = 0
    last_stats = None

    def closure():
        nonlocal last_stats
        opt.zero_grad(set_to_none=True)
        L, stats = loss_total(
            fields, grids, masks, weights,
            adv_scheme=adv_scheme,
            enforce_reflection_wall=enforce_reflection_wall,
        )
        L.backward()
        last_stats = stats
        return L

    # Outer loop to get more than max_iter=20 total LBFGS iterations
    outer_steps = math.ceil(max_lbfgs_steps / 20)
    for k in range(outer_steps):
        opt.step(closure)
        step += 20
        if (k % max(1, print_every // 20) == 0) and last_stats is not None:
            s = last_stats
            print(
                f"[lbfgs ~{step:04d}] L={s['L']:.3e}  "
                f"L_pde={s['L_pde']:.3e}  L_g={s['L_g']:.3e}  "
                f"L_wall={s['L_wall']:.3e}  L_far={s['L_far']:.3e}  "
                f"dy1Om00={s['dy1omega00']:.6f}  lam={s['lam']:.6f}"
            )

    return fields, grids


# ----------------------------
# Visualization (same style as before)
# ----------------------------

def visualize(fields: Fields, grids, y1_lim=(-20, 20), y2_lim=(0, 20), title_suffix=""):
    z1, z2, h1, h2, Z1, Z2, Y1, Y2, s1, s2 = grids

    # crop by physical coords
    y1_line = Y1[:, 0].detach().cpu().numpy()
    y2_line = Y2[0, :].detach().cpu().numpy()

    I = np.where((y1_line >= y1_lim[0]) & (y1_line <= y1_lim[1]))[0]
    J = np.where((y2_line >= y2_lim[0]) & (y2_line <= y2_lim[1]))[0]
    if I.size < 3 or J.size < 3:
        print("Visualization window too small; adjust y1_lim/y2_lim.")
        return

    def crop(T):
        A = T.detach().cpu().numpy()
        return A[np.ix_(I, J)]

    Y1c = crop(Y1)
    Y2c = crop(Y2)

    Om = crop(fields.Omega)
    Ph = crop(fields.Phi)
    Ps = crop(fields.Psi)
    U1 = crop(fields.U1)
    U2 = crop(fields.U2)

    fig = plt.figure(figsize=(14, 7))

    def surf(ax, Z, ttl):
        ax.plot_surface(Y1c, Y2c, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.set_title(ttl)
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.view_init(elev=18, azim=-60)

    ax1 = fig.add_subplot(2, 3, 1, projection="3d"); surf(ax1, Om, r"$\Omega$"+title_suffix)
    ax2 = fig.add_subplot(2, 3, 2, projection="3d"); surf(ax2, Ph, r"$\Phi$"+title_suffix)
    ax3 = fig.add_subplot(2, 3, 3, projection="3d"); surf(ax3, Ps, r"$\Psi$"+title_suffix)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d"); surf(ax4, U1, r"$U_1$"+title_suffix)
    ax5 = fig.add_subplot(2, 3, 5, projection="3d"); surf(ax5, U2, r"$U_2$"+title_suffix)
    ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
    ax6.text(0.05, 0.6, f"PyTorch solve\nlambda={fields.lam.detach().cpu().item():.6f}", fontsize=16)

    plt.tight_layout()
    plt.show()


def main():
    torch.set_default_dtype(torch.float64)

    #n1, n2 = 513, 257
    n1, n2 = 257, 129
    L1, L2 = 60.0, 60.0

    fields, grids = solve_torch(
        n1=n1, n2=n2, L1=L1, L2=L2,
        adv_scheme="upwind",              # try "central" to compare
        enforce_reflection_wall=True,
        max_lbfgs_steps=5000,
        print_every=50,
        lam0=3.0
    )

    print(f"Done. lambda={fields.lam.detach().cpu().item():.12g}")
    visualize(fields, grids, y1_lim=(-20, 20), y2_lim=(0, 20), title_suffix="")


if __name__ == "__main__":
    main()
