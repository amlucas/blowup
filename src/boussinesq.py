#!/usr/bin/env python3
"""
PyTorch loss-based solve for 2D Boussinesq self-similar profile (vorticity formulation, no pressure)
Half-plane only: y2 >= 0, via z-mapping:
  y1 = sinh(z1),  z1 ∈ [-L1/2, L1/2]
  y2 = sinh(z2),  z2 ∈ [0,     L2/2]

Unknowns on grid (with enforced y1 symmetries by construction):
  U1 (odd in y1), U2 (even in y1), Phi (odd in y1), Psi (even in y1)
  + scalar lambda

Derived quantity:
  Omega := d_y1 U2 - d_y2 U1   (computed each forward pass)

Residuals enforced on INTERIOR nodes only:
  R_om   = Omega + W·∇Omega - Phi
  R_phi  = (2 + d_y1 U1)*Phi + W·∇Phi + (d_y1 U2)*Psi
  R_psi  = (2 + d_y2 U2)*Psi + W·∇Psi + (d_y2 U1)*Phi
  R_div  = d_y1 U1 + d_y2 U2
  R_comp = d_y1 Psi - d_y2 Phi

where W = (1+lambda)*y + U.

Boundary conditions (losses only on boundary nodes):
  Wall z2=0:
    always: U2=0
    optional reflection-consistent:
      Psi=0, d_y2 U1=0, d_y2 Phi=0, d_y2 Omega=0

  Far boundary (i=0, i=n1-1, j=n2-1; excluding wall):
    Phi=0, Psi=0, and U gradients flat: U1_y1=U1_y2=U2_y1=U2_y2=0

Gauge:
  (d_y1 Omega)(0,0) = -1 at z1=0 center, z2=0 wall.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def make_grids(n1, n2, L1, L2, device, dtype):
    z1 = torch.linspace(-L1/2, L1/2, n1, device=device, dtype=dtype)
    z2 = torch.linspace(0.0,   L2/2, n2, device=device, dtype=dtype)
    dz1 = z1[1] - z1[0]
    dz2 = z2[1] - z2[0]
    Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")
    Y1 = torch.sinh(Z1)
    Y2 = torch.sinh(Z2)
    s1 = 1.0 / torch.cosh(z1)  # (n1,)
    s2 = 1.0 / torch.cosh(z2)  # (n2,)
    return z1, z2, dz1, dz2, Z1, Z2, Y1, Y2, s1, s2


def make_masks(n1, n2, device):
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
    far[:, 0] = False

    return interior, wall, far


def dz1_central(f, h1):
    out = torch.empty_like(f)
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*h1)
    out[0, :]    = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*h1)
    out[-1, :]   = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*h1)
    return out


def dz2_central(f, h2):
    out = torch.empty_like(f)
    out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*h2)
    out[:, 0]    = (-3*f[:, 0] + 4*f[:, 1] - f[:, 2]) / (2*h2)
    out[:, -1]   = (3*f[:, -1] - 4*f[:, -2] + f[:, -3]) / (2*h2)
    return out


def dy1_from_dz1(f, h1, s1):
    return dz1_central(f, h1) * s1[:, None]


def dy2_from_dz2(f, h2, s2):
    return dz2_central(f, h2) * s2[None, :]


def dy1_upwind(f, W1, h1, s1):
    df_b = torch.empty_like(f)
    df_f = torch.empty_like(f)

    df_b[1:, :] = (f[1:, :] - f[:-1, :]) / h1
    df_b[0, :]  = (f[1, :] - f[0, :]) / h1

    df_f[:-1, :] = (f[1:, :] - f[:-1, :]) / h1
    df_f[-1, :]  = (f[-1, :] - f[-2, :]) / h1

    return torch.where(W1 >= 0, df_b, df_f) * s1[:, None]


def dy2_upwind(f, W2, h2, s2):
    df_b = torch.empty_like(f)
    df_f = torch.empty_like(f)

    df_b[:, 1:] = (f[:, 1:] - f[:, :-1]) / h2
    df_b[:, 0]  = (f[:, 1] - f[:, 0]) / h2

    df_f[:, :-1] = (f[:, 1:] - f[:, :-1]) / h2
    df_f[:, -1]  = (f[:, -1] - f[:, -2]) / h2

    return torch.where(W2 >= 0, df_b, df_f) * s2[None, :]



class Fields(torch.nn.Module):
    """
    Enforce y1-parity exactly by parameterizing only the y1>=0 half
    and reflecting with +/- symmetry.

    Odd in y1:  U1, Phi
    Even in y1: U2, Psi
    Omega is derived from U and will then be odd automatically.
    """
    def __init__(self, n1, n2, lam0=1.9, device="cpu", dtype=torch.float64, learn_lambda=False):
        super().__init__()
        assert n1 % 2 == 1, "Need odd n1 so that y1=0 is a gridline."
        self.n1, self.n2 = n1, n2
        self.i0 = n1 // 2
        self.n1h = n1 - self.i0  # includes centerline + positive side

        self._U1_h  = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._U2_h  = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._Phi_h = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))
        self._Psi_h = torch.nn.Parameter(torch.zeros((self.n1h, n2), device=device, dtype=dtype))

        if learn_lambda:
            self.lam = torch.nn.Parameter(torch.tensor(float(lam0), device=device, dtype=dtype))
        else:
            self.lam = torch.tensor(float(lam0), device=device, dtype=dtype)

    @staticmethod
    def _reflect_even(pos):
        neg = torch.flip(pos[1:], dims=[0])
        return torch.cat([neg, pos], dim=0)

    @staticmethod
    def _reflect_odd(pos):
        neg = torch.flip(pos[1:], dims=[0])
        center = torch.zeros_like(pos[:1])
        return torch.cat([-neg, center, pos[1:]], dim=0)

    @property
    def U1(self):
        pos = self._U1_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd(pos)

    @property
    def Phi(self):
        pos = self._Phi_h.clone()
        pos[0, :] = 0.0
        return self._reflect_odd(pos)

    @property
    def U2(self):
        return self._reflect_even(self._U2_h)

    @property
    def Psi(self):
        return self._reflect_even(self._Psi_h)


@torch.no_grad()
def init_fields(fields: Fields, Y1: torch.Tensor, Y2: torch.Tensor,
                R1=20.0, R2=20.0, d1=2.0, d2=2.0):
    """
    Initialize:
      U1 = -y1 * chi(y1,y2),  U2 = y2 * chi(y1,y2)
    where chi ~ 1 inside |y1|<R1 and y2<R2, and smoothly decays to 0 outside.
    This is consistent with 'flat' far-field gradients.
    """
    device = Y1.device
    dtype = Y1.dtype

    # smooth cutoffs (tanh-based)
    absY1 = torch.abs(Y1)
    chi1 = 0.5 * (1.0 - torch.tanh((absY1 - torch.tensor(R1, device=device, dtype=dtype)) /
                                  torch.tensor(d1, device=device, dtype=dtype)))
    chi2 = 0.5 * (1.0 - torch.tanh((Y2 - torch.tensor(R2, device=device, dtype=dtype)) /
                                  torch.tensor(d2, device=device, dtype=dtype)))
    chi = chi1 * chi2

    U1_full = -Y1 * chi
    U2_full = Y2 * chi

    # copy to half-params (i>=i0)
    i0 = fields.i0
    fields._U1_h.data[:] = U1_full[i0:, :].clone()
    fields._U2_h.data[:] = U2_full[i0:, :].clone()

    # enforce odd centerline for U1
    fields._U1_h.data[0, :] = 0.0

    # Phi/Psi init
    fields._Phi_h.data.zero_()
    fields._Psi_h.data.zero_()


def pde_residuals(fields: Fields, Y1, Y2, dz1, dz2, s1, s2, adv_scheme="upwind"):
    U1, U2, Ph, Ps = fields.U1, fields.U2, fields.Phi, fields.Psi
    lam = fields.lam

    W1 = (1.0 + lam) * Y1 + U1
    W2 = (1.0 + lam) * Y2 + U2

    # central derivatives (for div/curl/comp and source terms)
    U1_y1 = dy1_from_dz1(U1, dz1, s1)
    U1_y2 = dy2_from_dz2(U1, dz2, s2)
    U2_y1 = dy1_from_dz1(U2, dz1, s1)
    U2_y2 = dy2_from_dz2(U2, dz2, s2)

    Ph_y1_c = dy1_from_dz1(Ph, dz1, s1)
    Ph_y2_c = dy2_from_dz2(Ph, dz2, s2)
    Ps_y1_c = dy1_from_dz1(Ps, dz1, s1)
    Ps_y2_c = dy2_from_dz2(Ps, dz2, s2)

    # Omega derived from U (SIGN HERE)
    Omega = U2_y1 - U1_y2

    # Omega derivatives for advection
    Om_y1_c = dy1_from_dz1(Omega, dz1, s1)
    Om_y2_c = dy2_from_dz2(Omega, dz2, s2)

    # advection derivatives
    if adv_scheme == "upwind":
        Om_y1 = dy1_upwind(Omega, W1, dz1, s1)
        Om_y2 = dy2_upwind(Omega, W2, dz2, s2)
        Ph_y1 = dy1_upwind(Ph, W1, dz1, s1)
        Ph_y2 = dy2_upwind(Ph, W2, dz2, s2)
        Ps_y1 = dy1_upwind(Ps, W1, dz1, s1)
        Ps_y2 = dy2_upwind(Ps, W2, dz2, s2)
    elif adv_scheme == "central":
        Om_y1, Om_y2 = Om_y1_c, Om_y2_c
        Ph_y1, Ph_y2 = Ph_y1_c, Ph_y2_c
        Ps_y1, Ps_y2 = Ps_y1_c, Ps_y2_c
    else:
        raise ValueError("adv_scheme must be 'upwind' or 'central'.")

    adv_Om = W1 * Om_y1 + W2 * Om_y2
    adv_Ph = W1 * Ph_y1 + W2 * Ph_y2
    adv_Ps = W1 * Ps_y1 + W2 * Ps_y2

    R_om   = Omega + adv_Om - Ph
    R_phi  = (2.0 + U1_y1) * Ph + adv_Ph + (U2_y1) * Ps
    R_psi  = (2.0 + U2_y2) * Ps + adv_Ps + (U1_y2) * Ph
    R_div  = U1_y1 + U2_y2
    R_comp = Ps_y1_c - Ph_y2_c

    return {
        "R_om": R_om,
        "R_phi": R_phi,
        "R_psi": R_psi,
        "R_div": R_div,
        "R_comp": R_comp,
        "U1_y2": U1_y2,
        "Ph_y2": Ph_y2_c,
        "Omega": Omega,
        "Omega_y2": Om_y2_c,
        "Omega_y1": Om_y1_c,
    }


# ----------------------------
# Loss
# ----------------------------

def loss_total(fields: Fields, grids, masks, weights, adv_scheme="upwind", enforce_reflection_wall=True):
    z1, z2, dz1, dz2, Z1, Z2, Y1, Y2, s1, s2 = grids
    interior, wall, far = masks

    res = pde_residuals(fields, Y1, Y2, dz1, dz2, s1, s2, adv_scheme=adv_scheme)

    def mse_mask(A, mask):
        v = A[mask]
        return torch.mean(v*v)

    L_pde = (
        mse_mask(res["R_om"], interior) +
        mse_mask(res["R_phi"], interior) +
        mse_mask(res["R_psi"], interior) +
        mse_mask(res["R_div"], interior) +
        mse_mask(res["R_comp"], interior)
    )

    # Gauge: (dy1 Omega)(0,0) = -1
    i0 = fields.n1 // 2
    j0 = 0
    dy1_Om = res["Omega_y1"]
    g = dy1_Om[i0, j0] + 1.0
    L_g = g*g

    # Wall BCs
    U1, U2, Ph, Ps = fields.U1, fields.U2, fields.Phi, fields.Psi
    L_wall = mse_mask(U2, wall)

    if enforce_reflection_wall:
        U1_y2 = res["U1_y2"]
        Ph_y2 = res["Ph_y2"]
        Om_y2 = res["Omega_y2"]
        L_wall = L_wall + mse_mask(Ps, wall) + mse_mask(U1_y2, wall) + mse_mask(Ph_y2, wall) + mse_mask(Om_y2, wall)

    # Far BCs
    U1_y1 = dy1_from_dz1(U1, dz1, s1)
    U1_y2 = dy2_from_dz2(U1, dz2, s2)
    U2_y1 = dy1_from_dz1(U2, dz1, s1)
    U2_y2 = dy2_from_dz2(U2, dz2, s2)

    L_far = (
        mse_mask(Ph, far) +
        mse_mask(Ps, far) +
        mse_mask(U1_y1, far) +
        mse_mask(U1_y2, far) +
        mse_mask(U2_y1, far) +
        mse_mask(U2_y2, far)
    )

    L = \
        weights["pde"]   * L_pde  + \
        weights["gauge"] * L_g    + \
        weights["wall"]  * L_wall + \
        weights["far"]   * L_far

    stats = {
        "L": L.detach().item(),
        "L_pde": L_pde.detach().item(),
        "L_g": L_g.detach().item(),
        "L_wall": L_wall.detach().item(),
        "L_far": L_far.detach().item(),
        "dy1omega00": dy1_Om[i0, j0].detach().item(),
        "lam": float(fields.lam.detach().item()),
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
    lam0=1.9,
    learn_lambda=False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    grids = make_grids(n1, n2, L1, L2, device=device, dtype=dtype)
    masks = make_masks(n1, n2, device=device)

    fields = Fields(n1, n2, lam0=lam0, device=device, dtype=dtype, learn_lambda=learn_lambda)

    _, _, _, _, _, _, Y1, Y2, _, _ = grids
    init_fields(fields, Y1, Y2)

    # weights (tune these)
    weights = {"pde": 100.0, "gauge": 10.0, "wall": 10.0, "far": 100.0}

    opt = torch.optim.LBFGS(
        fields.parameters(),
        lr=1.0,
        max_iter=20,
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
# Visualization
# ----------------------------

def visualize(fields: Fields, grids, y1_lim=(-20, 20), y2_lim=(0, 20), title_suffix=""):
    z1, z2, dz1, dz2, Z1, Z2, Y1, Y2, s1, s2 = grids

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

    U1 = crop(fields.U1)
    U2 = crop(fields.U2)
    Ph = crop(fields.Phi)
    Ps = crop(fields.Psi)

    U1t, U2t = fields.U1, fields.U2
    Omega_t = dy1_from_dz1(U2t, dz1, s1) - dy2_from_dz2(U1t, dz2, s2)
    Om = crop(Omega_t)

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

    n1, n2 = 257, 129
    L1, L2 = 60.0, 60.0

    fields, grids = solve_torch(
        n1=n1, n2=n2, L1=L1, L2=L2,
        adv_scheme="upwind",
        enforce_reflection_wall=True,
        max_lbfgs_steps=50000,
        print_every=50,
        lam0=1.9,
        learn_lambda=False,
    )

    print(f"Done. lambda={fields.lam.detach().cpu().item():.12g}")
    visualize(fields, grids, y1_lim=(-20, 20), y2_lim=(0, 20), title_suffix="")


if __name__ == "__main__":
    main()
