#!/usr/bin/env python3
"""
2D Boussinesq self-similar profile equations — vorticity formulation (no pressure)
Solve ONLY on the physical half-plane: y2 >= 0 (no extension to y2<0, no quadrant reduction).

Mapping infinite y to finite z-box:
  y1 = sinh(z1),   z1 ∈ [-L1/2, L1/2]
  y2 = sinh(z2),   z2 ∈ [0,     L2/2]   (half-plane)

Unknowns on the HALF-PLANE grid:
  fields: Omega, U1, U2, Phi, Psi   (each on grid)
  scalar: lambda

Equations (moved to LHS):
  R_om   = Omega + W·∇Omega - Phi
  R_phi  = (2 + ∂y1 U1)*Phi + W·∇Phi + (∂y1 U2)*Psi
  R_psi  = (2 + ∂y2 U2)*Psi + W·∇Psi + (∂y2 U1)*Phi
  R_div  = ∂y1 U1 + ∂y2 U2
  R_vor  = Omega - (∂y1 U2 - ∂y2 U1)
  R_comp = ∂y1 Psi - ∂y2 Phi
where W = (1+lambda) y + U.

Derivatives:
  ∂/∂y1 = (1/cosh(z1)) ∂/∂z1
  ∂/∂y2 = (1/cosh(z2)) ∂/∂z2

IMPORTANT: Consistent discretization
  - PDE residuals are enforced ONLY on INTERIOR nodes (exclude wall and outer box boundary).
  - Boundary nodes have ONLY BC residuals (wall + far-field).
  - ADVECTIVE derivatives W·∇(Omega,Phi,Psi) use FIRST-ORDER UPWIND, with stencil chosen from current W.
    (Jacobian freezes the upwind stencil per Newton iteration.)

Boundary conditions used here:
  Wall z2=0 (y2=0):
    always enforce U2 = 0  (nonpenetration)
    optionally (reflection-consistent wall):
      Psi = 0,  ∂y2(U1)=0, ∂y2(Phi)=0, ∂y2(Omega)=0

  Far-field (outer box boundary: i=0, i=n1-1, j=n2-1; excluding wall j=0):
    Phi = 0, Psi = 0,
    U1_y1=U1_y2=U2_y1=U2_y2=0  (simple flatness)

Gauge (nondegeneracy):
  (∂y1 Omega)(0,0) = -1 at y1=0 (z1=0 center), y2=0 (z2=0 wall).

Newton-like iterations:
  Each iteration:
    - build interior PDE residual/Jacobian
    - append gauge + BC residual/Jacobian
    - solve J dx ≈ -r by damped LSQR
    - backtracking line-search on ||r||
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ----------------------------
# 1) 1D derivative matrices on uniform grids (z)
# ----------------------------

def D1_center_1d(n: int, h: float) -> sp.csr_matrix:
    """Centered first derivative: 2nd-order interior; 2nd-order one-sided at boundaries."""
    if n < 3:
        raise ValueError("Need n>=3.")
    rows, cols, data = [], [], []

    for i in range(1, n - 1):
        rows += [i, i]
        cols += [i - 1, i + 1]
        data += [-0.5 / h, 0.5 / h]

    # left boundary: (-3 f0 + 4 f1 - f2)/(2h)
    rows += [0, 0, 0]
    cols += [0, 1, 2]
    data += [-3 / (2 * h), 4 / (2 * h), -1 / (2 * h)]

    # right boundary: (3 f_{n-1} - 4 f_{n-2} + f_{n-3})/(2h)
    rows += [n - 1, n - 1, n - 1]
    cols += [n - 1, n - 2, n - 3]
    data += [3 / (2 * h), -4 / (2 * h), 1 / (2 * h)]

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def D1_forward_1d(n: int, h: float) -> sp.csr_matrix:
    """Forward first derivative: 1st-order interior; 1st-order backward at last point."""
    if n < 2:
        raise ValueError("Need n>=2.")
    rows, cols, data = [], [], []
    for i in range(n - 1):
        rows += [i, i]
        cols += [i, i + 1]
        data += [-1.0 / h, 1.0 / h]
    # last point: backward
    rows += [n - 1, n - 1]
    cols += [n - 2, n - 1]
    data += [-1.0 / h, 1.0 / h]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def D1_backward_1d(n: int, h: float) -> sp.csr_matrix:
    """Backward first derivative: 1st-order interior; 1st-order forward at first point."""
    if n < 2:
        raise ValueError("Need n>=2.")
    rows, cols, data = [], [], []
    # first point: forward
    rows += [0, 0]
    cols += [0, 1]
    data += [-1.0 / h, 1.0 / h]
    for i in range(1, n):
        rows += [i, i]
        cols += [i - 1, i]
        data += [-1.0 / h, 1.0 / h]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# ----------------------------
# 2) 2D derivative operators in z (Kronecker structure)
# ----------------------------

def Dz_2d(n1: int, n2: int, h1: float, h2: float):
    """
    C-order flatten vec = F.ravel(order="C"), index = i*n2 + j.
    """
    I1 = sp.eye(n1, format="csr")
    I2 = sp.eye(n2, format="csr")

    D1c_z1 = D1_center_1d(n1, h1)
    D1c_z2 = D1_center_1d(n2, h2)
    D1f_z1 = D1_forward_1d(n1, h1)
    D1f_z2 = D1_forward_1d(n2, h2)
    D1b_z1 = D1_backward_1d(n1, h1)
    D1b_z2 = D1_backward_1d(n2, h2)

    Dz1_c = sp.kron(D1c_z1, I2, format="csr")
    Dz2_c = sp.kron(I1, D1c_z2, format="csr")

    Dz1_f = sp.kron(D1f_z1, I2, format="csr")
    Dz2_f = sp.kron(I1, D1f_z2, format="csr")

    Dz1_b = sp.kron(D1b_z1, I2, format="csr")
    Dz2_b = sp.kron(I1, D1b_z2, format="csr")

    return Dz1_c, Dz2_c, Dz1_f, Dz2_f, Dz1_b, Dz2_b


def Dy_from_Dz(z1: np.ndarray, z2: np.ndarray, Dz1: sp.csr_matrix, Dz2: sp.csr_matrix):
    """Apply chain rule ∂/∂y = (1/cosh(z)) ∂/∂z."""
    n1 = z1.size
    n2 = z2.size
    s1 = 1.0 / np.cosh(z1)
    s2 = 1.0 / np.cosh(z2)

    S1 = np.repeat(s1[:, None], n2, axis=1).ravel(order="C")
    S2 = np.repeat(s2[None, :], n1, axis=0).ravel(order="C")

    Dy1 = sp.diags(S1, 0, format="csr") @ Dz1
    Dy2 = sp.diags(S2, 0, format="csr") @ Dz2
    return Dy1, Dy2


def build_upwind_Dy(z1, z2, Dz1_f, Dz2_f, Dz1_b, Dz2_b, W1, W2):
    """
    Build upwind physical derivatives for advection:
      if W>0 use backward difference, else forward.
    Stencil is frozen for the current Newton iteration.
    """
    n1 = z1.size
    n2 = z2.size
    s1 = 1.0 / np.cosh(z1)
    s2 = 1.0 / np.cosh(z2)

    S1 = np.repeat(s1[:, None], n2, axis=1).ravel(order="C")
    S2 = np.repeat(s2[None, :], n1, axis=0).ravel(order="C")

    W1v = W1.ravel(order="C")
    W2v = W2.ravel(order="C")

    m1p = (W1v >= 0.0).astype(float)
    m1m = 1.0 - m1p
    m2p = (W2v >= 0.0).astype(float)
    m2m = 1.0 - m2p

    M1p = sp.diags(m1p, 0, format="csr")
    M1m = sp.diags(m1m, 0, format="csr")
    M2p = sp.diags(m2p, 0, format="csr")
    M2m = sp.diags(m2m, 0, format="csr")

    # upwind in z, then scale to y
    Dy1u = sp.diags(S1, 0, format="csr") @ (M1p @ Dz1_b + M1m @ Dz1_f)
    Dy2u = sp.diags(S2, 0, format="csr") @ (M2p @ Dz2_b + M2m @ Dz2_f)
    return Dy1u, Dy2u


# ----------------------------
# 3) Packing/unpacking (no symmetry reduction)
# ----------------------------

def pack_full(om, u1, u2, ph, ps, lam) -> np.ndarray:
    return np.concatenate(
        [
            om.ravel(order="C"),
            u1.ravel(order="C"),
            u2.ravel(order="C"),
            ph.ravel(order="C"),
            ps.ravel(order="C"),
            np.array([lam], dtype=float),
        ]
    )


def unpack_full(x, n1, n2):
    N = n1 * n2
    om = x[0:N].reshape((n1, n2), order="C")
    u1 = x[N:2 * N].reshape((n1, n2), order="C")
    u2 = x[2 * N:3 * N].reshape((n1, n2), order="C")
    ph = x[3 * N:4 * N].reshape((n1, n2), order="C")
    ps = x[4 * N:5 * N].reshape((n1, n2), order="C")
    lam = float(x[5 * N])
    return om, u1, u2, ph, ps, lam


# ----------------------------
# 4) Node sets: interior vs boundaries
# ----------------------------

def flat_ids_from_mask(mask_2d: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask_2d.ravel(order="C")).astype(int)


def interior_ids(n1: int, n2: int) -> np.ndarray:
    m = np.ones((n1, n2), dtype=bool)
    m[:, 0] = False
    m[0, :] = False
    m[-1, :] = False
    m[:, -1] = False
    return flat_ids_from_mask(m)


def wall_ids(n1: int, n2: int) -> np.ndarray:
    m = np.zeros((n1, n2), dtype=bool)
    m[:, 0] = True
    return flat_ids_from_mask(m)


def far_ids(n1: int, n2: int) -> np.ndarray:
    m = np.zeros((n1, n2), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, -1] = True
    m[:, 0] = False
    return flat_ids_from_mask(m)


def build_selector_matrix(nodes: np.ndarray, N: int) -> sp.csr_matrix:
    m = nodes.size
    return sp.csr_matrix((np.ones(m), (np.arange(m), nodes)), shape=(m, N))


# ----------------------------
# 5) Gauge + BC assembly (rows)
# ----------------------------

def gauge_row_dy1_omega_origin(Dy1_center: sp.csr_matrix, n1: int, n2: int):
    N = n1 * n2
    i0 = n1 // 2
    j0 = 0
    origin = i0 * n2 + j0

    eT = sp.csr_matrix(([1.0], ([0], [origin])), shape=(1, N))
    row_om = eT @ Dy1_center  # use centered derivative for gauge (more accurate)
    ZN = sp.csr_matrix((1, N))
    Z1 = sp.csr_matrix((1, 1))
    row_full = sp.hstack([row_om, ZN, ZN, ZN, ZN, Z1], format="csr")
    return row_full, origin


def bc_rows_and_residuals(x: np.ndarray, Dy1: sp.csr_matrix, Dy2: sp.csr_matrix,
                          n1: int, n2: int,
                          w_wall: float = 10.0, w_far: float = 10.0,
                          enforce_reflection_wall: bool = True):
    N = n1 * n2
    om, u1, u2, ph, ps, _lam = unpack_full(x, n1, n2)

    omv = om.ravel(order="C")
    u1v = u1.ravel(order="C")
    u2v = u2.ravel(order="C")
    phv = ph.ravel(order="C")
    psv = ps.ravel(order="C")

    u1_y1 = Dy1 @ u1v
    u1_y2 = Dy2 @ u1v
    u2_y1 = Dy1 @ u2v
    u2_y2 = Dy2 @ u2v
    om_y2 = Dy2 @ omv
    ph_y2 = Dy2 @ phv

    wall = wall_ids(n1, n2)
    far = far_ids(n1, n2)
    Sw = build_selector_matrix(wall, N)
    Sf = build_selector_matrix(far, N)

    mw = wall.size
    mf = far.size
    ZNw = sp.csr_matrix((mw, N))
    ZNf = sp.csr_matrix((mf, N))
    Z1w = sp.csr_matrix((mw, 1))
    Z1f = sp.csr_matrix((mf, 1))

    rows = []
    rhs = []

    # Wall: U2=0
    r_u2w = Sw @ u2v
    J_u2w = sp.hstack([ZNw, ZNw, Sw, ZNw, ZNw, Z1w], format="csr")
    rows.append(w_wall * J_u2w)
    rhs.append(w_wall * r_u2w)

    if enforce_reflection_wall:
        # Psi=0
        r_psw = Sw @ psv
        J_psw = sp.hstack([ZNw, ZNw, ZNw, ZNw, Sw, Z1w], format="csr")
        rows.append(w_wall * J_psw)
        rhs.append(w_wall * r_psw)

        SDy2w = Sw @ Dy2

        # U1_y2=0
        r_u1y2w = Sw @ u1_y2
        J_u1y2w = sp.hstack([ZNw, SDy2w, ZNw, ZNw, ZNw, Z1w], format="csr")
        rows.append(w_wall * J_u1y2w)
        rhs.append(w_wall * r_u1y2w)

        # Phi_y2=0
        r_phy2w = Sw @ ph_y2
        J_phy2w = sp.hstack([ZNw, ZNw, ZNw, SDy2w, ZNw, Z1w], format="csr")
        rows.append(w_wall * J_phy2w)
        rhs.append(w_wall * r_phy2w)

        # Omega_y2=0
        r_omy2w = Sw @ om_y2
        J_omy2w = sp.hstack([SDy2w, ZNw, ZNw, ZNw, ZNw, Z1w], format="csr")
        rows.append(w_wall * J_omy2w)
        rhs.append(w_wall * r_omy2w)

    # Far: Phi=0, Psi=0
    r_phf = Sf @ phv
    r_psf = Sf @ psv
    J_phf = sp.hstack([ZNf, ZNf, ZNf, Sf,  ZNf, Z1f], format="csr")
    J_psf = sp.hstack([ZNf, ZNf, ZNf, ZNf, Sf,  Z1f], format="csr")
    rows.append(w_far * J_phf); rhs.append(w_far * r_phf)
    rows.append(w_far * J_psf); rhs.append(w_far * r_psf)

    # Far: flatness of U gradients
    SDy1f = Sf @ Dy1
    SDy2f = Sf @ Dy2
    r_u1y1f = Sf @ u1_y1
    r_u1y2f = Sf @ u1_y2
    r_u2y1f = Sf @ u2_y1
    r_u2y2f = Sf @ u2_y2

    J_u1y1f = sp.hstack([ZNf, SDy1f, ZNf,  ZNf,  ZNf, Z1f], format="csr")
    J_u1y2f = sp.hstack([ZNf, SDy2f, ZNf,  ZNf,  ZNf, Z1f], format="csr")
    J_u2y1f = sp.hstack([ZNf, ZNf,  SDy1f, ZNf,  ZNf, Z1f], format="csr")
    J_u2y2f = sp.hstack([ZNf, ZNf,  SDy2f, ZNf,  ZNf, Z1f], format="csr")

    rows.append(w_far * J_u1y1f); rhs.append(w_far * r_u1y1f)
    rows.append(w_far * J_u1y2f); rhs.append(w_far * r_u1y2f)
    rows.append(w_far * J_u2y1f); rhs.append(w_far * r_u2y1f)
    rows.append(w_far * J_u2y2f); rhs.append(w_far * r_u2y2f)

    r_bc = np.concatenate(rhs) if rhs else np.zeros(0)
    J_bc = sp.vstack(rows, format="csr") if rows else sp.csr_matrix((0, 5 * N + 1))
    return r_bc, J_bc


# ----------------------------
# 6) PDE residual/Jacobian with UPWIND advection (frozen stencil)
# ----------------------------

def residual_full_vorticity_upwind(om, u1, u2, ph, ps, lam, Y1, Y2,
                                   Dy1_c, Dy2_c, Dy1_u, Dy2_u):
    """
    Uses centered derivatives for all non-advective derivatives,
    and upwind derivatives for advected fields Omega,Phi,Psi in W·∇.
    """
    n1, n2 = om.shape
    N = n1 * n2

    omv = om.ravel(order="C")
    u1v = u1.ravel(order="C")
    u2v = u2.ravel(order="C")
    phv = ph.ravel(order="C")
    psv = ps.ravel(order="C")

    Y1v = Y1.ravel(order="C")
    Y2v = Y2.ravel(order="C")

    # centered derivatives (for coupling terms)
    u1_y1 = Dy1_c @ u1v
    u1_y2 = Dy2_c @ u1v
    u2_y1 = Dy1_c @ u2v
    u2_y2 = Dy2_c @ u2v

    # upwind derivatives for advected scalars
    om_y1u = Dy1_u @ omv
    om_y2u = Dy2_u @ omv
    ph_y1u = Dy1_u @ phv
    ph_y2u = Dy2_u @ phv
    ps_y1u = Dy1_u @ psv
    ps_y2u = Dy2_u @ psv

    W1v = (1.0 + lam) * Y1v + u1v
    W2v = (1.0 + lam) * Y2v + u2v

    adv_om = W1v * om_y1u + W2v * om_y2u
    adv_ph = W1v * ph_y1u + W2v * ph_y2u
    adv_ps = W1v * ps_y1u + W2v * ps_y2u

    R_om   = omv + adv_om - phv
    R_phi  = (2.0 + u1_y1) * phv + adv_ph + (u2_y1) * psv
    R_psi  = (2.0 + u2_y2) * psv + adv_ps + (u1_y2) * phv
    R_div  = u1_y1 + u2_y2
    R_vor  = omv - (u2_y1 - u1_y2)
    R_comp = (Dy1_c @ psv) - (Dy2_c @ phv)  # compatibility uses centered

    return np.concatenate([R_om, R_phi, R_psi, R_div, R_vor, R_comp])


def jacobian_full_vorticity_upwind(om, u1, u2, ph, ps, lam, Y1, Y2,
                                   Dy1_c, Dy2_c, Dy1_u, Dy2_u):
    """
    Jacobian for the UPWIND residual, freezing the upwind stencil (Dy*_u).
    """
    n1, n2 = om.shape
    N = n1 * n2

    omv = om.ravel(order="C")
    u1v = u1.ravel(order="C")
    u2v = u2.ravel(order="C")
    phv = ph.ravel(order="C")
    psv = ps.ravel(order="C")

    Y1v = Y1.ravel(order="C")
    Y2v = Y2.ravel(order="C")

    # centered derivatives (non-advective)
    u1_y1 = Dy1_c @ u1v
    u1_y2 = Dy2_c @ u1v
    u2_y1 = Dy1_c @ u2v
    u2_y2 = Dy2_c @ u2v

    # upwind derivatives (advective)
    om_y1u = Dy1_u @ omv
    om_y2u = Dy2_u @ omv
    ph_y1u = Dy1_u @ phv
    ph_y2u = Dy2_u @ phv
    ps_y1u = Dy1_u @ psv
    ps_y2u = Dy2_u @ psv

    W1v = (1.0 + lam) * Y1v + u1v
    W2v = (1.0 + lam) * Y2v + u2v

    D_W1 = sp.diags(W1v, 0, format="csr")
    D_W2 = sp.diags(W2v, 0, format="csr")

    # advective operator A_u = diag(W1) Dy1_u + diag(W2) Dy2_u
    A_u = D_W1 @ Dy1_u + D_W2 @ Dy2_u

    I = sp.eye(N, format="csr")
    Z = sp.csr_matrix((N, N))

    # diag of upwind gradients for coupling into u-variations
    D_om_y1u = sp.diags(om_y1u, 0, format="csr")
    D_om_y2u = sp.diags(om_y2u, 0, format="csr")
    D_ph_y1u = sp.diags(ph_y1u, 0, format="csr")
    D_ph_y2u = sp.diags(ph_y2u, 0, format="csr")
    D_ps_y1u = sp.diags(ps_y1u, 0, format="csr")
    D_ps_y2u = sp.diags(ps_y2u, 0, format="csr")

    D_ph = sp.diags(phv, 0, format="csr")
    D_ps = sp.diags(psv, 0, format="csr")
    D_u2y1 = sp.diags(u2_y1, 0, format="csr")
    D_u1y2 = sp.diags(u1_y2, 0, format="csr")

    # R_om = om + (W·∇_u om) - ph
    J1_om = I + A_u
    J1_u1 = D_om_y1u
    J1_u2 = D_om_y2u
    J1_ph = -I
    J1_ps = Z
    J1_l  = (Y1v * om_y1u) + (Y2v * om_y2u)

    # R_phi = (2+u1_y1)*ph + W·∇_u ph + (u2_y1)*ps
    # d/du1: (ph)*Dy1_c  + diag(ph_y1u)   (from W1 variation)
    # d/du2: diag(ph_y2u) + (ps)*Dy1_c    (from W2 variation + u2_y1*ps term)
    J2_om = Z
    J2_u1 = D_ph @ Dy1_c + D_ph_y1u
    J2_u2 = D_ph_y2u + D_ps @ Dy1_c
    J2_ph = sp.diags(2.0 + u1_y1, 0, format="csr") + A_u
    J2_ps = D_u2y1
    J2_l  = (Y1v * ph_y1u) + (Y2v * ph_y2u)

    # R_psi = (2+u2_y2)*ps + W·∇_u ps + (u1_y2)*ph
    J3_om = Z
    J3_u1 = D_ps_y1u + D_ph @ Dy2_c
    J3_u2 = D_ps @ Dy2_c + D_ps_y2u
    J3_ph = D_u1y2
    J3_ps = sp.diags(2.0 + u2_y2, 0, format="csr") + A_u
    J3_l  = (Y1v * ps_y1u) + (Y2v * ps_y2u)

    # R_div = u1_y1 + u2_y2
    J4_om = Z
    J4_u1 = Dy1_c
    J4_u2 = Dy2_c
    J4_ph = Z
    J4_ps = Z
    J4_l  = np.zeros(N)

    # R_vor = om - (u2_y1 - u1_y2)
    J5_om = I
    J5_u1 = Dy2_c
    J5_u2 = -Dy1_c
    J5_ph = Z
    J5_ps = Z
    J5_l  = np.zeros(N)

    # R_comp = ps_y1 - ph_y2 (centered)
    J6_om = Z
    J6_u1 = Z
    J6_u2 = Z
    J6_ph = -Dy2_c
    J6_ps = Dy1_c
    J6_l  = np.zeros(N)

    J = sp.bmat(
        [
            [J1_om, J1_u1, J1_u2, J1_ph, J1_ps],
            [J2_om, J2_u1, J2_u2, J2_ph, J2_ps],
            [J3_om, J3_u1, J3_u2, J3_ph, J3_ps],
            [J4_om, J4_u1, J4_u2, J4_ph, J4_ps],
            [J5_om, J5_u1, J5_u2, J5_ph, J5_ps],
            [J6_om, J6_u1, J6_u2, J6_ph, J6_ps],
        ],
        format="csr",
    )

    Jlam = np.concatenate([J1_l, J2_l, J3_l, J4_l, J5_l, J6_l])
    J = sp.hstack([J, sp.csr_matrix(Jlam).T], format="csr")
    return J


# ----------------------------
# 7) Newton loop with interior-PDE enforcement
# ----------------------------

def newton_solve(
    n1=257,
    n2=129,
    L1=60.0,
    L2=60.0,
    maxit=50,
    lsqr_damp=1e-1,
    step_cap=1.0,
    w_gauge=100.0,
    w_wall=10.0,
    w_far=10.0,
    enforce_reflection_wall=True,
):
    # z grids
    z1 = np.linspace(-L1 / 2, L1 / 2, n1)
    z2 = np.linspace(0.0, L2 / 2, n2)
    h1 = z1[1] - z1[0]
    h2 = z2[1] - z2[0]

    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")
    Y1 = np.sinh(Z1)
    Y2 = np.sinh(Z2)
    N = n1 * n2

    # z-derivative operators
    Dz1_c, Dz2_c, Dz1_f, Dz2_f, Dz1_b, Dz2_b = Dz_2d(n1, n2, h1, h2)

    # centered physical derivatives (used for non-advective terms + BCs + gauge)
    Dy1_c, Dy2_c = Dy_from_Dz(z1, z2, Dz1_c, Dz2_c)

    # node sets
    ids_int = interior_ids(n1, n2)
    m_int = ids_int.size
    pde_rows = np.concatenate([ids_int + k * N for k in range(6)]).astype(int)

    # gauge row (centered derivative)
    gauge_row, origin_idx = gauge_row_dy1_omega_origin(Dy1_c, n1, n2)

    # initial guess
    rng = np.random.default_rng(0)
    eps = 1e-3
    om0 = -Y1 / (1.0 + Y1**2 + Y2**2)
    u10 = eps * rng.standard_normal((n1, n2))
    u20 = eps * rng.standard_normal((n1, n2))
    ph0 = eps * rng.standard_normal((n1, n2))
    ps0 = eps * rng.standard_normal((n1, n2))
    lam0 = 0.3
    x = pack_full(om0, u10, u20, ph0, ps0, lam0)

    def build_rJ(xcur):
        om, u1, u2, ph, ps, lam = unpack_full(xcur, n1, n2)

        # build W and upwind Dy matrices for this iteration
        W1 = (1.0 + lam) * Y1 + u1
        W2 = (1.0 + lam) * Y2 + u2
        Dy1_u, Dy2_u = build_upwind_Dy(z1, z2, Dz1_f, Dz2_f, Dz1_b, Dz2_b, W1, W2)

        # full PDE (upwind advection)
        r_pde_full = residual_full_vorticity_upwind(
            om, u1, u2, ph, ps, lam, Y1, Y2, Dy1_c, Dy2_c, Dy1_u, Dy2_u
        )
        J_pde_full = jacobian_full_vorticity_upwind(
            om, u1, u2, ph, ps, lam, Y1, Y2, Dy1_c, Dy2_c, Dy1_u, Dy2_u
        )

        # restrict PDE to interior only
        r_pde = r_pde_full[pde_rows]
        J_pde = J_pde_full[pde_rows, :]

        # gauge residual (centered)
        omega_vec = om.ravel(order="C")
        dy1_om = (Dy1_c @ omega_vec)[origin_idx]
        r_g = np.array([dy1_om + 1.0])

        # BC residual/J (centered)
        r_bc, J_bc = bc_rows_and_residuals(
            xcur, Dy1_c, Dy2_c, n1, n2,
            w_wall=w_wall, w_far=w_far,
            enforce_reflection_wall=enforce_reflection_wall,
        )

        # stack
        r = np.concatenate([r_pde, w_gauge * r_g, r_bc])
        J = sp.vstack([J_pde, w_gauge * gauge_row, J_bc], format="csr")

        # diagnostics
        blocks = [r_pde[k * m_int:(k + 1) * m_int] for k in range(6)]
        bnorms = np.array([np.linalg.norm(b) for b in blocks], dtype=float)
        nr_pde = float(np.linalg.norm(r_pde))
        nr_bc = float(np.linalg.norm(r_bc))
        nr_g = float(np.linalg.norm(w_gauge * r_g))

        return r, J, dy1_om, nr_pde, nr_bc, nr_g, bnorms

    bnames = ["om", "phi", "psi", "div", "vor", "comp"]

    for it in range(maxit):
        r, J, dy1om, nr_pde, nr_bc, nr_g, bnorms = build_rJ(x)
        nr = float(np.linalg.norm(r))

        print(f"[it {it:02d}] ||r||={nr:.3e}  ||PDE||={nr_pde:.3e}  ||BC||={nr_bc:.3e}  ||G||={nr_g:.3e}  dy1_omega(0,0)={dy1om:.6f}")
        print("          PDE blocks:", " ".join([f"{n}={v:.2e}" for n, v in zip(bnames, bnorms)]))

        sol = spla.lsqr(J, -r, damp=lsqr_damp, atol=1e-10, btol=1e-10, iter_lim=8000)
        dx = sol[0]
        dxn = float(np.linalg.norm(dx))
        print(f"          lsqr: iters={sol[2]} exit={sol[1]}  ||dx||={dxn:.3e}")

        if not np.isfinite(dxn) or dxn == 0.0:
            print("          dx invalid/zero; stopping.")
            break

        if dxn > step_cap:
            dx *= (step_cap / dxn)

        alpha = 1.0
        r0 = nr
        nrt = nr
        for _ in range(14):
            xt = x + alpha * dx
            rt, _, _, _, _, _, _ = build_rJ(xt)
            nrt = float(np.linalg.norm(rt))
            if nrt <= (1.0 - 1e-4 * alpha) * r0:
                break
            alpha *= 0.5

        x = x + alpha * dx
        print(f"          line search: alpha={alpha:.3f}  ||r|| {r0:.3e} -> {nrt:.3e}")

        if nrt < 1e-10:
            print("Converged.")
            break

        if alpha < 1e-6:
            print("          line search stalled (alpha ~ 0). Consider adjusting weights/damping or initial guess.")
            break

    # return solution and grids for plotting
    return x, z1, z2, Y1, Y2


# ----------------------------
# 8) Visualization
# ----------------------------

def plot_surface(ax, X, Y, Z, title, xlabel, ylabel):
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.view_init(elev=18, azim=-60)


def visualize_solution(om, u1, u2, ph, ps, y1, y2, y1_lim=(-20, 20), y2_lim=(0, 20)):
    mask_y1 = (y1[:, 0] >= y1_lim[0]) & (y1[:, 0] <= y1_lim[1])
    mask_y2 = (y2[0, :] >= y2_lim[0]) & (y2[0, :] <= y2_lim[1])

    I = np.where(mask_y1)[0]
    J = np.where(mask_y2)[0]
    if I.size < 3 or J.size < 3:
        print("Visualization window too small for chosen limits; skipping plots.")
        return

    Y1 = y1[np.ix_(I, J)]
    Y2 = y2[np.ix_(I, J)]

    Om = om[np.ix_(I, J)]
    Ph = ph[np.ix_(I, J)]
    Ps = ps[np.ix_(I, J)]
    U1 = u1[np.ix_(I, J)]
    U2 = u2[np.ix_(I, J)]

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    plot_surface(ax1, Y1, Y2, Om, r"$\Omega$", r"$y_1$", r"$y_2$")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    plot_surface(ax2, Y1, Y2, Ph, r"$\Phi$", r"$y_1$", r"$y_2$")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    plot_surface(ax3, Y1, Y2, Ps, r"$\Psi$", r"$y_1$", r"$y_2$")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    plot_surface(ax4, Y1, Y2, U1, r"$U_1$", r"$y_1$", r"$y_2$")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    plot_surface(ax5, Y1, Y2, U2, r"$U_2$", r"$y_1$", r"$y_2$")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(0.05, 0.6, "Self-similar\nsolution", fontsize=18)

    plt.tight_layout()
    plt.show()


# ----------------------------
# 9) Main
# ----------------------------

def main():
    n1, n2 = 257, 129
    L1, L2 = 60.0, 60.0

    x, z1, z2, y1, y2 = newton_solve(
        n1=n1, n2=n2,
        L1=L1, L2=L2,
        maxit=100,
        lsqr_damp=1e-1,
        step_cap=1.0,
        enforce_reflection_wall=True,
    )

    om, u1, u2, ph, ps, lam = unpack_full(x, n1, n2)
    print(f"Done. lambda={lam:.12g}")

    visualize_solution(
        om, u1, u2, ph, ps,
        y1, y2,
        y1_lim=(-20, 20),
        y2_lim=(0, 20),
    )


if __name__ == "__main__":
    main()
