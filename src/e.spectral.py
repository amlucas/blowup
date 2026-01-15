import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyfftw
def fftn(a):
    return pyfftw.interfaces.numpy_fft.fftn(a, threads=os.cpu_count(), planner_effort="FFTW_MEASURE")

def ifftn(a):
    return pyfftw.interfaces.numpy_fft.ifftn(a, threads=os.cpu_count(), planner_effort="FFTW_MEASURE")

N = 1500
T = 3.16
cfl = 0.5
plot_every = 20
dpi = 400
filt_alpha = 36.0
filt_mf = 10
r_levels = 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
outdir = "out"
os.makedirs(outdir, exist_ok=True)

L = 2.0 * np.pi
d = L / N
x = np.arange(N) * d
y = np.arange(N) * d
X, Y = np.meshgrid(x, y, indexing="ij")

dy_ic = Y - np.pi
r1sq = X * X + dy_ic * dy_ic
p1 = np.zeros_like(X, dtype=float)
m1 = r1sq < (np.pi**2)
denom1 = (np.pi**2) - r1sq[m1]
p1[m1] = np.exp(1.0 - (np.pi**2) / denom1)

a = 1.95 * np.pi
dx2 = X - 2.0 * np.pi
p2 = np.zeros_like(X, dtype=float)
m2 = np.abs(dx2) < a
denom2 = (a**2) - (dx2[m2] ** 2)
p2[m2] = np.exp(1.0 - (a**2) / denom2)
rho = 50.0 * p1 * p2 * (1.0 - p1)
omega = np.zeros_like(rho)

k = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
KX, KY = np.meshgrid(k, k, indexing="ij")
k2 = KX**2 + KY**2
k2[0, 0] = 1.0
k_int = np.fft.fftfreq(N) * N
KX_int, KY_int = np.meshgrid(k_int, k_int, indexing="ij")
r = np.sqrt(KX_int**2 + KY_int**2)
rmax = np.max(r)
filt = np.exp(-filt_alpha * (r / rmax) ** filt_mf)
k_cut = N // 3
dealias = ((np.abs(KX_int) <= k_cut) & (np.abs(KY_int) <= k_cut)).astype(float)

def Dx(f):
    f_hat = fftn(f) * dealias
    return ifftn((1j * KX) * (filt * f_hat)).real


def Dy(f):
    f_hat = fftn(f) * dealias
    return ifftn((1j * KY) * (filt * f_hat)).real


def vel_from_omega(omega_):
    w_hat = fftn(omega_)
    psi_hat = w_hat / k2
    psi_hat[0, 0] = 0.0
    u_hat = +(1j * KY) * (filt * psi_hat)
    v_hat = -(1j * KX) * (filt * psi_hat)
    u_ = ifftn(u_hat).real
    v_ = ifftn(v_hat).real
    return u_, v_


def rhs(rho_, omega_):
    u_, v_ = vel_from_omega(omega_)
    drho = -(Dx(u_ * rho_) + Dy(v_ * rho_))
    adv_omega = Dx(u_ * omega_) + Dy(v_ * omega_)
    domega = -adv_omega - Dy(rho_)
    return drho, domega
E0 = None
t = 0.0
step = 0
while t < T:
    u, v = vel_from_omega(omega)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    dt_adv = cfl * min(d / (umax + 1e-12), d / (vmax + 1e-12))
    dt = min(dt_adv, cfl * d)
    if t + dt > T:
        dt = T - t
    dA = d * d
    w_hat_E = fftn(omega)
    psi_hat_E = w_hat_E / k2
    psi_hat_E[0, 0] = 0.0
    psi_E = ifftn(psi_hat_E).real
    KE = 0.5 * np.sum(psi_E * omega) * dA
    PE = -np.sum(rho * X) * dA
    E_total = KE + PE
    if E0 is None:
        E0 = E_total
    dE_rel = (E_total - E0) / (abs(E0) + 1e-14) * 100.0
    rho_max = np.max(rho)
    rho_min = np.min(rho)
    if step % plot_every == 0:
        rp = np.maximum(rho, 0.0)
        m = np.sum(rp)
        xcm = np.sum(rp * X) / m
        ycm = np.sum(rp * Y) / m

        print(
            f"plot step={step:06d} t={t:12.8e} xcm={xcm:.4f} ycm={ycm:.4f} "
            f"E={E_total:.6f} dE%={dE_rel:.4f} rho=[{rho_min:.4f},{rho_max:.4f}]"
        )
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), constrained_layout=True)
        ax.contour(Y, X, rho, levels=r_levels, colors="k", linewidths=0.8)
        ax.set_title(f"step: {step:08d}, time: {t:6.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.savefig(os.path.join(outdir, f"rho_{step:06d}.png"), dpi=dpi)
        plt.close(fig)

    k1_r, k1_w = rhs(rho, omega)
    r1 = rho + dt * k1_r
    w1 = omega + dt * k1_w

    k2_r, k2_w = rhs(r1, w1)
    r2 = 0.75 * rho + 0.25 * (r1 + dt * k2_r)
    w2 = 0.75 * omega + 0.25 * (w1 + dt * k2_w)

    k3_r, k3_w = rhs(r2, w2)
    rho = (1.0 / 3.0) * rho + (2.0 / 3.0) * (r2 + dt * k3_r)
    omega = (1.0 / 3.0) * omega + (2.0 / 3.0) * (w2 + dt * k3_w)

    t += dt
    step += 1
