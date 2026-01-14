import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
N = 128
T = 3.0
cfl = 0.5
plot_every = 20
levels = 40
filt_alpha = 36.0
filt_mf = 10
os.makedirs("out", exist_ok=True)
L = 2.0 * np.pi
dx = L / N
dy = L / N
x = np.arange(N) * dx
y = np.arange(N) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
x_ic = X - np.pi
y_ic = Y
dy_ic = y_ic - np.pi
r1sq = x_ic * x_ic + dy_ic * dy_ic
p1 = np.zeros_like(X, dtype=float)
m1 = r1sq < (np.pi**2)
denom1 = (np.pi**2) - r1sq[m1]
p1[m1] = np.exp(1.0 - (np.pi**2) / denom1)
a = 1.95 * np.pi
dx2 = x_ic
p2 = np.zeros_like(X, dtype=float)
m2 = np.abs(dx2) < a
denom2 = (a**2) - (dx2[m2] ** 2)
p2[m2] = np.exp(1.0 - (a**2) / denom2)
rho = 50.0 * p1 * p2 * (1.0 - p1)
omega = np.zeros_like(rho)
k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
k_filt = np.fft.fftfreq(N) * N
KX, KY = np.meshgrid(k_filt, k_filt, indexing="ij")
r = np.sqrt(KX**2 + KY**2)
rmax = np.max(r)
filt = np.exp(-filt_alpha * (r / rmax) ** filt_mf)
t = 0.0
step = 0
E0 = None
rho_min0 = float(np.min(rho))
rho_max0 = float(np.max(rho))
while t < T:
    w_hat = np.fft.fftn(omega)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0
    psi_hat = w_hat / k2
    psi_hat[0, 0] = 0.0
        psi_hat = psi_hat * filt
    u_hat = +(1j * KY) * psi_hat
    v_hat = -(1j * KX) * psi_hat
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
        umax = float(np.max(np.abs(u)))
        vmax = float(np.max(np.abs(v)))
        dt_adv = cfl * min(dx / (umax + 1e-12), dy / (vmax + 1e-12))
        dt = min(dt_adv, cfl * min(dx, dy))
        if t + dt > T:
            dt = T - t
        dA = dx * dy
        KE = 0.5 * float(np.sum(u * u + v * v)) * dA
        PE = float(np.sum(rho * Y)) * dA
        E_total = KE + PE
        if E0 is None:
            E0 = E_total
        dE_rel = (E_total - E0) / (abs(E0) + 1e-14) * 100.0
        rho_max = float(np.max(rho))
        rho_min = float(np.min(rho))
    if step % plot_every == 0:
            rp = np.maximum(rho, 0.0)
            m = float(np.sum(rp))
            if m > 0:
                xcm = float(np.sum(rp * X) / m)
                ycm = float(np.sum(rp * Y) / m)
            else:
                xcm = float("nan")
                ycm = float("nan")
        print(f"plot step={step:06d} t={t:.6f} xcm={xcm:.4f} ycm={ycm:.4f} E={E_total:.6f} dE%={dE_rel:.4f} rho=[{rho_min:.4f},{rho_max:.4f}]")
        ds = 1
        rr = rho
        XX = X
        YY = Y
        rmin = float(np.min(rr))
        rmax = float(np.max(rr))
        if abs(rmax - rmin) < 1e-14:
            r_levels = np.linspace(rmin - 1.0, rmax + 1.0, levels + 1)
        else:
            r_levels = np.linspace(rmin, rmax, levels + 1)
        L_plot = 2.0 * np.pi
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), constrained_layout=True)
        cs = ax.contour(YY, XX, rr, levels=r_levels, colors="k", linewidths=0.8)
        ax.set_title(f"rho contours (step={step}, t={t:.4f})")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, L_plot)
        ax.set_ylim(0.0, L_plot)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        path = os.path.join("out", f"rho_{step:06d}.png")
        fig.savefig(path)
        plt.close(fig)
    def rhs(rho_, omega_):
        w_hat = np.fft.fftn(omega_)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0
        psi_hat = w_hat / k2
        psi_hat[0, 0] = 0.0
        psi_hat = psi_hat * filt
        u_hat = +(1j * KY) * psi_hat
        v_hat = -(1j * KX) * psi_hat
        u_ = np.fft.ifftn(u_hat).real
        v_ = np.fft.ifftn(v_hat).real
        N_div = rho_.shape[0]
        out_div_rho = np.zeros_like(rho_)
        for j in range(N_div):
            q_div = rho_[:, j]
            a_half_div = 0.5 * (u_[:, j] + np.roll(u_[:, j], -1))
            qm2_div = np.roll(q_div, 2)
            qm1_div = np.roll(q_div, 1)
            q0_div = q_div
            qp1_div = np.roll(q_div, -1)
            qp2_div = np.roll(q_div, -2)
            qp3_div = np.roll(q_div, -3)
            sL0_div = np.abs(qm2_div - 2.0 * qm1_div + q0_div)
            sL1_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            sL2_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            mL01_div = sL0_div <= sL1_div
            mL_div = np.where(mL01_div, 0, 1)
            mL_div = np.where((sL2_div < np.minimum(sL0_div, sL1_div)), 2, mL_div)
            qL0_div = 0.375 * qm2_div - 1.25 * qm1_div + 1.875 * q0_div
            qL1_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qL2_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qL_div = np.where(mL_div == 0, qL0_div, np.where(mL_div == 1, qL1_div, qL2_div))
            sR0_div = np.abs(qp1_div - 2.0 * qp2_div + qp3_div)
            sR1_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            sR2_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            mR01_div = sR0_div <= sR1_div
            mR_div = np.where(mR01_div, 0, 1)
            mR_div = np.where((sR2_div < np.minimum(sR0_div, sR1_div)), 2, mR_div)
            qR0_div = 1.875 * qp1_div - 1.25 * qp2_div + 0.375 * qp3_div
            qR1_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qR2_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qR_div = np.where(mR_div == 0, qR0_div, np.where(mR_div == 1, qR1_div, qR2_div))
            fhat_div = np.where(a_half_div >= 0.0, a_half_div * qL_div, a_half_div * qR_div)
            out_div_rho[:, j] += (fhat_div - np.roll(fhat_div, 1)) / dx
        for i in range(N_div):
            q_div = rho_[i, :]
            a_half_div = 0.5 * (v_[i, :] + np.roll(v_[i, :], -1))
            qm2_div = np.roll(q_div, 2)
            qm1_div = np.roll(q_div, 1)
            q0_div = q_div
            qp1_div = np.roll(q_div, -1)
            qp2_div = np.roll(q_div, -2)
            qp3_div = np.roll(q_div, -3)
            sL0_div = np.abs(qm2_div - 2.0 * qm1_div + q0_div)
            sL1_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            sL2_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            mL01_div = sL0_div <= sL1_div
            mL_div = np.where(mL01_div, 0, 1)
            mL_div = np.where((sL2_div < np.minimum(sL0_div, sL1_div)), 2, mL_div)
            qL0_div = 0.375 * qm2_div - 1.25 * qm1_div + 1.875 * q0_div
            qL1_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qL2_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qL_div = np.where(mL_div == 0, qL0_div, np.where(mL_div == 1, qL1_div, qL2_div))
            sR0_div = np.abs(qp1_div - 2.0 * qp2_div + qp3_div)
            sR1_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            sR2_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            mR01_div = sR0_div <= sR1_div
            mR_div = np.where(mR01_div, 0, 1)
            mR_div = np.where((sR2_div < np.minimum(sR0_div, sR1_div)), 2, mR_div)
            qR0_div = 1.875 * qp1_div - 1.25 * qp2_div + 0.375 * qp3_div
            qR1_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qR2_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qR_div = np.where(mR_div == 0, qR0_div, np.where(mR_div == 1, qR1_div, qR2_div))
            fhat_div = np.where(a_half_div >= 0.0, a_half_div * qL_div, a_half_div * qR_div)
            out_div_rho[i, :] += (fhat_div - np.roll(fhat_div, 1)) / dy
        drho = -out_div_rho
        N_dy = rho_.shape[0]
        out_dy = np.zeros_like(rho_)
        a_dy = np.ones(N_dy)
        for i in range(N_dy):
            q_dy = rho_[i, :]
            a_half_dy = 0.5 * (a_dy + np.roll(a_dy, -1))
            qm2_dy = np.roll(q_dy, 2)
            qm1_dy = np.roll(q_dy, 1)
            q0_dy = q_dy
            qp1_dy = np.roll(q_dy, -1)
            qp2_dy = np.roll(q_dy, -2)
            qp3_dy = np.roll(q_dy, -3)
            sL0_dy = np.abs(qm2_dy - 2.0 * qm1_dy + q0_dy)
            sL1_dy = np.abs(qm1_dy - 2.0 * q0_dy + qp1_dy)
            sL2_dy = np.abs(q0_dy - 2.0 * qp1_dy + qp2_dy)
            mL01_dy = sL0_dy <= sL1_dy
            mL_dy = np.where(mL01_dy, 0, 1)
            mL_dy = np.where((sL2_dy < np.minimum(sL0_dy, sL1_dy)), 2, mL_dy)
            qL0_dy = 0.375 * qm2_dy - 1.25 * qm1_dy + 1.875 * q0_dy
            qL1_dy = -0.125 * qm1_dy + 0.75 * q0_dy + 0.375 * qp1_dy
            qL2_dy = 0.375 * q0_dy + 0.75 * qp1_dy - 0.125 * qp2_dy
            qL_dy = np.where(mL_dy == 0, qL0_dy, np.where(mL_dy == 1, qL1_dy, qL2_dy))
            sR0_dy = np.abs(qp1_dy - 2.0 * qp2_dy + qp3_dy)
            sR1_dy = np.abs(q0_dy - 2.0 * qp1_dy + qp2_dy)
            sR2_dy = np.abs(qm1_dy - 2.0 * q0_dy + qp1_dy)
            mR01_dy = sR0_dy <= sR1_dy
            mR_dy = np.where(mR01_dy, 0, 1)
            mR_dy = np.where((sR2_dy < np.minimum(sR0_dy, sR1_dy)), 2, mR_dy)
            qR0_dy = 1.875 * qp1_dy - 1.25 * qp2_dy + 0.375 * qp3_dy
            qR1_dy = 0.375 * q0_dy + 0.75 * qp1_dy - 0.125 * qp2_dy
            qR2_dy = -0.125 * qm1_dy + 0.75 * q0_dy + 0.375 * qp1_dy
            qR_dy = np.where(mR_dy == 0, qR0_dy, np.where(mR_dy == 1, qR1_dy, qR2_dy))
            fhat_dy = np.where(a_half_dy >= 0.0, a_half_dy * qL_dy, a_half_dy * qR_dy)
            out_dy[i, :] = (fhat_dy - np.roll(fhat_dy, 1)) / dy
        out_div_omega = np.zeros_like(omega_)
        for j in range(N_div):
            q_div = omega_[:, j]
            a_half_div = 0.5 * (u_[:, j] + np.roll(u_[:, j], -1))
            qm2_div = np.roll(q_div, 2)
            qm1_div = np.roll(q_div, 1)
            q0_div = q_div
            qp1_div = np.roll(q_div, -1)
            qp2_div = np.roll(q_div, -2)
            qp3_div = np.roll(q_div, -3)
            sL0_div = np.abs(qm2_div - 2.0 * qm1_div + q0_div)
            sL1_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            sL2_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            mL01_div = sL0_div <= sL1_div
            mL_div = np.where(mL01_div, 0, 1)
            mL_div = np.where((sL2_div < np.minimum(sL0_div, sL1_div)), 2, mL_div)
            qL0_div = 0.375 * qm2_div - 1.25 * qm1_div + 1.875 * q0_div
            qL1_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qL2_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qL_div = np.where(mL_div == 0, qL0_div, np.where(mL_div == 1, qL1_div, qL2_div))
            sR0_div = np.abs(qp1_div - 2.0 * qp2_div + qp3_div)
            sR1_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            sR2_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            mR01_div = sR0_div <= sR1_div
            mR_div = np.where(mR01_div, 0, 1)
            mR_div = np.where((sR2_div < np.minimum(sR0_div, sR1_div)), 2, mR_div)
            qR0_div = 1.875 * qp1_div - 1.25 * qp2_div + 0.375 * qp3_div
            qR1_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qR2_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qR_div = np.where(mR_div == 0, qR0_div, np.where(mR_div == 1, qR1_div, qR2_div))
            fhat_div = np.where(a_half_div >= 0.0, a_half_div * qL_div, a_half_div * qR_div)
            out_div_omega[:, j] += (fhat_div - np.roll(fhat_div, 1)) / dx
        for i in range(N_div):
            q_div = omega_[i, :]
            a_half_div = 0.5 * (v_[i, :] + np.roll(v_[i, :], -1))
            qm2_div = np.roll(q_div, 2)
            qm1_div = np.roll(q_div, 1)
            q0_div = q_div
            qp1_div = np.roll(q_div, -1)
            qp2_div = np.roll(q_div, -2)
            qp3_div = np.roll(q_div, -3)
            sL0_div = np.abs(qm2_div - 2.0 * qm1_div + q0_div)
            sL1_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            sL2_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            mL01_div = sL0_div <= sL1_div
            mL_div = np.where(mL01_div, 0, 1)
            mL_div = np.where((sL2_div < np.minimum(sL0_div, sL1_div)), 2, mL_div)
            qL0_div = 0.375 * qm2_div - 1.25 * qm1_div + 1.875 * q0_div
            qL1_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qL2_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qL_div = np.where(mL_div == 0, qL0_div, np.where(mL_div == 1, qL1_div, qL2_div))
            sR0_div = np.abs(qp1_div - 2.0 * qp2_div + qp3_div)
            sR1_div = np.abs(q0_div - 2.0 * qp1_div + qp2_div)
            sR2_div = np.abs(qm1_div - 2.0 * q0_div + qp1_div)
            mR01_div = sR0_div <= sR1_div
            mR_div = np.where(mR01_div, 0, 1)
            mR_div = np.where((sR2_div < np.minimum(sR0_div, sR1_div)), 2, mR_div)
            qR0_div = 1.875 * qp1_div - 1.25 * qp2_div + 0.375 * qp3_div
            qR1_div = 0.375 * q0_div + 0.75 * qp1_div - 0.125 * qp2_div
            qR2_div = -0.125 * qm1_div + 0.75 * q0_div + 0.375 * qp1_div
            qR_div = np.where(mR_div == 0, qR0_div, np.where(mR_div == 1, qR1_div, qR2_div))
            fhat_div = np.where(a_half_div >= 0.0, a_half_div * qL_div, a_half_div * qR_div)
            out_div_omega[i, :] += (fhat_div - np.roll(fhat_div, 1)) / dy
        domega = -out_div_omega + out_dy
        return drho, domega
    k1_r, k1_w = rhs(rho, omega)
    r1 = rho + dt * k1_r
    w1 = omega + dt * k1_w
    k2_r, k2_w = rhs(r1, w1)
    r2 = 0.75 * rho + 0.25 * (r1 + dt * k2_r)
    w2 = 0.75 * omega + 0.25 * (w1 + dt * k2_w)
    k3_r, k3_w = rhs(r2, w2)
    rho_new = (1.0 / 3.0) * rho + (2.0 / 3.0) * (r2 + dt * k3_r)
    omega_new = (1.0 / 3.0) * omega + (2.0 / 3.0) * (w2 + dt * k3_w)
        if step % max(1, plot_every) == 0:
        print(f"step={step:06d} t={t:.6f} dt={dt:.3e} max|u|={umax:.3e} max|v|={vmax:.3e} max|rho|={np.max(np.abs(rho)):.3e} max|omega|={np.max(np.abs(omega)):.3e}")
rr = rho
XX = X
YY = Y
rmin = float(np.min(rr))
rmax = float(np.max(rr))
if abs(rmax - rmin) < 1e-14:
    r_levels = np.linspace(rmin - 1.0, rmax + 1.0, levels + 1)
else:
    r_levels = np.linspace(rmin, rmax, levels + 1)
L_plot = 2.0 * np.pi
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), constrained_layout=True)
cs = ax.contour(YY, XX, rr, levels=r_levels, colors="k", linewidths=0.8)
ax.set_title(f"rho contours (step={step}, t={t:.4f})")
ax.set_aspect("equal")
ax.set_xlim(0.0, L_plot)
ax.set_ylim(0.0, L_plot)
ax.set_xlabel("y")
ax.set_ylabel("x")
path = os.path.join("out", f"rho_{step:06d}.png")
fig.savefig(path)
plt.close(fig)
