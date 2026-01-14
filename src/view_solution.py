#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def surf(ax, X, Y, Z, title):
    ax.plot_surface(X, Y, Z, cmap="jet", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.view_init(elev=18, azim=-60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", help="solution.npz")
    p.add_argument("--y1", nargs=2, type=float, default=[-20, 20])
    p.add_argument("--y2", nargs=2, type=float, default=[0, 20])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    d = np.load(args.file)
    y1 = d["y1_c"]
    y2 = d["y2_c"]

    I = np.where((y1 >= args.y1[0]) & (y1 <= args.y1[1]))[0]
    J = np.where((y2 >= args.y2[0]) & (y2 <= args.y2[1]))[0]

    Y1, Y2 = np.meshgrid(y1[I], y2[J], indexing="ij")

    Om  = d["Omega"][np.ix_(I, J)]
    Ph  = d["Phi"][np.ix_(I, J)]
    Ps  = d["Psi"][np.ix_(I, J)]
    U1  = d["U1"][np.ix_(I, J)]
    U2  = d["U2"][np.ix_(I, J)]
    lam = d["lam"]

    fig = plt.figure(figsize=(14, 7))

    surf(fig.add_subplot(2, 3, 1, projection="3d"), Y1, Y2, Om,  r"$\Omega$")
    surf(fig.add_subplot(2, 3, 2, projection="3d"), Y1, Y2, Ph,  r"$\Phi$")
    surf(fig.add_subplot(2, 3, 3, projection="3d"), Y1, Y2, Ps,  r"$\Psi$")
    surf(fig.add_subplot(2, 3, 4, projection="3d"), Y1, Y2, U1,  r"$U_1$")
    surf(fig.add_subplot(2, 3, 5, projection="3d"), Y1, Y2, U2,  r"$U_2$")

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ax.text(0.05, 0.6, rf"$\lambda = {lam:.6f}$", fontsize=16)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
