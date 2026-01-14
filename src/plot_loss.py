#!/usr/bin/env python3
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    it, L, L_pde, L_g, L_wall, L_far, lr = [], [], [], [], [], [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            it.append(int(row["it"]))
            L.append(float(row["L"]))
            L_pde.append(float(row["L_pde"]))
            L_g.append(float(row["L_g"]))
            L_wall.append(float(row["L_wall"]))
            L_far.append(float(row["L_far"]))
            lr.append(float(row["lr"]))
    return (
        np.asarray(it),
        np.asarray(L),
        np.asarray(L_pde),
        np.asarray(L_g),
        np.asarray(L_wall),
        np.asarray(L_far),
        np.asarray(lr),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="loss.csv produced by the solver")
    ap.add_argument("--out", default=None, help="output image (png/pdf). default: show")
    ap.add_argument("--logy", action="store_true", help="log-scale y axis")
    ap.add_argument("--every", type=int, default=1, help="plot every k-th sample")
    ap.add_argument("--components", action="store_true", help="also plot loss components")
    args = ap.parse_args()

    it, L, L_pde, L_g, L_wall, L_far, lr = read_csv(args.csv)

    k = max(1, args.every)
    it = it[::k]; L = L[::k]; L_pde = L_pde[::k]; L_g = L_g[::k]
    L_wall = L_wall[::k]; L_far = L_far[::k]; lr = lr[::k]

    plt.figure(figsize=(8, 4.5))
    plt.plot(it, L, label="L")
    if args.components:
        plt.plot(it, L_pde, label="L_pde")
        #plt.plot(it, L_g, label="L_g")
        #plt.plot(it, L_wall, label="L_wall")
        #plt.plot(it, L_far, label="L_far")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    if args.logy:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()

    # optional: show LR changes on a second axis (only if it varies)
    if np.nanmax(lr) / max(np.nanmin(lr), 1e-300) > 1.01:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(it, lr, linestyle="--", linewidth=1, label="lr")
        ax2.set_ylabel("learning rate")
        ax2.set_yscale("log")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()

    if args.out:
        out = os.path.expanduser(args.out)
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
