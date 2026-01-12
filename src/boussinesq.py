#!/usr/bin/env python

import numpy as np

def main():
    n1, n2 = 64, 64
    L1, L2 = 60, 60 # same as in paper: [-30, 30]^2

    z1 = np.linspace(-L1/2, L1/2, n1)
    z2 = np.linspace(-L2/2, L2/2, n2)

    y1 = np.sinh(z1)

    print(y1)


if __name__ == '__main__':
    main()
