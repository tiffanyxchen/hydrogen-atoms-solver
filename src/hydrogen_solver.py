#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hydrogen_solver.py

Core numerical solvers for the radial Schrödinger equation
of the hydrogen atom.

Includes:
- ODE solver (SciPy odeint)
- Numerov method (high-accuracy solver)
"""

import numpy as np
from scipy import integrate
from numba import jit


@jit(nopython=True)
def schrodinger_deriv(y, r, l, En):
    u, up = y
    dudr = up
    d2udr2 = (l * (l + 1) / r**2 - 2.0 / r - En) * u
    return np.array([dudr, d2udr2])


def solve_schrodinger(En, l, R):
    ur = integrate.odeint(
        schrodinger_deriv,
        [0.0, -1e-7],
        R[::-1],
        args=(l, En)
    )[:, 0][::-1]

    norm = integrate.simpson(ur**2, x=R)
    ur /= np.sqrt(norm)

    return ur


@jit(nopython=True)
def numerovc(f, x0, dx, dh):
    """ Given precomputed function f(x), solves for x(t), 
        which satisties:
            x''(t) = f(t) x(t)
            dx     = (dx(t) / dt)_{t = 0}
            x0     = x(t = 0)
    """
    x = np.zeros(len(f))
    
    x[0] = x0
    x[1] = x0 + dh * dx
    
    h2   = dh**2
    h12  = h2/12
    w0   = x0   * (1 - h12 * f[0])
    w1   = x[1] * (1 - h12 * f[1])
    
    xi = x[1]
    fi = f[1]
    
    for i in range (2, f.size):
        w2 = 2 * w1 - w0 + h2 * fi * xi
        fi = f[i]
        xi = w2 / (1 - h12 * fi)
        x[i] = xi
        w0, w1 = w1, w2
        
    return x


def f_schrodinger(En, l, R):
    return l * (l + 1.) / R**2 - 2.0 / R - En

# uses numerov to compute ur
def compute_schrodinger(En, l, R):
    "Computes Schrod Eq."
    f  = f_schrodinger(En, l, R[::-1])
    ur = numerovc(f, 0.0, -1e-7, -R[1]+R[0])[::-1]
    norm = integrate.simpson(ur**2, x=R)
    return ur * 1 / np.sqrt(abs(norm))
