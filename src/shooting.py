#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shooting.py

Eigenvalue finder using shooting method for the radial
Schrödinger equation.
"""

import numpy as np
from scipy import integrate, optimize
from src.hydrogen_solver import solve_schrodinger, compute_schrodinger


# =========================================================
# Basic Shooting Function (odeint version)
# =========================================================

def shoot(En, l, R):
    ur = solve_schrodinger(En, l, R)
    ur = ur / R**l

    f0, f1 = ur[0], ur[1]
    f_at_0 = f0 + (f1 - f0) * (0.0 - R[0]) / (R[1] - R[0])

    return f_at_0


# =========================================================
# Improved Shooting (cubic extrapolation, Numerov)
# =========================================================

def shoot2(En, l, R):
    ur = compute_schrodinger(En, l, R)
    ur = ur / R**l
    poly = np.polyfit(R[:4], ur[:4], deg=3)
    return np.polyval(poly, 0.0)


# =========================================================
# Bound State Finder
# =========================================================

def find_bound_states(R, l, nmax, Esearch):
    """
    Parameters
    ----------
    R : TYPE
        real space mesh.
    l : TYPE
        orbital quantum number.
    nmax : TYPE
        maximum number of bounds states we require.
    Esearch : TYPE
        energy mesh, which brackets all bound-states, i.e., 
        [every sign change of hte wave function at u(0).

    Returns
    -------
    None.
    """
    n    = 0
    Ebnd = []                               # save all bound states
    u0   = shoot2(Esearch[0], l, R)         # u(r=0) for the first energy Esearch[0]
    
    for i in range(1, len(Esearch)):
        u1 = shoot2(Esearch[i], l, R)       # evaluate u(r=0) and all Esearch points
        
        if u0 * u1 < 0:
            Ebound = optimize.brentq(shoot2, Esearch[i-1], Esearch[i], xtol = 1e-16, args=(l, R)) # root finding routine
            Ebnd.append((l, Ebound))
            if len(Ebnd) > nmax: break
            n += 1
            print(f"Found bound state at E={Ebound:14.9f} E_exact={-1.0/(n+l)**2:14.9f} l={l}")
        u0 = u1
        
    return Ebnd


# =========================================================
# Sorting helper
# =========================================================

def cmp_key(x):
    """
    Sort by energy, then slightly by angular momentum.
    """
    return x[1] + x[0] / 10000.0
