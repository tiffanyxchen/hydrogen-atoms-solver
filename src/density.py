#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
density.py

Construct electron charge density from hydrogenic orbitals
using independent-electron (non-interacting) approximation.

Uses orbitals from hydrogen_solver and energies from shooting.
"""

import numpy as np
from scipy import integrate

# Expect these to be available in your project
# from src.hydrogen_solver import solve_schrodinger


def orbital_occupancy(l):
    """Maximum electrons per (n,l) orbital."""
    return 2 * (2*l + 1)


def build_density(R, bound_states, solve_fn, Z):
    rho = np.zeros(len(R))
    N   = 0
    info = []

    for (l, En) in bound_states:
        ur = solve_fn(En, l, R)

        dN = orbital_occupancy(l)

        if N + dN <= Z:
            ferm = 1.0
        else:
            ferm = (Z - N) / float(dN)

        electrons_added = ferm * dN

        drho = ur**2 * electrons_added / (4*np.pi*R**2)
        rho += drho

        N += dN

        info.append((l, En, ferm, electrons_added))

        if N >= Z:
            break

    return rho, info


def total_electrons(R, rho):
    """Check normalization: should return ~Z"""
    return integrate.simpson(rho * 4*np.pi*R**2, x=R)


def radial_distribution(R, rho):
    """Return 4πr²ρ(r) for plotting."""
    return 4*np.pi * R**2 * rho
