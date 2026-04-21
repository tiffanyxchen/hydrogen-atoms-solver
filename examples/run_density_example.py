#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from src.hydrogen_solver import solve_schrodinger, compute_schrodinger
from src.density import build_density, total_electrons, radial_distribution
from src.shooting import find_bound_states, cmp_key

# radial grid
R = np.linspace(1e-6, 100, 2000)

# energy search grid
Esearch = -1.2 / np.arange(1, 20, 0.2)**2
nmax    = 7
# find bound states (l = 0..3)
Bnd = []

for l in range(nmax - 1):
    Bnd += find_bound_states(R, l, nmax-l, Esearch)

# sort states
Bnd = sorted(Bnd, key=cmp_key)

# choose atomic number
Z = 46  # Nickel-like

# build density
rho, info = build_density(R, Bnd, compute_schrodinger, Z)

print("Total electrons:", total_electrons(R, rho))
print("States used:")


for item in info:
    print(item)

# plot radial distribution
plt.figure()
plt.plot(R, radial_distribution(R, rho))

plt.xscale("linear")
plt.xlim(0, 50)  # for log scale, avoid 0
plt.ylim(0, 2.5)
plt.xlabel("r")
plt.ylabel("4π r² ρ(r)")
plt.title(f"Radial Distribution (Z={Z})")
plt.show()
