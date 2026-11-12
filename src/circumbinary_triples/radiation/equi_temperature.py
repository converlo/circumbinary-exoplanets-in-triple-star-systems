from circumbinary_triples.constants import (L_SUN, SIGMA, AU)

import numpy as np

def temperature(system, position):
    L = system.luminosities()  # (L_A, L_B, L_C) in solar units

    stars = position.pos[:, [0, 1, 3], :]
    planet = position.pos[:, 2:3, :]
    distances = np.linalg.norm(stars - planet, axis=-1)

    flux = np.sum(L / distances**2, axis=1) * L_SUN / AU**2

    T_eq = system.planet.greenhouse_factor * (
        (1 - system.planet.albedo) * flux / (16 * np.pi * SIGMA)
    )**(1/4)

    return T_eq
    
def mean_temperature(system, position):
    return np.nanmean(temperature(system, position))

def std_temperature(system, position):
    return np.nanstd(temperature(system, position))