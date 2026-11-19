import numpy as np

def is_stable(system, trajectory):
    G = 4 * np.pi**2
    mu = G * (system.binary.m_A + system.binary.m_B + system.planet.m_p)
    a_C = system.companion.orbit.a

    rp = trajectory.pos[:, 2, :]
    vp = trajectory.v[:, 2, :]

    energy = 0.5 * np.sum(vp**2, axis=1) - mu / np.linalg.norm(rp, axis=1)

    if not np.all(np.isfinite(energy)) or np.any(energy >= 0):
        return 0 
    
    if np.any(-mu / (2 * energy) > a_C):
        return 0   

    return 1

def e_p(system, trajectory):
    e_p = trajectory.e[:, 2, 0]
    return e_p

def a_p(system, trajectory):
    a_p = trajectory.a[:, 2, 0]
    return a_p

def Delta_a(system, trajectory):
    a_p = trajectory.a[:, 2, 0]
    Delta_a = np.max(a_p) - np.min(a_p)
    return Delta_a

def Delta_e(system, trajectory):
    e_p = trajectory.e[:, 2, 0]
    Delta_e = np.max(e_p) - np.min(e_p)
    return Delta_e

