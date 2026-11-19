import rebound
import numpy as np
from circumbinary_triples.config import Trajectory


def build_system(system):
    sim = rebound.Simulation()
    sim.units = ("AU", "msun", "year")

    sim.add(m=system.binary.m_A)
    sim.add(
        m=system.binary.m_B,
        a=system.binary.orbit.a,
        e=system.binary.orbit.e,
        inc=system.binary.orbit.i,
        omega=system.binary.phases.omega,
        Omega=system.binary.phases.Omega,
        M=system.binary.phases.M,
    )
    sim.add(
        m=system.planet.m_p,
        a=system.planet.orbit.a,
        e=system.planet.orbit.e,
        inc=system.planet.orbit.i,
        omega=system.planet.phases.omega,
        Omega=system.planet.phases.Omega,
        M=system.planet.phases.M,
    )
    sim.add(
        m=system.companion.m_C,
        a=system.companion.orbit.a,
        e=system.companion.orbit.e,
        inc=system.companion.orbit.i,
        omega=system.companion.phases.omega,
        Omega=system.companion.phases.Omega,
        M=system.companion.phases.M,
    )

    sim.move_to_com()
    return sim


def sim_parameters(sim, integrator="whfast"):
    B = sim.particles[1]
    sim.integrator = integrator
    sim.dt = B.P/20
    return sim


def trajectory(sim, n_orbits=10, res_orbits=100):
    p = sim.particles[2]

    times = np.linspace(sim.t, sim.t + n_orbits*p.P, res_orbits)
    pos = np.zeros((len(times), sim.N, 3))
    v = np.zeros((len(times), sim.N, 3))
    e = np.zeros((len(times), sim.N, 1))
    a = np.zeros((len(times), sim.N, 1))

    t_real = np.zeros(len(times))

    for k, t in enumerate(times):
        sim.integrate(t, exact_finish_time=0)
        t_real[k] = sim.t
        com = sim.com(0, 2)

        pos[k] = np.array([[i.x, i.y, i.z] for i in sim.particles]) - np.array([com.x, com.y, com.z])
        v[k] = np.array([[i.vx, i.vy, i.vz] for i in sim.particles]) - np.array([com.vx, com.vy, com.vz])
        for j in range(1, sim.N):
            e[k, j, 0] = sim.particles[j].e
            a[k, j, 0] = sim.particles[j].a

    return Trajectory(t=t_real, pos=pos, v=v, e=e, a=a)


