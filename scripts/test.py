import warnings
import numpy as np
import circumbinary_triples as ct
import matplotlib.pyplot as plt


system = ct.SystemConfig(
    binary=ct.BinaryConfig(
        m_A=1, m_B=1.51, R_A=2.0, R_B=1.0,
        orbit=ct.OrbitConfig(a=0.384, e=0.11, i=0.0),
    ),
    planet=ct.PlanetConfig(
        m_p=0.001, R_p=0.1,
        albedo=0.30, greenhouse_factor=1.13,
        orbit=ct.OrbitConfig(a=1.5, e=0.0, i=0.0),
    ),
    companion=ct.CompanionConfig(
        m_C=2.26, R_C=1.0,
        orbit=ct.OrbitConfig(a=26.17, e=0.169, i=0.0),
    ),
)
 
# --- Réglages du balayage ---------------------------------------------------
A_MIN, A_MAX, N_POINTS = 2*system.binary.orbit.a, system.companion.orbit.a, 30     # a_p (AU)
N_ORBITS = 500                              # périodes planétaires intégrées
RES_ORBITS = 10000                           # instantanés par intégration
N_REAL = 5                                 # réalisations de phases par a_p
# -----------------------------------------------------------------------------


def simulate(system, rng):
    system = ct.random_phases(system, rng)
    sim = ct.sim_parameters(ct.build_system(system), integrator = "whfast")
    return ct.trajectory(sim, n_orbits=N_ORBITS, res_orbits=RES_ORBITS)

observables = {
    "t": lambda system, traj: traj.t,
    "T": lambda system, traj: ct.temperature(system, traj),
    "T_mean": lambda system, traj: ct.mean_temperature(system, traj),
    "T_std": lambda system, traj: ct.std_temperature(system, traj),
    "Stability": lambda system, traj: ct.is_stable(system, traj),
    "e_p": lambda system, traj: ct.e_p(system, traj),
    "a_p": lambda system, traj: ct.a_p(system, traj),
    "Delta_a": lambda system, traj: ct.Delta_a(system, traj),
    "Delta_e": lambda system, traj: ct.Delta_e(system, traj),
}

axis = ct.SweepAxis.lin("planet.orbit.a", A_MIN, A_MAX, N_POINTS)

results = ct.sweep(system, axis, simulate=simulate, observables=observables,
                   n_realizations=N_REAL, seed=12345)


grid = np.array(axis.values)                 # (n_points)
stability = results.array("Stability")           # (n_points, n_real), valeur 0 ou 1
f_stable = stability.mean(axis=1)                # (n_points)fraction stable par a_p
stable = f_stable > 0.95                # (n_points_stable)


# delta_a, delta_e : (n_points, n_real)
Delta_a = results.array("Delta_a")
Delta_a_stable = np.where(stability == 1, Delta_a, np.nan)
Delta_a_med = np.nanmedian(Delta_a_stable, axis=1)
Delta_a_std = np.nanstd(Delta_a_stable, axis=1, ddof=1)

Delta_e = results.array("Delta_e")
Delta_e_stable = np.where(stability == 1, Delta_e, np.nan)
Delta_e_med = np.nanmedian(Delta_e_stable, axis=1)
Delta_e_std = np.nanstd(Delta_e_stable, axis=1, ddof=1)

# a_p, e_p, T, t : (n_points, n_real, n_times)
a_p = results.array("a_p")[:, 0, :]
e_p = results.array("e_p")[:, 0, :] 
T = results.array("T")[:, 0, :]
t = results.array("t")[:, 0, :]


colors = plt.cm.tab20(np.linspace(0, 1, len(grid)))

# --- 1. Fraction stable -------------------------------------------------------
#a_min, a_max = system.holman_wiegert()

fig_s, ax_s = plt.subplots(figsize=(8, 4))
ax_s.bar(grid, f_stable, width=2, color="steelblue")

#ax_s.axvline(a_min, color="red", linestyle="--", linewidth=1.5, label=rf"$a_{{\mathrm{{HW,min}}}} = {a_min:.2f}$ AU")
#ax_s.axvline(a_max, color="darkorange", linestyle="--", linewidth=1.5, label=rf"$a_{{\mathrm{{HW,max}}}} = {a_max:.2f}$ AU")
#ax_s.legend()

ax_s.set_xlabel("a_p (AU)")
ax_s.set_ylabel("Stable fraction")
ax_s.set_ylim(0, 1.1)

plt.show()


# --- 2. et 3. Eccentricity and semimajor axis variation -------------------------------------------------------
fig_Delta_a_p, ax_Delta_a_p = plt.subplots(figsize=(12, 8))
ax_Delta_a_p.plot(grid[stable], Delta_a_med[stable])
ax_Delta_a_p.errorbar(
    grid[stable], Delta_a_med[stable],
    yerr=Delta_a_std[stable],
    fmt="o-", capsize=3
)
ax_Delta_a_p.set_xlabel(r"$a_p$ initial (AU)")
ax_Delta_a_p.set_ylabel(r"Moyenne de $\Delta a_p$ (AU), réalisations stables")
ax_Delta_a_p.set_title(r"$\Delta a_p$ moyen pour les points avec $f_{\rm stable} > 0.95$")
ax_Delta_a_p.grid(True, alpha=0.3)

fig_Delta_e_p, ax_Delta_e_p = plt.subplots(figsize=(12, 8))
ax_Delta_e_p.plot(grid[stable], Delta_e_med[stable])
ax_Delta_e_p.errorbar(
    grid[stable], Delta_e_med[stable],
    yerr=Delta_e_std[stable],
    fmt="o-", capsize=3
)
ax_Delta_e_p.set_xlabel(r"$a_p$ initial (AU)")
ax_Delta_e_p.set_ylabel(r"Moyenne de $\Delta e_p$, réalisations stables")
ax_Delta_e_p.set_title(r"$\Delta e_p$ moyen pour les points avec $f_{\rm stable} > 0.95$")
ax_Delta_e_p.grid(True, alpha=0.3)

stable_idx = np.where(stable)[0]

fig_a, ax_a = plt.subplots(figsize=(12, 8))
for k in stable_idx:
    ax_a.plot(t[k], a_p[k], color=colors[k], lw=0.8, label=f"a_p = {grid[k]:.2f} AU")
    ax_a.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
ax_a.set_xlabel("Time (yr)")
ax_a.set_ylabel("a_p (AU)")

fig_e, ax_e = plt.subplots(figsize=(12, 8))
for k in stable_idx:
    ax_e.plot(t[k], e_p[k], color=colors[k], lw=0.8, label=f"e_p = {grid[k]:.2f}")
    ax_e.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
ax_e.set_xlabel("Time (yr)")
ax_e.set_ylabel("e_p")

plt.show()

'''
# --- 4. T(t) pour les orbites stables (réalisation 0) --------------------------

fig_T, ax_T = plt.subplots(figsize=(8, 4))

for k in stable_idx:
    ax_T.plot(t[k], T[k] - 273.15, color=colors[k], lw=0.8, label=f"a_p = {grid[k]:.2f} AU")
    ax_T.axhspan(0, 100, color="deepskyblue", alpha=0.2, label="HZ (0-100 °C)")
    ax_T.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

ax_T.set_xlabel("Time (yr)")
ax_T.set_ylabel("Temperature (°C)")
fig_T.tight_layout()
 
# --- 5. et 6. T_mean et T_std, uniquement sur les réalisations stables ---------
T_mean = np.where(stability == 1, results.array("T_mean"), np.nan)
T_std = np.where(stability == 1, results.array("T_std"), np.nan)
 
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # points sans aucune réalisation stable
    T_mean_a = np.nanmean(T_mean, axis=1) - 273.15
    T_std_a = np.nanmean(T_std, axis=1)
 
fig_meanT, ax_meanT = plt.subplots(figsize=(6, 4))
ax_meanT.plot(grid, T_mean_a, lw=1, color="blue", marker="+", markersize=5)
ax_meanT.axhspan(0, 100, color="deepskyblue", alpha=0.2)
ax_meanT.set_xlabel("a_p (AU)")
ax_meanT.set_ylabel("Mean Temperature (°C)")
 
fig_stdT, ax_stdT = plt.subplots(figsize=(6, 4))
ax_stdT.plot(grid, T_std_a, lw=1, color="red", marker="+", markersize=5)
ax_stdT.set_xlabel("a_p (AU)")
ax_stdT.set_ylabel("Temperature standard deviation (K)")

plt.show()
'''