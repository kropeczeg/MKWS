"""
mechanism_comparison.py
=======================
Supporting module — mechanism validation.

The three production modules all use GRI-Mech 3.0 so that the whole
H2/CH4 blend range is treated with a single, consistent mechanism that
also carries the nitrogen chemistry needed for NOx. GRI-Mech, however, is
optimised primarily for natural gas. To check that it is trustworthy at
the pure-hydrogen end of the blend range, this module repeats two of the
calculations for 100 % H2 with a dedicated, detailed H2/O2 mechanism
(``h2o2.yaml``, 10 species / 29 reactions) and overlays the results.

Two quantities are compared:

  * ignition delay vs temperature (constant-volume reactor),
  * laminar burning velocity vs equivalence ratio (1-D free flame).

A close agreement gives confidence that the GRI-Mech results reported in
the main study are not an artefact of the chosen mechanism. Note that the
``h2o2`` mechanism treats N2 as inert and therefore cannot predict NOx;
the NOx study necessarily stays with GRI-Mech.

Outputs
-------
  results/data/mech_comparison_ignition.csv
  results/data/mech_comparison_flame.csv
  results/figures/mech_comparison_ignition.png
  results/figures/mech_comparison_flame.png
"""

from __future__ import annotations

import csv
import os
import numpy as np
import cantera as ct

import common as cm

#: Mechanisms compared. GRI-Mech is the project default; h2o2 is the
#: detailed reference for hydrogen.
MECHS = {
    "GRI-Mech 3.0": "gri30.yaml",
    "h2o2 (detailed H2/O2)": "h2o2.yaml",
}


# --------------------------------------------------------------------------
# Helpers that take an explicit mechanism (so we can swap it)
# --------------------------------------------------------------------------

def _set_h2_air(gas: ct.Solution, phi: float, T: float, p: float) -> None:
    """Set a pure-hydrogen / air mixture on the given gas object."""
    gas.set_equivalence_ratio(phi, "H2:1.0", "O2:1.0, N2:3.76")
    gas.TP = T, p


def ignition_delay_mech(mech: str, phi: float, T0: float, p0: float,
                        t_end: float = 1.0, max_steps: int = 100_000) -> float:
    """Constant-volume ignition delay for pure H2 with a chosen mechanism."""
    gas = ct.Solution(mech)
    _set_h2_air(gas, phi, T0, p0)
    reactor = ct.IdealGasReactor(gas, clone=False)
    net = ct.ReactorNet([reactor])
    net.rtol, net.atol = 1.0e-9, 1.0e-15

    times, temps = [], []
    t, steps = 0.0, 0
    while t < t_end and steps < max_steps:
        t = net.step()
        times.append(t)
        temps.append(reactor.T)
        steps += 1
        if reactor.T > T0 + 600.0 and len(temps) > 5:
            if temps[-1] - temps[-2] < temps[-2] - temps[-3]:
                break

    times, temps = np.asarray(times), np.asarray(temps)
    if temps.size < 3 or (temps.max() - T0) < 200.0:
        return float("nan")
    return float(times[int(np.argmax(np.gradient(temps, times)))])


def flame_speed_mech(mech: str, phi: float, T0: float, p0: float,
                     width: float = 0.03) -> float:
    """Laminar burning velocity for pure H2 with a chosen mechanism."""
    gas = ct.Solution(mech)
    _set_h2_air(gas, phi, T0, p0)
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3.0, slope=0.08, curve=0.16)
    flame.transport_model = "mixture-averaged"
    try:
        flame.solve(loglevel=0, auto=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! non-converged ({mech}, phi={phi}): {exc}")
        return float("nan")
    return float(flame.velocity[0])


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------

def compare_ignition(T_array, phi, p0):
    print("Mechanism comparison — ignition delay (100% H2) ...")
    results = {}
    for label, mech in MECHS.items():
        taus = []
        for T0 in T_array:
            tau = ignition_delay_mech(mech, phi, T0, p0)
            taus.append(tau)
            print(f"  {label:>24}  T0={T0:6.1f} K  tau={tau*1e3:9.4f} ms")
        results[label] = np.asarray(taus)
    return results


def compare_flame(phi_array, T0, p0):
    print("Mechanism comparison — flame speed (100% H2) ...")
    results = {}
    for label, mech in MECHS.items():
        speeds = []
        prev = None  # simple continuation within a mechanism
        gas = None
        flame = None
        for phi in phi_array:
            g = ct.Solution(mech)
            _set_h2_air(g, phi, T0, p0)
            fl = ct.FreeFlame(g, width=0.03)
            fl.set_refine_criteria(ratio=3.0, slope=0.08, curve=0.16)
            fl.transport_model = "mixture-averaged"
            if flame is not None:
                try:
                    fl.set_initial_guess(data=flame.to_array())
                except Exception:
                    pass
            try:
                fl.solve(loglevel=0, auto=True)
                s = float(fl.velocity[0])
                flame = fl
            except Exception:
                s = float("nan")
            speeds.append(s)
            print(f"  {label:>24}  phi={phi:4.2f}  S_L={s*100:7.2f} cm/s")
        results[label] = np.asarray(speeds)
    return results


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_ignition(T_array, results, phi, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    markers = {"GRI-Mech 3.0": "o", "h2o2 (detailed H2/O2)": "x"}
    inv_T = 1000.0 / T_array
    for label, taus in results.items():
        ax.semilogy(inv_T, taus * 1e3, marker=markers.get(label, "o"),
                    label=label, linestyle="-")
    ax.set_xlabel(r"$1000\,/\,T_0$  [1/K]")
    ax.set_ylabel(r"Ignition delay  $\tau_{ign}$  [ms]")
    ax.set_title(f"Mechanism comparison — ignition delay, 100% H$_2$ "
                 f"($\\phi$={phi}, p={p0/1e5:.0f} bar)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_flame(phi_array, results, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    markers = {"GRI-Mech 3.0": "o", "h2o2 (detailed H2/O2)": "x"}
    for label, speeds in results.items():
        ax.plot(phi_array, speeds * 100, marker=markers.get(label, "o"),
                label=label)
    ax.set_xlabel(r"Equivalence ratio  $\phi$")
    ax.set_ylabel(r"Laminar burning velocity  $S_L$  [cm/s]")
    ax.set_title(f"Mechanism comparison — flame speed, 100% H$_2$ "
                 f"($T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def save_ignition_csv(T_array, results, phi, p0,
                      fname="mech_comparison_ignition.csv"):
    path = os.path.join(cm.DATA_DIR, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "phi", "T0_K", "p0_bar", "tau_ign_ms"])
        for label, taus in results.items():
            for T0, tau in zip(T_array, taus):
                w.writerow([label, phi, T0, p0 / 1e5, tau * 1e3])
    print(f"  saved {path}")


def save_flame_csv(phi_array, results, T0, p0,
                   fname="mech_comparison_flame.csv"):
    path = os.path.join(cm.DATA_DIR, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "phi", "T0_K", "p0_bar", "S_L_cm_s"])
        for label, speeds in results.items():
            for phi, s in zip(phi_array, speeds):
                w.writerow([label, phi, T0, p0 / 1e5, s * 100])
    print(f"  saved {path}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg in ("ignition", "all"):
        phi_i, p_i = 1.0, 20.0e5
        T_array = np.linspace(900.0, 1400.0, 9)
        ign = compare_ignition(T_array, phi_i, p_i)
        plot_ignition(T_array, ign, phi_i, p_i,
                      "mech_comparison_ignition.png")
        save_ignition_csv(T_array, ign, phi_i, p_i)

    if arg in ("flame", "all"):
        T_f, p_f = 450.0, 5.0e5
        phi_array = np.linspace(0.6, 1.4, 7)
        fla = compare_flame(phi_array, T_f, p_f)
        plot_flame(phi_array, fla, T_f, p_f, "mech_comparison_flame.png")
        save_flame_csv(phi_array, fla, T_f, p_f)

    print("Mechanism comparison complete.")


if __name__ == "__main__":
    main()
