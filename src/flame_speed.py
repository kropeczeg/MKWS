"""
flame_speed.py
==============
Module 2 of 3.

Computes the laminar burning velocity S_L of premixed H2/CH4-air flames
and studies how it depends on:

  * the equivalence ratio,
  * the hydrogen energy fraction of the fuel.

Method
------
A freely-propagating, one-dimensional, adiabatic premixed flame is solved
with Cantera's ``FreeFlame`` model. The laminar burning velocity is the
unburned-gas inlet velocity of the converged solution. Mixture-averaged
transport is used as a good compromise between accuracy and run time;
the grid is refined adaptively through ``set_refine_criteria``.

The unburned mixture is set at the combustor inlet temperature so that
S_L is reported at conditions relevant to the primary zone of an aero
gas turbine.

Outputs
-------
  results/data/flame_speed.csv
  results/figures/flame_speed_phi.png
  results/figures/flame_speed_h2.png
"""

from __future__ import annotations

import csv
import os
import numpy as np
import cantera as ct

import common as cm


# --------------------------------------------------------------------------
# Core single-point calculation
# --------------------------------------------------------------------------

def laminar_flame_speed(phi: float,
                        x_h2_energy: float,
                        T0: float = 450.0,
                        p0: float = 5.0e5,
                        width: float = 0.03,
                        loglevel: int = 0) -> float:
    """Compute the laminar burning velocity of one premixed flame.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    x_h2_energy : float
        Hydrogen energy fraction in [0, 1].
    T0, p0 : float
        Unburned-gas temperature [K] and pressure [Pa].
    width : float
        Width of the computational domain [m].
    loglevel : int
        Cantera solver verbosity (0 = silent).

    Returns
    -------
    float
        Laminar burning velocity S_L [m/s], or ``np.nan`` if the solver
        fails to converge.
    """
    gas = ct.Solution(cm.MECHANISM)
    cm.set_mixture(gas, phi, x_h2_energy, T0, p0)

    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3.0, slope=0.07, curve=0.14)
    flame.transport_model = "mixture-averaged"

    try:
        flame.solve(loglevel=loglevel, auto=True)
    except Exception as exc:  # noqa: BLE001 - report and skip non-converged
        print(f"    ! flame did not converge (phi={phi}, "
              f"x_E={100*x_h2_energy:.0f}%): {exc}")
        return float("nan")

    # Inlet (unburned) velocity = laminar burning velocity.
    return float(flame.velocity[0])


def _make_flame(phi, x_h2_energy, T0, p0, width):
    """Build and configure a FreeFlame without solving (helper for sweeps)."""
    gas = ct.Solution(cm.MECHANISM)
    cm.set_mixture(gas, phi, x_h2_energy, T0, p0)
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3.0, slope=0.08, curve=0.16)
    flame.transport_model = "mixture-averaged"
    return gas, flame


def solve_series(states, T0, p0, width=0.03, loglevel=0):
    """Solve a sequence of flames using continuation (warm starts).

    Each converged solution is reused as the initial guess for the next
    state in the list. This is dramatically faster than solving every
    flame from scratch and is the standard way to sweep a parameter.

    Parameters
    ----------
    states : list of (phi, x_h2_energy)
        Sequence of operating points, ordered so that neighbouring
        points are physically close.
    T0, p0 : float
        Unburned-gas temperature [K] and pressure [Pa].

    Returns
    -------
    list of float
        Laminar burning velocities [m/s], aligned with ``states``.
    """
    speeds = []
    flame = None
    for i, (phi, xe) in enumerate(states):
        gas = ct.Solution(cm.MECHANISM)
        cm.set_mixture(gas, phi, xe, T0, p0)
        new_flame = ct.FreeFlame(gas, width=width)
        new_flame.set_refine_criteria(ratio=3.0, slope=0.08, curve=0.16)
        new_flame.transport_model = "mixture-averaged"

        # Warm start from the previous converged profile when available.
        if flame is not None:
            try:
                new_flame.set_initial_guess(data=flame.to_array())
            except Exception:
                pass  # fall back to a cold start

        try:
            new_flame.solve(loglevel=loglevel, auto=True)
            s = float(new_flame.velocity[0])
            flame = new_flame
        except Exception as exc:  # noqa: BLE001
            print(f"    ! non-converged (phi={phi}, x_E={100*xe:.0f}%): {exc}")
            s = float("nan")
        speeds.append(s)
    return speeds


# --------------------------------------------------------------------------
# Parameter sweeps
# --------------------------------------------------------------------------

def sweep_phi(blends, phi_array, T0, p0):
    """S_L vs equivalence ratio for several blends (continuation per blend)."""
    out = {}
    for xe in blends:
        states = [(phi, xe) for phi in phi_array]
        speeds = solve_series(states, T0, p0)
        for phi, s in zip(phi_array, speeds):
            print(f"  [phi-sweep] {cm.blend_label(xe):>16} "
                  f"phi={phi:4.2f}  S_L={s*100:7.2f} cm/s")
        out[xe] = np.asarray(speeds)
    return out


def sweep_h2(h2_array, phi, T0, p0):
    """S_L vs hydrogen energy fraction at fixed phi (continuation)."""
    states = [(phi, xe) for xe in h2_array]
    speeds = solve_series(states, T0, p0)
    for xe, s in zip(h2_array, speeds):
        print(f"  [H2-sweep] x_E={100*xe:5.1f}%  S_L={s*100:7.2f} cm/s")
    return np.asarray(speeds)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_phi_sweep(phi_array, data, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    for xe, speeds in data.items():
        ax.plot(phi_array, speeds * 100, marker="o",
                label=cm.blend_label(xe))
    ax.set_xlabel(r"Equivalence ratio  $\phi$")
    ax.set_ylabel(r"Laminar burning velocity  $S_L$  [cm/s]")
    ax.set_title(f"Laminar flame speed vs equivalence ratio "
                 f"($T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_h2_sweep(h2_array, speeds, phi, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    ax.plot(100 * h2_array, speeds * 100, marker="D", color="C2")
    ax.set_xlabel("Hydrogen energy fraction  [%]")
    ax.set_ylabel(r"Laminar burning velocity  $S_L$  [cm/s]")
    ax.set_title(f"Laminar flame speed vs H$_2$ enrichment "
                 f"($\\phi$={phi}, $T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

def _append_csv(rows, fname="flame_speed.csv", header=False):
    path = os.path.join(cm.DATA_DIR, fname)
    mode = "w" if header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["sweep", "phi", "x_h2_energy",
                             "T0_K", "p0_bar", "S_L_cm_s"])
        writer.writerows(rows)


def run_phi_blend(xe, T0, p0):
    """Run and checkpoint a single-blend phi sweep (one CLI stage)."""
    phi_array = np.linspace(0.5, 1.4, 8)
    states = [(phi, xe) for phi in phi_array]
    speeds = solve_series(states, T0, p0)
    rows = []
    for phi, s in zip(phi_array, speeds):
        print(f"  [phi-sweep] {cm.blend_label(xe):>16} "
              f"phi={phi:4.2f}  S_L={s*100:7.2f} cm/s", flush=True)
        rows.append(["phi", phi, xe, T0, p0 / 1e5, s * 100])
    _append_csv(rows)
    # save/update the phi figure from the full CSV so far
    _replot_phi(T0, p0)


def run_h2(phi_fixed, T0, p0):
    """Run and checkpoint the hydrogen sweep (one CLI stage)."""
    h2_array = np.linspace(0.0, 1.0, 11)
    states = [(phi_fixed, xe) for xe in h2_array]
    speeds = solve_series(states, T0, p0)
    rows = []
    for xe, s in zip(h2_array, speeds):
        print(f"  [H2-sweep] x_E={100*xe:5.1f}%  S_L={s*100:7.2f} cm/s",
              flush=True)
        rows.append(["h2", phi_fixed, xe, T0, p0 / 1e5, s * 100])
    _append_csv(rows)
    plot_h2_sweep(h2_array, np.asarray(speeds), phi_fixed, T0, p0,
                  "flame_speed_h2.png")


def _replot_phi(T0, p0):
    """Rebuild the phi figure from whatever is in the CSV so far."""
    path = os.path.join(cm.DATA_DIR, "flame_speed.csv")
    if not os.path.exists(path):
        return
    data = {}
    phis = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["sweep"] != "phi":
                continue
            xe = float(r["x_h2_energy"])
            data.setdefault(xe, []).append(float(r["S_L_cm_s"]) / 100.0)
            phis.setdefault(xe, []).append(float(r["phi"]))
    if not data:
        return
    data = {k: np.asarray(v) for k, v in data.items()}
    any_phi = np.asarray(sorted(next(iter(phis.values()))))
    plot_phi_sweep(any_phi, data, T0, p0, "flame_speed_phi.png")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    """CLI driver with staged execution to fit time limits.

    Usage:
        python flame_speed.py init          # write CSV header
        python flame_speed.py phi <x_h2>    # one blend phi sweep
        python flame_speed.py h2            # hydrogen sweep
        python flame_speed.py all           # everything (may be slow)
    """
    import sys

    # --- conditions representative of an aero gas turbine combustor inlet ---
    T0 = 450.0
    p0 = 5.0e5
    blends = [0.0, 0.3, 0.6, 1.0]

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "init":
        _append_csv([], header=True)
        print("CSV initialised.", flush=True)
        return

    if arg == "phi":
        xe = float(sys.argv[2])
        run_phi_blend(xe, T0, p0)
        print(f"phi sweep for x_E={100*xe:.0f}% done.", flush=True)
        return

    if arg == "h2":
        run_h2(1.0, T0, p0)
        print("H2 sweep done.", flush=True)
        return

    # arg == "all"
    _append_csv([], header=True)
    for xe in blends:
        run_phi_blend(xe, T0, p0)
    run_h2(1.0, T0, p0)
    print("Module 2 (laminar flame speed) complete.", flush=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
