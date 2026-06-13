"""
ignition_delay.py
=================
Module 1 of 3.

Computes auto-ignition delay times tau_ign for H2/CH4-air mixtures in a
constant-volume, adiabatic reactor, and studies how tau_ign responds to:

  * the hydrogen energy fraction of the fuel,
  * the initial temperature,
  * the initial pressure.

Method
------
A homogeneous mixture is enclosed in an adiabatic, constant-volume
reactor (Cantera ``IdealGasReactor``). The reactor is integrated in time
and the ignition delay is defined as the instant of maximum temperature
rise rate, d(T)/dt|max -- a standard and mechanism-independent definition.

The constant-volume assumption corresponds to the classic shock-tube /
rapid-compression-machine measurement of ignition delay and is the
quantity most often reported in the kinetics literature, which makes the
results directly comparable.

Outputs
-------
  results/data/ignition_delay.csv
  results/figures/ignition_T_sweep.png
  results/figures/ignition_p_sweep.png
  results/figures/ignition_h2_sweep.png
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

def ignition_delay(phi: float,
                   x_h2_energy: float,
                   T0: float,
                   p0: float,
                   t_end: float = 1.0,
                   max_steps: int = 100_000) -> float:
    """Compute the ignition delay time of one mixture.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    x_h2_energy : float
        Hydrogen energy fraction in [0, 1].
    T0, p0 : float
        Initial temperature [K] and pressure [Pa].
    t_end : float
        Maximum integration time [s]. Mixtures that do not ignite within
        this window return ``np.nan``.
    max_steps : int
        Safety cap on the number of integration steps.

    Returns
    -------
    float
        Ignition delay [s], defined as the time of maximum dT/dt, or
        ``np.nan`` if no ignition is detected.
    """
    gas = ct.Solution(cm.MECHANISM)
    cm.set_mixture(gas, phi, x_h2_energy, T0, p0)

    reactor = ct.IdealGasReactor(gas, clone=False)
    net = ct.ReactorNet([reactor])
    net.rtol = 1.0e-9
    net.atol = 1.0e-15

    times = []
    temps = []
    t = 0.0
    steps = 0
    while t < t_end and steps < max_steps:
        t = net.step()
        times.append(t)
        temps.append(reactor.T)
        steps += 1
        # Early stop once clearly ignited and the gradient is falling: the
        # temperature has risen well above its initial value.
        if reactor.T > T0 + 600.0 and len(temps) > 5:
            # Continue a little to make sure the dT/dt peak is captured.
            if temps[-1] - temps[-2] < temps[-2] - temps[-3]:
                break

    times = np.asarray(times)
    temps = np.asarray(temps)
    if temps.size < 3 or (temps.max() - T0) < 200.0:
        return float("nan")  # no meaningful ignition

    dTdt = np.gradient(temps, times)
    i_peak = int(np.argmax(dTdt))
    return float(times[i_peak])


# --------------------------------------------------------------------------
# Parameter sweeps
# --------------------------------------------------------------------------

def sweep_temperature(blends, T_array, phi, p0):
    """tau_ign vs initial temperature for several blends."""
    out = {}
    for xe in blends:
        taus = []
        for T0 in T_array:
            tau = ignition_delay(phi, xe, T0, p0)
            taus.append(tau)
            print(f"  [T-sweep] {cm.blend_label(xe):>16} "
                  f"T0={T0:6.1f} K  tau={tau*1e3:9.4f} ms")
        out[xe] = np.asarray(taus)
    return out


def sweep_pressure(blends, p_array, phi, T0):
    """tau_ign vs initial pressure for several blends."""
    out = {}
    for xe in blends:
        taus = []
        for p0 in p_array:
            tau = ignition_delay(phi, xe, T0, p0)
            taus.append(tau)
            print(f"  [p-sweep] {cm.blend_label(xe):>16} "
                  f"p0={p0/1e5:5.1f} bar  tau={tau*1e3:9.4f} ms")
        out[xe] = np.asarray(taus)
    return out


def sweep_h2(h2_array, phi, T0, p0):
    """tau_ign vs hydrogen energy fraction at fixed T0, p0, phi."""
    taus = []
    for xe in h2_array:
        tau = ignition_delay(phi, xe, T0, p0)
        taus.append(tau)
        print(f"  [H2-sweep] x_E={100*xe:5.1f}%  tau={tau*1e3:9.4f} ms")
    return np.asarray(taus)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_temperature_sweep(T_array, data, phi, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    inv_T = 1000.0 / T_array
    for xe, taus in data.items():
        ax.semilogy(inv_T, taus * 1e3, marker="o", label=cm.blend_label(xe))
    ax.set_xlabel(r"$1000\,/\,T_0$  [1/K]")
    ax.set_ylabel(r"Ignition delay  $\tau_{ign}$  [ms]")
    ax.set_title(f"Ignition delay vs temperature "
                 f"($\\phi$={phi}, p={p0/1e5:.0f} bar)")
    # Secondary axis with absolute temperature for readability.
    def _safe_inv(x):
        x = np.asarray(x, dtype=float)
        return np.divide(1000.0, x, out=np.full_like(x, np.nan), where=x != 0)

    secax = ax.secondary_xaxis("top", functions=(_safe_inv, _safe_inv))
    secax.set_xlabel(r"$T_0$  [K]")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_pressure_sweep(p_array, data, phi, T0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    for xe, taus in data.items():
        ax.loglog(p_array / 1e5, taus * 1e3, marker="s",
                  label=cm.blend_label(xe))
    ax.set_xlabel("Initial pressure  $p_0$  [bar]")
    ax.set_ylabel(r"Ignition delay  $\tau_{ign}$  [ms]")
    ax.set_title(f"Ignition delay vs pressure "
                 f"($\\phi$={phi}, $T_0$={T0:.0f} K)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_h2_sweep(h2_array, taus, phi, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    ax.semilogy(100 * h2_array, taus * 1e3, marker="D", color="C3")
    ax.set_xlabel("Hydrogen energy fraction  [%]")
    ax.set_ylabel(r"Ignition delay  $\tau_{ign}$  [ms]")
    ax.set_title(f"Ignition delay vs H$_2$ enrichment "
                 f"($\\phi$={phi}, $T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

def save_csv(rows, fname="ignition_delay.csv"):
    path = os.path.join(cm.DATA_DIR, fname)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "phi", "x_h2_energy",
                         "T0_K", "p0_bar", "tau_ign_ms"])
        writer.writerows(rows)
    print(f"  saved {path}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    # --- conditions representative of an aero gas turbine combustor ---
    phi = 0.7                      # lean primary zone
    p_ref = 20.0e5                 # 20 bar
    T_ref = 1100.0                 # K (autoignition-relevant)
    blends = [0.0, 0.3, 0.6, 1.0]  # hydrogen energy fractions

    rows = []

    print("Temperature sweep ...")
    T_array = np.linspace(900.0, 1400.0, 11)
    T_data = sweep_temperature(blends, T_array, phi, p_ref)
    plot_temperature_sweep(T_array, T_data, phi, p_ref,
                           "ignition_T_sweep.png")
    for xe, taus in T_data.items():
        for T0, tau in zip(T_array, taus):
            rows.append(["T", phi, xe, T0, p_ref / 1e5, tau * 1e3])

    print("Pressure sweep ...")
    p_array = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0]) * 1e5
    p_data = sweep_pressure(blends, p_array, phi, T_ref)
    plot_pressure_sweep(p_array, p_data, phi, T_ref,
                        "ignition_p_sweep.png")
    for xe, taus in p_data.items():
        for p0, tau in zip(p_array, taus):
            rows.append(["p", phi, xe, T_ref, p0 / 1e5, tau * 1e3])

    print("Hydrogen sweep ...")
    h2_array = np.linspace(0.0, 1.0, 11)
    h2_taus = sweep_h2(h2_array, phi, T_ref, p_ref)
    plot_h2_sweep(h2_array, h2_taus, phi, T_ref, p_ref,
                  "ignition_h2_sweep.png")
    for xe, tau in zip(h2_array, h2_taus):
        rows.append(["h2", phi, xe, T_ref, p_ref / 1e5, tau * 1e3])

    save_csv(rows)
    print("Module 1 (ignition delay) complete.")


if __name__ == "__main__":
    main()
