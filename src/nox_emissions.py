"""
nox_emissions.py
================
Module 3 of 3.

Computes the adiabatic flame temperature and the NOx emission of
H2/CH4-air mixtures, and studies the trade-off introduced by hydrogen
enrichment:

  * hydrogen raises the adiabatic flame temperature, which
  * increases thermal (Zeldovich) NO formation,

while leaner operation lowers temperature and therefore NOx. The module
quantifies this competition.

Method
------
For each operating point the burned gas is first brought to the
constant-pressure adiabatic equilibrium temperature (T_ad). NOx is then
obtained from a finite-residence-time, constant-pressure, adiabatic
reactor: the mixture is ignited and integrated for a residence time
typical of a gas turbine primary zone. This kinetic treatment is more
realistic than assuming full chemical equilibrium, because thermal NO
formation is rate-limited and does not reach its equilibrium value within
the available residence time.

NOx is reported as an emission index EINOx [g NO2 / kg fuel], the standard
metric for gas turbine emissions, with all NO converted to an equivalent
mass of NO2.

Outputs
-------
  results/data/nox_emissions.csv
  results/figures/nox_phi.png
  results/figures/nox_h2.png
  results/figures/tad_phi.png
"""

from __future__ import annotations

import csv
import os
import numpy as np
import cantera as ct

import common as cm


# Molar masses [kg/kmol] used for the emission index.
_M_NO = 30.006
_M_NO2 = 46.006


# --------------------------------------------------------------------------
# Adiabatic flame temperature
# --------------------------------------------------------------------------

def adiabatic_flame_temperature(phi: float,
                                x_h2_energy: float,
                                T0: float,
                                p0: float) -> float:
    """Constant-pressure adiabatic equilibrium (flame) temperature [K]."""
    gas = ct.Solution(cm.MECHANISM)
    cm.set_mixture(gas, phi, x_h2_energy, T0, p0)
    gas.equilibrate("HP")
    return float(gas.T)


# --------------------------------------------------------------------------
# NOx from a finite-residence-time reactor
# --------------------------------------------------------------------------

def einox(phi: float,
          x_h2_energy: float,
          T0: float,
          p0: float,
          residence_time: float = 5.0e-3) -> tuple[float, float]:
    """Emission index of NOx and the burned-gas temperature.

    Parameters
    ----------
    phi : float
        Equivalence ratio.
    x_h2_energy : float
        Hydrogen energy fraction in [0, 1].
    T0, p0 : float
        Inlet temperature [K] and pressure [Pa].
    residence_time : float
        Reactor residence time [s], representative of a gas-turbine
        primary zone (a few milliseconds).

    Returns
    -------
    (EINOx, T_burned) : tuple of float
        EINOx in g(NO2-equivalent) per kg of fuel, and the final
        burned-gas temperature [K].
    """
    gas = ct.Solution(cm.MECHANISM)
    cm.set_mixture(gas, phi, x_h2_energy, T0, p0)

    # Fuel mass fraction of the fresh charge, needed to normalise EINOx.
    y_fuel = 0.0
    for sp in ("CH4", "H2"):
        if sp in gas.species_names:
            y_fuel += gas.Y[gas.species_index(sp)]

    # Step 1 -- combustion. Bring the charge to the constant-pressure
    # adiabatic burned state. This represents the fast main heat-release
    # chemistry of the flame and gives the post-flame temperature and the
    # radical pool that seed NO formation. Equilibrating only the fast
    # (non-nitrogen) chemistry would be ideal; equilibrating "HP" here is
    # a robust proxy that sets the correct burned temperature and major
    # species, after which NO is re-grown kinetically.
    gas.equilibrate("HP")

    # Reset NOx to zero so that what we measure is the NO *formed* during
    # the residence time in the hot products, not the equilibrium value.
    Y = gas.Y.copy()
    for sp in ("NO", "NO2", "N2O", "NH", "N"):
        if sp in gas.species_names:
            Y[gas.species_index(sp)] = 0.0
    # Renormalise and re-impose temperature & pressure.
    T_burned_eq = gas.T
    gas.TPY = T_burned_eq, p0, Y

    # Step 2 -- finite-rate NO formation in the hot products over the
    # residence time, at constant pressure and adiabatic.
    reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
    net = ct.ReactorNet([reactor])
    net.rtol = 1.0e-9
    net.atol = 1.0e-18
    net.advance(residence_time)

    # Mass fractions of NO and NO2 in the burned gas.
    y_no = reactor.thermo["NO"].Y[0] if "NO" in gas.species_names else 0.0
    y_no2 = reactor.thermo["NO2"].Y[0] if "NO2" in gas.species_names else 0.0

    # Convert NO to NO2-equivalent mass and form the emission index
    # (mass of pollutant per mass of fuel burned), in g/kg.
    y_nox_as_no2 = y_no * (_M_NO2 / _M_NO) + y_no2
    ei = 1000.0 * y_nox_as_no2 / y_fuel if y_fuel > 0 else float("nan")
    return float(ei), float(reactor.T)


# --------------------------------------------------------------------------
# Parameter sweeps
# --------------------------------------------------------------------------

def sweep_phi(blends, phi_array, T0, p0, residence_time):
    """EINOx and T_ad vs equivalence ratio for several blends."""
    ei_out, tad_out = {}, {}
    for xe in blends:
        eis, tads = [], []
        for phi in phi_array:
            ei, _ = einox(phi, xe, T0, p0, residence_time)
            tad = adiabatic_flame_temperature(phi, xe, T0, p0)
            eis.append(ei)
            tads.append(tad)
            print(f"  [phi-sweep] {cm.blend_label(xe):>16} "
                  f"phi={phi:4.2f}  T_ad={tad:6.1f} K  "
                  f"EINOx={ei:8.3f} g/kg")
        ei_out[xe] = np.asarray(eis)
        tad_out[xe] = np.asarray(tads)
    return ei_out, tad_out


def sweep_h2(h2_array, phi, T0, p0, residence_time):
    """EINOx and T_ad vs hydrogen energy fraction at fixed phi."""
    eis, tads = [], []
    for xe in h2_array:
        ei, _ = einox(phi, xe, T0, p0, residence_time)
        tad = adiabatic_flame_temperature(phi, xe, T0, p0)
        eis.append(ei)
        tads.append(tad)
        print(f"  [H2-sweep] x_E={100*xe:5.1f}%  T_ad={tad:6.1f} K  "
              f"EINOx={ei:8.3f} g/kg")
    return np.asarray(eis), np.asarray(tads)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_nox_phi(phi_array, ei_data, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    for xe, eis in ei_data.items():
        ax.semilogy(phi_array, eis, marker="o", label=cm.blend_label(xe))
    ax.set_xlabel(r"Equivalence ratio  $\phi$")
    ax.set_ylabel(r"EINOx  [g NO$_2$ / kg fuel]")
    ax.set_title(f"NOx emission index vs equivalence ratio "
                 f"($T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_tad_phi(phi_array, tad_data, T0, p0, fname):
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax = plt.subplots()
    for xe, tads in tad_data.items():
        ax.plot(phi_array, tads, marker="s", label=cm.blend_label(xe))
    ax.set_xlabel(r"Equivalence ratio  $\phi$")
    ax.set_ylabel(r"Adiabatic flame temperature  $T_{ad}$  [K]")
    ax.set_title(f"Adiabatic flame temperature vs equivalence ratio "
                 f"($T_0$={T0:.0f} K, p={p0/1e5:.0f} bar)")
    ax.legend()
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


def plot_nox_h2(h2_array, eis, tads, phi, T0, p0, fname):
    """Dual-axis plot: EINOx and T_ad vs hydrogen fraction."""
    import matplotlib.pyplot as plt
    cm.apply_plot_style()
    fig, ax1 = plt.subplots()

    color1 = "C3"
    ax1.plot(100 * h2_array, eis, marker="D", color=color1, label="EINOx")
    ax1.set_xlabel("Hydrogen energy fraction  [%]")
    ax1.set_ylabel(r"EINOx  [g NO$_2$ / kg fuel]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.spines.right.set_visible(True)
    color2 = "C0"
    ax2.plot(100 * h2_array, tads, marker="s", color=color2,
             label=r"$T_{ad}$")
    ax2.set_ylabel(r"Adiabatic flame temperature  $T_{ad}$  [K]",
                   color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    ax1.set_title(f"NOx / flame-temperature trade-off vs H$_2$ enrichment "
                  f"($\\phi$={phi}, p={p0/1e5:.0f} bar)")
    fig.savefig(os.path.join(cm.FIG_DIR, fname))
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

def save_csv(rows, fname="nox_emissions.csv"):
    path = os.path.join(cm.DATA_DIR, fname)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "phi", "x_h2_energy", "T0_K", "p0_bar",
                         "T_ad_K", "EINOx_g_per_kg"])
        writer.writerows(rows)
    print(f"  saved {path}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    # --- conditions representative of an aero gas turbine combustor ---
    T0 = 700.0                      # K (preheated charge)
    p0 = 20.0e5                     # 20 bar
    residence_time = 5.0e-3         # s (primary-zone residence time)
    blends = [0.0, 0.3, 0.6, 1.0]   # hydrogen energy fractions

    rows = []

    print("Equivalence-ratio sweep (NOx and T_ad) ...")
    phi_array = np.linspace(0.5, 1.3, 9)
    ei_data, tad_data = sweep_phi(blends, phi_array, T0, p0, residence_time)
    plot_nox_phi(phi_array, ei_data, T0, p0, "nox_phi.png")
    plot_tad_phi(phi_array, tad_data, T0, p0, "tad_phi.png")
    for xe in blends:
        for phi, ei, tad in zip(phi_array, ei_data[xe], tad_data[xe]):
            rows.append(["phi", phi, xe, T0, p0 / 1e5, tad, ei])

    print("Hydrogen sweep (lean, phi=0.7) ...")
    phi_fixed = 0.7
    h2_array = np.linspace(0.0, 1.0, 11)
    eis, tads = sweep_h2(h2_array, phi_fixed, T0, p0, residence_time)
    plot_nox_h2(h2_array, eis, tads, phi_fixed, T0, p0, "nox_h2.png")
    for xe, ei, tad in zip(h2_array, eis, tads):
        rows.append(["h2", phi_fixed, xe, T0, p0 / 1e5, tad, ei])

    save_csv(rows)
    print("Module 3 (NOx emissions) complete.")


if __name__ == "__main__":
    main()
