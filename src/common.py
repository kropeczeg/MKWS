"""
common.py
=========
Shared utilities for the MKWS 2026 project:

    "Effect of hydrogen enrichment on ignition delay, laminar flame speed
     and NOx emissions of methane-air mixtures under aero gas turbine
     combustor conditions"

This module centralises the physics that the three computational modules
(ignition_delay.py, flame_speed.py, nox_emissions.py) all depend on:

  * construction of an H2/CH4 fuel blend defined by an ENERGY fraction,
  * construction of the corresponding fuel + air mixture at a given
    equivalence ratio,
  * lower heating values and a couple of small helpers,
  * a single place to set the chemical mechanism and plotting style.

Defining the blend by energy fraction (rather than mole fraction) is the
physically meaningful choice: it answers the engineering question
"if I replace x% of the chemical energy of the fuel with hydrogen, what
happens?", which is exactly the decarbonisation scenario for aero gas
turbines.

Author: Jan Kalak, Faculty of Power and Aeronautical Engineering (MEiL),
        Warsaw University of Technology
Course : Computational Methods in Combustion (MKWS), 2026
"""

from __future__ import annotations

import os
import numpy as np
import cantera as ct

# --------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------

#: Chemical mechanism. GRI-Mech 3.0 is used because it contains both the
#: C1 chemistry (CH4) and the N-chemistry (extended Zeldovich) required for
#: thermal-NOx predictions, as well as H2/O2 sub-chemistry.
MECHANISM = "gri30.yaml"

#: Molar lower heating values [J/kmol] at 298.15 K, used to convert between
#: energy fraction and mole fraction of the H2/CH4 blend.
#: Values from standard thermochemical data.
LHV_CH4 = 802.3e6   # J/kmol  (CH4 + 2 O2 -> CO2 + 2 H2O, gaseous water)
LHV_H2 = 241.8e6   # J/kmol  (H2 + 0.5 O2 -> H2O, gaseous water)

#: Project root and output folders (resolved relative to this file).
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "results", "data")

for _d in (FIG_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------
# Fuel blend construction
# --------------------------------------------------------------------------

def blend_mole_fractions(x_h2_energy: float) -> dict[str, float]:
    """Return the {CH4, H2} mole fractions of a blend defined by energy share.

    Parameters
    ----------
    x_h2_energy : float
        Fraction of the fuel's chemical energy supplied by hydrogen,
        in the range [0, 1]. 0.0 is pure methane, 1.0 is pure hydrogen.

    Returns
    -------
    dict
        Mole fractions {"CH4": ..., "H2": ...} that sum to 1.

    Notes
    -----
    If n_H2 and n_CH4 are the moles of each fuel, the hydrogen energy
    fraction is

        x_E = n_H2 * LHV_H2 / (n_H2 * LHV_H2 + n_CH4 * LHV_CH4).

    Solving for the mole ratio and normalising gives the result below.
    """
    if not 0.0 <= x_h2_energy <= 1.0:
        raise ValueError("x_h2_energy must be in [0, 1]")

    # Edge cases avoid division by zero and floating-point fuzz.
    if x_h2_energy == 0.0:
        return {"CH4": 1.0, "H2": 0.0}
    if x_h2_energy == 1.0:
        return {"CH4": 0.0, "H2": 1.0}

    # Energy:  x_E = n_H2 LHV_H2 / (n_H2 LHV_H2 + n_CH4 LHV_CH4)
    # Let r = n_H2 / n_CH4. Then x_E = r LHV_H2 / (r LHV_H2 + LHV_CH4)
    #  =>  r = x_E * LHV_CH4 / ((1 - x_E) * LHV_H2)
    r = x_h2_energy * LHV_CH4 / ((1.0 - x_h2_energy) * LHV_H2)
    n_ch4 = 1.0
    n_h2 = r * n_ch4
    total = n_ch4 + n_h2
    return {"CH4": n_ch4 / total, "H2": n_h2 / total}


def fuel_string(x_h2_energy: float) -> str:
    """Cantera fuel composition string for a given hydrogen energy share."""
    xb = blend_mole_fractions(x_h2_energy)
    return f"CH4:{xb['CH4']:.6f}, H2:{xb['H2']:.6f}"


def set_mixture(gas: ct.Solution,
                phi: float,
                x_h2_energy: float,
                T: float,
                p: float) -> ct.Solution:
    """Set a gas object to a fuel/air mixture at the requested state.

    Parameters
    ----------
    gas : ct.Solution
        Cantera solution object to be configured (modified in place).
    phi : float
        Equivalence ratio.
    x_h2_energy : float
        Hydrogen energy fraction of the fuel blend, in [0, 1].
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].

    Returns
    -------
    ct.Solution
        The same object, for convenience in call chains.
    """
    fuel = fuel_string(x_h2_energy)
    # Air as 21% O2 / 79% N2 by mole; Cantera computes the stoichiometric
    # oxidiser demand internally from the fuel composition.
    gas.set_equivalence_ratio(phi, fuel, "O2:1.0, N2:3.76")
    gas.TP = T, p
    return gas


def blend_label(x_h2_energy: float) -> str:
    """Human-readable label such as '30% H2 (energy)'."""
    return f"{int(round(100 * x_h2_energy))}% H2 (energy)"


# --------------------------------------------------------------------------
# Plotting style
# --------------------------------------------------------------------------

def apply_plot_style() -> None:
    """Apply a clean, publication-style Matplotlib configuration.

    Kept deliberately simple and non-flashy so that figures read like a
    diploma-thesis report rather than a marketing deck.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (7.0, 4.5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


if __name__ == "__main__":
    # Quick sanity check of the energy-fraction <-> mole-fraction mapping.
    print("Hydrogen blend composition check")
    print(f"{'x_E [%]':>8} | {'X_CH4':>8} | {'X_H2':>8} | recovered x_E [%]")
    for xe in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        xb = blend_mole_fractions(xe)
        # Recover energy fraction to verify the inverse relationship.
        e_h2 = xb["H2"] * LHV_H2
        e_ch4 = xb["CH4"] * LHV_CH4
        rec = e_h2 / (e_h2 + e_ch4) if (e_h2 + e_ch4) > 0 else 0.0
        print(f"{100*xe:8.1f} | {xb['CH4']:8.4f} | {xb['H2']:8.4f} | {100*rec:8.2f}")
