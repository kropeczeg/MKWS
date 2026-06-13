"""
Build the project Jupyter notebook (MKWS_hydrogen_enrichment.ipynb).

The notebook is a guided, presentation-ready walkthrough of the study. By
default it *loads* the pre-computed CSV results (fast, good for a live
demo) and reproduces the figures; every heavy computation is also given as
an optional, clearly-marked cell that can be re-run from scratch.

Run:  python build_notebook.py
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


# --------------------------------------------------------------------------
md(r"""# Effect of hydrogen enrichment on methane–air combustion

**Computational Methods in Combustion (MKWS), 2026**
Faculty of Power and Aeronautical Engineering (MEiL), Warsaw University of Technology

This notebook studies how blending hydrogen into methane changes three
combustion quantities relevant to an aero gas turbine combustor:

1. **Ignition delay** $\tau_{ign}$
2. **Laminar burning velocity** $S_L$
3. **NOx emission index** EINOx (and the adiabatic flame temperature behind it)

The hydrogen content is defined as a **fraction of the fuel chemical energy**,
which is the meaningful variable for the decarbonisation question. All
chemistry uses **GRI-Mech 3.0**; a separate section validates this choice
against a detailed H$_2$/O$_2$ mechanism.

> The heavy computations are pre-computed and stored as CSV files in
> `../results/data/`. This notebook reloads them so it runs in seconds.
> Cells marked **(slow, optional)** re-run a calculation from scratch.""")

# --- setup ---
md("## 0. Setup")
code(r"""import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the project modules importable regardless of the working directory
# the notebook is launched from. We look for the 'src' folder next to (or
# one level above) the current directory.
_here = os.getcwd()
for _cand in (os.path.join(_here, "src"),
              os.path.join(_here, "..", "src"),
              _here):
    if os.path.isfile(os.path.join(_cand, "common.py")):
        sys.path.insert(0, os.path.abspath(_cand))
        _root = os.path.abspath(os.path.join(_cand, os.pardir))
        break
else:
    raise RuntimeError("could not locate the project's src/ directory")

import common as cm
cm.apply_plot_style()

DATA = os.path.join(_root, "results", "data")
print("Project root   :", _root)
print("Data directory :", DATA)
print("Mechanism      :", cm.MECHANISM)""")

# --- blend definition ---
md(r"""## 1. The fuel blend: energy fraction vs mole fraction

Mixing hydrogen and methane on an **energy** basis is not the same as mixing
on a **mole** basis, because the two fuels carry very different heating
values per mole. The table below shows the mole fractions that correspond to
a given hydrogen energy share — note that 30 % of the *energy* from hydrogen
already means almost 60 % of the *moles*.""")
code(r"""rows = []
for xe in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    xb = cm.blend_mole_fractions(xe)
    rows.append([100*xe, xb["CH4"], xb["H2"]])
pd.DataFrame(rows, columns=["H2 energy [%]", "X_CH4", "X_H2"]).round(4)""")

# --- module 1 ---
md(r"""## 2. Ignition delay

Auto-ignition delay in a constant-volume, adiabatic reactor, defined as the
time of maximum temperature-rise rate. We reload the pre-computed sweeps over
temperature, pressure and hydrogen fraction.""")
code(r"""ign = pd.read_csv(os.path.join(DATA, "ignition_delay.csv"))
ign.head()""")

md("### 2a. $\\tau_{ign}$ vs temperature (Arrhenius plot)")
code(r"""sub = ign[ign["sweep"] == "T"]
fig, ax = plt.subplots()
for xe, g in sub.groupby("x_h2_energy"):
    ax.semilogy(1000.0/g["T0_K"], g["tau_ign_ms"], marker="o",
                label=cm.blend_label(xe))
ax.set_xlabel(r"$1000/T_0$ [1/K]"); ax.set_ylabel(r"$\tau_{ign}$ [ms]")
ax.set_title("Ignition delay vs temperature"); ax.legend(); plt.show()""")

md("### 2b. $\\tau_{ign}$ vs hydrogen fraction")
code(r"""sub = ign[ign["sweep"] == "h2"]
fig, ax = plt.subplots()
ax.semilogy(100*sub["x_h2_energy"], sub["tau_ign_ms"], marker="D", color="C3")
ax.set_xlabel("Hydrogen energy fraction [%]"); ax.set_ylabel(r"$\tau_{ign}$ [ms]")
ax.set_title("Ignition delay vs H2 enrichment"); plt.show()""")

md(r"""**(slow, optional)** Recompute a single ignition-delay point to confirm
the stored data.""")
code(r"""import ignition_delay as igmod
tau = igmod.ignition_delay(phi=0.7, x_h2_energy=0.5, T0=1100.0, p0=20e5)
print(f"tau_ign(50% H2, 1100 K, 20 bar) = {tau*1e3:.4f} ms")""")

# --- module 2 ---
md(r"""## 3. Laminar burning velocity

Freely-propagating 1-D premixed flame (`FreeFlame`, mixture-averaged
transport). $S_L$ is the unburned-gas inlet velocity of the converged
solution.""")
code(r"""fla = pd.read_csv(os.path.join(DATA, "flame_speed.csv"))
fla.head()""")

md("### 3a. $S_L$ vs equivalence ratio")
code(r"""sub = fla[fla["sweep"] == "phi"]
fig, ax = plt.subplots()
for xe, g in sub.groupby("x_h2_energy"):
    g = g.sort_values("phi")
    ax.plot(g["phi"], g["S_L_cm_s"], marker="o", label=cm.blend_label(xe))
ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$S_L$ [cm/s]")
ax.set_title("Laminar flame speed vs equivalence ratio"); ax.legend(); plt.show()""")

md("### 3b. $S_L$ vs hydrogen fraction (stoichiometric)")
code(r"""sub = fla[fla["sweep"] == "h2"].sort_values("x_h2_energy")
fig, ax = plt.subplots()
ax.plot(100*sub["x_h2_energy"], sub["S_L_cm_s"], marker="D", color="C2")
ax.set_xlabel("Hydrogen energy fraction [%]"); ax.set_ylabel(r"$S_L$ [cm/s]")
ax.set_title("Laminar flame speed vs H2 enrichment"); plt.show()""")

# --- module 3 ---
md(r"""## 4. NOx emissions and the flame-temperature trade-off

The burned gas is brought to its constant-pressure equilibrium temperature;
NO is then grown kinetically over a primary-zone residence time. NOx is
reported as an emission index in g(NO$_2$-equivalent) per kg fuel.""")
code(r"""nox = pd.read_csv(os.path.join(DATA, "nox_emissions.csv"))
nox.head()""")

md("### 4a. EINOx vs equivalence ratio")
code(r"""sub = nox[nox["sweep"] == "phi"]
fig, ax = plt.subplots()
for xe, g in sub.groupby("x_h2_energy"):
    g = g.sort_values("phi")
    ax.semilogy(g["phi"], g["EINOx_g_per_kg"], marker="o",
                label=cm.blend_label(xe))
ax.set_xlabel(r"$\phi$"); ax.set_ylabel("EINOx [g NO2 / kg fuel]")
ax.set_title("NOx emission index vs equivalence ratio"); ax.legend(); plt.show()""")

md(r"""### 4b. The headline result: NOx / temperature trade-off

Hydrogen raises the flame temperature almost linearly, but thermal NOx
responds *exponentially* to temperature — so a modest temperature rise
produces a large NOx penalty.""")
code(r"""sub = nox[nox["sweep"] == "h2"].sort_values("x_h2_energy")
fig, ax1 = plt.subplots()
ax1.plot(100*sub["x_h2_energy"], sub["EINOx_g_per_kg"], marker="D", color="C3")
ax1.set_xlabel("Hydrogen energy fraction [%]")
ax1.set_ylabel("EINOx [g NO2 / kg fuel]", color="C3")
ax1.tick_params(axis="y", labelcolor="C3")
ax2 = ax1.twinx(); ax2.spines.right.set_visible(True); ax2.grid(False)
ax2.plot(100*sub["x_h2_energy"], sub["T_ad_K"], marker="s", color="C0")
ax2.set_ylabel(r"$T_{ad}$ [K]", color="C0")
ax2.tick_params(axis="y", labelcolor="C0")
ax1.set_title("NOx / flame-temperature trade-off vs H2 enrichment"); plt.show()""")

# --- validation ---
md(r"""## 5. Mechanism validation

GRI-Mech 3.0 is optimised mainly for natural gas. To confirm it is reliable
at the pure-hydrogen end of the blend range, the ignition delay and flame
speed of 100 % H$_2$ are recomputed with a dedicated detailed H$_2$/O$_2$
mechanism and overlaid. Close agreement justifies using a single mechanism
across the whole study.""")
code(r"""mi = pd.read_csv(os.path.join(DATA, "mech_comparison_ignition.csv"))
mf = pd.read_csv(os.path.join(DATA, "mech_comparison_flame.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, g in mi.groupby("mechanism"):
    axes[0].semilogy(1000.0/g["T0_K"], g["tau_ign_ms"], marker="o", label=label)
axes[0].set_xlabel(r"$1000/T_0$ [1/K]"); axes[0].set_ylabel(r"$\tau_{ign}$ [ms]")
axes[0].set_title("Ignition delay, 100% H2"); axes[0].legend()

for label, g in mf.groupby("mechanism"):
    g = g.sort_values("phi")
    axes[1].plot(g["phi"], g["S_L_cm_s"], marker="o", label=label)
axes[1].set_xlabel(r"$\phi$"); axes[1].set_ylabel(r"$S_L$ [cm/s]")
axes[1].set_title("Flame speed, 100% H2"); axes[1].legend()
plt.tight_layout(); plt.show()""")

# --- conclusions ---
md(r"""## 6. Conclusions

- **Ignition:** hydrogen markedly shortens the ignition delay, and the effect
  grows with temperature — beneficial for relight and flame stabilisation.
- **Flame speed:** $S_L$ rises strongly and non-linearly with hydrogen content
  (roughly an order of magnitude from pure CH$_4$ to pure H$_2$).
- **NOx:** the price of hydrogen is a steep, near-exponential rise in thermal
  NOx, driven by the higher flame temperature. Lean operation is the main
  lever to keep NOx acceptable.
- **Mechanism:** GRI-Mech 3.0 agrees closely with a detailed H$_2$/O$_2$
  mechanism for both ignition delay and flame speed at 100 % H$_2$, so it is
  an appropriate single mechanism for the whole blend range.

Overall, hydrogen enrichment improves ignition and burning characteristics
but demands lean, carefully temperature-controlled combustion to manage NOx —
the central trade-off for hydrogen in aero gas turbines.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out = "../MKWS_hydrogen_enrichment.ipynb"
import os as _os
out_abs = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), out))
with open(out_abs, "w") as f:
    nbf.write(nb, f)
print("Notebook written to", out_abs)
print("Cells:", len(cells))
