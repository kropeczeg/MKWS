# Effect of hydrogen enrichment on the combustion of methane–air mixtures

**Computational Methods in Combustion (MKWS) — project, summer semester 2026**
Jan Kalak (333459)
Faculty of Power and Aeronautical Engineering (MEiL), Warsaw University of Technology

This project quantifies how blending hydrogen into methane changes three
combustion quantities that matter for an aero gas turbine combustor:

1. **Ignition delay time** `τ_ign`
2. **Laminar burning velocity** `S_L`
3. **NOx emission index** `EINOx` (and the adiabatic flame temperature behind it)

The hydrogen content is always specified as a **fraction of the fuel's
chemical energy** (not mole fraction), because that is the physically
meaningful variable for the decarbonisation question "what happens if I
replace *x*% of the fuel energy with hydrogen?".

All chemistry uses **GRI-Mech 3.0**, which contains the C1, H2/O2 and
nitrogen sub-mechanisms needed for these three quantities.

---

## Key finding

Hydrogen enrichment is a genuine trade-off. It shortens the ignition delay
and strongly raises the flame speed (both helpful for stability and
relight), but because it pushes the flame temperature up, the **thermal
(Zeldovich) NOx grows roughly exponentially** — at φ = 0.7, going from pure
methane to pure hydrogen raises EINOx by more than an order of magnitude
for only a ~190 K rise in flame temperature. Staying lean is the lever that
keeps NOx in check.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── MKWS_hydrogen_enrichment.ipynb       # guided notebook (loads results, replots)
├── src/
│   ├── common.py                        # blend definition, mixture setup, plot style
│   ├── ignition_delay.py                # Module 1 — constant-volume reactor
│   ├── flame_speed.py                   # Module 2 — 1-D freely-propagating flame
│   ├── nox_emissions.py                 # Module 3 — burned-gas NOx + flame temperature
│   ├── mechanism_comparison.py          # Validation — GRI-Mech vs detailed H2/O2
│   ├── build_notebook.py                # regenerates the .ipynb
│   └── run_all.py                       # master driver
├── results/
│   ├── data/                            # CSV output of every sweep
│   └── figures/                         # PNG figures
└── report/
    ├── MKWS_report_Kalak.tex            # LaTeX source (English)
    ├── MKWS_report_Kalak.pdf            # compiled report
    └── figures/                         # figures used by the report
```

## How to run

```bash
pip install -r requirements.txt

cd src
python run_all.py              # run all three modules
python run_all.py --no-flame   # skip the slow 1-D flame module
```

Individual modules can also be run on their own:

```bash
python ignition_delay.py       # ~1 min
python nox_emissions.py        # ~1 min
python flame_speed.py all      # several minutes (1-D flames)
```

The flame-speed module additionally supports **staged execution** so a
single sweep can be run at a time (useful on time-limited machines):

```bash
python flame_speed.py init     # write the CSV header
python flame_speed.py phi 0.0  # equivalence-ratio sweep, pure methane
python flame_speed.py phi 0.3  # ... 30 % H2 (energy)
python flame_speed.py h2       # hydrogen-fraction sweep
```

## Method summary

| Module | Cantera model | Quantity | Definition |
|--------|---------------|----------|------------|
| 1 | `IdealGasReactor` (const. V, adiabatic) | `τ_ign` | time of max d*T*/d*t* |
| 2 | `FreeFlame` (mixture-averaged transport) | `S_L` | unburned-gas inlet velocity |
| 3 | `equilibrate('HP')` + `IdealGasConstPressureReactor` | `EINOx`, `T_ad` | NO formed over a primary-zone residence time |
| val. | both models, 100 % H2 | `τ_ign`, `S_L` | GRI-Mech vs detailed `h2o2` |

A separate **mechanism validation** (`mechanism_comparison.py`) repeats the
ignition-delay and flame-speed calculations for pure hydrogen with a detailed
H2/O2 mechanism and confirms that GRI-Mech 3.0 agrees to within a few percent,
justifying its use across the whole blend range.

## The notebook

`MKWS_hydrogen_enrichment.ipynb` is a guided, presentation-ready walkthrough.
It reloads the pre-computed CSV results so it runs in seconds, reproduces all
figures, and includes optional cells that re-run individual calculations from
scratch.

```bash
jupyter notebook MKWS_hydrogen_enrichment.ipynb
# or regenerate it:  cd src && python build_notebook.py
```

## The report

`report/MKWS_report_Kalak.pdf` is the written report (in English), with the
LaTeX source in `report/MKWS_report_Kalak.tex`. It follows the structure
expected for the course (Introduction, State of the art, Model description,
Results, Conclusions) and embeds the figures from `report/figures/`.

To rebuild the PDF from source:

```bash
cd report
pdflatex MKWS_report_Kalak.tex
pdflatex MKWS_report_Kalak.tex   # second pass resolves references / page count
```

This requires a LaTeX distribution (e.g. TeX Live) with the `mhchem`,
`siunitx`, `fancyhdr` and `lastpage` packages.

## Notes / limitations

- GRI-Mech 3.0 is optimised mainly for natural gas; for pure-hydrogen flames
  a dedicated H2 mechanism would be slightly more accurate, but GRI-Mech
  keeps the whole blend range on a single consistent mechanism.
- NOx is computed in a homogeneous, perfectly-mixed sense; real combustors
  also produce NOx in stratified and prompt pathways not captured by a 0-D
  residence-time model.
- The 1-D flames use mixture-averaged (not multicomponent) transport as a
  speed/accuracy compromise.
