"""
run_all.py
==========
Master driver for the MKWS 2026 project. Runs the three computational
modules in sequence and reports timing.

    Module 1 -- ignition_delay.py  (constant-volume reactor)
    Module 2 -- flame_speed.py     (freely-propagating 1-D flame)
    Module 3 -- nox_emissions.py   (burned-gas reactor + equilibrium)

Module 2 (the 1-D flames) is by far the most expensive. If you only want a
quick check, run modules 1 and 3 alone, or run flame_speed.py in its
staged mode (see its docstring).

Usage
-----
    python run_all.py            # run every module
    python run_all.py --no-flame # skip the slow flame-speed module
"""

from __future__ import annotations

import sys
import time

import ignition_delay
import flame_speed
import nox_emissions


def main():
    skip_flame = "--no-flame" in sys.argv
    t0 = time.time()

    print("=" * 70)
    print("MKWS 2026 -- Hydrogen enrichment of methane-air combustion")
    print("=" * 70)

    print("\n[1/3] Ignition delay ...")
    t = time.time()
    ignition_delay.main()
    print(f"      module 1 finished in {time.time() - t:.1f} s")

    if not skip_flame:
        print("\n[2/3] Laminar flame speed ... (this is the slow one)")
        t = time.time()
        flame_speed.main()  # 'all' stage
        print(f"      module 2 finished in {time.time() - t:.1f} s")
    else:
        print("\n[2/3] Laminar flame speed ... SKIPPED (--no-flame)")

    print("\n[3/3] NOx emissions ...")
    t = time.time()
    nox_emissions.main()
    print(f"      module 3 finished in {time.time() - t:.1f} s")

    print("\n" + "=" * 70)
    print(f"All modules complete in {time.time() - t0:.1f} s.")
    print("Figures -> results/figures/   Data -> results/data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
