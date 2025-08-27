# Re-Entry Flip-Flop Grids (RFFG) — Event Synchronization for aw4null

**Short idea (TL;DR):**
This paper proposes **Re-Entry Flip-Flop Grids (RFFGs)** — a logic-based, decentralized method to **synchronize discrete events** (edges, triggers, packet boundaries) across multiple, independent USB-driven instruments. The system, designed by **Lukas Jakubczyk**, targets **multi-oscilloscope measurements in the autowerkstatt4null (aw4null)** context and supports **both live (online) and offline** synchronization of recorded oscilloscope data.

---

## Authors & Roles

* **First Author:** **Lukas Jakubczyk** ([@lj440](https://github.com/lj440)) — [lukas.jakubczyk@thga.de](mailto:lukas.jakubczyk@thga.de)
* **Editor:** **Meihui Huang** ([@kathamatician](https://github.com/kathamatician)) — [meihui@nabla-b.engineering](mailto:meihui@nabla-b.engineering)

---

## Repository Overview

This repository hosts the source for the **RFFG synchronization paper** and its simulation code. It contains:

* `src/` with the LaTeX source (`main.tex`)
* `simulation/` with a Python reference simulation (`RFFG.sim.py`) and a notebook (`BeepSync.ipynb`)
* `assets/` for figures and supplementary material
* A top-level `Makefile` for building PDFs
* This `README.md`

```
.
├── assets
│   ├── nice_conference_submission.pdf
│   ├── Zeichnungen.zip
│   └── zeichnung.py
├── Makefile
├── README.md
├── simulation
│   ├── BeepSync.ipynb
│   ├── requirements.txt
│   └── RFFG.sim.py
└── src
    └── main.tex
```

---

## What’s in the Paper (very short)

* Introduces **Re-Entry Flip-Flop Grids (RFFGs)** as a distributed, logic-level synchronization layer.
* Achieves **emergent, swarm-like convergence** via three rules: **Stability, Adjustment, Oscillation**.
* Complements bit-level hardware schemes (PLLs/USB-inSync): RFFGs synchronize **events**, not raw bitstreams.
* Target use-case: **multi-oscilloscope automotive diagnostics** (aw4null), **online/offline** alignment of recorded data.

---

## Build Instructions (PDF)

To compile the paper PDF from `src/main.tex`:

```bash
make all
```

Fast draft (ignores missing graphics / placeholders):

```bash
make draft
```

The resulting PDF will be placed in the `build/` directory.

### Dependency

We recommend **TeX Live Full** (includes `latexmk` and common packages).

**Debian/Ubuntu:**

```bash
sudo apt update
sudo apt install texlive-full
```

**Arch:**

```bash
sudo pacman -S texlive-most
```

**macOS (Homebrew):**

```bash
brew install --cask mactex
# after install, ensure latexmk is on PATH (restart shell if needed)
```

---

## Simulation (generate figures / sanity checks)

The simulation demonstrates convergence behavior of RFFGs (random init, two-peaks, outside-window, oscillation regime).

1. Create a virtual environment and install deps:

```bash
cd simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the reference script:

```bash
python RFFG.sim.py
```

3. (Optional) Explore the notebook:

```bash
jupyter notebook BeepSync.ipynb
```

> The LaTeX project expects figures under `build/simulation_results/` (see `\graphicspath` in `main.tex`).
> If your pipeline doesn’t generate them automatically, run the simulation first, then `make all`.

---

## File Map

* `src/main.tex` — main LaTeX source of the paper
* `simulation/RFFG.sim.py` — reference simulation producing histograms/phase/temporal plots
* `simulation/BeepSync.ipynb` — interactive exploration
* `assets/` — supplementary drawings and the late-breaking abstract PDF
* `Makefile` — convenience targets: `all`, `draft`

---

## Reproducibility Notes

* If you change figure names/paths, keep them in sync with `\graphicspath{...}` in `main.tex`.
* For deterministic runs, set seeds inside `RFFG.sim.py`.
* Large figure sets can be generated offline and versioned (or stored via LFS if needed).

---

## Contact

* **Lukas Jakubczyk** — [lukas.jakubczyk@thga.de](mailto:lukas.jakubczyk@thga.de)
* **Meihui Huang** — [meihui@nabla-b.engineering](mailto:meihui@nabla-b.engineering)
