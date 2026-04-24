# petropt — The Python Library for Petroleum Engineering

[![PyPI](https://img.shields.io/pypi/v/petropt)](https://pypi.org/project/petropt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/petropt/petropt/actions/workflows/test.yml/badge.svg)](https://github.com/petropt/petropt/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/petropt/)

> Free, open-source petroleum engineering tools for Python. MIT licensed — use anywhere, no restrictions.

**[Online Calculators](https://tools.petropt.com)** · **[Documentation](https://petropt.com/docs)** · **[Advanced Analytics](https://petropt.com/suite)**

## Install

```bash
pip install petropt
```

## Quick Start

```python
import petropt

# Load a production dataset
df = petropt.datasets.load_sample_production()

# PVT: Standing bubble point pressure
pb = petropt.correlations.standing_bubble_point(api=35, gas_sg=0.65, temp=200)

# Decline: Arps hyperbolic forecast
import numpy as np
t = np.arange(0, 60)
q = petropt.correlations.arps_decline(qi=1000, di=0.05, b=0.5, t=t)

# IPR: Vogel inflow performance
ipr = petropt.correlations.vogel_ipr(qmax=1500, pr=3500, num_points=50)

# Petrophysics: water saturation from resistivity
sw = petropt.petrophysics.archie_sw(rt=20.0, phi=0.20, rw=0.05)

# RTA: Blasingame variables for type-curve matching
bg = petropt.rta.blasingame_variables(t, q, cum, pwf, pi=3000)

# Economics: NPV of a well
npv = petropt.economics.npv(cash_flows=[-500000, 80000, 70000, 60000], discount_rate=0.10, periods_per_year=1)
```

## What's Inside

### Correlations (50+ functions)
- **PVT** — Standing bubble point/Rs/Bo, Beggs-Robinson viscosity, Sutton & Piper pseudocritical, Hall-Yarborough & Dranchuk Z-factor, Lee-Gonzalez-Eakin gas viscosity, gas Bg/density/compressibility
- **Water PVT** — McCain Bw/viscosity/density, Osif compressibility, gas solubility
- **IPR** — Vogel, Fetkovich, Rawlins-Schellhardt (C&n), PI-based, composite (Vogel + PI)
- **Decline Curves** — Arps (exponential, hyperbolic, harmonic), cumulative production, EUR
- **Multiphase Flow** — Beggs-Brill pressure gradient (all flow patterns, any inclination)
- **Hydraulics** — Darcy-Weisbach with Churchill friction factor
- **Relative Permeability** — Corey, Brooks-Corey, LET models (oil/water/gas)
- **Material Balance** — Gas P/Z, Havlena-Odeh, OOIP estimation, drive indices
- **Volumetrics** — STOIIP, GIIP, drainage radius, recovery factor

### Petrophysics
- **Vshale** — linear, Larionov (Tertiary / older), Clavier
- **Porosity** — density, sonic (Wyllie / Raymer-Hunt-Gardner), neutron-density, effective
- **Water saturation** — Archie, Simandoux, Indonesian (Poupon-Leveaux)
- **Permeability** — Timur, Coates (NMR)
- **Pay** — cutoff-based net pay, NTG, pay-weighted averages, hydrocarbon pore thickness

### Rate Transient Analysis (RTA)
- **Transforms** — pressure-normalized rate, material balance time
- **Type curves** — Blasingame, Agarwal-Gardner, NPI variables
- **Flowing material balance** — contacted OOIP/OGIP from producing-well data
- **Linear flow** — sqrt(t) analysis, sqrt(k)·xf extraction from fracture wells

### Drilling
- **Well control** — hydrostatic, ECD, MAASP, kill mud weight, ICP/FCP (Driller's / Wait-and-Weight)
- **Hydraulics** — annular velocity, nozzle TFA, bit pressure drop
- **Tubulars** — Barlow burst with API 0.875 factor, full API 5C3 collapse (yield / plastic / transition / elastic regimes)

### Production Engineering
- **Liquid loading** — Turner (1969) and Coleman (1991) critical droplet-lift velocities
- **Flow assurance** — Katz hydrate formation temperature, Hammerschmidt methanol/MEG/ethanol inhibitor dosing
- **Piping** — API RP 14E erosional velocity
- **Choke flow** — Gilbert (1954) critical-flow rate correlation

### Economics
- NPV, IRR, payback period, oil & gas monthly cashflow (WI/NRI/severance/opex/capex)

### I/O
- **LAS files** — Read well logs to pandas DataFrame (wraps lasio)
- **Production CSV** — Auto-detect date/oil/gas/water columns from any naming convention

### Datasets
- **Petrobras 3W** — Labeled well events for fault detection (CC BY 4.0)
- **NPD Wellbore** — Norwegian Continental Shelf well metadata (NLOD)
- **Sample Production** — Bundled 2-well, 12-month dataset for demos

### Notebooks
- `student_intro.ipynb` — Your first petroleum dataset in Python
- `decline_analysis.ipynb` — Arps decline curve analysis tutorial

## Why petropt?

| | petropt | pyResToolbox | DIY scripts |
|---|---|---|---|
| **License** | **MIT** (use anywhere) | GPL (copyleft) | N/A |
| **Datasets** | Built-in (3W, NPD, samples) | No | No |
| **LAS reader** | Yes | No | Manual |
| **Student notebooks** | Yes | No | No |
| **Web calculators** | [tools.petropt.com](https://tools.petropt.com) | No | No |
| **Economics** | NPV, IRR, cashflow | No | Manual |

petropt is **MIT licensed** — use it freely in commercial projects, consulting, research, and AI applications. No copyleft restrictions.

## History

This repository was first initiated 8 years ago, in July 2018 (see [`master` branch](https://github.com/petropt/petropt/tree/master) for the original placeholder). Most of the code was developed and used privately over the years. We consolidated the collected work and released the open-source version on April 24, 2026.

## For AI Builders

```bash
pip install petro-mcp  # MCP server that wraps petropt for Claude/ChatGPT/LLMs
```

## Links

- [tools.petropt.com](https://tools.petropt.com) — Free online PE calculators
- [petropt.com/suite](https://petropt.com/suite) — Advanced analytics (Bayesian DCA, ML, anomaly detection)
- [petropt.com](https://petropt.com) — Groundwork Analytics

## Citation

If you use petropt in academic work, please cite:

```
Shirangi, M.G. (2026). petropt: The Python Library for Petroleum Engineering.
https://github.com/petropt/petropt
```

## License

MIT — see [LICENSE](LICENSE) for details.
