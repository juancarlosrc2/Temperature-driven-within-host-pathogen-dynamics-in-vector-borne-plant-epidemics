# A mechanistic framework linking within-host pathogen progression to vector-mediated transmission under climate forcing

This repository contains the source code, processed climate data, numerical outputs, and figure-generation notebooks associated with the manuscript:

> Juan Carlos Rodríguez-Cabanillas, Manuel A. Matías, and Àlex Giménez-Romero. **A mechanistic framework linking within-host pathogen progression to vector-mediated transmission under climate forcing.**

## Overview

The study develops a stage-structured host–vector epidemiological model in which temperature affects transmission through within-host pathogen progression. Infected hosts move through ordered infection stages with increasing pathogen load and infectiousness. Temperature-dependent pathogen growth advances hosts through these stages, whereas cold-induced pathogen decline moves them backwards and may lead to recovery.

The framework is parameterized for Pierce's disease of grapevine, caused by *Xylella fastidiosa*. The repository contains analyses under:

- constant temperature;
- idealized seasonal temperature forcing;
- seasonal forcing with Gaussian temperature perturbations; and
- empirical ERA5-Land temperature series from locations with reported Pierce's disease or *X. fastidiosa* subsp. *fastidiosa* ST1.

## Repository structure

```text
.
├── vector_borne_functions.py
├── dynamics.ipynb
├── Results _.ipynb
├── R_0_constant_temperature/
├── R_0_sinusoidal_temperature/
├── adding_noise/
├── real_data/
├── animation/
├── figures/
├── requirements.txt
├── LICENSE
└── README.md
```

### Main files

- `vector_borne_functions.py` — core thermal-response functions, host–vector model equations, and numerical integration routines.
- `dynamics.ipynb` — exploratory simulations of the stage-structured epidemic dynamics.
- `Results _.ipynb` — main notebook used to assemble the analyses and generate the manuscript figures. Run this notebook from the repository root so that relative paths resolve correctly.

### Analysis directories

- `R_0_constant_temperature/` — constant-temperature calculations of the basic reproduction number, final epidemic size, and epidemic timing.
- `R_0_sinusoidal_temperature/` — analyses under idealized seasonal temperature forcing, including precomputed parameter-sweep outputs.
- `adding_noise/` — robustness analyses in which independent Gaussian perturbations are added to the seasonal temperature series.
- `real_data/` — empirical-temperature analysis based on ERA5-Land data, including the processed temperature series and summary statistics.
- `animation/` — notebook and exported animation illustrating progression, stasis, and regression across infected-host stages.
- `figures/` — figure files generated for the manuscript.

Precomputed arrays used in the manuscript are included so that the figures can be reproduced without rerunning the most computationally intensive parameter sweeps.

## Installation

The analyses were developed with Python 3.12. A virtual environment is recommended.

```bash
git clone https://github.com/SrJuan212/Temperature-driven-within-host-pathogen-dynamics-in-vector-borne-plant-epidemics.git
cd Temperature-driven-within-host-pathogen-dynamics-in-vector-borne-plant-epidemics

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some analysis and visualization notebooks additionally use `numba`, `seaborn`, `geopandas`, and `cartopy`. Install them when running the complete workflow:

```bash
pip install numba seaborn geopandas cartopy
```

## Reproducing the manuscript figures

Start Jupyter from the repository root:

```bash
jupyter lab
```

Then open `Results _.ipynb` and run the cells in order. The notebook reads the archived numerical outputs from the analysis directories and generates the main manuscript figures.

The lower-level scripts and notebooks in `R_0_constant_temperature/`, `R_0_sinusoidal_temperature/`, `adding_noise/`, and `real_data/` document the corresponding calculations and data-processing steps. Large parameter sweeps can require substantial computation time and memory.

## Model implementation

The core model is implemented in `vector_borne_functions.py`. It includes:

- the seasonal temperature function;
- the temperature-dependent pathogen growth function, `f(T)`;
- the cold-driven pathogen decline function, `g(T)`;
- the mapping from accumulated thermal progression to stage-specific infectiousness;
- progression and regression rates across infected-host stages;
- deterministic host–vector epidemic dynamics; and
- routines for constant, seasonal, noisy, and empirical temperature forcing.

The implementation tracks susceptible, infected, chronically infected, and removed hosts together with susceptible and infected vectors. Infected-host infectiousness depends on the host's within-host progression stage.

## Data

The model parameterization is based on previously published experimental data on temperature-dependent *Xylella fastidiosa* growth, symptom progression, and recovery in grapevine.

The file `real_data/selected_points_.nc` contains the processed ERA5-Land temperature series used in the empirical analysis. ERA5-Land data were obtained from the Copernicus Climate Change Service Climate Data Store. The corresponding site-level summary statistics are provided in `real_data/df_summary.csv`.

Users of the ERA5-Land data should also cite the original dataset and comply with the Copernicus Climate Data Store terms of use.

## Reproducibility

The repository contains:

- the model source code;
- processed temperature data used in the empirical analysis;
- parameter files;
- numerical outputs used in the manuscript;
- analysis and figure-generation notebooks; and
- final figure files.

A permanent archival version of the repository is available through Zenodo:

> **Zenodo DOI:** add DOI here

## Citation

Please cite the accompanying manuscript when using this code or data:

> Rodríguez-Cabanillas, J. C., Matías, M. A., and Giménez-Romero, À. *A mechanistic framework linking within-host pathogen progression to vector-mediated transmission under climate forcing*. Citation details to be added upon publication.

Please also cite the archived software release using the citation information provided by Zenodo.

## License

This repository is distributed under the GNU Affero General Public License v3.0. See `LICENSE` for the full terms.

## Contact

For questions about the model or analyses, contact:

**Àlex Giménez-Romero**  
Centro de Estudios Avanzados de Blanes (CEAB-CSIC)  
`alex.gimenez@csic.es`
