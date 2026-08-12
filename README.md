# Bounded distance preference in metro mobility

This repository contains the processed data and code used to estimate a bounded
Zipf-like distance relation in the metro systems of London, New York City,
Tokyo, Taipei and Tianjin.

For each 0.5 km distance bin, the data provide:

- `f_r`: total observed metro flow;
- `p_r`: number of represented directed station pairs;
- `pi_r = f_r / p_r`: flow per represented station pair.

The analysis searches every contiguous interval containing at least 12 bins.
For each candidate interval, it fits

```text
pi(r) proportional to r^(-alpha)
```

by discrete maximum likelihood and selects the interval with the smallest
Kolmogorov-Smirnov distance. Uncertainty and goodness of fit are evaluated with
800 parametric bootstrap replicates using a fixed random seed.

## Files

```text
analysis.py                         Complete analysis pipeline
reproduce.ipynb                    Short annotated reproduction notebook
data/metro_distance_profiles.csv   Model-ready f(r), p(r) and pi(r)
data/effective_sample_sizes.csv    OD-derived bootstrap sample sizes
results/metro_law_estimates.csv    Reproduced city-level estimates
Figures/plot_statistical_figures.py Migrated plotting code for Figures 2--4
Figures/plot_network_maps.py        Migrated plotting code for Figure 5
Figures/Figure_1.png                Conceptual figure exported from the manuscript
Figures/Figure_2.*                  Distance-preference curves and fitted windows
Figures/Figure_3.*                  Cross-city estimates and validity domains
Figures/Figure_4.png                Flow, pair availability and preference profiles
Figures/Figure_5_<City>.png         Short-, middle- and long-range OD networks
```

## Run

Install Python 3.10 or later and the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the complete analysis:

```bash
python analysis.py
```

The result table is written to `results/metro_law_estimates.csv`. A complete run
usually takes about one minute on a standard desktop.

Generate the figures after running the analysis:

```bash
python Figures/plot_statistical_figures.py
```

The statistical plotting script reads the existing data and result tables; it
does not repeat or modify the statistical estimation. Figure 1 is the assembled
conceptual figure as embedded in the current manuscript. Its exact composite
source file was not retained in the working project, so the manuscript export is
included without claiming full code reproducibility.

The Figure 5 code is a cleaned migration of the final map-rendering cell in
`Geo_Plots.ipynb`. Regenerating those maps requires the source-specific OD tables
and internet access for the Esri basemap:

```bash
python Figures/plot_network_maps.py --raw-data-dir /path/to/od_tables
```

The final manuscript outputs are included in `Figures/`, so the figures can be
inspected without redistributing the raw OD records.

## Figure provenance

| Paper figure | Included output | Migrated source |
|---|---|---|
| Figure 1 | `Figure_1.png` | Composite exported from the current manuscript; the original assembled source was not retained |
| Figure 2 | `Figure_2.png`, `Figure_2.pdf` | Final distance-profile cell in `MLE-KS_0708.ipynb`, consolidated in `plot_statistical_figures.py` |
| Figure 3 | `Figure_3.png`, `Figure_3.pdf` | Final cross-city plotting script, consolidated in `plot_statistical_figures.py` |
| Figure 4 | `Figure_4.png` | Final distribution cell in `Geo_Plots.ipynb`, consolidated in `plot_statistical_figures.py` |
| Figure 5 | Five city PNG files | Final network-map cell in `Geo_Plots.ipynb`, consolidated in `plot_network_maps.py` |

The files in `Figures/` are copies of the outputs used in the current manuscript,
renamed to match the manuscript numbering. Figure 4 regeneration uses the
released binned `f(r)` values rather than rebuilding the same bins from raw OD
records. Figure 5 retains the original raw-OD dependency because station
coordinates are required for the maps.

## Data scope

The released table contains aggregated analytical inputs rather than passenger-
level records. `p_r` reproduces the represented station-pair counts used in the
current manuscript analysis. It is not yet a reconstruction of every possible
directed station pair including zero-flow pairs. A separate audit also identified
incomplete pair coverage across cities and duplicate pair records in the Tokyo
source. The repository therefore reproduces the current results exactly, while a
future complete-pair reconstruction may update the numerical estimates.

Raw origin-destination records are not redistributed because of their size and
source-specific access conditions. Only the Kish effective sample sizes required
by the bootstrap are included.

## Citation

Please cite the associated article when using these data or methods. Full citation
information will be added after publication.
