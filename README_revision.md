# Code for the revised Node2Vec2Rank paper

This is the node2vec2rank repository as used for the revision of the paper in Scientific Reports.
README.md describes the package. This file says which script gives each new or changed result in
the revised paper. The scripts are in `benchmarks/` and write their tables to `benchmarks/results/`
and their figures to `benchmarks/results/figures/`. The result files used in the paper are included.

## Setup

```sh
pip install -e ".[enrichment]"
pip install pecanpy pyreadr     # node2vec and DeepWalk; reading the Revelio .rda files
```

The HeLa scripts read the Revelio data folder
(https://github.com/danielschw188/Revelio/tree/master/data), given as `--revelio-dir`. The Reactome
gene sets are in `data/gene_set_libraries/`. `hela_original_networks.py` also needs the paper's four
hdWGCNA networks (G1.txt, S.txt, G2.txt, M.txt, from https://doi.org/10.5281/zenodo.10558426),
given as `--networks-dir`.

Scripts skip replicates that are already in their result file, so delete the file first to rerun
from scratch. Use `OMP_NUM_THREADS=1` when several runs share a machine.

## Results and scripts

| In the paper | Script | Result files |
|---|---|---|
| Fig. 3 | `paper_sims.py` | `paper_sims_*.csv` |
| Fig. S7, Table S1 (accuracy) | `competitor_replicates.py` | `competitor_replicates_*.csv` |
| Table S1 (stability, runtime) | `competitor_seeds.py` | `competitor_seeds_*.csv` |
| HeLa subsamples of 80% of the cells | `competitor_hela.py --part subsample` | `competitor_hela_subsample.csv` |
| Fig. S8 | `revision_sims.py --part ablation` and `--part ablation8` | `revision_ablation.csv`, `revision_ablation8.csv` |
| Top 20 genes and phase markers (Fig. 5 networks) | `hela_original_networks.py` | `hela_original_networks_markers.csv` |
| Fig. S15 | `hela_original_networks.py --dimensions` | `revision_hela_spectrum_original.csv`, `revision_hela_rankings_original.csv` |
| HeLa permutation test | `cell_cycle.py` | `cell_cycle*_paper-setting_power5_1000perm.csv` |
| Fig. S16 | `revision_hela.py --part grid` | `revision_hela_grid.csv` |
| Fig. S17 | `runtime.py` | `runtime.csv` |
| Table S2 | `two_truths.py` | `two_truths.csv`, `two_truths_mixed.csv` |
| 79% to 88% of accepted pairs (Methods, rejection sampling) | `model_validity.py` | `model_validity_paper_design.csv` |
| Figures of all the above | `revision_figures.py` | `figures/*.pdf` |

## Commands

`R` is the Revelio data folder and `N` the folder with the four hdWGCNA networks.

```sh
# Fig. 3 (50 graph pairs per setting)
for regime in naive random; do for n in 50 100 500 1000; do
  python benchmarks/paper_sims.py --regime $regime --nodes $n
done; done
python benchmarks/paper_sims.py --summarise

# Fig. S7 and Table S1
python benchmarks/competitor_replicates.py --part sims
python benchmarks/competitor_replicates.py --part planted --kind partner --replicates 10 --revelio-dir $R
python benchmarks/competitor_replicates.py --part planted --kind loss --replicates 10 --revelio-dir $R
python benchmarks/competitor_replicates.py --summarise
python benchmarks/competitor_seeds.py --dataset dcsbm
python benchmarks/competitor_seeds.py --dataset coexpression
python benchmarks/competitor_seeds.py --dataset hela --revelio-dir $R --runs 10 --plexi-runs 6
python benchmarks/competitor_seeds.py --summarise
python benchmarks/competitor_hela.py --part subsample --revelio-dir $R

# Fig. S8
python benchmarks/revision_sims.py --part ablation
python benchmarks/revision_sims.py --part ablation8

# Phase markers and Fig. S15, on the networks of Fig. 5
python benchmarks/hela_original_networks.py --networks-dir $N --revelio-dir $R
python benchmarks/hela_original_networks.py --networks-dir $N --revelio-dir $R --dimensions

# HeLa permutation test (1,000 permutations, networks with soft power 5)
python benchmarks/cell_cycle.py --revelio-dir $R --setting paper --paper-power 5 \
    --num-permutations 1000 --n-jobs 4 --methods gsea-n2v2r gsea-dedi perm-n2v2r perm-dedi \
    --comparisons G1-S G2-M --suffix _paper-setting_power5_1000perm

# Fig. S16, S17, Table S2 and the rejection-sampling check
python benchmarks/revision_hela.py --part grid --revelio-dir $R
python benchmarks/runtime.py
python benchmarks/two_truths.py
python benchmarks/model_validity.py

# Figures
python benchmarks/revision_figures.py
```

Each script describes its settings at the top of the file.
