# Benchmarks

These benchmarks check how well node2vec2rank's ranking choices and the new significance test recover
nodes whose connectivity changes between two graphs. They also check how the methods behave when
nothing changes, on simulated networks, simulated expression and real single-cell networks. To
reproduce the network simulations (about 3 minutes on 4 cores):

```sh
pip install -e ".[plot]"
python benchmarks/simulations.py   # 20 replicates per setting -> results/*.csv
python benchmarks/figures.py       # -> results/*.png
```

## Setup

Every setting uses two graphs on 1,000 nodes. The "changed" nodes are 10% or 30% of the nodes, which
move to another community in the second graph. The 0% setting (no change) is used to check calibration
and degree bias. The four scenarios are:

| Scenario | Graphs |
|---|---|
| Binary, degree-corrected SBM | 4 blocks, Pareto(2.5) degree parameters, Bernoulli edges |
| Binary SBM | 4 blocks, no degree heterogeneity |
| Weighted, degree-corrected SBM | as above with Gaussian edge weights (sd 0.1) |
| Co-expression (WGCNA-style) | 150 samples of 5 latent module factors with heterogeneous loadings; 30% of genes in no module; adjacency \|cor\|^6 |

The demo network in `data/networks/demo` (1,000 nodes, 10 communities, where one community rewires
without changing degree) is included as an external check.

Methods compared, all computed from one joint embedding of 24 dimensions:

- **DeDi**: absolute degree difference.
- **n2v2r (default)**: Borda over dimensions 4, 6, …, 24 × {euclidean, cosine}, as in the paper.
- **euclidean only / cosine only**: the same Borda restricted to one metric.
- **ULSE**: the default Borda on the regularised unfolded Laplacian embedding.
- **elbow dimension**: both metrics at the single dimension picked by the Zhu–Ghodsi elbow.
- **degree-adjusted z**: the new `N2V2R.significance()`, which combines the default dimensions and
  metrics after removing each distance's trend with degree.
- **degree-adjusted z (elbow)**: `significance(dimensions="elbow")`.

## Findings

**1. Any n2v2r variant beats degree difference, as intended.** DeDi is close to random (AUROC
0.51–0.71) because moving between communities barely changes degree.

![AUROC](results/auroc.png)

**2. The embedding dimension matters most, and no automatic choice is safe.** At 10% change the
default Borda reaches AUROC 0.965–1.0. At 30% change it drops to 0.78–0.83 in the binary and
co-expression scenarios. The elbow dimension recovers 1.0 in all three SBM scenarios, because the
dimensions beyond the true rank are mostly noise and dilute the Borda. On the demo network, however,
the elbow sits at dimension 2 while the change lives in weaker dimensions (the spectrum has several
gaps). There the elbow finds none of the changed community, while the default Borda recovers 68%.
Averaging over dimensions 4–24 is therefore a reasonable hedge, and it remains the default. The elbow
is offered as an option (`embed_dimensions: "auto"`, `significance(dimensions="elbow")`) and should be
checked against the scree plot (`plotting.plot_scree`).

![Demo](results/demo.png)

**3. Raw distances are degree-biased, in opposite directions for the two metrics.** With no change,
Euclidean rankings correlate positively with degree (Spearman up to 0.98 in co-expression networks,
0.59 in degree-corrected SBMs): hubs have large embedding norms, so their noise is large in absolute
terms. Cosine rankings correlate negatively (down to −0.76), because low-degree nodes have noisy
directions. The default Borda over both metrics partly cancels these biases, but it keeps whichever
dominates (−0.48 to 0.43). This justifies mixing the metrics better than either alone, but not fully.
The degree-adjusted z removes the bias in every scenario (|rho| ≤ 0.04) and ranks as well as or better
than the default Borda in 7 of 8 settings (it loses 0.03 AUROC in co-expression with 30% change). It
also recovers 59% of the demo's changed community, against 68% for the default Borda.

![Degree bias](results/degree_bias.png)

**4. The p-values are valid but conservative, and powerful only when the signal is concentrated.**
With no change, 2.4–3.9% of nodes have p < 0.05, and no scenario had a false call at q < 0.1 in any
of its 20 replicates (see the next section for harder co-expression nulls). Power at q < 0.1 with 10%
change is 1.0 with the elbow dimension (0.78 for co-expression). With the default dimensions it
is 0.95 for weighted graphs, 0.26 for co-expression and near 0 for binary graphs, where the noisy high
dimensions dilute the combined score. With 30% change power drops for both variants (at best 0.70,
for weighted graphs with the elbow dimension). The empirical null assumes that most nodes
do not change, so it absorbs part of a change that affects a third of the graph; calls stay
conservative (no false discoveries) rather than inflated.

![Calibration](results/calibration.png)

**Calibration of the significance test.** An independent review found that the test is conservative
and that the elbow variant gave false calls on co-expression networks in which many genes belong to
no module. `benchmarks/calibration.py` reproduces its null scenarios with 30 replicates each:
degree-corrected SBMs with Pareto or lognormal degrees, a weighted one, and co-expression networks
from 60 or 150 samples with 40% of genes in no module.

| Null scenario (30 replicates) | default dims, p < 0.05 (spline) | elbow, p < 0.05 (spline) | elbow, replicates with a call at q < 0.1: quadratic trend | spline trend |
|---|---|---|---|---|
| DCSBM, Pareto degrees | 1.4% | 2.2% | 0% | 0% |
| DCSBM, lognormal degrees | 1.8% | 1.4% | 0% | 0% |
| weighted DCSBM | 1.6% | 1.7% | 0% | 0% |
| co-expression, 150 samples | 1.5% | 3.5% | 13% | 13% |
| co-expression, 60 samples | 1.7% | 3.4% | **77%** | **3%** |

- The false calls came from the degree trend, not from the tails. The lowest-degree genes (those in no
  module) have embeddings close to zero, so their distances rise steeply at the bottom of the degree
  range. The quadratic trend bent at the ends and missed that rise. The trend is now a natural cubic
  spline with knots at the degree quartiles. It has as many parameters as the quadratic, so a changed
  group that shares a degree keeps its signal (a unit test checks this), but it is linear beyond the
  outer knots. It removes almost all of those false calls and changes the ranking AUROCs by at most
  0.01 elsewhere. `trend="polynomial"` in `empirical_null_test` keeps the old fit.
- The conservativeness was not fixed. The log distances are left-skewed, so the combined scores have a
  thinner upper tail than a normal distribution, and 1.4–3.9% of null nodes have p < 0.05. Three
  corrections were tried on these nulls, on the network simulations and on the resampled locCSN null:
  a Box-Cox transform per distance chosen to make the scores symmetric, a Yeo-Johnson transform of the
  combined score, and a scale estimated from the upper half of the scores. Each moved the rate at
  p < 0.05 toward 5%, but the first two gave false calls. The Box-Cox version made about 9 calls per
  replicate on the resampled locCSN null, and the Yeo-Johnson version about 9 per replicate with the
  elbow dimension on the simulated co-expression null. These exploratory runs are not in the repository. The upper-half scale lost all power when 30% of nodes changed. The conservative
  version is kept: its calls can be trusted, but it misses changes that a calibrated test would find.
  The permutation test (below) is calibrated when the samples are available.

**5. The regularised Laplacian embedding (ULSE) brings no consistent gain.** Its AUROC is within ±0.02
of UASE everywhere. It remains available as `embedding_method: "ulse"` for networks with extreme hubs,
but it is not the default.

**6. A simulation-based (parametric bootstrap) null was tried and not adopted.** Re-embedding graphs
simulated from the fitted low-rank model gives per-node p-values whose calibration depends heavily on
the assumed rank. With rank 8 instead of the true 3, 10% of null nodes had p < 0.05; at the true rank
the share was 1.5%. It is also expensive at 20,000 nodes and needs a noise model for weighted networks.

## Simulated expression data

`benchmarks/expression.py` (about 10 minutes) simulates gene expression rather than networks. It uses
`node2vec2rank.simulate`: 1,000 genes, 100 samples per condition, 5 modules, and 10% of genes rewired
(module switch, loss or gain). Another 10% of genes are differential-expression decoys, whose mean
shifts but whose co-expression does not. Networks are built as the paper does (`|cor|^6`), with 10
replicates. Three tests are compared: `significance()` with all dimensions and with the elbow
dimension, and the new sample-label `permutation_test` (100 permutations).

**Calibration with shuffled sample labels.** Pooling both conditions and splitting the samples at
random leaves no true difference between the groups. The permutation test is close to uniform: 5.4%
of genes have p < 0.05 (3.9–8.6% per replicate), with one false call at q < 0.1 in 10 replicates. The
empirical null is conservative (2.2% all dimensions, 3.6% elbow), with no false calls in 10
replicates. A first version of the permutation test standardised distances by
their permutation mean and sd. It failed on one random split, where two near-equal singular values
made a single dimension unstable, and called 202 genes. Its scores are now rank-based, so no single
dimension can dominate.

![Expression calibration](results/expression_calibration.png)

**Power against the known rewiring.** The tests answer different questions:

- The permutation test calls 82% of rewired genes. It also calls 25% of their module partners,
  whose co-expression neighbourhood genuinely changed when a gene joined or left their module. Only
  2% of genes that are never in a module, and 2% of differential-expression decoys, are called.
- The empirical null calls only the genes that stand out from genes of similar degree: 53% of
  rewired genes with the elbow dimension and 24% with all dimensions. It calls almost no partners or
  decoys (at most 0.4%).
- For ranking the rewired genes, AUROC is 0.98 for the elbow z-score, 0.96 for the all-dimension
  z-score, 0.91 for the default Borda and 0.90 for the permutation z-score. The permutation score
  ranks partners high too.

![Expression calls](results/expression_calls.png)

In practice, use `permutation_test` when the samples are available and "did this gene's co-expression
change at all" is the question. Use `significance()` when only the networks are available, or to
prioritise the most rewired genes.

## Real single-cell networks

`benchmarks/real_data.py` (about 2 minutes) uses the locCSN networks of autism (ASD) and control
(CTL) brain cells that used to be in `data/networks/locscn` (942 genes; read from git history, or from
`--data-dir`). Each edge weight is the fraction of cells in which the edge is present, over 211 control
and 238 ASD cells. The per-cell networks are not in the repository, so the permutation test cannot be
run, and a null has to be built from the averaged networks. The resampled null redraws both networks
from the pooled edge frequencies, Binomial(cells, pooled frequency) / cells for every edge. It keeps
the degree structure and edge-level sampling noise, while no gene differs between the groups. It
treats edges as independent across cells, so it is less noisy than a cell-level resampling would be.
The spike-ins swap the neighbourhoods of 5% of genes in pairs, either between genes adjacent in
degree (degree-preserving) or between random genes, on top of the resampled null or the real pair.
There are 20 replicates of each.

![Real data](results/real_data.png)

**1. The elbow is too aggressive on real networks.** The spectrum has one dominant singular value (480,
then 43, 26, 20) and a long smooth tail, so the elbow is dimension 2. The elbow ranking agrees poorly
with the default one (Spearman 0.35, 12 genes in common in the top 50). On the real pair it also
recovers spike-ins much worse: AUROC 0.68 against 0.88 for the default Borda with random swaps, and
0.54 against 0.69 with degree-matched swaps. On the resampled null, where the only structure is
the dominant one, the two are close (0.95 against 0.96). This is the demo network's failure again, and
it supports keeping the averaged dimensions as the default.

**2. The degree biases seen in simulations hold on real data.** Under the resampled null, Euclidean
rankings correlate with degree at 0.57 and cosine rankings at −0.41. The default Borda is at 0.12, and
the degree-adjusted z at 0.08. On ASD vs CTL itself the cosine bias is −0.65, the default Borda −0.26
and the elbow Borda 0.42, while the z-scores stay at 0.06–0.09.

**3. The degree-adjusted test is calibrated on the resampled null.** With all dimensions, 4.3% of genes
have p < 0.05 (at most 5.4% in a replicate), and two genes were called at q < 0.1 in 20 replicates. The
elbow variant is conservative (0.8%, no calls). The excess of p-values near 1 comes from genes that
change less than their degree peers, which only makes the one-sided test conservative. Genes isolated
in both networks (23 here) are now left untested by `significance()`, instead of entering the
degree trend with a zero distance.

**4. A naive null that swaps edges between the networks is not valid per gene.** Swapping every edge's
weight between ASD and CTL with probability ½ keeps the size of each gene's edge differences and only
randomises their direction. The genes that differ most in the real data therefore keep large
distances, and the same 8–15 genes are called in every replicate (6.5% of genes with p < 0.05). It is
included in the results as a warning, not as a calibration check.

**5. ASD vs CTL has no significant gene.** No gene reaches q < 0.1 with either variant, and 3.6% of
genes have p < 0.05 (2.9% with the elbow), which is close to what the null gives. The top of the
default ranking is led by NLGN4Y (p = 0.0007, q = 0.66) and USP9Y (p = 0.002), two Y-chromosome genes.
NLGN4Y's degree drops from 270 in CTL to 112 in ASD. This looks like a difference in the sex
composition of the donors or cells, rather than autism biology. This is inferred from the gene names;
the donor metadata is not in the repository. The next genes (STXBP1, CAMK4, CNTN3, SCN8A, GRIN2B) are
neuronal genes with nominal p-values between 0.008 and 0.03.

**6. The real differences dwarf the resampled noise.** Spike-ins on the resampled null are easy (AUROC
0.93–1.0 for the n2v2r rankings, power at q < 0.1 of 0.83–0.91 for the z-score). On the real pair, the same spike-ins rank at
0.69–0.88 and are almost never called. The real ASD–CTL differences are much larger than the edge-level
sampling noise of the averaged networks. They include biology, donor effects and the within-cell
correlation of edges that the resampled null ignores. So the resampled null checks the degree
adjustment, but it cannot say whether the ASD–CTL differences exceed what two random groups of cells
would show. That needs the per-cell networks and `permutation_test`.

![Real data spike-in](results/real_data_spike_in.png)

The hdWGCNA cell-cycle networks used in the paper are on Zenodo (10.5281/zenodo.10558426) and are
not in the repository, so they were not tested.

## Pathway tests on single cells: HeLa cell cycle

`benchmarks/cell_cycle.py` uses the HeLa S3 scRNA-seq of
[Revelio](https://github.com/danielschw188/Revelio) (1,564 cells, WT and Ago2KO batches), the cells
behind the paper's cell-cycle networks. Phases are assigned with Revelio's marker-gene method (87
outlier cells dropped) and merged into G1 (M/G1 + G1/S, 838 cells), S (258), G2 (186) and M (195).
Networks are built hdWGCNA style from the 2,000 most variable genes: metacells (k = 25, at most 10
shared cells) per phase, signed WGCNA adjacency with soft power 10 (the lowest with scale-free fit
above 0.8). For the paper's sequential comparisons, the Reactome library (534 sets with 5–500 genes
in the data, 64 of them cell-cycle sets by name) is tested with GSEA prerank on the n2v2r Borda
ranking and on absDeDi, as in the paper's notebook, and with `gene_set_test` (1,000 cell-label
permutations within batch) for n2v2r and for DeDi. As a null control, the G1 cells of each batch are
split at random into two halves.

| comparison | method | pathways at FDR 0.1 | cell-cycle pathways at FDR 0.1 | AUROC of cell-cycle pathways |
|---|---|---|---|---|
| G1 → S | GSEA prerank, n2v2r | 112 | 38 | 0.83 |
| | GSEA prerank, DeDi | 74 | 34 | 0.84 |
| | permutation test, n2v2r | 146 | **59** | **0.91** |
| | permutation test, DeDi | 72 | 31 | 0.87 |
| S → G2 | GSEA prerank, n2v2r | 75 | 18 | 0.54 |
| | GSEA prerank, DeDi | 32 | 22 | 0.59 |
| | permutation test, n2v2r | 135 | **36** | **0.72** |
| | permutation test, DeDi | 38 | 16 | 0.65 |
| G2 → M | GSEA prerank, n2v2r | 153 | 27 | 0.57 |
| | GSEA prerank, DeDi | 90 | 20 | 0.61 |
| | permutation test, n2v2r | 104 | **34** | **0.74** |
| | permutation test, DeDi | 0 | 0 | 0.62 |
| **null: G1 halves** | GSEA prerank, n2v2r | **23** | 11 | 0.52 |
| | GSEA prerank, DeDi | **12** | 8 | 0.67 |
| | permutation test, n2v2r | 0 | 0 | 0.47 |
| | permutation test, DeDi | 0 | 0 | 0.55 |

Full numbers: `results/cell_cycle.csv`; top sets per comparison and method:
`results/cell_cycle_top_sets.csv`.

**1. GSEA prerank on network rankings is not calibrated on real data.** Two random halves of the same
G1 cells give 23 Reactome pathways at FDR 0.1 with the n2v2r ranking and 12 with DeDi, led by
REACTOME_CELL_CYCLE, M_PHASE and MITOTIC_PROMETAPHASE at q ≈ 0: the pathways the paper reports for
the real comparisons. They most likely come out because their genes are strongly co-expressed
(cells labelled G1 still spread along the cycle), not because anything differs. The cell-label permutation
test calls nothing on the same split, in line with the simulations (no false call at FDR 0.1 in 12
null replicates; nominal p < 0.05 for 4–7% of sets).

**2. With a calibrated test, n2v2r clearly beats DeDi.** n2v2r calls more cell-cycle pathways in
every comparison (59 vs 31, 36 vs 16, 34 vs 0) and ranks them higher (AUROC 0.91/0.72/0.74 vs
0.87/0.65/0.62). Under GSEA prerank the two look similar, because both are dominated by the same
co-expression artefact. At G2 → M, DeDi finds nothing, while n2v2r's top sets are the expected ones
(condensation of prometaphase chromosomes, G2/M phases, M phase, APC/C degradation, PLK1 at G2/M).

**Caveats.** Phases are assigned from cell-cycle marker genes, so cell-cycle pathways are the easiest
positives (every gene is z-scored within phase and batch before building the networks, so mean
expression differences do not enter them). The cell-cycle label is a regular expression on Reactome
names, so "pathways at FDR 0.1" also includes real, non-cell-cycle changes between phases and
is not a false-positive count. Cells of one cell line are treated as exchangeable within batch; with
several donors, shuffle donors instead. The paper's own hdWGCNA networks (Zenodo) could not be
downloaded here, so these networks follow the hdWGCNA recipe but are not the same files, and the
merge of Revelio's five phases into four is an assumption.

### Which calls are true? Planted changes and split halves

"More cell-cycle pathways" is only a proxy: the cell-cycle label comes from pathway names, and a
pathway without it is not necessarily false. `benchmarks/cell_cycle_validation.py` adds two checks
with a real answer (1,000 permutations; about 75 minutes on 4 cores).

**Planted changes in real cells.** The G1 cells of each batch are split at random, so nothing differs
between the halves. In the second half, 30% or 60% of the genes of 6–8 disjoint Reactome sets
(15–100 genes) lose their co-expression: each gene's values are shuffled across the cells of a batch,
which keeps its expression level. Planted sets are the true positives. Sets that contain none of the
shuffled genes are counted as negatives (sets that share some are left out). Three replicates per
strength (`results/cell_cycle_spike_in*.csv`):

| method | recall at FDR 0.1 (30% / 60%) | calls among negatives (30% / 60%) | AUROC planted vs negatives | median rank of planted sets |
|---|---|---|---|---|
| GSEA prerank, n2v2r | 0.11 / 0.25 | 5.0 / 6.0 | 0.68 / 0.67 | 68 / 35 |
| GSEA prerank, DeDi | 0.11 / 0.21 | 2.7 / 0.3 | 0.69 / 0.79 | 57 / 25 |
| permutation test, n2v2r | 0.06 / **0.42** | 1.3 / 0.7 | **0.83 / 0.95** | **20 / 5** |
| permutation test, DeDi | 0.06 / 0.38 | 2.7 / 0.7 | 0.80 / **0.98** | 28 / 5.5 |
| fast test, n2v2r | 0.15 / 0.29 | 3.7 / 2.0 | 0.81 / 0.91 | 34 / 11 |
| fast test, DeDi | 0.06 / 0.29 | 2.0 / 0.3 | 0.77 / 0.97 | 34 / 7 |

The "negatives" are not a clean null: when a gene loses its co-expression, its partners lose edges
too, so sets of partners do change in network terms. Even so, GSEA on the n2v2r ranking makes the most
of these calls and ranks the planted sets far lower. With the permutation test, the planted sets rank
near the top (median 5th of about 140–250 sets at 60%). At 30% every method has little power at FDR
0.1, but the permutation tests still rank the planted sets better (AUROC 0.80–0.83 vs 0.68–0.69). With
this loss-of-co-expression change, n2v2r and DeDi perform similarly under a calibrated test. The
larger n2v2r advantage in the phase comparisons likely comes from changes that are not simple degree
losses.

**Split-half reproducibility.** The cells of each phase are split in two, and each comparison is run
on both halves (metacells of 10 cells, at most 5 shared, since half a G2 or M phase is about 90 cells)
(`results/cell_cycle_reproducibility*.csv`):

| comparison | method | Spearman of set scores | calls (half 1 / half 2 / both) | Jaccard of calls | top-20 overlap |
|---|---|---|---|---|---|
| G1 → S | GSEA prerank, n2v2r | 0.69 | 97 / 90 / 61 | 0.48 | 16 |
| | GSEA prerank, DeDi | 0.28 | 13 / 33 / 10 | 0.28 | 9 |
| | permutation test, n2v2r | 0.65 | 142 / 125 / 104 | **0.64** | **17** |
| | permutation test, DeDi | 0.02 | 0 / 0 / 0 | – | 0 |
| | fast test, n2v2r | 0.72 | 139 / 136 / 117 | **0.74** | 13 |
| | fast test, DeDi | −0.05 | 0 / 2 / 0 | 0 | 0 |
| G2 → M | all methods | −0.08 to 0.30 | GSEA 9–55 per half, tests 0–7 | ≤ 0.21 | ≤ 9 |

With half the cells, n2v2r's G1 → S calls reproduce well under both calibrated tests (104–117 sets
called in both halves), while DeDi finds nothing. Reproducibility alone does not prove the calls are
true: GSEA's artefacts are reproducible too, since co-expression is a stable property of the cells.
For G2 → M, halves of about 90 cells are too few for any method to be reproducible, so this
comparison is not informative.

### A faster test

`node2vec2rank.fast_gene_sets.fast_gene_set_test` gets its null from a formula instead of hundreds of
permutations, in the spirit of CAMERA. It uses the degree-adjusted z-scores of `significance()` as node
scores. The correlation between node scores is modelled as a function of expression correlation, and
both are learnt from about 10 label permutations. It supports the same `strata` and `ranking` options.
On HeLa it takes about 20 seconds per comparison, against 6–12 minutes for 1,000 permutations.

Checks:
- No calls on the HeLa G1 null split.
- On 10 simulated null splits with batch effects (strata by batch), no false calls at FDR 0.1 with
  n2v2r and 3 single calls with DeDi; nominal p < 0.05 for 3.8–5.5% of sets.
- Phase comparisons (`results/cell_cycle_fast.csv`): 53, 6 and 23 cell-cycle pathways at FDR 0.1 for
  G1 → S, S → G2 and G2 → M, against 59, 36 and 34 for the permutation test.
- Planted changes: lower recall and a worse median rank than the permutation test (table above).

It asks a narrower question, whether a set's genes changed more than genes of similar degree, and it
relies on the correlation model. Use it to explore; use `gene_set_test` for the final calls.

## Limitations

The simulations are small (1,000-node), two-graph settings with community-switch changes. Real
regulatory and co-expression networks have more gradual changes, larger size and no clean rank. The
simulated-expression benchmark covers the sample-level null, but the locCSN networks come without
their cells. The shuffled-label calibration on real per-cell data is covered by the HeLa null split above; per-sample (bulk) data remain untested.
