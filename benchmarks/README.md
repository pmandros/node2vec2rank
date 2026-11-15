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
- **degree-adjusted z**: the new `N2V2R.significance()`. It removes each distance's trend with degree,
  averages the scores within windows of two neighbouring dimensions (both metrics) and combines the
  windows with a Cauchy combination test.
- **degree-adjusted z (mean)**: `significance(combine="mean")`, which averages all default dimensions
  and metrics instead (the first version of the test).
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
The degree-adjusted z removes the bias in every scenario (mean |rho| ≤ 0.04) and ranks as well as or
better than the default Borda in 7 of 8 settings (it loses 0.03 AUROC in co-expression with 30%
change). It also recovers 60% of the demo's changed community, against 68% for the default Borda.

![Degree bias](results/degree_bias.png)

**4. The p-values are valid but conservative, and combining dimensions by windows keeps most of the
signal.** With no change, 1.5–1.8% of nodes have p < 0.05 (2.6–3.6% with the mean over dimensions,
2.4–3.9% with the elbow dimension), and no variant made a false call at q < 0.1 in any of the 20
replicates of any scenario (see the next section for harder co-expression nulls). Power at q < 0.1
with 10% change:

| Scenario | all dimensions, Cauchy (default) | all dimensions, mean | elbow dimension |
|---|---|---|---|
| Binary, degree-corrected SBM | 0.45 | 0.02 | 1.00 |
| Binary SBM | 0.87 | 0.00 | 1.00 |
| Weighted, degree-corrected SBM | 0.96 | 0.95 | 1.00 |
| Co-expression | 0.18 | 0.26 | 0.79 |

In the binary SBMs the change lives in the first few dimensions. Averaging all 22 combinations of
dimension and metric dilutes it with the noise of the higher dimensions, so the mean calls almost
nothing. The Cauchy combination is dominated by the strongest window, so the signal survives, and the
windows keep a single unstable dimension from dominating. In co-expression networks the change is
spread over many dimensions, and the mean is somewhat more powerful. The z-score ranking is the same
for both (AUROC within 0.01). With 30% change power drops for every variant (at best 0.70, for
weighted graphs with the elbow dimension; 0.14 with the Cauchy combination and 0 with the mean). The
empirical null assumes that most nodes do not change, so it absorbs part of a change that affects a
third of the graph; calls stay conservative (no false discoveries) rather than inflated.

Two details were needed to keep the windows calibrated. First, a single window is more sensitive than
the mean to how well the degree trend fits the largest hubs. The trend uses the rank of each node's
degree, which squeezes hubs of very different degree together, so above the 95th percentile the
position now grows with log degree instead (`log_tail` in `covariate_adjusted_zscores`). Without it
the weighted SBM null gave false calls in 20% of replicates. Second, windows of two dimensions work
better than single dimensions, which gave false calls when two near-equal singular values made one
cosine distance unstable, and better than windows of three, which lost most of the binary SBM gain.
Disjoint blocks of dimensions (1–4, 5–8, ...) were also tried; the noisy high blocks swamped the
signal. These exploratory runs are not in the repository.

Neither combination finds pure hub changes. In the mixed hub-and-community simulations of the
two-truths analysis (4-block degree-corrected SBM, 2,000 nodes, 5% of nodes doubling or halving their
degree parameter), both made no or almost no calls (at most 3% of the changed nodes) with the default
distances or with the radial (norm) distance, while the same test on the degree difference called 65%
of hub changes in Poisson-weighted graphs (0% in binary ones), with no false calls in either case.
A node that only gains or loses connections within its community moves along its own direction, and
the degree adjustment compares it with nodes of its new degree, which absorbs most of that move.

![Calibration](results/calibration.png)

**Calibration of the significance test.** An independent review found that the test is conservative
and that the elbow variant gave false calls on co-expression networks in which many genes belong to
no module. `benchmarks/calibration.py` reproduces its null scenarios with 30 replicates each:
degree-corrected SBMs with Pareto or lognormal degrees, a weighted one, and co-expression networks
from 60 or 150 samples with 40% of genes in no module. In every replicate both graphs come from the
same model, so any call is false. At q < 0.1, Benjamini-Hochberg allows a call in up to 10% of such
replicates.

| Null scenario (30 replicates) | p < 0.05: Cauchy | mean | replicates with a call at q < 0.1: Cauchy | mean | elbow, quadratic trend | elbow, spline trend |
|---|---|---|---|---|---|---|
| DCSBM, Pareto degrees | 0.5% | 1.4% | 0% | 0% | 0% | 0% |
| DCSBM, lognormal degrees | 0.8% | 1.8% | 0% | 0% | 0% | 0% |
| weighted DCSBM | 0.6% | 1.6% | 3% | 0% | 0% | 0% |
| co-expression, 150 samples | 1.2% | 1.5% | 13% | 0% | 13% | 13% |
| co-expression, 60 samples | 0.9% | 1.7% | 0% | 0% | **77%** | **3%** |

The Cauchy and mean columns use all default dimensions and the spline trend. The Cauchy combination
made a call in 4 of the 30 co-expression replicates with 150 samples, one or two genes each time. That
is in line with the 10% that the FDR level allows under a complete null (the mean made none, because
it is more conservative). With the elbow dimension the two combinations are the same up to the hub
adjustment and give the same calls.

- The false calls came from the degree trend, not from the tails. The lowest-degree genes (those in no
  module) have embeddings close to zero, so their distances rise steeply at the bottom of the degree
  range. The quadratic trend bent at the ends and missed that rise. The trend is now a natural cubic
  spline with knots at the degree quartiles. It has as many parameters as the quadratic, so a changed
  group that shares a degree keeps its signal (a unit test checks this), but it is linear beyond the
  outer knots. It removes almost all of those false calls and changes the ranking AUROCs by at most
  0.01 elsewhere. `trend="polynomial"` in `empirical_null_test` keeps the old fit.
- The conservativeness was not fixed. The log distances are left-skewed, so the combined scores have a
  thinner upper tail than a normal distribution, and 0.5–3.9% of null nodes have p < 0.05. Three
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
replicates. Four tests are compared: `significance()` with all dimensions (Cauchy combination and
mean) and with the elbow dimension, and the new sample-label `permutation_test` (100 permutations).

**Calibration with shuffled sample labels.** Pooling both conditions and splitting the samples at
random leaves no true difference between the groups. The permutation test is close to uniform: 5.4%
of genes have p < 0.05 (3.9–8.6% per replicate), with one false call at q < 0.1 in 10 replicates. The
empirical null is conservative (0.8% with the Cauchy combination, 2.2% with the mean, 3.4% with the
elbow), with no false calls in 10 replicates. A first version of the permutation test standardised distances by
their permutation mean and sd. It failed on one random split, where two near-equal singular values
made a single dimension unstable, and called 202 genes. Its scores are now rank-based, so no single
dimension can dominate.

![Expression calibration](results/expression_calibration.png)

**Power against the known rewiring.** The tests answer different questions:

- The permutation test calls 82% of rewired genes. It also calls 25% of their module partners,
  whose own connections did not change: a gene joining or leaving their module changes their
  neighbourhood and moves them in the joint embedding. Only 2% of genes that are never in a module,
  and 2% of differential-expression decoys, are called. Its null is that the two groups do not differ
  at all, so under a real difference it is not a per-gene test and its q-values do not control the
  false discovery rate among genes.
- The empirical null calls only the genes that stand out from genes of similar degree: 53% of
  rewired genes with the elbow dimension, 24% with the mean over all dimensions and 12% with the
  Cauchy combination. Here the change is spread over many dimensions, which favours the mean, as in
  the simulated co-expression networks above. It calls almost no partners or decoys (at most 0.4%).
- For ranking the rewired genes, AUROC is 0.98 for the elbow z-score, 0.96 for both all-dimension
  z-scores, 0.91 for the default Borda and 0.90 for the permutation z-score. The permutation score
  ranks partners high too.

![Expression calls](results/expression_calls.png)

In practice, use `permutation_test` when the samples are available and the question is whether the
networks differ at all and around which genes. Use `significance()` when only the networks are available, or to
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
the degree-adjusted z at 0.07. On ASD vs CTL itself the cosine bias is −0.65, the default Borda −0.26
and the elbow Borda 0.42, while the z-scores stay at 0.06–0.09.

**3. The degree-adjusted test is calibrated on the resampled null.** With all dimensions, 3.2% of genes
have p < 0.05 (at most 5.2% in a replicate) and no gene was called at q < 0.1 in 20 replicates. The mean
over dimensions gives 4.3% and called two genes in 20 replicates. The elbow variant is conservative
(0.8%, no calls). The excess of p-values near 1 comes from genes that
change less than their degree peers, which only makes the one-sided test conservative. Genes isolated
in both networks (23 here) are now left untested by `significance()`, instead of entering the
degree trend with a zero distance.

**4. A naive null that swaps edges between the networks is not valid per gene.** Swapping every edge's
weight between ASD and CTL with probability ½ keeps the size of each gene's edge differences and only
randomises their direction. The genes that differ most in the real data therefore keep large
distances, and the same 8–15 genes are called in every replicate (5.5–6.5% of genes with p < 0.05). It is
included in the results as a warning, not as a calibration check.

**5. ASD vs CTL has no significant gene.** No gene reaches q < 0.1 with any variant, and 3.8% of
genes have p < 0.05 (3.6% with the mean, 2.8% with the elbow), which is close to what the null gives.
The top of the default ranking is led by NLGN4Y (p = 0.001, q = 0.64) and USP9Y (p = 0.001), two
Y-chromosome genes.
NLGN4Y's degree drops from 270 in CTL to 112 in ASD. This looks like a difference in the sex
composition of the donors or cells, rather than autism biology. This is inferred from the gene names;
the donor metadata is not in the repository. The next genes (STXBP1, CAMK4, CNTN3, SCN8A, GRIN2B) are
neuronal genes with nominal p-values between 0.009 and 0.03.

**6. The real differences dwarf the resampled noise.** Spike-ins on the resampled null are easy (AUROC
0.93–1.0 for the n2v2r rankings, power at q < 0.1 of 0.89–0.93 for the z-score, against 0.83–0.91 with
the mean over dimensions). On the real pair, the same spike-ins rank at 0.69–0.88 and are rarely
called (power 0.09–0.10, against 0.05 with the mean). The real ASD–CTL differences are much larger than the edge-level
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

**The phase merge.** The paper does not say how Revelio's five phases were merged into four, so
`--phase-merge` tries three merges with the fast test (`results/cell_cycle_fast*.csv`):
"default" (G1 = M/G1 + G1/S, M = G2/M), "later" (every transition goes to the phase it leads into)
and "earlier" (every transition goes to the phase it leaves). Cell-cycle pathways at FDR 0.1, n2v2r
vs DeDi:

| merge | cells G1 / S / G2 / M | G1 → S | S → G2 | G2 → M | null split |
|---|---|---|---|---|---|
| default | 838 / 258 / 186 / 195 | 53 vs 39 | 6 vs 11 | 23 vs 0 | 0 vs 0 |
| later | 320 / 776 / 186 / 195 | 44 vs 0 | 36 vs 22 | 20 vs 0 | 0 vs 0 |
| earlier | 518 / 258 / 381 / 320 | 40 vs 7 | 13 vs 10 | 37 vs 0 | 0 vs 0 |

The conclusions do not depend on the merge: no calls on the null split, and n2v2r finds more
cell-cycle pathways than DeDi in all comparisons but one, and far more at G2 → M. The "earlier" merge
gives the most balanced phases.

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

## Run-to-run variation of competitor rankings

`benchmarks/competitor_seeds.py` ranks the same network pair many times with a new seed each time: n2v2r,
node2vec and DeepWalk (separate embeddings + Procrustes), a Python port of PLEX.I, and the deterministic DeDi
and a DGCA-style correlation-difference score. Mean share of the top 100 shared by two runs (block model /
co-expression / HeLa G1 → S): n2v2r 1.00 / 1.00 / 1.00; node2vec 0.96 / 0.92 / 0.61; DeepWalk 0.94 / 0.92 /
0.60; PLEX.I with its default 50 trainings 0.24 / 0.70 / 0.75. In these two simulations node2vec and DeepWalk
rank the changed nodes better than n2v2r (AUROC 0.998-0.999 vs 0.963-0.975; one network pair each). Full
numbers: `results/competitor_seeds_*.csv`.

## Revision runs (split/merge, ablation, HeLa sensitivity, runtime)

- `revision_sims.py --part splitmerge`: community split and merge (and switch) in the degree-corrected block model
  and the co-expression simulation; `revision_splitmerge.csv`.
- `revision_sims.py --part ablation`: one component of n2v2r varied at a time (embedding, dimension, distance,
  aggregation) on the `simulations.py` settings; `revision_ablation.csv`.
- `revision_hela.py --part dimensions --revelio-dir ...`: HeLa singular values, single-dimension rankings and the
  fast set test per dimension choice, next to the null split; `revision_hela_{spectrum,rankings,set_test}.csv`.
- `revision_hela.py --part grid --revelio-dir ...`: metacell size x soft power; `revision_hela_grid.csv`.
- `runtime.py`: runtime and peak memory of n2v2r and DeDi up to 20,000 nodes; `runtime.csv`.

## Alignment, the radial metric and the chi-square test

`alignment_wald.py` (10 replicates per setting, about 15 minutes on 4 cores; `alignment_wald.csv`,
`alignment_wald_summary.csv`, `alignment_wald_dimension.csv`) and `alignment_hela.py --revelio-dir ...`
(`alignment_hela.csv`). "Default" is the paper's Borda over dimensions 4–24 with Euclidean and cosine distances;
"test" is `significance()`; "chi-square" is `wald_test()` at the elbow dimension. Calls are at q < 0.1.

### Single nodes change (truth: the nodes whose own position changed)

AUROC:

| scenario | changed | default | default, aligned | test | test, aligned | chi-square | chi-square, aligned | DeDi |
|---|---|---|---|---|---|---|---|---|
| binary block model, heavy-tailed degrees | 10% | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 |
| | 30% | 0.83 | 1.00 | 0.83 | 1.00 | 0.98 | 1.00 | 0.52 |
| binary block model | 10% | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.52 |
| | 30% | 0.83 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 0.51 |
| weighted block model | 10% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 |
| | 30% | 0.96 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.56 |
| co-expression \|cor\|^6 | 10% | 0.96 | 0.99 | 0.99 | 1.00 | | | 0.67 |
| | 30% | 0.78 | 0.84 | 0.74 | 0.79 | | | 0.70 |
| co-expression ((1+cor)/2)^10 | 10% | 0.89 | 1.00 | 0.99 | 1.00 | | | 0.60 |
| | 30% | 0.80 | 0.93 | 0.78 | 0.84 | | | 0.68 |
| mixed changes: hub status | 5% | 0.55 | 0.66 | 0.56 | 0.66 | 0.96 | 0.97 | 0.95 |
| mixed changes: community | 5% | 0.71 | 0.87 | 0.72 | 0.88 | 0.90 | 0.91 | 0.49 |
| mixed changes: both | 5% | 0.73 | 0.89 | 0.74 | 0.89 | 0.98 | 0.98 | 0.95 |

Changed nodes called / unchanged nodes called (1,000 nodes; 2,000 in the mixed design):

| scenario | changed | test | test, aligned | chi-square | chi-square, aligned |
|---|---|---|---|---|---|
| binary block model, heavy-tailed degrees | none | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0.1 |
| | 100 | 72 / 0 | 99 / 0.1 | 100 / 4.4 | 100 / 4.8 |
| | 300 | 0 / 0 | 260 / 0 | 129 / 2.0 | 299 / 2.9 |
| binary block model | none | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0.3 |
| | 100 | 82 / 0 | 100 / 0 | 100 / 2.9 | 100 / 4.2 |
| | 300 | 0 / 0 | 297 / 0 | 154 / 0 | 300 / 1.8 |
| weighted block model | none | 0 / 0 | 0 / 0 | 0 / 0.1 | 0 / 0.2 |
| | 100 | 100 / 0.4 | 100 / 0.3 | 100 / 12.5 | 100 / 6.5 |
| | 300 | 19 / 0 | 300 / 0.6 | 299 / 7.1 | 300 / 4.7 |
| co-expression \|cor\|^6 | none | 0 / 0.5 | 0 / 0.4 | | |
| | 100 | 14 / 0 | 72 / 0.8 | | |
| | 300 | 0 / 0.1 | 0 / 0 | | |
| co-expression ((1+cor)/2)^10 | none | 0 / 0 | 0 / 0 | | |
| | 100 | 74 / 3.3 | 73 / 0.7 | | |
| | 300 | 0.1 / 0 | 0 / 1.6 | | |
| mixed: hub / community / both (about 100 each) | | 0 / 0 | 0.3 / 0 | 69, 54, 82 / 5.5 | 71, 56, 85 / 6.2 |

- Alignment never lowered the AUROC when single nodes change, and the gain grows with the share of changed nodes.
  Without it `significance()` calls nothing at 30% in the binary models: every node has moved, so no node
  stands out from nodes of its degree.
- The chi-square test keeps its level without change (3.6–5.1% of nodes with p < 0.05, at most 0.3 calls per
  1,000 nodes) and finds nearly every changed node at an observed false discovery share of 1–6% (11% without
  alignment in the weighted model at 10%).
- It is the only node2vec2rank ranking that finds both kinds of change in the mixed design. The radial-only Borda
  has 0.87 on hub changes and 0.48 on community changes; the cosine-only Borda 0.49 and 0.73 (0.88 aligned).
- With 30% of genes switching module no test makes calls in co-expression networks, aligned or not.
- For co-expression networks the chi-square test is not reported: its inflation (median statistic over the
  chi-square median) is 6–79 there, against 0.9–1.5 in the block models after alignment.

### Whole communities change (truth: every node of the affected communities)

AUROC:

| scenario | default | default, aligned | test | test, aligned | chi-square | chi-square, aligned | DeDi |
|---|---|---|---|---|---|---|---|
| block model, 8 communities: one splits | 0.76 | 0.74 | 0.77 | 0.75 | 0.58 | 0.55 | 0.65 |
| block model: two merge | 0.94 | 0.64 | 0.90 | 0.58 | 0.90 | 0.68 | 0.86 |
| co-expression: one module splits | 0.87 | 0.83 | 0.91 | 0.86 | | | 0.68 |
| co-expression: two modules merge | 0.92 | 0.77 | 0.94 | 0.75 | | | 0.76 |
| paper's design, degree-naive (500 nodes, 6 communities) | 0.96 | 0.93 | 0.88 | 0.92 | 0.93 | 0.93 | 0.77 |
| paper's design, random | 0.97 | 0.99 | 0.90 | 0.94 | 0.94 | 0.92 | 0.82 |
| ((1+cor)/2)^10: one module weakens | 0.89 | 0.34 | 0.96 | 0.26 | | | 0.87 |
| \|cor\|^6: one module weakens | 0.40 | 0.17 | 0.34 | 0.14 | | | 0.77 |

- Alignment removes changes that a linear map of the embedding explains: a merge and a module changing strength.
  The nodes of those communities did not change relative to their community, which is what alignment keeps.
  Splits and the paper's design (one row of the block matrix re-drawn) are barely affected.
- The chi-square test misses the split because the elbow is 2 with 8 communities.
- In these designs nodes outside the affected communities change their edges to them, so calls outside are not
  false; counts are in `alignment_wald_summary.csv`.

### HeLa cell cycle (2,000 genes, hdWGCNA-style signed networks, fast gene-set test with 20 label shuffles)

| comparison | pathways called at FDR 0.1 (cell-cycle): default | aligned | cell-cycle genes in the top 100: default | aligned |
|---|---|---|---|---|
| G1 → S | 131 (50) | 0 | 22 | 3 |
| S → G2 | 44 (10) | 0 | 1 | 4 |
| G2 → M | 31 (20) | 0 | 12 | 14 |
| G1 vs G1, three random splits | 0, 0, 0 | 0, 0, 0 | 3, 6, 1 | 10, 6, 10 |

`significance()` called no single gene in any comparison, with or without alignment. On these cells the pathway
signal is a change shared by groups of genes, and alignment removes it. This is why alignment is an option and
not the default.

### The dimension of the chi-square test

No change, 4 communities, nodes with p < 0.05 (target 5%), without / with genomic control:

| scenario | d = 2 | 3 | 4 | 6 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| binary block model | 7.2 / 6.8 | 7.2 / 6.3 | 6.0 / 4.8 | 27 / 8.3 | 43 / 7.1 | 69 / 6.4 | 80 / 7.2 |
| binary, heavy-tailed degrees | 4.7 / 4.7 | 4.9 / 4.8 | 4.7 / 4.6 | 23 / 11 | 36 / 12 | 55 / 16 | 64 / 19 |
| weighted | 1.7 / 1.7 | 3.1 / 3.1 | 4.6 / 4.6 | 8.3 / 7.1 | 18 / 8.7 | 63 / 12 | 85 / 13 |

Up to the rank (4) the test is valid. Beyond it, the singular vectors are fitted to the noise they are multiplied
with, and the statistic is inflated for every node; genomic control only partly repairs that (46–141 false calls
per 1,000 nodes with heavy-tailed degrees). Two things were tried to lift this limit and dropped: a dimension rule
that counts singular values above the noise level (it admits weak dimensions, which are inflated too: 16–75 false
calls in the weighted model), and combining dimensions 4–24 with the Cauchy combination (75–85% of unchanged nodes
with p < 0.05).

### Also tried and not included

A chi-square test for networks built from samples, with each gene's covariance from a jackknife over the samples
(30 blocks, projection held fixed). It kept its level without change (0.1–1.6% at p < 0.05 up to the population
rank), but with 10% of genes rewired it called 150–350 unchanged genes: their partners moved, and 150 samples are
enough to detect that. With alignment this was cured only for the signed network at exactly its rank (AUROC 0.99,
4 false calls); at other dimensions and for |cor|^6 it ranked worse (0.67–0.96) than the degree-adjusted test
(0.99). A test of "this gene's edges changed at all" is not a ranking of rewired genes.
