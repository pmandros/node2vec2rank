"""Known-biology benchmark on single-cell data: HeLa cells across the cell cycle.

Data: the HeLa S3 scRNA-seq of Revelio (Schwabe et al., 2020), 1,564 cells in
two batches (WT and Ago2KO), from https://github.com/danielschw188/Revelio
(data/revelioTestData_rawDataMatrix.rda and revelioTestData_cyclicGenes.rda;
reading .rda files needs ``pip install pyreadr``). These are the cells behind
the paper's hdWGCNA cell-cycle networks.

Steps:

1. Phases: every cell is assigned a phase with Revelio's method (marker-gene
   scores z-scored per phase and per cell, per batch; low-confidence cells and
   suspected doublets removed). Revelio's five phases are merged into the
   paper's four: G1 = M/G1 + G1/S, S, G2, M = G2/M.
2. Networks, hdWGCNA style: log-normalised counts, the 2,000 most variable
   genes detected in at least 5% of cells, metacells (k = 25, at most 10
   shared cells) per phase, signed WGCNA network ((1 + cor) / 2)^power with
   the power picked by the scale-free fit. The counts stay sparse until the
   gene subset is taken.
3. For the paper's comparisons G1 -> S and G2 -> M (and S -> G2), the Reactome
   library is tested in three ways:
   - GSEA prerank (gseapy, as in the paper's notebook) on the n2v2r Borda
     ranking and on absDeDi;
   - the cell-label permutation gene-set test (``gene_set_test``), with n2v2r
     and with DeDi, labels shuffled within batch.
   Reactome cell-cycle pathways are the expected positives: we report how many
   pathways each method calls at FDR 0.1, how many of those are cell-cycle
   pathways, and the AUROC of cell-cycle pathways by the method's score.
4. Null control on real data: the G1 cells of each batch are split at random
   into two halves. Nothing should be called.

Cells of one cell line are treated as exchangeable units within batch. With
donors (patients), shuffle donors instead of cells.

Caveat: phases are assigned from cell-cycle marker genes, so cell-cycle genes
differ in expression between the groups by construction. Every node is
standardised within group (and batch) before building the networks, so mean
differences (and batch differences) do not enter the networks, but cell-cycle pathways are still the
easiest positives.

Run from the repository root (about 30 minutes on 4 cores):

    OMP_NUM_THREADS=1 python benchmarks/cell_cycle.py --revelio-dir path/to/Revelio/data
"""

import argparse
import functools
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from node2vec2rank import N2V2R  # noqa: E402
from node2vec2rank.fast_gene_sets import fast_gene_set_test  # noqa: E402
from node2vec2rank.permutation import _standardize_within, gene_set_test, read_gmt  # noqa: E402
from node2vec2rank.singlecell import (highly_variable_genes, metacell_network, metacells,  # noqa: E402
                                      normalize_log1p)
from node2vec2rank.simulate import coexpression_network  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
REACTOME = os.path.join(REPO, "data", "gene_set_libraries", "human", "c2.cp.reactome.v7.5.1.symbols.gmt")
PHASE_MERGES = {
    # transitions into and out of G1 go to G1, G2/M to M (the default here)
    "default": {"M.G1": "G1", "G1.S": "G1", "S": "S", "G2": "G2", "G2.M": "M"},
    # every transition goes to the phase it leads into
    "later": {"M.G1": "G1", "G1.S": "S", "S": "S", "G2": "G2", "G2.M": "M"},
    # every transition goes to the phase it leaves
    "earlier": {"M.G1": "M", "G1.S": "G1", "S": "S", "G2": "G2", "G2.M": "G2"},
    # the paper's script: M = M/G1 + G2/M, G1 = G1/S
    "paper": {"M.G1": "M", "G1.S": "G1", "S": "S", "G2": "G2", "G2.M": "M"},
}
# The paper's hdWGCNA networks (scripts/single_cell_cell_cycle/cs_cc_Revelio_Seurat_hdWGCNA.R in the authors'
# scripts): variable genes, metacells with k = 20 and at most 10 shared cells per phase, signed network with soft
# power 6, topological overlap (TOM, rounded to 4 digits), and only genes that vary in the metacells of every phase.
# The paper's networks have 2,420 genes (from Seurat's 3,000 SCTransform variable genes); here they are the 2,420
# most variable genes among those that vary in every phase's metacells.
PAPER_NETWORK = dict(num_genes=2420, candidates=4000, k=20, max_shared=10, power=6)
PHASES = PHASE_MERGES["default"]
COMPARISONS = [("G1", "S"), ("S", "G2"), ("G2", "M")]
CELL_CYCLE = re.compile(
    r"CELL_CYCLE|MITOTIC|MITOSIS|G1_S|G2_M|S_PHASE|M_PHASE|G0_AND_EARLY_G1|DNA_REPLICATION|"
    r"REPLICATION_|KINETOCHORE|CENTROSOME|CENTRIOLE|SPINDLE|CHECKPOINT|CYCLIN|APC_C|CDC20|CDH1|"
    r"SISTER_CHROMATID|CONDENSATION_OF|PROMETAPHASE|METAPHASE|ANAPHASE|TELOPHASE|CYTOKINESIS|"
    r"POLO_LIKE|PLK1|AURKA|E2F|RB1|ORC|MCM|UNWINDING_OF_DNA|LAGGING_STRAND|LEADING_STRAND|"
    r"NUCLEAR_ENVELOPE_BREAKDOWN|CHROMOSOME_MAINTENANCE|TELOMERE")
DIMENSIONS = list(range(4, 25, 2))
PAPER_PARAMS = dict(embed_dimensions=DIMENSIONS, distance_metrics=["euclidean", "cosine"], seed=42)


def read_revelio(directory):
    import pyreadr
    counts = next(iter(pyreadr.read_r(os.path.join(directory, "revelioTestData_rawDataMatrix.rda")).values()))
    markers = next(iter(pyreadr.read_r(os.path.join(directory, "revelioTestData_cyclicGenes.rda")).values()))
    genes = np.asarray(counts.index)
    cells = np.asarray(counts.columns)
    matrix = sp.csr_matrix(counts.to_numpy().T.astype(np.float32))  # cells x genes
    return matrix, genes, cells, markers


def assign_phases(counts, genes, cells, markers, min_marker_cor=0.2, min_highest=0.75, max_second=0.5,
                  merge=PHASES):
    """Revelio's getCellCyclePhaseAssignInformation, per batch."""
    batch = np.array([c.split("_")[0] for c in cells])
    library = np.asarray(counts.sum(axis=1)).ravel()
    logged = normalize_log1p(counts, scale=np.median(library))
    position = {g: i for i, g in enumerate(genes)}
    names = list(markers.columns)
    phase = np.empty(len(cells), dtype=object)
    outlier = np.zeros(len(cells), dtype=bool)
    for value in np.unique(batch):
        rows = batch == value
        scores = []
        for name in names:
            index = [position[g] for g in markers[name].dropna() if g in position]
            block = logged[rows][:, index].toarray()
            block = block[:, block.sum(axis=0) > 0]
            average = block.mean(axis=1)
            cor = np.array([np.corrcoef(block[:, j], average)[0, 1] for j in range(block.shape[1])])
            scores.append(block[:, cor > min_marker_cor].mean(axis=1))
        score = np.array(scores).T
        score = (score - score.mean(axis=0)) / score.std(axis=0, ddof=1)
        score = (score - score.mean(axis=1, keepdims=True)) / score.std(axis=1, ddof=1, keepdims=True)
        order = np.argsort(-score, axis=1)
        highest = score[np.arange(len(score)), order[:, 0]]
        second = score[np.arange(len(score)), order[:, 1]]
        gap = np.abs(order[:, 0] - order[:, 1])
        gap = np.minimum(gap, len(names) - gap)
        phase[rows] = np.array(names)[order[:, 0]]
        outlier[rows] = (highest < min_highest) | ((gap > 1) & (second > max_second))
    return pd.DataFrame({"cell": cells, "batch": batch, "revelio_phase": phase,
                         "phase": pd.Series(phase).map(merge).to_numpy(), "outlier": outlier})


def topological_overlap(adjacency):
    """WGCNA's topological overlap of a non-negative adjacency matrix:
    (sum_u a_iu a_uj + a_ij) / (min(k_i, k_j) + 1 - a_ij), with 1 on the diagonal."""
    a = np.array(adjacency, dtype=np.float64)
    np.fill_diagonal(a, 0)
    degree = a.sum(axis=0)
    tom = (a @ a + a) / (np.minimum.outer(degree, degree) + 1 - a)
    np.fill_diagonal(tom, 1)
    return tom


def tom_network(expression, k=20, max_shared=10, num_pcs=30, power=6, random_state=0):
    """The paper's network: metacells, signed soft-thresholded correlation, TOM."""
    aggregated = metacells(expression, k=k, max_shared=max_shared, num_pcs=num_pcs, random_state=random_state)
    tom = np.round(topological_overlap(coexpression_network(np.asarray(aggregated), power=power, signed=True)), 4)
    if isinstance(expression, pd.DataFrame):
        return pd.DataFrame(tom, index=expression.columns, columns=expression.columns)
    return tom


def scale_free_fit(network, num_bins=10):
    degree = np.asarray(network).sum(axis=0)
    counts, edges = np.histogram(degree, bins=num_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    keep = counts > 0
    x, y = np.log10(centres[keep]), np.log10(counts[keep] / counts.sum())
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r2 = 1 - residual.var() / y.var()
    return -np.sign(slope) * r2


def pick_power(groups, powers=(4, 6, 8, 10, 12, 14, 16, 18, 20), threshold=0.8):
    """Lowest signed soft power whose mean scale-free fit (over groups) reaches
    the threshold, as in WGCNA/hdWGCNA; the best power otherwise."""
    aggregated = [metacells(group) for group in groups]
    fits = {}
    for power in powers:
        fits[power] = np.mean([scale_free_fit(coexpression_network(m, power=power, signed=True))
                               for m in aggregated])
        if fits[power] >= threshold:
            return power, fits
    return max(fits, key=fits.get), fits


def auroc(labels, scores):
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(np.nan_to_num(scores, nan=-np.inf))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def prerank(ranking, gene_sets, seed=42):
    import gseapy
    result = gseapy.prerank(rnk=pd.DataFrame(ranking), gene_sets=gene_sets, threads=4, min_size=5,
                            max_size=500, permutation_num=1000, outdir=None, seed=seed, verbose=False,
                            weight=0, no_plot=True).res2d
    result = result.set_index("Term")
    return pd.DataFrame({"score": result["NES"].astype(float),
                         "qvalue": np.where(result["NES"].astype(float) > 0,
                                            result["FDR q-val"].astype(float), 1.0)})


def summarise(name, comparison, result, is_cell_cycle, q_threshold=0.1):
    called = result["qvalue"] <= q_threshold
    positives = is_cell_cycle.reindex(result.index).fillna(False).to_numpy()
    return {"comparison": comparison, "method": name, "tested": len(result),
            "cell_cycle_tested": int(positives.sum()), "called": int(called.sum()),
            "cell_cycle_called": int((called & positives).sum()),
            "precision": (called & positives).sum() / max(called.sum(), 1),
            "auroc_cell_cycle": auroc(positives, result["score"].to_numpy())}


METHODS = {"gsea-n2v2r": "GSEA prerank, n2v2r Borda", "gsea-dedi": "GSEA prerank, absDeDi",
           "perm-n2v2r": "permutation set test, n2v2r", "perm-dedi": "permutation set test, DeDi",
           "fast-n2v2r": "fast set test, n2v2r", "fast-dedi": "fast set test, DeDi"}


def score_methods(group_a, group_b, strata, gene_sets, build, num_permutations, n_jobs, methods=tuple(METHODS),
                  num_fast_permutations=10):
    """Every method's set results (columns ``score``, larger is more
    differential, and ``qvalue``) for one comparison."""
    results = {}
    if {"gsea-n2v2r", "gsea-dedi"} & set(methods):
        network_a, network_b = build(group_a), build(group_b)
        model = N2V2R([network_a, network_b], nodes=list(group_a.columns), verbose=-1, **PAPER_PARAMS)
        model.fit_transform_rank()
        if "gsea-n2v2r" in methods:
            results[METHODS["gsea-n2v2r"]] = prerank(model.aggregate_transform()["1"]["borda_ranks"], gene_sets)
        if "gsea-dedi" in methods:
            results[METHODS["gsea-dedi"]] = prerank(model.degree_difference_ranking()["1"]["absDeDi"], gene_sets)
    for ranking, key in (("n2v2r", "perm-n2v2r"), ("degree_difference", "perm-dedi")):
        if key in methods:
            test = gene_set_test(group_a, group_b, gene_sets, build_network=build,
                                 num_permutations=num_permutations, min_size=5, max_size=500, strata=strata,
                                 random_state=0, n_jobs=n_jobs, ranking=ranking, **PAPER_PARAMS)
            results[METHODS[key]] = test.drop(columns="score").rename(columns={"nes": "score"})
    for ranking, key in (("n2v2r", "fast-n2v2r"), ("degree_difference", "fast-dedi")):
        if key in methods:
            test = fast_gene_set_test(group_a, group_b, gene_sets, build_network=build,
                                      num_permutations=num_fast_permutations, min_size=5, max_size=500,
                                      strata=strata, random_state=0, ranking=ranking, **PAPER_PARAMS)
            results[METHODS[key]] = test.drop(columns="score").rename(columns={"z": "score"})
    return results


def run_comparison(label, group_a, group_b, strata, gene_sets, is_cell_cycle, build, num_permutations,
                   n_jobs, top_sets, methods=tuple(METHODS)):
    records, tops = [], []
    tic = time.time()
    results = score_methods(group_a, group_b, strata, gene_sets, build, num_permutations, n_jobs, methods)
    for name, result in results.items():
        records.append(summarise(name, label, result, is_cell_cycle))
        top = result.sort_values(["qvalue", "score"], ascending=[True, False]).head(top_sets)
        tops.append(pd.DataFrame({"comparison": label, "method": name, "gene_set": top.index,
                                  "score": top["score"].to_numpy(), "qvalue": top["qvalue"].to_numpy(),
                                  "cell_cycle": is_cell_cycle.reindex(top.index).fillna(False).to_numpy()}))
    print(f"{label}: {time.time() - tic:.0f}s", flush=True)
    return records, tops


def random_halves(strata, rng):
    """A boolean mask selecting a random half of the samples of every stratum."""
    half = np.zeros(len(strata), dtype=bool)
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        half[rng.choice(members, len(members) // 2, replace=False)] = True
    return half


def prepare(revelio_dir, num_genes=2000, merge="default", setting="default", paper_power=None, paper_genes=None):
    """Phases, expression of the most variable genes standardised within phase
    and batch, and the network builder. setting "default" is this benchmark's
    pipeline (signed correlation network, metacells k = 25, power picked by the
    scale-free fit); "paper" is the paper's (PAPER_NETWORK: TOM, k = 20,
    power 6, phases merged as in the paper's script); paper_power overrides
    its soft power, and paper_genes (gene names, e.g. the genes of the paper's
    own networks, written as Seurat writes them, with "-" for "_") replaces its
    gene selection."""
    paper = setting == "paper"
    if paper:
        merge, num_genes = "paper", PAPER_NETWORK["candidates"]
    counts, genes, cells, markers = read_revelio(revelio_dir)
    cells_table = assign_phases(counts, genes, cells, markers, merge=PHASE_MERGES[merge])
    print(pd.crosstab([cells_table.batch, cells_table.outlier], cells_table.revelio_phase), flush=True)
    keep = ~cells_table["outlier"].to_numpy()
    counts, cells_table = counts[keep], cells_table[keep].reset_index(drop=True)

    logged = normalize_log1p(counts)
    if paper and paper_genes is not None:
        index = {g.replace("_", "-"): i for i, g in enumerate(genes)}
        missing = [g for g in paper_genes if g not in index]
        if missing:
            raise ValueError(f"{len(missing)} genes not in the data, e.g. {missing[:5]}")
        selected = np.array([index[g] for g in paper_genes])
    else:
        selected = highly_variable_genes(logged, counts, num_genes=num_genes, min_detected=0.0 if paper else 0.05)
    expression = pd.DataFrame(logged[:, selected].toarray(), columns=genes[selected])
    del logged, counts

    # every gene is z-scored within phase and batch, as gene_set_test does, so
    # that the GSEA rankings are computed on the same networks
    batches = {phase: cells_table.loc[cells_table["phase"].eq(phase), "batch"].to_numpy()
               for phase in ("G1", "S", "G2", "M")}
    groups = {phase: _standardize_within(expression[cells_table["phase"].eq(phase).to_numpy()], batches[phase])
              for phase in batches}
    if paper:
        k, max_shared = PAPER_NETWORK["k"], PAPER_NETWORK["max_shared"]
        aggregated = {phase: metacells(group, k=k, max_shared=max_shared) for phase, group in groups.items()}
        varying = np.logical_and.reduce([np.asarray(a).std(axis=0) > 0 for a in aggregated.values()])
        chosen = np.flatnonzero(varying)
        if paper_genes is None:
            chosen = chosen[:PAPER_NETWORK["num_genes"]]  # columns are most variable first
        groups = {phase: group.iloc[:, chosen] for phase, group in groups.items()}
        print(f"paper setting: {sum(len(a) for a in aggregated.values())} metacells "
              f"({ {p: len(a) for p, a in aggregated.items()} }) from {len(cells_table)} cells; "
              f"{varying.sum()} of {len(varying)} candidate genes vary in every phase, kept {len(chosen)}",
              flush=True)
        power = PAPER_NETWORK["power"] if paper_power is None else paper_power
        print(f"paper setting: signed soft power {power} with topological overlap", flush=True)
        build = functools.partial(tom_network, k=k, max_shared=max_shared, power=power)
        return cells_table, groups, batches, build
    power, fits = pick_power(list(groups.values()))
    print(f"signed soft power {power}; scale-free fits {fits}", flush=True)
    build = functools.partial(metacell_network, power=power, signed=True)
    return cells_table, groups, batches, build


def read_genes(path):
    """Gene names, one per line; None for no file."""
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def expressions_and_networks(build, *groups):
    """Metacell expression (for expression-based methods) and networks of each
    group with the builder from prepare."""
    k, max_shared = build.keywords.get("k", 25), build.keywords.get("max_shared", 10)
    return ([np.asarray(metacells(g, k=k, max_shared=max_shared)) for g in groups],
            [np.asarray(build(g)) for g in groups])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revelio-dir", required=True, help="the data folder of the Revelio repository")
    parser.add_argument("--num-genes", type=int, default=2000)
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--top-sets", type=int, default=15)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--suffix", default="", help="appended to the result file names")
    parser.add_argument("--phase-merge", choices=list(PHASE_MERGES), default="default",
                        help="how Revelio's five phases are merged into four")
    parser.add_argument("--comparisons", nargs="+", default=[f"{a}-{b}" for a, b in COMPARISONS],
                        help="phase pairs to compare, e.g. G1-S G2-M")
    parser.add_argument("--skip-null", action="store_true", help="skip the null split of the G1 cells")
    parser.add_argument("--setting", choices=["default", "paper"], default="default",
                        help="'paper': the paper's networks (TOM, k = 20, power 6, 2,420 genes; see prepare)")
    parser.add_argument("--paper-power", type=int, default=None,
                        help="soft power for --setting paper (default: PAPER_NETWORK's)")
    parser.add_argument("--paper-genes", default=None,
                        help="for --setting paper: file with one gene per line to use instead of the gene selection")
    args = parser.parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cells_table, groups, batches, build = prepare(args.revelio_dir, args.num_genes, args.phase_merge, args.setting,
                                                  args.paper_power, read_genes(args.paper_genes))

    gene_sets = read_gmt(REACTOME)
    names = pd.Index(list(gene_sets))
    is_cell_cycle = pd.Series(np.asarray(names.str.contains(CELL_CYCLE)), index=names)

    records, tops = [], []
    for a, b in (c.split("-") for c in args.comparisons):
        r, t = run_comparison(f"{a} -> {b}", groups[a], groups[b], (batches[a], batches[b]), gene_sets,
                              is_cell_cycle, build, args.num_permutations, args.n_jobs, args.top_sets,
                              args.methods)
        records += r
        tops += t

    if not args.skip_null:
        # null control: random halves of the G1 cells, within batch
        half = random_halves(batches["G1"], np.random.default_rng(0))
        r, t = run_comparison("null: G1 halves", groups["G1"][half], groups["G1"][~half],
                              (batches["G1"][half], batches["G1"][~half]), gene_sets, is_cell_cycle, build,
                              args.num_permutations, args.n_jobs, args.top_sets, args.methods)
        records += r
        tops += t

    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(RESULTS_DIR, f"cell_cycle{args.suffix}.csv"), index=False)
    pd.concat(tops).to_csv(os.path.join(RESULTS_DIR, f"cell_cycle_top_sets{args.suffix}.csv"), index=False)
    cells_table.groupby(["batch", "phase"]).size().rename("cells").to_csv(
        os.path.join(RESULTS_DIR, f"cell_cycle_phases{args.suffix}.csv"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3))


if __name__ == "__main__":
    main()
