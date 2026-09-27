"""Can UASE be adapted so that co-expression networks satisfy its model?

The consistency theory of UASE assumes independent edge weights around a low-rank mean. WGCNA-style
networks ``|cor|^b`` break both parts: the entries of a sample correlation matrix are dependent, and
the embedded target ``E|r|^b`` is not ``|rho|^b`` but depends on the number of samples, so two
conditions with different sample sizes have different targets even when nothing changed.

Two inputs that fix this under a Gaussian factor model ``x = L f + e`` are compared with the paper's
``|cor|^b``, all embedded with the unchanged :func:`node2vec2rank.embedding.uase`:

* ``signed``: the Pearson correlation matrix itself with a zero diagonal. Its population is exactly
  ``L L^T`` off the diagonal (rank = number of factors), and given the factor scores the genes are
  independent, so the hollowed Gram-matrix perturbation theory (Abbe, Fan & Wang 2022) gives
  row-wise bounds. The target does not depend on the number of samples up to O(1/n).
* ``debiased power``: ``|r|^b`` minus its delta-method bias, an entrywise function of r and the
  number of samples. Its target is ``|rho|^b``, the population WGCNA network, the same for every
  sample size up to O(1/n^2); for even b it is exactly low rank (at most C(r + b - 1, b)) under the
  factor model. (A U-statistic for rho^b is unbiased only with known means and variances; after
  standardising with sample moments it was more biased than |r|^b, so it was dropped.)

Designs: disjoint modules (each gene loads on one factor, as in simulate.py) and mixed loadings
(each gene loads on two factors with either sign). Changes: module switches (rewiring) and sign
flips (a gene becomes anti-correlated with its module, invisible to ``|cor|``).

Run with ``python benchmarks/correlation_uase.py``; writes ``benchmarks/results/correlation_uase_*.csv``.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from node2vec2rank.embedding import uase  # noqa: E402
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances  # noqa: E402
from node2vec2rank.significance import empirical_null_test  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
DIMS = list(range(4, 25, 2))
POWER = 6
NUM_FACTORS = 5
METHODS = ("power", "signed", "debiased power")


def roc_auc_score(labels, scores):
    labels = np.asarray(labels, bool)
    ranks = rankdata(scores)
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


# --- networks ------------------------------------------------------------------------------------

def standardise(values):
    values = values - values.mean(axis=0)
    return values / values.std(axis=0)


def debiased_power(r, m, power):
    """|r|^power minus its second-order (delta-method) bias, so its mean is |rho|^power + O(1/m^2).

    With Var(r) ~ (1 - rho^2)^2 / m and E(r) - rho ~ -rho (1 - rho^2) / (2m) for bivariate normal
    data, E f(r) ~ f(rho) + f''(rho) Var(r) / 2 + f'(rho) (E(r) - rho). Subtracting the estimate of
    the two correction terms is an entrywise function of r and m, so it costs nothing.
    """
    a = np.abs(r)
    bias = (power * (power - 1) / 2 * a ** (power - 2) * (1 - a * a) ** 2
            - power / 2 * a ** power * (1 - a * a)) / m
    return a ** power - bias


def network(values, method):
    z = standardise(values)
    if method == "power":
        adjacency = np.abs(z.T @ z / len(z)) ** POWER
    elif method == "signed":
        adjacency = z.T @ z / len(z)
    elif method == "debiased power":
        adjacency = debiased_power(z.T @ z / len(z), len(z), POWER)
    else:
        raise ValueError(method)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def population_network(load, method):
    correlation = load @ load.T
    adjacency = correlation if method == "signed" else correlation ** POWER
    np.fill_diagonal(adjacency, 0)
    return adjacency


# --- factor models -------------------------------------------------------------------------------

def loadings(n, rng, design, change, frac_changed=0.1):
    """Loadings before and after the change, and which genes changed."""
    before = np.zeros((n, NUM_FACTORS))
    strength = rng.uniform(0.3, 0.9, n)
    in_module = rng.random(n) < 0.7
    factors = [rng.choice(NUM_FACTORS, 1 if design == "disjoint" else 2, replace=False) for _ in range(n)]
    for i in np.flatnonzero(in_module):
        w = rng.uniform(0.5, 1.0, len(factors[i])) * (1 if design == "disjoint" else rng.choice([-1, 1], len(factors[i])))
        before[i, factors[i]] = strength[i] * w / np.linalg.norm(w)
    after = before.copy()
    changed = np.zeros(n, bool)
    chosen = rng.choice(np.flatnonzero(in_module), int(frac_changed * n), replace=False)
    changed[chosen] = True
    for i in chosen:
        if change == "switch":
            after[i] = np.roll(before[i], rng.integers(1, NUM_FACTORS))
        elif change == "sign flip":
            after[i] = -before[i]
    return before, after, changed, strength * in_module


def sample(load, m, rng):
    n = len(load)
    return rng.standard_normal((m, NUM_FACTORS)) @ load.T \
        + rng.standard_normal((m, n)) * np.sqrt(1 - (load ** 2).sum(axis=1))


# --- read-outs -----------------------------------------------------------------------------------

def distance_table(graphs, dims, seed=0):
    embeddings = uase(graphs, max(dims), random_state=seed)
    return pd.DataFrame({(d, metric): compute_pairwise_distances(embeddings[0, :, :d], embeddings[1, :, :d], metric)
                         for d in dims for metric in ("euclidean", "cosine")})


def readout(graphs, seed):
    table = distance_table(graphs, sorted(set(DIMS) | {NUM_FACTORS}), seed)
    default = table[DIMS].to_numpy()
    degree = np.mean([np.abs(g).sum(axis=1) for g in graphs], axis=0)
    _, _, q = empirical_null_test(default, degree)
    return dict(borda=borda_aggregate(default), rank_d=table[(NUM_FACTORS, "euclidean")].to_numpy(),
                q=q)


def change_experiment(rng, reps, n=1000):
    rows = []
    for design in ("disjoint", "mixed"):
        for change in ("switch", "sign flip"):
            for rep in range(reps):
                before, after, changed, _ = loadings(n, rng, design, change)
                data = {}
                for m1, m2 in ((50, 50), (150, 150), (500, 500), (50, 500), (150, 600)):
                    data[(m1, m2)] = sample(before, m1, rng), sample(after, m2, rng)
                for method in METHODS:
                    target = distance_table([population_network(before, method),
                                             population_network(after, method)], [NUM_FACTORS])
                    target = target[(NUM_FACTORS, "euclidean")].to_numpy()
                    for (m1, m2), (x1, x2) in data.items():
                        r = readout([network(x1, method), network(x2, method)], rep)
                        rows.append(dict(
                            design=design, change=change, method=method, samples=f"{m1} vs {m2}", rep=rep,
                            auroc_borda=roc_auc_score(changed, r["borda"]),
                            auroc_rank_d=roc_auc_score(changed, r["rank_d"]),
                            spearman_vs_target_changed=spearmanr(target[changed], r["rank_d"][changed])[0],
                            true_calls=int(np.sum((r["q"] < 0.1) & changed)),
                            false_calls=int(np.sum((r["q"] < 0.1) & ~changed))))
    return pd.DataFrame(rows)


def null_experiment(rng, reps, n=1000):
    """Nothing changes: the same loadings in both conditions, equal or unequal sample sizes."""
    rows = []
    for design in ("disjoint", "mixed"):
        for rep in range(reps):
            load, _, _, strength = loadings(n, rng, design, "switch")
            for m1, m2 in ((50, 50), (50, 500), (150, 600)):
                pairs = [(sample(load, m1, rng), sample(load, m2, rng)) for _ in range(2)]
                for method in METHODS:
                    outs = [readout([network(a, method), network(b, method)], rep) for a, b in pairs]
                    top = [rankdata(-o["borda"]) <= 0.05 * n for o in outs]
                    rows.append(dict(
                        design=design, method=method, samples=f"{m1} vs {m2}", rep=rep,
                        # a ranking that replicates although nothing changed is a systematic artefact
                        replicate_agreement=spearmanr(outs[0]["borda"], outs[1]["borda"])[0],
                        top5pct_overlap=np.mean(top[0][top[1]]),
                        null_order_vs_strength=spearmanr(outs[0]["borda"], strength)[0],
                        false_calls=int(np.sum(outs[0]["q"] < 0.1))))
    return pd.DataFrame(rows)


def main(reps=5):
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(2026)
    summaries = {}
    for name, experiment, keys in (
            ("change", change_experiment, ["design", "change", "samples", "method"]),
            ("null", null_experiment, ["design", "samples", "method"])):
        results = experiment(rng, reps)
        results.to_csv(os.path.join(RESULTS, f"correlation_uase_{name}.csv"), index=False)
        summaries[name] = results.drop(columns="rep").groupby(keys, sort=False).mean().round(3)
        print(summaries[name].to_string(), "\n", flush=True)
    return summaries


if __name__ == "__main__":
    main()
