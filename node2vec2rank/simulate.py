"""Simulated gene expression with known co-expression rewiring, and co-expression networks.

Useful to check node2vec2rank end to end when no suitable real data (or no
ground truth) is available: simulate expression in two conditions, build a
co-expression network per condition, run n2v2r and compare against the
genes that were rewired.

Model: every gene either belongs to one of ``num_modules`` modules or to none.
A module gene's expression is ``loading * factor_module + sqrt(1 - loading^2) *
noise``, with a latent standard normal factor per module and sample, so genes of
the same module are correlated and genes with larger loadings are hubs. In the
second condition the rewired genes change membership:

- ``switch``: the gene moves to another module,
- ``loss``: the gene leaves its module and becomes independent,
- ``gain``: an independent gene joins a module.

Differentially expressed genes only shift their mean in the second condition.
They are decoys: their co-expression does not change, so a co-expression
method should not call them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

REWIRING_TYPES = ("switch", "loss", "gain")


@dataclass
class SimulatedExpression:
    """Expression of two conditions with the ground truth of the simulation.

    Attributes:
        expression: list of two DataFrames (samples x genes), one per condition.
        modules: DataFrame (genes x conditions) with every gene's module, -1 for none.
        rewired: boolean Series, True for the genes whose module changes.
        rewiring_type: Series with 'switch', 'loss', 'gain' or 'none' per gene.
        differentially_expressed: boolean Series, True for mean-shift decoys.
    """
    expression: list
    modules: pd.DataFrame
    rewired: pd.Series
    rewiring_type: pd.Series
    differentially_expressed: pd.Series


def simulate_expression(num_genes=1000, num_samples=100, num_modules=5, frac_in_modules=0.7,
                        frac_rewired=0.1, rewiring=REWIRING_TYPES, frac_differentially_expressed=0.1,
                        loading_range=(0.3, 0.9), fold_change=1.0, random_state=None):
    """Simulates expression of two conditions with a known set of rewired genes.

    Args:
        num_genes: number of genes.
        num_samples: number of samples per condition (an int, or a pair).
        num_modules: number of co-expression modules.
        frac_in_modules: fraction of genes that belong to a module in the first condition.
        frac_rewired: fraction of all genes that are rewired, split evenly
            between the rewiring types.
        rewiring: the rewiring types to use, from 'switch', 'loss' and 'gain'.
        frac_differentially_expressed: fraction of the non-rewired genes whose
            mean shifts by ``fold_change`` standard deviations in the second condition.
        loading_range: range of the (uniform) module loadings; larger loadings
            mean stronger co-expression.
        fold_change: mean shift of the differentially expressed genes.
        random_state: seed or numpy Generator.

    Returns:
        SimulatedExpression
    """
    rng = np.random.default_rng(random_state)
    rewiring = tuple(rewiring)
    unknown = set(rewiring) - set(REWIRING_TYPES)
    if unknown or not rewiring:
        raise ValueError(f"rewiring must be chosen from {REWIRING_TYPES}, got {rewiring}")
    samples = (num_samples, num_samples) if np.isscalar(num_samples) else tuple(num_samples)

    in_module = rng.random(num_genes) < frac_in_modules
    before = np.where(in_module, rng.integers(0, num_modules, num_genes), -1)
    after = before.copy()
    rewiring_type = np.full(num_genes, "none", dtype=object)

    num_rewired = int(round(frac_rewired * num_genes))
    per_type = np.diff(np.linspace(0, num_rewired, len(rewiring) + 1).round().astype(int))
    available = np.ones(num_genes, bool)
    for kind, count in zip(rewiring, per_type):
        pool = np.flatnonzero(available & (in_module if kind in ("switch", "loss") else ~in_module))
        if count > len(pool):
            raise ValueError(f"Not enough genes for {count} '{kind}' rewirings; lower frac_rewired")
        chosen = rng.choice(pool, count, replace=False)
        if kind == "switch":
            after[chosen] = (before[chosen] + rng.integers(1, num_modules, count)) % num_modules
        elif kind == "loss":
            after[chosen] = -1
        else:
            after[chosen] = rng.integers(0, num_modules, count)
        rewiring_type[chosen] = kind
        available[chosen] = False

    candidates = np.flatnonzero(available)
    num_de = int(round(frac_differentially_expressed * len(candidates)))
    differentially_expressed = np.zeros(num_genes, bool)
    differentially_expressed[rng.choice(candidates, num_de, replace=False)] = True

    loadings = rng.uniform(*loading_range, num_genes)
    genes = [f"gene{i}" for i in range(num_genes)]
    expression = []
    for condition, (modules, size) in enumerate(zip((before, after), samples)):
        factors = rng.standard_normal((size, num_modules))
        values = rng.standard_normal((size, num_genes))
        member = modules >= 0
        values[:, member] = (loadings[member] * factors[:, modules[member]]
                             + np.sqrt(1 - loadings[member] ** 2) * values[:, member])
        if condition == 1:
            values[:, differentially_expressed] += fold_change
        expression.append(pd.DataFrame(values, index=[f"c{condition + 1}_s{i}" for i in range(size)],
                                       columns=genes))

    index = pd.Index(genes, name="gene")
    return SimulatedExpression(
        expression=expression,
        modules=pd.DataFrame({"condition_1": before, "condition_2": after}, index=index),
        rewired=pd.Series(rewiring_type != "none", index=index, name="rewired"),
        rewiring_type=pd.Series(rewiring_type, index=index, name="rewiring_type"),
        differentially_expressed=pd.Series(differentially_expressed, index=index,
                                           name="differentially_expressed"))


def coexpression_network(expression, power=6, signed=False):
    """Builds a WGCNA-style co-expression network from an expression table.

    Args:
        expression: DataFrame (samples x genes) or array of the same shape.
        power: soft-thresholding power; 1 gives the (absolute) correlation network.
        signed: if True, uses the signed adjacency ((1 + cor) / 2)^power,
            otherwise the unsigned |cor|^power.

    Returns:
        DataFrame (genes x genes) if ``expression`` is a DataFrame, else an array,
        with a zero diagonal.
    """
    values = np.asarray(expression, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    # constant genes have undefined correlation; treat them as unconnected
    correlation = np.nan_to_num(correlation, nan=0.0)
    adjacency = ((1 + correlation) / 2) ** power if signed else np.abs(correlation) ** power
    np.fill_diagonal(adjacency, 0)
    if isinstance(expression, pd.DataFrame):
        return pd.DataFrame(adjacency, index=expression.columns, columns=expression.columns)
    return adjacency
