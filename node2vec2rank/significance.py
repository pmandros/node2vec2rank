"""Per-node significance of embedding shifts with a degree-adjusted empirical null.

A node's distance between two graph embeddings mixes a true shift with
estimation noise, and that noise depends strongly on the node's degree:
low-degree nodes have noisy embeddings, while hubs have large embedding norms.
Raw distances (and rankings built on them) therefore correlate with degree even
when nothing changes.

This module standardises every node's (log) distance by the expected value and
spread of distances at the node's degree, estimated with robust regression so
that the minority of truly differential nodes does not distort the null. The standardised
scores of the different embedding dimensions and distance metrics are then
combined into one-sided p-values and Benjamini-Hochberg q-values. By default
they are averaged within windows of two consecutive dimensions and the windows
are combined with the Cauchy combination test (Liu & Xie, 2020), which keeps
the signal of a change that lives in only a few dimensions; averaging all
dimensions dilutes it. The underlying assumption, as for other empirical-null
methods, is that most nodes do not change between the graphs.

The p-values are conservative: the log distances are left-skewed, so the
standardised scores have a thinner upper tail than a normal distribution, and
typically 0.5-3.5% of unchanged nodes have p < 0.05. Corrections of the tail
that were tried (Box-Cox or Yeo-Johnson symmetrisation, a scale from the upper
half) produced false discoveries in some null benchmarks, so the conservative
version is kept (see benchmarks/README.md).
"""

import numpy as np
from scipy.stats import norm, rankdata

MAD_TO_SD = 1.4826


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are kept as NaN."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    qvalues = np.full_like(pvalues, np.nan)
    valid = ~np.isnan(pvalues)
    p = pvalues[valid]
    m = len(p)
    if m == 0:
        return qvalues
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    qvalues[valid] = adjusted
    return qvalues


def _robust_location_scale(values):
    location = np.median(values)
    scale = MAD_TO_SD * np.median(np.abs(values - location))
    return location, scale


def _huber_regression(design, response, num_iterations=50, tuning=1.345):
    """Robust linear regression with Huber weights (iteratively reweighted least squares)."""
    weights = np.ones(len(response))
    fitted = np.zeros(len(response))
    for _ in range(num_iterations):
        sqrt_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design * sqrt_weights[:, None], response * sqrt_weights,
                                       rcond=None)[0]
        new_fitted = design @ coefficients
        residuals = response - new_fitted
        scale = MAD_TO_SD * np.median(np.abs(residuals - np.median(residuals)))
        if scale <= 0:
            return new_fitted
        scaled = np.abs(residuals) / (tuning * scale)
        weights = np.where(scaled <= 1, 1.0, 1.0 / np.maximum(scaled, 1e-12))
        if np.allclose(new_fitted, fitted, rtol=0, atol=1e-10 * max(scale, 1e-12)):
            break
        fitted = new_fitted
    return new_fitted


def _natural_spline_basis(position, knots):
    """Natural cubic spline basis with an intercept (Hastie et al., ESL eq. 5.4-5.5).

    The fitted curve is cubic between the knots and linear beyond the outer
    knots, so it can follow the steep changes of noise with degree near the
    lowest degrees without the end effects of a global polynomial.
    """
    knots = np.asarray(knots, dtype=np.float64)

    def truncated(j):
        return ((np.maximum(position - knots[j], 0) ** 3 - np.maximum(position - knots[-1], 0) ** 3)
                / (knots[-1] - knots[j]))

    last = truncated(len(knots) - 2)
    return np.column_stack([np.ones_like(position), position]
                           + [truncated(j) - last for j in range(len(knots) - 2)])


SPLINE_KNOTS = (0.25, 0.5, 0.75)
LOG_TAIL = 0.95


def _covariate_position(covariate, log_tail=None):
    """Rank quantile of the covariate, optionally extended linearly in log scale for its top tail.

    Rank quantiles make the trend insensitive to the scale and skew of degrees,
    but they squeeze the largest hubs together: a hub with ten times the degree
    of the 95th percentile sits at the same position as a node just above it.
    With ``log_tail``, positions above that quantile grow linearly with the log
    of the covariate instead, continuing the slope of the log covariate between
    the 75th percentile and ``log_tail``.
    """
    position = (rankdata(covariate) - 0.5) / max(len(covariate), 1)
    if log_tail is None or len(covariate) < 20:
        return position
    positive = covariate[covariate > 0]
    log_covariate = np.log(np.maximum(covariate, positive.min() if positive.size else 1.0))
    upper_quartile, top = np.quantile(log_covariate, [0.75, log_tail])
    if top <= upper_quartile:
        return position
    slope = (log_tail - 0.75) / (top - upper_quartile)
    return np.where(log_covariate > top, log_tail + slope * (log_covariate - top), position)


def covariate_adjusted_zscores(statistic, covariate, trend="spline", polynomial_degree=2,
                               knots=SPLINE_KNOTS, log_tail=None):
    """Robust z-scores of a statistic after removing its trend with a covariate.

    The expected value and the spread of the statistic are modelled as smooth
    functions of the covariate's rank quantile, fitted with robust Huber
    regression so that the minority of differential nodes does not pull the
    trend. Using a smooth global trend, rather than comparing each node only
    to nodes of the same degree, keeps the signal of a group of changed nodes
    that happen to share a degree.

    Args:
        statistic: array of shape (n,); non-finite values give NaN scores.
        covariate: array of shape (n,), e.g., node degrees.
        trend: "spline" for a natural cubic spline with three knots (the
            default), or "polynomial". The spline has as many parameters as a
            quadratic, so a group of changed nodes sharing a degree keeps its
            signal as well, but it is linear beyond the outer knots. A
            quadratic bends at the ends and misfits the lowest-degree nodes
            (e.g., genes outside any co-expression module), giving them too
            large z-scores.
        polynomial_degree: degree of the polynomial trend.
        knots: knots of the spline, as rank quantiles of the covariate.
        log_tail: if given (e.g., 0.95), the covariate positions above this
            quantile grow with the log of the covariate rather than its rank,
            so that the trend extrapolates to the largest hubs.

    Returns:
        np.ndarray of shape (n,) with the z-scores.
    """
    statistic = np.asarray(statistic, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    valid = np.isfinite(statistic) & np.isfinite(covariate)
    zscores = np.full_like(statistic, np.nan)
    num_valid = valid.sum()

    position = _covariate_position(covariate[valid], log_tail)
    if trend == "spline":
        design = _natural_spline_basis(position, knots)
    elif trend == "polynomial":
        design = np.column_stack([position ** k for k in range(polynomial_degree + 1)])
    else:
        raise ValueError(f'trend must be "spline" or "polynomial", got {trend!r}')
    if num_valid <= design.shape[1] + 1:
        return zscores
    values = statistic[valid]

    residuals = values - _huber_regression(design, values)
    # E|X| = sd * sqrt(2 / pi) for a normal variable
    spread = _huber_regression(design, np.abs(residuals)) * np.sqrt(np.pi / 2)
    global_scale = _robust_location_scale(residuals)[1]
    floor = max(global_scale, np.finfo(float).eps) * 0.1
    zscores[valid] = residuals / np.maximum(spread, floor)
    return zscores


def cauchy_combination(pvalues):
    """Combines the p-values in each row with the Cauchy combination test (ACAT).

    The combined p-value stays valid under arbitrary dependence between the
    p-values of a row, and is dominated by the smallest ones, so a signal in a
    few of them is not diluted by the rest (Liu & Xie, JASA 2020). NaNs are
    ignored; rows without any p-value give NaN.
    """
    pvalues = np.clip(np.asarray(pvalues, dtype=np.float64), 1e-300, 1 - 1e-16)
    if pvalues.ndim == 1:
        pvalues = pvalues[:, None]
    combined = np.full(pvalues.shape[0], np.nan)
    rows = ~np.all(np.isnan(pvalues), axis=1)
    # tan((0.5 - p) * pi), written as 1 / tan(p * pi) to stay precise for tiny p
    statistic = np.nanmean(1 / np.tan(pvalues[rows] * np.pi), axis=1)
    combined[rows] = 0.5 - np.arctan(statistic) / np.pi
    # the arctangent loses precision for very large statistics; use its tail instead
    large = rows.copy()
    large[rows] = statistic > 1e15
    combined[large] = 1 / (statistic[statistic > 1e15] * np.pi)
    return combined


def _restandardize(scores):
    """Centres and scales scores by their median and MAD (NaNs are kept)."""
    finite = np.isfinite(scores)
    if finite.sum() < 3:
        return scores
    location, scale = _robust_location_scale(scores[finite])
    return (scores - location) / max(scale, np.finfo(float).eps)


def _row_nanmean(values):
    combined = np.full(values.shape[0], np.nan)
    rows = ~np.all(np.isnan(values), axis=1)
    combined[rows] = np.nanmean(values[rows], axis=1)
    return combined


def empirical_null_test(distances, covariate, trend="spline", combine="cauchy", dimensions=None, window=2):
    """Tests every node for a larger than expected shift between two graphs.

    Args:
        distances: array of shape (n,) or (n, c) with a node's distance between
            the two embeddings for c combinations of dimension and metric.
        covariate: array of shape (n,) to adjust for, typically node degree.
        trend: shape of the degree trend, see :func:`covariate_adjusted_zscores`.
        combine: how the z-scores of the c columns are combined.

            - "cauchy" (default): the columns of ``window`` consecutive
              dimensions (with all their metrics) are averaged, each window is
              re-standardised, and the windows are combined with
              :func:`cauchy_combination`. A change that shows in only a few
              dimensions, typically the leading ones in binary networks, keeps
              its signal. The trend extrapolates to the largest hubs in log
              degree (``log_tail`` of :func:`covariate_adjusted_zscores`),
              since a single window is more sensitive to their misfit.
            - "mean": all columns are averaged, as in the first version of
              this test. It misses most changes that show in only a few
              dimensions, but is somewhat more powerful when a change spreads
              over many dimensions, as in co-expression networks.
        dimensions: array of shape (c,) with the embedding dimension of every
            column, used to form the windows; by default every column is its
            own dimension.
        window: number of consecutive dimensions averaged in a window.

    Returns:
        tuple ``(z, pvalues, qvalues)`` of arrays of shape (n,). Larger z means
        a larger shift than nodes of similar degree; p-values are one-sided.
        Nodes without a positive distance (e.g., isolated in both graphs) get NaN.
    """
    if combine not in ("cauchy", "mean"):
        raise ValueError(f'combine must be "cauchy" or "mean", got {combine!r}')
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[:, None]
    # a node with no positive distance in any combination carries no
    # information (typically it is isolated in both graphs, so its embedding is
    # zero); it is left untested instead of pulling down the low-degree trend
    with np.errstate(invalid="ignore"):
        untestable = ~np.any(distances > 0, axis=1)
    distances = np.where(untestable[:, None], np.nan, distances)

    zscores = []
    for column in distances.T:
        positive = column[np.isfinite(column) & (column > 0)]
        # distances of exactly zero get the smallest observed positive distance
        floor = positive.min() if positive.size else 1.0
        with np.errstate(invalid="ignore"):
            log_distance = np.log(np.where(np.isfinite(column), np.maximum(column, floor), np.nan))
        z = covariate_adjusted_zscores(log_distance, covariate, trend=trend,
                                       log_tail=LOG_TAIL if combine == "cauchy" else None)
        zscores.append(_restandardize(z) if combine == "cauchy" else z)
    zscores = np.column_stack(zscores)

    if combine == "mean":
        # averaging correlated z-scores changes their spread; re-standardise robustly
        combined = _restandardize(_row_nanmean(zscores))
    else:
        if dimensions is None:
            dimensions = np.arange(zscores.shape[1])
        dimensions = np.asarray(dimensions)
        if dimensions.shape != (zscores.shape[1],):
            raise ValueError(f"dimensions must have one entry per column, got {dimensions.shape} "
                             f"for {zscores.shape[1]} columns")
        levels = np.unique(dimensions)
        width = min(window, len(levels))
        windows = [_restandardize(_row_nanmean(zscores[:, np.isin(dimensions, levels[start:start + width])]))
                   for start in range(len(levels) - width + 1)]
        pvalues = cauchy_combination(norm.sf(np.column_stack(windows)))
        # the Cauchy p-values are valid but conservative for positively
        # dependent windows; re-standardising their z-scores calibrates them
        # to the bulk of the nodes, as for the mean
        with np.errstate(divide="ignore"):
            combined = _restandardize(norm.isf(np.clip(pvalues, 1e-300, 1)))

    pvalues = norm.sf(combined)
    return combined, pvalues, benjamini_hochberg(pvalues)
