"""Figures for the revision, in the style of the paper's simulation figures
(plotly default template, 800 x 600 px = 600 x 450 pt PDFs, violin + box per method with the
mean as a dotted line, grey dashed lines joining the same network pair across
methods, titles like "deg-naive, 100 nodes, 6 communities").

Reads benchmarks/results/*.csv and writes PDFs to benchmarks/results/figures/.

    python benchmarks/revision_figures.py
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "figures")
os.environ.setdefault("BROWSER_PATH", "/opt/pw-browsers/chromium")

WIDTH, HEIGHT = 800, 600  # px; kaleido writes 600 x 450 pt, the size of the paper's PDFs
FONT = dict(size=30, family="DejaVu Sans")
# the paper's colours for its three methods, then new ones for the competitors
COLOURS = {
    "n2v2r_b": "green", "n2v2r_e1": "darkorange", "DeDi": "mediumpurple",
    "node2vec": "royalblue", "DeepWalk": "deepskyblue", "PLEX.I": "crimson", "DGCA": "goldenrod",
    "n2v2r_elbow": "darkorange", "n2v2r_eucl": "steelblue", "n2v2r_cos": "teal",
}
NAMES = {  # result-file names -> figure labels
    "n2v2r (default)": "n2v2r_b", "node2vec + Procrustes": "node2vec", "DeepWalk + Procrustes": "DeepWalk",
    "PLEX.I (50 trainings)": "PLEX.I", "DGCA-style": "DGCA", "DeDi": "DeDi",
    "n2v2r elbow dimension": "n2v2r_elbow", "n2v2r euclidean only": "n2v2r_eucl", "n2v2r cosine only": "n2v2r_cos",
}
EXTRA_COLOURS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
                 "#FF97FF", "#FECB52"]


def colour(name, i):
    return COLOURS.get(name, "steelblue")  # ablation variants other than the default


def violins(table, methods, value, pair, title, ylabel, path, yrange=(0, 1), angle=None):
    """One violin per method; table has columns method, value and pair. Tick
    labels are tilted when they would not fit side by side."""
    if angle is None:
        angle = 0 if sum(len(m) for m in methods) <= 30 else -35
    wide = table.pivot_table(index=pair, columns="method", values=value)
    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(go.Violin(x=[method] * len(wide), y=wide[method], name=method, line_color=colour(method, i),
                                line_width=2, box_visible=True, meanline_visible=True, points=False,
                                showlegend=False, spanmode="hard"))
    for _, row in wide[methods].iterrows():  # the same pair joined across methods, over the violins
        fig.add_trace(go.Scatter(x=methods, y=row.to_numpy(), mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(60,60,60,0.7)", width=0.8, dash="dash")))
    fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=FONT, title=dict(text=title, font_size=33),
                      margin=dict(l=90, r=20, t=60, b=70 if angle == 0 else 140))
    fig.update_yaxes(title=ylabel, range=[yrange[0] - 0.05, yrange[1] + 0.05])
    fig.update_xaxes(tickangle=angle, tickfont=dict(size=30 if len(methods) <= 3 else 24))
    fig.write_image(path)


def paper_sims():
    for path in sorted(glob.glob(os.path.join(RESULTS, "paper_sims_*-com.csv"))):
        table = pd.read_csv(path)
        regime, n, m = table.regime.iloc[0], int(table.nodes.iloc[0]), int(table.communities.iloc[0])
        label = "deg-naive" if regime == "naive" else "random"
        methods = ["n2v2r_b", "n2v2r_e1", "DeDi", "node2vec", "DeepWalk", "PLEX.I"]
        violins(table, methods, "recall", "pair", f"{label}, {n} nodes, {m} communities", "recall",
                os.path.join(OUT, f"paper_sims_{n}-nodes_{m}-com_degNaive-{regime == 'naive'}_recall.pdf"))


def competitor_replicates():
    files = [f for f in glob.glob(os.path.join(RESULTS, "competitor_replicates_*.csv"))
             if "summary" not in f and "_paper-setting" not in f]
    table = pd.concat([pd.read_csv(f) for f in files])
    table = table[table.method.isin(NAMES)].assign(method=lambda t: t.method.map(NAMES))
    for setting, block in table.groupby("setting"):
        methods = [m for m in ["n2v2r_b", "node2vec", "DeepWalk", "PLEX.I", "DGCA", "DeDi"] if m in set(block.method)]
        lo = min(0.4, np.floor(block.auroc.min() * 10) / 10)
        violins(block, methods, "auroc", "replicate", setting, "AUROC",
                os.path.join(OUT, "competitors_" + setting.replace(" ", "_") + "_auroc.pdf"), yrange=(lo, 1))


def split_merge():
    table = pd.read_csv(os.path.join(RESULTS, "revision_splitmerge.csv"))
    table = table.assign(method=table.method.map(NAMES))
    for (setting, change), block in table.groupby(["setting", "change"]):
        methods = [m for m in ["n2v2r_b", "n2v2r_elbow", "n2v2r_eucl", "n2v2r_cos", "DeDi", "DGCA"]
                   if m in set(block.method)]
        violins(block, methods, "auroc", "replicate", f"{setting}, community {change}", "AUROC",
                os.path.join(OUT, f"splitmerge_{setting.replace(' ', '_')}_{change}_auroc.pdf"), yrange=(0.4, 1))


ABLATION_LABELS = {
    "embedding: joint UASE (default)": "n2v2r_b", "embedding: ULSE": "ULSE", "embedding: OMNI": "OMNI",
    "embedding: separate ASE + Procrustes": "ASE+Procr.", "embedding: none (raw adjacency rows)": "raw rows",
    "distance: euclidean only": "eucl only", "distance: cosine only": "cos only",
    "aggregation: median rank": "median rank", "dimension: elbow": "elbow", "DeDi": "DeDi",
    **{f"dimension: {d} only": f"dim {d}" for d in (4, 8, 16, 24)},
}


def ablation():
    table = pd.read_csv(os.path.join(RESULTS, "revision_ablation.csv"))
    table = table[table.variant.isin(ABLATION_LABELS)].assign(method=lambda t: t.variant.map(ABLATION_LABELS))
    for scenario, block in table.groupby("scenario"):
        title = {"dcsbm": "block model", "coexpression": "co-expression"}[scenario] + ", 10% switch"
        violins(block, list(ABLATION_LABELS.values()), "auroc", "replicate", title, "AUROC",
                os.path.join(OUT, f"ablation_{scenario}_auroc.pdf"), yrange=(0.4, 1), angle=-45)


def ablation8():
    """The same variants on the 8-community split, merge and switch settings."""
    table = pd.read_csv(os.path.join(RESULTS, "revision_ablation8.csv"))
    table = table[table.variant.isin(ABLATION_LABELS)].assign(method=lambda t: t.variant.map(ABLATION_LABELS))
    for (setting, change), block in table.groupby(["setting", "change"]):
        lo = min(0.4, np.floor(block.auroc.min() * 10) / 10)
        violins(block, list(ABLATION_LABELS.values()), "auroc", "replicate", f"{setting}, community {change}",
                "AUROC", os.path.join(OUT, f"ablation8_{setting.replace(' ', '_')}_{change}_auroc.pdf"),
                yrange=(lo, 1), angle=-45)


def hela_dimensions():
    spectrum = pd.read_csv(os.path.join(RESULTS, "revision_hela_spectrum.csv"))
    fig = go.Figure()
    for i, (comparison, block) in enumerate(spectrum.groupby("comparison", sort=False)):
        fig.add_trace(go.Scatter(x=block.dimension, y=block.singular_value, mode="lines+markers", name=comparison,
                                 line=dict(color=EXTRA_COLOURS[i], width=2), marker=dict(size=8)))
        elbow = int(block.elbow.iloc[0])
        fig.add_trace(go.Scatter(x=[elbow], y=block.singular_value.iloc[[elbow - 1]], mode="markers", showlegend=False,
                                 marker=dict(size=16, symbol="circle-open", color=EXTRA_COLOURS[i], line_width=3)))
    fig.add_vrect(x0=4, x1=24, fillcolor="green", opacity=0.08, line_width=0)
    fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=FONT, title="HeLa, singular values",
                      legend=dict(x=0.45, y=0.95), margin=dict(l=90, r=20, t=60, b=80))
    fig.update_xaxes(title="dimension")
    fig.update_yaxes(title="singular value", type="log")
    fig.write_image(os.path.join(OUT, "hela_spectrum.pdf"))

    rankings = pd.read_csv(os.path.join(RESULTS, "revision_hela_rankings.csv"))
    rankings = rankings[rankings.ranking.str.endswith(" only")]
    fig = go.Figure()
    for i, (comparison, block) in enumerate(rankings.groupby("comparison", sort=False)):
        dims = block.ranking.str.replace(" only", "").astype(int)
        fig.add_trace(go.Scatter(x=dims, y=block.top100_shared_with_default / 100, mode="lines+markers",
                                 name=comparison, line=dict(color=EXTRA_COLOURS[i], width=2), marker=dict(size=8)))
    fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=FONT,
                      title="HeLa, single dimension vs n2v2r_b", legend=dict(x=0.45, y=0.1),
                      margin=dict(l=90, r=20, t=60, b=80))
    fig.update_xaxes(title="dimension")
    fig.update_yaxes(title="top-100 overlap with n2v2r_b", range=[0, 1.03])
    fig.write_image(os.path.join(OUT, "hela_dimensions_overlap.pdf"))

    tests = pd.read_csv(os.path.join(RESULTS, "revision_hela_set_test.csv"))
    fig = go.Figure()
    for i, (comparison, block) in enumerate(tests.groupby("comparison", sort=False)):
        fig.add_trace(go.Bar(x=block.dimensions.str.replace("default (Borda 4-24)", "n2v2r_b", regex=False),
                             y=block.called, name=comparison + " (all)", marker_color=EXTRA_COLOURS[i], opacity=0.5))
        fig.add_trace(go.Bar(x=block.dimensions.str.replace("default (Borda 4-24)", "n2v2r_b", regex=False),
                             y=block.cell_cycle_called, name=comparison + " (cell cycle)",
                             marker_color=EXTRA_COLOURS[i]))
    fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=dict(size=24, family="DejaVu Sans"), barmode="group",
                      title="HeLa, Reactome sets called (q ≤ 0.1)",
                      legend=dict(orientation="h", x=0, y=-0.3, yanchor="top", font=dict(size=16)),
                      margin=dict(l=90, r=20, t=60, b=170))
    fig.update_yaxes(title="sets called")
    fig.update_xaxes(tickangle=-30)
    fig.write_image(os.path.join(OUT, "hela_dimensions_set_test.pdf"))


def hela_grid():
    grid = pd.read_csv(os.path.join(RESULTS, "revision_hela_grid.csv"))
    for value, title, scale in (("top100_shared_with_paper", "top 100 shared with reference", "Greens"),
                                ("G1_S_called", "G1 → S sets called", "Greens"),
                                ("null_called", "null split sets called", "Reds")):
        wide = grid.pivot(index="metacell_k", columns="soft_power", values=value).sort_index(ascending=False)
        fig = go.Figure(go.Heatmap(z=wide.to_numpy(), x=[str(c) for c in wide.columns],
                                   y=[str(i) for i in wide.index], colorscale=scale, text=wide.to_numpy(),
                                   texttemplate="%{text}", textfont=dict(size=28), showscale=False))
        fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=FONT, title="HeLa, " + title,
                          margin=dict(l=90, r=20, t=60, b=70))
        fig.update_xaxes(title="soft power")
        fig.update_yaxes(title="metacell size k")
        fig.write_image(os.path.join(OUT, f"hela_grid_{value}.pdf"))


def runtime():
    table = pd.read_csv(os.path.join(RESULTS, "runtime.csv"))
    nodes = sorted(table.nodes.unique())
    for value, label in (("seconds", "seconds"), ("peak_memory_gb", "peak memory (GB)")):
        fig = go.Figure()
        for (method, storage, k), block in table.groupby(["method", "storage", "networks"]):
            name = "n2v2r_b" if method == "n2v2r" else method
            fig.add_trace(go.Scatter(x=block.nodes, y=block[value], mode="lines+markers", name=f"{name} {storage} K={k}",
                                     line=dict(color=COLOURS[name], width=2.5, dash="solid" if storage == "dense" else "dot"),
                                     marker=dict(size=10, symbol="circle" if k == 2 else "square")))
        fig.update_layout(template="plotly", width=WIDTH, height=HEIGHT, font=dict(size=24, family="DejaVu Sans"),
                          title="runtime" if value == "seconds" else "memory",
                          legend=dict(font=dict(size=16), orientation="h", x=0, y=-0.22, yanchor="top",
                                      traceorder="reversed"),
                          margin=dict(l=100, r=20, t=60, b=150))
        fig.update_xaxes(title="nodes", type="log", tickvals=nodes, ticktext=[f"{n // 1000}k" for n in nodes])
        fig.update_yaxes(title=label, type="log", dtick=1, exponentformat="power")
        fig.write_image(os.path.join(OUT, f"runtime_{value}.pdf"))


def main():
    os.makedirs(OUT, exist_ok=True)
    for make in (paper_sims, competitor_replicates, split_merge, ablation, ablation8, hela_dimensions, hela_grid, runtime):
        make()
        print(make.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
