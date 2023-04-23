import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import umap
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA


# plotting the 3D components with a color and overlaying names
def plot_3d(component1, component2, component3, color=None, names=None, color_type='nominal', save_dir=None, show=True):
    if color is None:
        color = np.ones(len(component1))

    if color_type == 'nominal':
        colorscale = 'Rainbow'
    elif color_type == 'numeric':
        colorscale = 'Solar'
    else:
        raise Exception('Wrong color type, can be nominal or numeric')

    if names is not None:
        fig = go.Figure(data=go.Scatter3d(
            x=component1,
            y=component2,
            z=component3,
            mode='markers',
            hovertemplate='<b>(%{x},%{y})</b>' + "<br>" +
            '<b>%{text}</b>' + '<extra></extra>',
            text=[f'Name: {names[i]} <br>Cluster: {color[i]}' for i in range(
                len(names))],
            marker=dict(
                size=4,
                color=color,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                showscale=True,
                line_width=1
            )
        ))
    else:
        fig = go.Figure(data=go.Scatter3d(
            x=component1,
            y=component2,
            z=component3,
            mode='markers',
            marker=dict(
                size=4,
                color=color,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                showscale=True,
                line_width=1
            )
        ))

    fig.update_layout(margin=dict(l=100, r=100, b=100,
                      t=100), width=800, height=600)

    if show:
        fig.show()

    if save_dir:
        fig.write_html(save_dir)


# plotting the 2D components with a color and overlaying names
def plot_2d(component1, component2, color=None, names=None, color_type='nominal', save_dir=None, show=True):

    if color is None:
        color = np.ones(len(component1))

    if color_type == 'nominal':
        colorscale = 'Rainbow'
    elif color_type == 'numeric':
        colorscale = 'Solar'
    else:
        raise Exception('Wrong color type, can be nominal or numeric')

    if names is not None:
        fig = go.Figure(data=go.Scatter(
            x=component1,
            y=component2,
            mode='markers',
            hovertemplate='<b>(%{x},%{y})</b>' + "<br>" +
            '<b>%{text}</b>' + '<extra></extra>',
            text=[f'Name: {names[i]} <br>Cluster: {color[i]}' for i in range(
                len(names))],
            marker=dict(
                size=6,
                color=color,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                showscale=True,
                line_width=1
            )
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=component1,
            y=component2,
            mode='markers',
            marker=dict(
                size=6,
                color=color,  # set color equal to a variable
                colorscale=colorscale,  # one of plotly colorscales
                showscale=True,
                line_width=1
            )
        ))
    fig.update_layout(margin=dict(l=100, r=100, b=100,
                      t=100), width=800, height=600)

    if show:
        fig.show()

    if save_dir:
        fig.write_html(save_dir)


def plot_embeddings(embeddings, color=None, names=None, color_type='nominal',  save_dir=None, show=True):
    if (embeddings.shape[1] == 2):
        plot_2d(embeddings[:, 0], embeddings[:, 1], color=color,
                color_type=color_type, names=names,  save_dir=save_dir, show=show)
    else:
        plot_3d(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], color=color,
                color_type=color_type, names=names,  save_dir=save_dir, show=show)


def dim_reduction(data, algorithm, n_components=3):

    if np.size(data, axis=1) < n_components:
        n_components = np.size(data, axis=1)

    if algorithm.casefold() == 'tsne'.casefold():
        reducer = TSNE(n_components=n_components,
                       perplexity=np.sqrt(np.size(data, axis=0)), n_iter=2000, n_jobs=-1)
    elif algorithm.casefold() == 'umap'.casefold():
        reducer = umap.UMAP(n_components=n_components)
    elif algorithm.casefold() == 'isomap'.casefold():
        reducer = Isomap(n_components=n_components)
    elif algorithm.casefold() == 'pca'.casefold():
        reducer = PCA(n_components=n_components)
    else:
        raise Exception(
            'Unknown dimensionality reduction algorithm. Options are tsne, umap, isomap, pca')

    return reducer.fit_transform(data)
