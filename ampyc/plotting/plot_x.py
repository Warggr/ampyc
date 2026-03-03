'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ampyc.utils import Polytope
import math


def get_x_figure(fig_number: int, expected_num_axes: int = 2) -> tuple[Figure, list[Axes]]:
    # check if the figure number is already open
    if plt.fignum_exists(fig_number):
        fig = plt.figure(fig_number)
        axes = fig.axes
    else:
        if expected_num_axes <= 5:
            rows, cols = expected_num_axes, 1
        elif expected_num_axes <= 12:
            rows, cols = math.ceil(expected_num_axes / 2), 2
        else:
            rows = math.ceil(math.sqrt(expected_num_axes))
            cols = math.ceil(expected_num_axes / rows)
        fig, axes = plt.subplots(rows, cols, num=fig_number, sharex=True, sharey=True)
        if expected_num_axes == 1:
            axes = [axes]
        else:
            axes = list(axes.flatten())
    return fig, axes


def plot_x_predictions(
    fig_number: int,
    predictions: list[np.ndarray],
    alpha: float = 0.5,
    **kwargs,
) -> None:
    prediction_dim = predictions[0].shape[1]
    _, axes = get_x_figure(fig_number, prediction_dim)
    for i, pred in enumerate(predictions):
        for j in range(prediction_dim):
            axes[j].plot(range(i, i+pred.shape[0]), pred[:, j], **kwargs, alpha=alpha)


def plot_constraints(
    fig: Figure,
    X: Polytope,
):
    mins, maxes = X.bounding_box
    for i, ax in enumerate(fig.axes):
        if np.isfinite(mins[i, 0]):
            ax.axline((-1, mins[i, 0]), slope=0, color='k', linewidth=2)
        if np.isfinite(maxes[i, 0]):
            ax.axline((-1, maxes[i, 0]), slope=0, color='k', linewidth=2)


def plot_x_state_time(
        fig_number: int,
        x: np.ndarray,
        X: Polytope | None,
        label: str | None = None,
        legend_loc: str = 'upper right',
        title: str | None = None,
        axes_labels: list[str] | None = None,
        x_label: str = 'time',
        **kwargs,
        ) -> None:
    '''
    Plots the state trajectory x over time, including the state constraint set X.
    This function plots the state variables against time.

    Args:
        fig_number (int): The figure number to use for the plot. This allows multiple plots in the same figure.
        x (np.ndarray): The state trajectory, shape (N, n, T), where N is the number of time steps,
                        n is the state dimension, and T is the number of trajectories.
        X (Polytope | None): The state constraint set.
        label (str | None): Label for the plot line.
        legend_loc (str): Location of the legend in the plot.
        title (str | None): Title of the plot.
        axes_labels (list[str]): Labels for the x and y axes.
        **kwargs: All other arguments are passed to ax.plot()
    '''

    n = x.shape[1]
    if axes_labels is None:
        axes_labels = [f'x_{{{i+1}}}' for i in range(n)]

    fig, axes = get_x_figure(fig_number, expected_num_axes=n)

    num_steps = x.shape[0]

    for i, ax in enumerate(axes):
        kwargs = {}
        if i == 0:
            kwargs['label'] = label
        ax.plot(x[:,i], **kwargs)
        if i == n - 1:
            ax.set_xlabel(x_label)
        ax.set_ylabel(axes_labels[i])
        ax.set_xlim([0, num_steps])
        ax.grid(visible=True)

    if X is not None:
        plot_constraints(fig, X)

    ax1 = axes[0]
    if title is not None:
        ax1.set_title(title)

    if label is not None:
        # remove duplicate legend entries
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc=legend_loc)

    fig.tight_layout()


def plot_x_state_state(fig_number: int,
                       x: np.ndarray,
                       X: Polytope | None,
                       label: str | None = None,
                       legend_loc: str = 'upper right',
                       title: str | None = None,
                       axes_labels: list[str] = ['x_1', 'x_2'],
                       **kwargs,
                       ) -> None:
    '''
    Plots the state trajectory x in the state space, including the state constraint set X.
    This function assumes a 2D state space (n=2) and plots the two state variables against each other.

    Args:
        fig_number (int): The figure number to use for the plot. This allows multiple plots in the same figure.
        x (np.ndarray): The state trajectory, shape (N, n=2, T), where N is the number of time steps,
                        n is the state dimension, and T is the number of trajectories.
        X (Polytope): The state constraint set.
        params (Params): Parameters for plotting, e.g., color, alpha, and linewidth.
        label (str | None): Label for the plot line.
        legend_loc (str): Location of the legend in the plot.
        title (str | None): Title of the plot.
        axes_labels (list[str]): Labels for the x and y axes.
    '''
    # check if the figure number is already open
    if plt.fignum_exists(fig_number):
        fig = plt.figure(fig_number)
        ax = fig.axes[0]
    else:
        fig = plt.figure(fig_number)
        ax = plt.gca()

    ax.scatter(x[0,0], x[0,1], marker='o', facecolors='none', color='k', label='initial state')
    ax.plot(x[:,0], x[:,1], label=label, **kwargs)
    if X is not None:
        X.plot(ax=ax, fill=False, edgecolor="k", alpha=1, linewidth=2, linestyle='-') 
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.grid(visible=True)

    if hasattr(params, 'zoom_out'):
        ax.set_xlim([i * params.zoom_out for i in X.xlim])
        ax.set_ylim([i * params.zoom_out for i in X.ylim])
    else:
        ax.set_xlim(X.xlim)
        ax.set_ylim(X.ylim)

    if title is not None:
        ax.set_title(title)

    # remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc)

    fig.tight_layout()
