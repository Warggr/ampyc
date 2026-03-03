'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np
import matplotlib.pyplot as plt

from ampyc.utils import Polytope
from .plot_x import plot_constraints

def plot_u(fig_number: int,
           u: np.ndarray,
           U: Polytope | None,
           label: str | None = None,
           legend_loc: str = 'upper right',
           title: str | None = None,
           axes_labels: list[str] = ['u'],
           x_label: str = 'time',
           **kwargs,
           ) -> None:
    '''
    Plots the control input u over time, including the input constraint set U.
    This function assumes a 1D control input (m=1) and plots the control variable against time.

    Args:
        fig_number (int): The figure number to use for the plot. This allows multiple plots in the same figure.
        u (np.ndarray): The control input trajectory, shape (N, m=1, T), where N is the number of trajectories,
                        m is the control dimension, and T is the number of time steps.
        U (Polytope | None): The input constraint set.
        label (str | None): Label for the plot line.
        legend_loc (str): Location of the legend in the plot.
        title (str | None): Title of the plot.
        axes_labels (list[str]): Label for the y axes.
        kwargs: All other arguments are passed to ax.plot().
    '''
    # check if the figure number is already open
    if plt.fignum_exists(fig_number):
        fig = plt.figure(fig_number)
        ax = fig.axes[0]
    else:
        fig = plt.figure(fig_number)
        ax = plt.gca()

    num_steps = u.shape[0]

    ax.plot(u, **kwargs, label=label)
    if U is not None:
        plot_constraints(fig, U)
    ax.set_xlabel(x_label)
    ax.set_ylabel(axes_labels[0])
    ax.set_xlim([0, num_steps])
    ax.grid(visible=True)

    if title is not None:
        ax.set_title(title)
    
    if label is not None:
        # remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=legend_loc)
    
    fig.tight_layout()
