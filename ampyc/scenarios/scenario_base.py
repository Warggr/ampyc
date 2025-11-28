'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from dataclasses import dataclass, field
from pprint import pformat
import numpy as np

from ampyc.systems import SystemBase


class Scenario:
    '''
    Base class for parameters. This class holds all parameters as dataclasses for a single experiment, i.e.,
    - sys: system parameters
    - sim: simulation parameters
    '''

    @dataclass
    class sim:
        '''
        Minimally required simulation parameters, more can be added as needed.

        Parameters:
            num_steps (int): Number of simulation steps.
            num_traj (int): Number of trajectories to simulate.
            x_0 (np.ndarray): Initial state for the simulation.
        '''
        x_0: np.ndarray
        y_reference: np.ndarray | float = 0.0
        num_traj: int = 1
        num_steps: int = 150

        def __post_init__(self):
            print('Call Scenario.sim __post_init__')
            if np.isscalar(self.y_reference):
                self.y_reference = self.y_reference * np.ones((self.num_steps, 1))


    def __init__(self, sys: SystemBase, sim: dict | None = None):
        """
        Build dataclasses for controller, system, simulation, and plotting parameters.

        Args:
            sys: System
            sim (dict | None): Simulation parameters. If None, default parameters are used.
        """
        self.sys = sys
        print('Call self.sim() @', self.sim)
        self.sim = self.sim() if sim is None else self.sim(**sim)

        print(f'Successfully initialized experiment \'{self.name}\'.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n' + \
            f'    sys={self.sys},\n' + \
            f'    sim={self.sim},\n' + \
            ')'

    def __str__(self) -> str:
        """String representation of the parameters.

        Returns:
            str: Formatted string representation of the stored parameters.
        """
        return f'Parameters:\n' + \
            f'    sys:\n {pformat(self.sys.__dict__, indent=8, width=60).strip("{}")}\n' + \
            f'    sim:\n {pformat(self.sim.__dict__, indent=8, width=50).strip("{}")}\n'
