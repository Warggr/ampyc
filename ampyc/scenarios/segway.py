'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from dataclasses import dataclass, field
import numpy as np
import casadi

from ampyc.noise import NoiseBase, PolytopeNoise
from ampyc.utils import Polytope
from ampyc.scenarios.scenario_base import Scenario
from ampyc.systems.nonlinear_system import NonlinearSystem


class NonlinearSegwaySystem(NonlinearSystem):
    @dataclass
    class Params:
        """
        dt (float): Time step.
        k (float): Spring constant.
        g (float): Gravitational acceleration.
        l (float): Length of the pendulum.
        c (float): Damping coefficient.
        """

        dt: float = 0.1
        k: float = 4.0
        g: float = 9.81
        l: float = 1.3
        c: float = 1.5

        # state constraints
        A_x: np.ndarray | None = field(default_factory=lambda: np.array(
            [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]
            ]))
        b_x: np.ndarray | None = field(default_factory=lambda: np.array(
            [np.deg2rad(45), np.deg2rad(45), np.deg2rad(60), np.deg2rad(60)]).reshape(-1, 1))

        # input constraints
        A_u: np.ndarray | None = field(
            default_factory=lambda: np.array([1, -1]).reshape(-1, 1))
        b_u: np.ndarray | None = field(
            default_factory=lambda: np.array([5, 5]).reshape(-1, 1))

        # noise description
        A_w: np.ndarray | None = field(default_factory=lambda: np.array(
            [
                [1, 0], 
                [-1, 0],
                [0, 1],
                [0, -1]
            ]))
        b_w: np.ndarray | None = field(default_factory=lambda: np.array(
            [1e-6, 1e-6, 1e-6, 1e-6]).reshape(-1, 1))

        # noise generator
        noise_generator: NoiseBase = field(init=False)

    def __init__(self,
        params: "NonlinearSegwaySystem.Params",
    ):
        noise_generator = PolytopeNoise(Polytope(params.A_w, params.b_w))

        super().__init__(n=2, m=1, p=2, noise_generator=noise_generator)
        self.params = params

    def _f(self, x, u, w=None, **kwargs):
        '''
        Nonlinear dynamics function for the inverted pendulum (segway) system.

        Args:
            x (casadi.SX or casadi.MX): State vector [theta, theta_dot].
            u (casadi.SX or casadi.MX): Control input (force).
        '''
        params = self.params
        x_next = casadi.vertcat(
            x[0] + params.dt*x[1],
            x[1] + params.dt*(-params.k*x[0] - params.c*x[1] + casadi.sin(x[0])*params.g/params.l + u)
        )
        return x_next

    def _h(self, x, u, w=None, **kwargs):
        return x


class NonlinearSegway(Scenario):
    '''
    Default parameters for experiments with a nominal nonlinear MPC controller.
    '''
    name = "Nonlinear segway dynamics"


    @dataclass
    class sim(Scenario.sim):
        num_steps: int = 30
        num_traj: int = 20
        x_0: np.ndarray = field(default_factory=lambda: np.array([np.deg2rad(20), 0]))
        y_reference = 0

    def __init__(self, sys: dict|None = None):
        sys_args = sys or {}
        super().__init__(sys=NonlinearSegwaySystem(**sys_args), sim=None)
