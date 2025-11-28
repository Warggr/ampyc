'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import cvxpy as cp

from ampyc.controllers import ControllerBase

class MPC(ControllerBase):
    '''
    Implements a standard linear nominal MPC controller, see e.g. Example 2.5 in:

    J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, "Model Predictive Control: Theory and Design",
    2nd edition, Nob Hill Publishing, 2009.

    More information is provided in Chapter 1 of the accompanying notes:
    https://github.com/IntelligentControlSystems/ampyc/notes/01_intro.pdf
    '''

    def __init__(
        self,
        N: int = 10,
        Q: np.ndarray | float = 1,
        R: np.ndarray | float = 10,
        name: str = 'nominal linear MPC',
    ):
        self.name = name
        self.N = N
        self.Q = Q
        self.R = R

    def _init_problem(
        self, sys: LinearSystem,
    ):
        Q, R = self.Q, self.R
        if isinstance(Q, (float, int)):
            Q = Q * np.eye(sys.n)
        if isinstance(self.R, (float, int)):
            R = R * np.eye(sys.m)

        # define optimization variables
        self.x = cp.Variable((sys.n, self.N+1))
        self.u = cp.Variable((sys.m, self.N))
        self.x_0 = cp.Parameter((sys.n))

        # define the objective
        objective = 0.0
        for i in range(self.N):
            objective += cp.quad_form(self.x[:, i], Q) + cp.quad_form(self.u[:, i], R)
        # NOTE: terminal cost is trivially zero due to terminal constraint

        # define the constraints
        constraints = [self.x[:, 0] == self.x_0]
        for i in range(self.N):
            constraints += [self.x[:, i+1] == sys.A @ self.x[:, i] + sys.B @ self.u[:, i]]
            if sys.X is not None:
                constraints += [sys.X.A @ self.x[:, i] <= sys.X.b]
            if sys.U is not None:
                constraints += [sys.U.A @ self.u[:, i] <= sys.U.b]
        constraints += [self.x[:, -1] == 0.0]

        # define the CVX optimization problem object
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _define_output_mapping(self):
        return {
            'control': self.u,
            'state': self.x,
        }

