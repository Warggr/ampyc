'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import casadi
import numpy as np

from ampyc.controllers import ControllerBase
from ampyc.utils import Polytope
from typing import Literal


class CasadiWrapper:
    @staticmethod
    def array(*args, **kwargs):
        return casadi.vcat(*args, **kwargs)

    def __getattr__(self, attr):
        """Forward every attribute lookup to casadi

        e.g, CasadiWrapper.sin() := casadi.sin()
        """
        return getattr(casadi, attr)


class NonlinearMPC(ControllerBase):
    '''
    Implements a standard nonlinear nominal MPC controller, see e.g. Section 2.5.5 in:

    J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, "Model Predictive Control: Theory and Design",
    2nd edition, Nob Hill Publishing, 2009.

    More information is provided in Chapter 2 of the accompanying notes:
    https://github.com/IntelligentControlSystems/ampyc/notes/02_nominalMPC.pdf
    '''

    def __init__(
        self,
        N: int = 10,
        Q: np.ndarray | float = 1,
        R: np.ndarray | float = 10,
        #name: str = 'nominal linear MPC',
        X_f: Polytope | Literal[0] | None = 0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        #self.name = name
        self.N = N
        self.Q = Q
        self.R = R
        self.X_f = X_f

    def _init_problem(self, sys, track_x = 0):
        if isinstance(track_x, (float, int)) and track_x == 0:
            track_x = np.zeros(sys.m)

        Q, R = self.Q, self.R
        if isinstance(Q, (float, int)):
            Q = Q * np.eye(sys.n)
        if isinstance(self.R, (float, int)):
            R = R * np.eye(sys.m)
        X_f = self.X_f
        if isinstance(X_f, (float, int)) and X_f == 0:
            X_f = Polytope(A=np.concat([np.eye(sys.n), -np.eye(sys.n)]), b=np.zeros((2*sys.n,)))

        # init casadi Opti object which holds the optimization problem
        self.prob = casadi.Opti()

        # define optimization variables
        self.x = self.prob.variable(sys.n, self.N+1)
        self.u = self.prob.variable(sys.m, self.N)
        self.x_0 = self.prob.parameter(sys.n)

        # define the objective
        objective = 0.0
        def quad_form(x, A):
            return x.T @ A @ x

        for i in range(self.N):
            objective += quad_form(self.x[:, i] - track_x, Q) + quad_form(self.u[:, i].T, R)
        # NOTE: terminal cost is trivially zero due to terminal constraint
        self.objective = objective
        self.prob.minimize(objective)

        # define the constraints
        self.prob.subject_to(self.x[:, 0] == self.x_0)
        for i in range(self.N):
            self.prob.subject_to(self.x[:, i+1] == sys.f(self.x[:, i], self.u[:, i], array_backend=CasadiWrapper()))
            if sys.X is not None:
                self.prob.subject_to(sys.X.A @ self.x[:, i] <= sys.X.b)
            if sys.U is not None:
                self.prob.subject_to(sys.U.A @ self.u[:, i] <= sys.U.b)
        self.prob.subject_to(X_f.A @ (self.x[:, -1] - track_x) <= X_f.b)

    def _define_output_mapping(self):
        return {
            'control': self.u,
            'state': self.x,
        }

