'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from ampyc.noise import ZeroNoise

from ampyc.systems import SystemBase
from ampyc.typing import Noise


@dataclass
class LinearSystemParams:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray | Literal[0] = 0

    n: int = field(init=False)
    m: int = field(init=False)
    p: int = field(init=False)

    A_u: np.ndarray | None = None
    b_u: np.ndarray | None = None
    A_x: np.ndarray | None = None
    b_x: np.ndarray | None = None
    A_w: np.ndarray | None = None
    b_w: np.ndarray | None = None

    noise_generator: Noise = None

    def __post_init__(self):
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]
        assert self.A.shape == (self.n, self.n)
        assert self.B.shape == (self.n, self.m)
        assert self.C.shape == (self.p, self.n)
        if self.D == 0:
            self.D = np.zeros((self.p, self.m))
        else:
            assert self.D.shape == (self.p, self.m)
        if self.noise_generator is None:
            self.noise_generator = ZeroNoise(self.n)


class LinearSystem(SystemBase):
    '''
    Implements a linear system of the form:
    .. math::
        x_{k+1} = A x_k + B u_k + w_k \\
        y_k = C x_k + D u_k

    where :math:`x` is the state, :math:`u` is the input, :math:`y` is the output, and :math:`w` is a disturbance.
    '''
    Params = LinearSystemParams

    def update_params(self, params: LinearSystemParams):
        super().update_params(params)
        assert params.A.shape == (self.n, self.n), 'A must have shape (n,n)'
        assert params.B.shape == (self.n, self.m), 'B must have shape (n,m)'
        assert params.C.shape[1] == self.n, 'C must have shape (num_output, n)'
        assert params.D.shape[1] == self.m, 'D must have shape (num_output, m)'
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.A @ x.reshape(self.n, 1) + self.B @ u.reshape(self.m, 1)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.C @ x.reshape(self.n, 1) + self.D @ u.reshape(self.m, 1)

