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
    B_w: np.ndarray | Literal[0] = 0
    D_w: np.ndarray | Literal[0] = 0

    n: int = field(init=False)
    m: int = field(init=False)
    p: int = field(init=False)
    q: int | None = field(init=False)

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
        if isinstance(self.D, int) and self.D == 0:
            self.D = np.zeros((self.p, self.m))
        else:
            assert self.D.shape == (self.p, self.m)

        B_w_zero = isinstance(self.B_w, int) and self.B_w == 0
        D_w_zero = isinstance(self.D_w, int) and self.D_w == 0
        if B_w_zero and D_w_zero:
            self.q = 0
        else:
            self.q = self.B_w.shape[1] if not B_w_zero else self.D_w.shape[1]
            if B_w_zero:
                self.B_w = np.zeros((self.n, self.q))
            else:
                assert self.B_w.shape == (self.n, self.q)
            if D_w_zero:
                self.D_w = np.zeros((self.p, self.q))
            else:
                assert self.D_w.shape == (self.p, self.q), f"Expected D_w to have shape {self.p, self.q} but has shape {self.D_w.shape}"
        if self.noise_generator is None:
            self.noise_generator = ZeroNoise(self.q)


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
        self.B_w = params.B_w
        self.D_w = params.D_w

    def f(self, x: np.ndarray, u: np.ndarray):
        """Nominal dynamics."""
        return (
            self.A @ x.reshape(self.n, 1) +
            self.B @ u.reshape(self.m, 1)
        )

    def h(self, x: np.ndarray, u: np.ndarray):
        return (
            self.C @ x.reshape(self.n, 1) +
            self.D @ u.reshape(self.m, 1)
        )

    def step(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        if self.q != 0:
            w = self.noise_generator._generate()
            x_new = self.f(x, u) + self.B_w @ w
            y = self.h(x_new, u) + self.D_w @ w
        else:
            x_new = self.f(x, u)
            y = self.h(x, u)
        return x_new, y
