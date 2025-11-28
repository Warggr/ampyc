'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
from __future__ import annotations

import numpy as np
import sys

from ampyc.systems import SystemBase
from ampyc.noise import NoiseBase, ZeroNoise

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ampyc.systems.system_base import ArrayLike, ArrayBackend


class LinearSystem(SystemBase):
    '''
    Implements a linear system of the form:
    .. math::
        x_{k+1} = A x_k + B u_k + w_k \\
        y_k = C x_k + D u_k

    where :math:`x` is the state, :math:`u` is the input, :math:`y` is the output, and :math:`w` is a disturbance.
    '''

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray | Literal[1] = 1,
        D: np.ndarray | Literal[0] = 0,
        B_w: np.ndarray | Literal[0] = 0,
        D_w: np.ndarray | Literal[0] = 0,

        noise_generator: NoiseBase | None = None,
        **kwargs,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        n = self.A.shape[0]
        m = self.B.shape[1]
        assert self.A.shape == (n, n)
        assert self.B.shape == (n, m)
        if isinstance(self.C, int) and self.C == 1:
            p = n
            self.C = np.eye(n)
        else:
            p = self.C.shape[0]
            assert self.C.shape == (p, n)
        if isinstance(self.D, int) and self.D == 0:
            self.D = np.zeros((p, m))
        else:
            assert self.D.shape == (p, m)

        B_w_zero = isinstance(B_w, int) and B_w == 0
        D_w_zero = isinstance(D_w, int) and D_w == 0
        if B_w_zero and D_w_zero:
            q = 0
            self.B_w = 0
            self.D_w = 0
            if self.noise_generator is not None:
                print('Warning: noise generator provided but not used', file=sys.stderr)
        else:
            q = B_w.shape[1] if not B_w_zero else D_w.shape[1]
            if B_w_zero:
                self.B_w = np.zeros((n, q))
            else:
                self.B_w = B_w
                assert self.B_w.shape == (n, q)
            if D_w_zero:
                self.D_w = np.zeros((p, q))
            else:
                self.D_w = D_w
                assert self.D_w.shape == (p, q), f"Expected D_w to have shape {p, q} but has shape {self.D_w.shape}"
        self.q = q
        super().__init__(n=n, m=m, p=p, noise=noise_generator, **kwargs)

    def _f(self, x: ArrayLike, u: ArrayLike, w: ArrayLike | None = None, **kwargs):
        result = (
            self.A @ x.reshape(self.n, 1) +
            self.B @ u.reshape(self.m, 1)
        )
        if w is not None:
            result += self.B_w @ w.reshape(self.q, 1)
        return result

    def _h(self, x: ArrayLike, u: ArrayLike, w: ArrayLike | None = None, **kwargs):
        result = (
            self.C @ x.reshape(self.n, 1) +
            self.D @ u.reshape(self.m, 1)
        )
        if w is not None:
            result += self.D_w @ w.reshape(self.q, 1)
        return result

    def step(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        if self.q != 0:
            w = self.noise_generator._generate()
        else:
            w = None
        x_new = self._f(x, u, w)
        y = self._h(x_new, u, w)
        return x_new, y
