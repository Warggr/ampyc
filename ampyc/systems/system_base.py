'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from abc import ABC, abstractmethod
import numpy as np
import functools
from typing import TypeVar, Callable, Sequence, Any


from ampyc.utils import Polytope
from ampyc.noise import NoiseBase


ArrayLike = TypeVar('ArrayLike')
ArrayBackend = Callable[[Sequence[Any]], ArrayLike]

class SystemBase(ABC):
    '''
    Base class for all systems. It defines the interface for the system and the methods
    that must be implemented by a derived system.

    It defines a general nonlinear system of the form:
    .. math::
        x_{k+1} = f(x_k, u_k) + w_k \\
        y_k = h(x_k, u_k)

    where :math:`x` is the state, :math:`u` is the input, :math:`y` is the output, and :math:`w` is a disturbance.

    The following methods must be implemented by derived classes:
    - update_params: update the system parameters, e.g., after a change in the system dimensions. This is also called
                     during initialization.
    - f: state update function to be implemented by the inherited class
    - h: output function to be implemented by the inherited class

    Usage:
    - get_state: Evaluates :math: `x_{k+1} = f(x_k, u_k) + w_k`
    - get_output: Evaluates :math: `y_k = h(x_k, u_k)`
    - step: Evaluates both get_state and get_output in sequence, i.e.,
        .. math::
            x_{k+1} = f(x_k, u_k) + w_k \\
            y_k = h(x_k, u_k)
        and returns both :math:`x_{k+1}` and :math:`y_k`
    '''

    def __init__(self, *args, **kwargs):
        '''
        Default constructor for the system base class. This method should not be overridden by derived systems, use
        update_params instead.

        Args:
            params: The system parameters derived from a ParamsBase dataclass.
        '''
        self.update_params(*args, **kwargs)

    def update_params(
        self,
        n: int, m: int, p: int,
        X: Polytope | tuple[np.ndarray, np.ndarray] | None = None,
        U: Polytope | tuple[np.ndarray, np.ndarray] | None = None,
        noise_generator: NoiseBase | None = None,
    ) -> None:
        '''
        Updates the system parameters, e.g., after a change in the system dimensions.

        Args:
            params: The new system parameters derived from a ParamsBase dataclass.
        '''

        # system dimensions
        self.n = n
        self.m = m
        self.p = p

        # store systems constraints as polytopes
        def get_constraint_polytope(p: Polytope | tuple[np.ndarray, np.ndarray] | None, expected_dim: int) -> Polytope | None:
            match p:
                case None:
                    return None
                    # return Polytope(np.zeros((0, expected_dim)), np.zeros((0,)))
                case A, b:
                    return Polytope(A, b)
                case c:
                    return c

        self.X = get_constraint_polytope(X, self.n)
        self.U = get_constraint_polytope(U, self.m)

        self.noise_generator = noise_generator

    def step(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Advances the system by one time step, given state x & input u and returns the output.
        This method calls get_state and get_output methods in sequence.

        Args:
            x: Current state of the system
            u: Input to the system
        Returns:
            x_next: Next state of the system after applying the input and adding a disturbance.
            output: Output of the system after evaluating the output function.
        '''
        w = None
        if self.noise_generator is not None:
            w = self.noise_generator.generate()
        x_next = self.f(x, u, w)
        output = self.h(x, u, w)
        return x_next, output

    def f(self, x, u, *args, **kwargs):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self._f(x, u, *args, **kwargs)

    def h(self, x, u, *args, **kwargs):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self._h(x, u, *args, **kwargs)

    @abstractmethod
    def _f(self, x: ArrayLike, u: ArrayLike, w: ArrayLike | None, *,
        array_backend: ArrayBackend = np.array,
    ) -> ArrayLike:
        '''
        Nominal system update function to be implemented by the inherited class.

        Args:
            x: Current state of the system
            u: Input to the system
            w: Noise, optional
        Returns:
            x_next: Next state of the system after applying the input.
        '''
        ...

    @abstractmethod
    def _h(self, x: ArrayLike, u: ArrayLike, w: ArrayLike | None, *,
        array_backend: ArrayBackend = np.array,
    ) -> ArrayLike:
        '''
        System output function to be implemented by the inherited class.

        Args:
            x: Current state of the system
            u: Input to the system
            w: Noise, optional.
        Returns:
            output: Output of the system after evaluating the output function.
        '''
        ...

    def _check_x_shape(self, x: np.ndarray) -> None:
        '''
        Verifies the shape of x.
        Usable if x is a float or an array type which defines a shape property (e.g. numpy and casadi arrays)
        '''
        if hasattr(x, 'shape') and self.n > 1:
            assert x.shape == (self.n, 1) or x.shape == (self.n,), 'x must be {0} dimensional, instead has shape {1}'.format(self.n, x.shape)

    def _check_u_shape(self, u: np.ndarray) -> None:
        '''
        Verifies the shape of u.
        Usable if u is a float or an array type which defines a shape property (e.g. numpy and casadi arrays)
        '''
        if hasattr(u, 'shape') and self.m > 1:
            assert u.shape == (self.m, 1) or u.shape == (self.m,), 'u must be {0} dimensional, instead has shape {1}'.format(self.m, u.shape)
