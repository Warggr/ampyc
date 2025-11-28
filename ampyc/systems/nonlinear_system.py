'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2025, Intelligent Control Systems Group, ETH Zurich
%
% This code is made available under an MIT License (see LICENSE file).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from ampyc.systems import SystemBase

class NonlinearSystem(SystemBase):
    '''
    Implements a nonlinear system of the form:
    .. math::
        x_{k+1} = f(x_k, u_k) + w_k \\
        y_k = h(x_k, u_k)

    where :math:`x` is the state, :math:`u` is the input, :math:`y` is the output, and :math:`w` is a disturbance.

    Additionally, the system can store the linear differential dynamics for the system as a list of A, B, C, and
    D matrices for different linearization points. (TODO)
    '''
    pass
