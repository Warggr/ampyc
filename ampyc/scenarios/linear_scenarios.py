import numpy as np

from ampyc.noise import GaussianNoise
from ampyc.scenarios.scenario_base import Scenario
from ampyc.systems.linear_system import LinearSystem


class DoubleIntegrator(Scenario):
    name: str = "Double integrator"

    def __init__(
        self,
        amplitude: float = 2,
        noise_ratio: float = 0,
        dt: float = 0.1,
        mass: float = 1,
        *,
        sim: dict | None = None,
    ):
        sim = sim or {}
        s_noise = noise_ratio * amplitude
        sys = LinearSystem(
            A=np.array([[1, dt], [0, 1]]),
            B=np.array([[(dt / mass) ** 2 / 2], [dt / mass]]),
            C=np.array([[1, 0]]),
            D_w=np.array([[1]]),
            noise_generator=GaussianNoise(mean=np.array([0]), covariance=np.array([[s_noise**2]])),
        )
        super().__init__(
            sys=sys,
            sim=dict(**sim, x_0=amplitude * np.ones((sys.n,)), y_reference=1),
        )


class DoubleIntegratorFullyObservable(Scenario):
    name: str = "Double integrator"

    def __init__(
        self,
        amplitude: float = 2,
        noise_ratio: float = 0,
        dt: float = 0.1,
        mass: float = 1,
        *,
        sim: dict | None = None,
    ):
        sim = sim or {}
        s_noise = noise_ratio * amplitude
        sys = LinearSystem(
            A=np.array([[1, dt], [0, 1]]),
            B=np.array([[(dt / mass) ** 2 / 2], [dt / mass]]),
            C=np.eye(2),
            D_w=np.eye(2),
            noise_generator=GaussianNoise(mean=np.array([0, 0]), covariance=s_noise**2*np.eye(2)),
        )
        super().__init__(
            sys=sys,
            sim=dict(**sim, x_0=amplitude, y_reference=1),
        )


class SingleIntegrator(Scenario):
    name = "Single integrator"

    def __init__(self):
        super().__init__(
            sys=LinearSystem(A=np.eye(1), B=np.eye(1), C=np.eye(1)),
            sim=dict(
                x_0=np.array([2.0]),
                num_steps=120,
            ),
        )
