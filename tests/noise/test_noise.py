from typing import Callable
from ampyc.noise import GaussianNoise, NoiseBase, PolytopeNoise, PolytopeVerticesNoise
from ampyc.utils import Polytope
import pytest
import numpy as np


def get_unit_cube_polytope(noise_dim):
    A = np.zeros((2 * noise_dim, noise_dim))
    b = np.zeros((2 * noise_dim,))
    for i in range(noise_dim):
        A[2 * i, i] = 1
        b[2 * i] = 1
        A[2 * i + 1, i] = -1
        b[2 * i + 1] = 1
    return Polytope(A, b)


@pytest.mark.parametrize(
    "noise_factory",
    [
        lambda noise_dim: GaussianNoise(
            mean=np.zeros((noise_dim,)), covariance=np.eye(noise_dim)
        ),
        lambda noise_dim: PolytopeNoise(get_unit_cube_polytope(noise_dim)),
        lambda noise_dim: PolytopeVerticesNoise(get_unit_cube_polytope(noise_dim)),
    ],
)
def test_noise_is_deterministic_with_seed(noise_factory: Callable[[int], NoiseBase]):
    noise_dim = 6
    noise1 = noise_factory(noise_dim)
    noise2 = noise_factory(noise_dim)
    noise1.seed(5)
    noise2.seed(5)
    assert np.all(noise1.generate() == noise2.generate())
