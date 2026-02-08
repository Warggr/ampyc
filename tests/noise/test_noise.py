from ampyc.noise import ZeroNoise, GaussianNoise, TruncGaussianNoise, PolytopeNoise
from ampyc.utils import Polytope
import polytope as pc
import numpy as np


def test_noise_shape():
    N = 6
    gen = ZeroNoise(dim=N)
    w = gen.generate()
    assert w.shape == (N, 1)

    gen = GaussianNoise(mean=np.zeros(N,), covariance=np.eye(N))
    w = gen.generate()
    assert w.shape == (N, 1)

    some_box = pc.box2poly([(-1, 1) for _ in range(N)])
    some_box = Polytope(some_box.A, some_box.b)
    gen = TruncGaussianNoise(mean=np.zeros(N,), covariance=np.eye(N), W=some_box)
    w = gen.generate()
    assert w.shape == (N, 1)

    gen = PolytopeNoise(some_box)
    w = gen.generate()
    assert w.shape == (N, 1)
