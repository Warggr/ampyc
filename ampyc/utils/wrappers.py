from typing import Protocol, Any


ArrayLike = Any


class ArrayBackend(Protocol):
    """
    Describes numpy or replacements (e.g. casadi, jax.numpy.)
    Only the function that are used somewhere in the code are listed here,
    but more can be added, as long as every library implements them.
    """

    def array(self, *args, **kwargs) -> ArrayLike:
        ...

    def sin(self, x: ArrayLike) -> ArrayLike:
        ...
