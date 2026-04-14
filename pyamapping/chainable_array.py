"""ChainableArray - a subclass of numpy.ndarray."""

from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike

NDArrayType = TypeVar("NDArrayType", bound=np.ndarray)


class ChainableArray(np.ndarray):
    """subclass for simpler numpy mapping by chaining syntax."""

    def __new__(cls, input_array, *args, **kwargs):
        """Create new instance."""
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Finalize array."""
        if obj is None:
            return

    def to_array(self):
        """Convert self to np.ndarray."""
        return np.array(self)

    def to_asig(self, sr=44100):
        """Convert self to pya.Asig."""
        from pya import Asig

        return Asig(self, sr=sr)

    def plot(self, *args, **kwargs):
        """Plot self via matplotlib."""
        import matplotlib.pyplot as plt

        sr = kwargs.pop("sr", None)
        if sr:
            xs = np.arange(0, self.shape[0]) / sr
            plt.plot(xs, self, *args, **kwargs)
            plt.xlabel("time [s]")
        else:
            xs = kwargs.pop("xs", None)
            if xs is not None:
                plt.plot(xs, self, *args, **kwargs)
            else:
                plt.plot(self, *args, **kwargs)

        return self

    def mapvec(
        self: NDArrayType, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> NDArrayType:
        """Map fn on self by using np.vectorize().

        Parameters
        ----------
            self (NDArrayType): array to map
            fn (Callable[..., Any]): function to call on each element

        Returns
        -------
            NDArrayType: mapping result as ChainableArray
        """
        return np.vectorize(fn)(self, *args, **kwargs)

    def map(
        self: NDArrayType, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> NDArrayType:
        """Apply function fn directly to self; on fail suggest to use mapvec().

        Parameters
        ----------
            self (np.ndarray): array used as input of fn
            fn (Callable[..., Any]): mapping function
            args and kwargs are passed on to fn

        Raises
        ------
            TypeError: if fn fails to operate on np.ndarray as first argument.
            mapvec() is then proposed as alternative.

        Returns
        -------
            ChainableArray: the mapping result as ChainableArray
        """
        try:
            return chain(fn(self, *args, **kwargs))
        except (TypeError, ValueError, AttributeError) as e:
            raise TypeError(
                f"Function {fn.__name__} does not support NumPy arrays directly. "
                "Use .mapvec() instead for np.vectorize elementwise mapping."
            ) from e

    def __getattr__(self, name: str) -> Callable:
        """Dynamically handle method calls."""
        if name.startswith("dynamic_"):
            return lambda *args: f"Called {name} with {args}"
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def chain(input_array: ArrayLike) -> ChainableArray:
    """Turn np.ndarray into ChainableArray."""
    # ToDo: check difference to input_array.view(ChainableArray)
    return ChainableArray(input_array)
