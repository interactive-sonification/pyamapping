import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyamapping.chainable_array import ChainableArray, chain

class TestChainableArrayCreation:
    """Test ChainableArray instantiation and type."""

    def test_from_list(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        assert isinstance(a, ChainableArray)
        assert isinstance(a, np.ndarray)

    def test_from_ndarray(self):
        a = ChainableArray(np.array([1.0, 2.0, 3.0]))
        assert isinstance(a, ChainableArray)

    def test_values_preserved(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        assert_allclose(a, [1.0, 2.0, 3.0])

    def test_dtype_preserved(self):
        a = ChainableArray(np.array([1, 2, 3], dtype=np.int32))
        assert a.dtype == np.int32

    def test_numpy_operations_return_chainable_array(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a * 2
        assert isinstance(result, ChainableArray)

    def test_shape_preserved(self):
        a = ChainableArray(np.ones((3, 4)))
        assert a.shape == (3, 4)


class TestChain:
    """Test the chain() factory function."""

    def test_returns_chainable_array(self):
        assert isinstance(chain(np.array([1.0, 2.0])), ChainableArray)

    def test_from_list(self):
        assert isinstance(chain([1.0, 2.0, 3.0]), ChainableArray)

    def test_values_preserved(self):
        assert_allclose(chain([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_from_scalar(self):
        assert isinstance(chain(1.0), ChainableArray)


class TestChainableArrayToArray:
    """Test to_array() conversion."""

    def test_returns_plain_ndarray(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.to_array()
        assert type(result) is np.ndarray
        assert not isinstance(result, ChainableArray)

    def test_values_preserved(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        assert_allclose(a.to_array(), [1.0, 2.0, 3.0])


class TestChainableArrayMap:
    """Test map() method."""

    def test_applies_function(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.map(lambda x: x * 2)
        assert_allclose(result, [2.0, 4.0, 6.0])

    def test_returns_chainable_array(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        assert isinstance(a.map(lambda x: x * 2), ChainableArray)

    def test_passes_args(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.map(np.add, 10.0)
        assert_allclose(result, [11.0, 12.0, 13.0])

    def test_raises_type_error_for_scalar_only_function(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="Use .mapvec()"):
            a.map(lambda x: x + 1 if isinstance(x, float) else (_ for _ in ()).throw(TypeError()))

    def test_chaining(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.map(lambda x: x * 2).map(lambda x: x + 1)
        assert_allclose(result, [3.0, 5.0, 7.0])


class TestChainableArrayMapvec:
    """Test mapvec() method."""

    def test_applies_scalar_function_elementwise(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.mapvec(lambda x: x ** 2)
        assert_allclose(result, [1.0, 4.0, 9.0])

    def test_returns_chainable_array(self):
        a = ChainableArray([1.0, 4.0, 9.0])
        assert isinstance(a.mapvec(np.sqrt), ChainableArray)

    def test_passes_args(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.mapvec(lambda x, y: x + y, 10.0)
        assert_allclose(result, [11.0, 12.0, 13.0])

    def test_chaining(self):
        a = ChainableArray([1.0, 2.0, 3.0])
        result = a.mapvec(lambda x: x * 2).mapvec(lambda x: x + 1)
        assert_allclose(result, [3.0, 5.0, 7.0])


class TestChainableArrayGetattr:
    """Test dynamic attribute handling via __getattr__."""

    def test_dynamic_prefix_returns_callable(self):
        a = ChainableArray([1.0, 2.0])
        assert callable(a.dynamic_foo)

    def test_dynamic_method_returns_string(self):
        a = ChainableArray([1.0, 2.0])
        result = a.dynamic_foo(1, 2)
        assert "dynamic_foo" in result

    def test_unknown_attribute_raises_attribute_error(self):
        a = ChainableArray([1.0, 2.0])
        with pytest.raises(AttributeError):
            _ = a.nonexistent_method