"""Tests for ``rig._internal.generators`` -- pure-Python iterator helpers."""

from rig._internal.generators import arguments, dictionaries, sequences
from rig._tests._base import MayaTestCase


class TestSequences(MayaTestCase):
    def test_all_scalars_yields_one_row(self):
        rows = list(sequences(1, 2, 3))
        self.assertEqual(rows, [[1, 2, 3]])

    def test_equal_length_lists(self):
        rows = list(sequences([1, 2, 3], [10, 20, 30], [100, 200, 300]))
        self.assertEqual(
            rows,
            [[1, 10, 100], [2, 20, 200], [3, 30, 300]],
        )

    def test_scalar_broadcasts(self):
        rows = list(sequences([1, 2, 3], 5))
        self.assertEqual(rows, [[1, 5], [2, 5], [3, 5]])

    def test_asymmetric_caps_to_last(self):
        # Non-strict iterator -- caps to last element, unlike vectorize().
        rows = list(sequences([1, 2, 3, 4, 5], [10, 20, 30]))
        self.assertEqual(
            rows,
            [[1, 10], [2, 20], [3, 30], [4, 30], [5, 30]],
        )

    def test_strings_are_scalars(self):
        rows = list(sequences("abc", [1, 2, 3]))
        self.assertEqual(rows, [["abc", 1], ["abc", 2], ["abc", 3]])

    def test_no_args_yields_nothing(self):
        self.assertEqual(list(sequences()), [])


class TestDictionaries(MayaTestCase):
    def test_simple(self):
        rows = list(dictionaries(a=[1, 2, 3], b=5))
        self.assertEqual(
            rows,
            [{"a": 1, "b": 5}, {"a": 2, "b": 5}, {"a": 3, "b": 5}],
        )

    def test_asymmetric(self):
        rows = list(dictionaries(a=[1, 2, 3, 4], b=[10, 20]))
        self.assertEqual(
            rows,
            [
                {"a": 1, "b": 10},
                {"a": 2, "b": 20},
                {"a": 3, "b": 20},
                {"a": 4, "b": 20},
            ],
        )


class TestArguments(MayaTestCase):
    def test_args_only(self):
        rows = list(arguments([1, 2, 3], 5))
        self.assertEqual(
            rows,
            [
                ([1, 5], {}),
                ([2, 5], {}),
                ([3, 5], {}),
            ],
        )

    def test_kwargs_only(self):
        rows = list(arguments(a=[1, 2], b=10))
        self.assertEqual(
            rows,
            [
                ([], {"a": 1, "b": 10}),
                ([], {"a": 2, "b": 10}),
            ],
        )

    def test_mixed_args_and_kwargs(self):
        rows = list(arguments([1, 2, 3], scale=10))
        self.assertEqual(
            rows,
            [
                ([1], {"scale": 10}),
                ([2], {"scale": 10}),
                ([3], {"scale": 10}),
            ],
        )

    def test_empty(self):
        self.assertEqual(list(arguments()), [])


class TestYieldEmptyContainers(MayaTestCase):
    """Regression: ``_yield`` raises IndexError on empty containers
    instead of silently returning the container itself.
    """

    def test_yield_empty_set_raises(self):
        from rig._internal.generators import _yield

        with self.assertRaises(IndexError):
            _yield(set(), 0)

    def test_yield_empty_dict_raises(self):
        from rig._internal.generators import _yield

        with self.assertRaises(IndexError):
            _yield({}, 0)

    def test_yield_empty_list_raises(self):
        from rig._internal.generators import _yield

        with self.assertRaises(IndexError):
            _yield([], 0)

    def test_yield_scalar_returns_self(self):
        # Non-sequence inputs still pass through unchanged.
        from rig._internal.generators import _yield

        self.assertEqual(_yield(42, 0), 42)
        self.assertEqual(_yield("hello", 0), "hello")