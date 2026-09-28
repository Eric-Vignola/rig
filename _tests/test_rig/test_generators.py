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


def _legacy_yield(obj, index):
    """Reference copy of the pre-fast-path ``_yield`` sequence branch."""
    from rig._internal.generators import _is_sequence

    if _is_sequence(obj):
        seq = list(obj)
        if not seq:
            raise IndexError("_yield: empty sequence has no element at any index")
        return seq[index]
    return obj


def _outcome(func, obj, index):
    """Return ``("ok", value)`` or ``("err", type, message)``."""
    try:
        return ("ok", func(obj, index))
    except Exception as e:
        return ("err", type(e), str(e))


class TestYieldFastPath(MayaTestCase):
    """``_yield`` indexes lists and tuples in place; every answer, including
    errors, matches the ``list(obj)[index]`` path it replaces.
    """

    TEST_CASE_START_NEW_SCENE = True

    def _assert_same(self, obj, indices):
        from rig._internal.generators import _yield

        for index in indices:
            self.assertEqual(
                _outcome(_yield, obj, index),
                _outcome(_legacy_yield, obj, index),
                f"{obj!r}[{index!r}]",
            )

    def test_yield_fast_path_equivalence(self):
        from maya import cmds

        from rig._internal.list import List

        class Sub(list):
            pass

        class TupleSub(tuple):
            pass

        a = cmds.createNode("transform", name="yield_a")
        b = cmds.createNode("transform", name="yield_b")
        indices = [0, 1, 2, -1, -2, -3, 3, 5, True, 1.0, "x", slice(0, 2)]
        for obj in (
            [10, 20, 30],
            (10, 20, 30),
            Sub([10, 20, 30]),
            TupleSub((10, 20, 30)),
            List([a, b, 3.5]),
            List([f"{a}.tx", f"{b}.ty"]),
            (),
            TupleSub(),
        ):
            self._assert_same(obj, indices)

    def test_yield_plug_list_returns_the_stored_element(self):
        from maya import cmds

        from rig._internal.generators import _yield
        from rig._internal.list import List

        a  = cmds.createNode("transform", name="yield_c")
        pl = List([f"{a}.tx", f"{a}.ty"])
        self.assertIs(_yield(pl, 0), list.__getitem__(pl, 0))
        self.assertIs(_yield(pl, 1), list(pl)[1])

    def test_yield_custom_iter_uses_legacy_path(self):
        from rig._internal.generators import _yield

        class Reversed(list):
            def __iter__(self):
                return iter(list.__getitem__(self, slice(None, None, -1)))

        class Short(list):
            def __len__(self):
                return 1

        class BadLen(list):
            def __len__(self):
                raise TypeError("no len")

        class TupleReversed(tuple):
            def __iter__(self):
                return iter(tuple.__getitem__(self, slice(None, None, -1)))

        self.assertEqual(_yield(Reversed([1, 2, 3]), 0), 3)
        self.assertEqual(_yield(TupleReversed((1, 2, 3)), 0), 3)
        # a raising __len__ makes the legacy code treat obj as a scalar
        bad = BadLen([1, 2])
        self.assertIs(_yield(bad, 0), bad)
        for obj in (Reversed([1, 2, 3]), Short([1, 2, 3]), Short(), bad):
            self._assert_same(obj, [0, 1, 2, 3, -1])

    def test_yield_empty_error_unchanged(self):
        from rig._internal.generators import _yield

        for obj in ([], (), type("Sub", (list,), {})()):
            with self.assertRaises(IndexError) as ctx:
                _yield(obj, 0)
            self.assertEqual(
                str(ctx.exception),
                "_yield: empty sequence has no element at any index",
            )
        with self.assertRaises(IndexError) as ctx:
            _yield((1, 2), 2)
        self.assertEqual(str(ctx.exception), "list index out of range")

    def test_yield_numpy_rows_unchanged(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy is not available")

        arr = numpy.arange(6).reshape(3, 2)
        rows = list(sequences(arr, [1, 2, 3]))
        self.assertEqual(len(rows), 3)
        for i, (row, scalar) in enumerate(rows):
            self.assertEqual(list(row), list(arr[i]))
            self.assertEqual(scalar, i + 1)

    def test_yield_does_not_copy_large_lists(self):
        # Broadcasting lists and tuples never reaches the generic
        # copy-then-index branch, so the cost stays linear in the length.
        from unittest import mock

        from rig._internal import generators

        big  = list(range(5000))
        real = generators._is_sequence
        with mock.patch.object(
            generators, "_is_sequence", side_effect=real
        ) as spy:
            rows = list(sequences(big, (7,), [3]))
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(len(rows), 5000)
        self.assertEqual(rows[-1], [4999, 7, 3])

    def test_yield_class_reported_as_str_or_dict_uses_legacy_path(self):
        from rig._internal.generators import _yield

        class ListAsStr(list):
            __class__ = property(lambda self: str)

        class TupleAsStr(tuple):
            __class__ = property(lambda self: str)

        class ListAsDict(list):
            __class__ = property(lambda self: dict)

        # isinstance(obj, str) holds, so the legacy code broadcasts it whole
        for obj in (ListAsStr([1, 2]), TupleAsStr((1, 2))):
            self.assertIs(_yield(obj, 0), obj)
            rows = list(sequences(obj, [1, 2]))
            self.assertEqual([row[1] for row in rows], [1, 2])
            self.assertTrue(all(row[0] is obj for row in rows))

        # isinstance(obj, dict) holds, so the legacy code walks it as keys
        as_dict = ListAsDict([1, 2])
        self.assertEqual(_yield(as_dict, 1), 2)
        self.assertEqual(_yield(as_dict, 1.0), 2)
        with self.assertRaises(IndexError) as ctx:
            _yield(as_dict, -1)
        self.assertEqual(
            str(ctx.exception), "_yield: index -1 out of range for dict of length 2"
        )
