"""Tests for ``rig._internal.node_ops`` -- version-keyed dispatch."""

from unittest import mock

from rig._internal.node_ops import NodeOp, SCOPE_COMPOUND, SCOPE_SCALAR
from rig._tests._base import MayaTestCase


def _patch_version(version: int):
    """Helper -- patch get_target_version to return ``version``."""
    return mock.patch(
        "rig._internal.node_ops.get_target_version", return_value=version
    )


class TestNodeOpScalarShortCircuit(MayaTestCase):
    def test_scalar_fn_called_for_all_numeric_args(self):
        op = NodeOp("dummy", scalar_fn=lambda a, b: a + b)
        self.assertEqual(op(2, 3), 5)
        self.assertEqual(op(2.5, 4.5), 7.0)

    def test_scalar_fn_skipped_for_non_numeric_args(self):
        # If at least one arg isn't a number, scalar_fn is bypassed
        # and the framework dispatches to a registered impl.
        op = NodeOp("dummy", scalar_fn=lambda a, b: a + b)

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl(a, b):
            return f"impl({a},{b})"

        self.assertEqual(op(2, "three"), "impl(2,three)")

    def test_no_scalar_fn_dispatches_to_impl(self):
        op = NodeOp("dummy")

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl(a):
            return a * 2

        self.assertEqual(op(5), 10)


class TestNodeOpVersionDispatch(MayaTestCase):
    def test_picks_highest_version_le_current(self):
        op = NodeOp("dummy")

        @op.impl(since=2024, scope=SCOPE_COMPOUND)
        def _impl_2024(x):
            return f"2024:{x}"

        @op.impl(since=2022, scope=SCOPE_COMPOUND)
        def _impl_2022(x):
            return f"2022:{x}"

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl_legacy(x):
            return f"legacy:{x}"

        with _patch_version(2026):
            self.assertEqual(op("foo"), "2024:foo")
        with _patch_version(2024):
            self.assertEqual(op("foo"), "2024:foo")
        with _patch_version(2023):
            self.assertEqual(op("foo"), "2022:foo")
        with _patch_version(2022):
            self.assertEqual(op("foo"), "2022:foo")
        with _patch_version(2020):
            self.assertEqual(op("foo"), "legacy:foo")

    def test_not_implemented_falls_through(self):
        op = NodeOp("dummy")

        @op.impl(since=2024, scope=SCOPE_COMPOUND)
        def _impl_2024_partial(x):
            raise NotImplementedError("only handles ints")

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl_fallback(x):
            return f"fallback:{x}"

        with _patch_version(2026):
            # 2024 impl raises NotImplementedError -> fall through to legacy
            self.assertEqual(op("foo"), "fallback:foo")

    def test_no_impl_raises(self):
        op = NodeOp("nonexistent")
        with self.assertRaises(RuntimeError) as cm:
            op("anything")
        self.assertIn("No impl", str(cm.exception))


class TestNodeOpPluginLoad(MayaTestCase):
    def test_plugin_loaded_lazily_once(self):
        op = NodeOp("dummy", requires_plugin="quatNodes")

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl(x):
            return x

        with mock.patch("rig._internal.node_ops.cmds.loadPlugin") as mock_load:
            op("a")
            op("b")
            op("c")
        # Only loaded once (idempotent) despite three calls.
        self.assertEqual(mock_load.call_count, 1)
        mock_load.assert_called_with("quatNodes", quiet=True)

    def test_no_plugin_no_load(self):
        op = NodeOp("dummy")

        @op.impl(since=0, scope=SCOPE_COMPOUND)
        def _impl(x):
            return x

        with mock.patch("rig._internal.node_ops.cmds.loadPlugin") as mock_load:
            op("a")
        mock_load.assert_not_called()


class TestNodeOpInvalidScope(MayaTestCase):
    def test_invalid_scope_raises(self):
        op = NodeOp("dummy")
        with self.assertRaises(ValueError):

            @op.impl(since=0, scope="invalid_scope_value")
            def _impl(x):
                return x