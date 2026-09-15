"""Tests for ``rig._internal.maya_version``."""

from unittest import mock

from maya import cmds
from rig import Node
from rig._internal import maya_version as _maya_version
from rig._tests._base import MayaTestCase


class TestMayaVersion(MayaTestCase):
    def setUp(self):
        super().setUp()
        _maya_version._reset_cache_for_test()

    def tearDown(self):
        _maya_version._reset_cache_for_test()
        super().tearDown()

    def test_get_maya_version_returns_int(self):
        version = _maya_version.get_maya_version()
        self.assertIsInstance(version, int)
        self.assertGreaterEqual(version, 2020)

    def test_get_maya_version_is_cached(self):
        # First call hits cmds.about; subsequent calls don't.
        with mock.patch(
            "rig._internal.maya_version.cmds.about", return_value="2022"
        ) as mock_about:
            v1 = _maya_version.get_maya_version()
            v2 = _maya_version.get_maya_version()
            v3 = _maya_version.get_maya_version()
        self.assertEqual(v1, 2022)
        self.assertEqual(v2, 2022)
        self.assertEqual(v3, 2022)
        # Only called once due to caching.
        self.assertEqual(mock_about.call_count, 1)

    def test_is_at_least_true(self):
        with mock.patch(
            "rig._internal.maya_version.cmds.about", return_value="2024"
        ):
            _maya_version._reset_cache_for_test()
            self.assertTrue(_maya_version.is_at_least(2022))
            self.assertTrue(_maya_version.is_at_least(2024))

    def test_is_at_least_false(self):
        with mock.patch(
            "rig._internal.maya_version.cmds.about", return_value="2022"
        ):
            _maya_version._reset_cache_for_test()
            self.assertFalse(_maya_version.is_at_least(2024))
            self.assertFalse(_maya_version.is_at_least(2026))

    def test_reset_cache_for_test_forces_requery(self):
        with mock.patch(
            "rig._internal.maya_version.cmds.about",
            side_effect=["2022", "2024"],
        ):
            _maya_version._reset_cache_for_test()
            self.assertEqual(_maya_version.get_maya_version(), 2022)
            _maya_version._reset_cache_for_test()
            self.assertEqual(_maya_version.get_maya_version(), 2024)


# --------------------------------------------------------------------- #
#  v3.E -- options(maya_version=N) override
# --------------------------------------------------------------------- #


class TestMayaVersionOption(MayaTestCase):
    """Verify that ``rig.options(maya_version=N)`` correctly overrides
    the dispatch target so the DSL emits legacy networks on demand.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Ensure clean state between tests.
        from rig import set_options

        set_options(maya_version=None)

    def tearDown(self):
        from rig import set_options

        set_options(maya_version=None)
        super().tearDown()

    def test_default_is_none(self):
        from rig._internal.container import ContainerOptions

        self.assertIsNone(ContainerOptions.maya_version)

    def test_target_2022_uses_legacy_pma(self):
        from rig import functions as f, set_options

        set_options(maya_version=2022)
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = f.sum([a.tx, b.tx])
        node_type = cmds.nodeType(str(result).split(".")[0])
        self.assertEqual(
            node_type,
            "plusMinusAverage",
            f"Expected legacy plusMinusAverage on target=2022, got {node_type}",
        )

    def test_target_2024_uses_native_sum(self):
        from rig import functions as f, set_options

        # Targeting 2024 only redirects dispatch -- the running Maya still has
        # to ship the ``sum`` node type for the network to build.
        if _maya_version.get_maya_version() < 2024:
            self.skipTest("native sum node requires Maya 2024+")
        set_options(maya_version=2024)
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = f.sum([a.tx, b.tx])
        node_type = cmds.nodeType(str(result).split(".")[0])
        self.assertEqual(
            node_type,
            "sum",
            f"Expected native sum on target=2024, got {node_type}",
        )

    def test_target_2022_pi_uses_constant(self):
        # v3.J: pi() now has a pre-2024 fallback (a network constant).
        from rig import functions as f, set_options

        set_options(maya_version=2022)
        result = f.pi()
        # Network node holding math.pi.
        node = str(result).split(".")[0]
        self.assertEqual(cmds.nodeType(node), "network")
        self.assertAlmostEqual(cmds.getAttr(str(result)), 3.141592653589793, places=4)

    def test_target_2022_log_raises(self):
        from rig import functions as f, set_options

        set_options(maya_version=2022)
        a = Node.create("transform", name="a")
        a.tx << 5.0
        # log() still raises -- no viable pre-2024 fallback.
        with self.assertRaises(RuntimeError):
            f.log(a.tx)

    def test_target_none_reverts_to_actual(self):
        from rig import set_options
        from rig._internal.maya_version import (
            get_maya_version,
            get_target_version,
        )

        set_options(maya_version=2022)
        self.assertEqual(get_target_version(), 2022)
        set_options(maya_version=None)
        self.assertEqual(get_target_version(), get_maya_version())

    def test_switching_clears_caches(self):
        # If we DON'T clear caches, the same call would return the
        # previously-cached node from the prior target.
        from rig import functions as f, set_options

        # Needs two targets that dispatch differently, which only exists once
        # the running Maya ships the native nodes.
        if _maya_version.get_maya_version() < 2024:
            self.skipTest("needs a target pair with differing dispatch (2024+)")

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")

        set_options(maya_version=None)
        r1 = f.sum([a.tx, b.tx])
        self.assertEqual(cmds.nodeType(str(r1).split(".")[0]), "sum")

        set_options(maya_version=2022)
        r2 = f.sum([a.tx, b.tx])
        self.assertEqual(cmds.nodeType(str(r2).split(".")[0]), "plusMinusAverage")

    def test_2024_only_matrix_axis_uses_vectorproduct_on_2022(self):
        # v3.J: matrix.axis() now falls back to vectorProduct(op=3) pre-2024.
        from rig import matrix as m, set_options

        # Disable publishing so result is the internal vectorProduct plug,
        # not the published container output (v4.P leaf-frame rule).
        set_options(maya_version=2022, publish_attributes=False)
        try:
            a      = Node.create("transform", name="a")
            result = m.axis(a.matrix, axis=0)
            node   = str(result).split(".")[0]
            self.assertEqual(cmds.nodeType(node), "vectorProduct")
            self.assertEqual(cmds.getAttr(f"{node}.operation"), 3)
        finally:
            set_options(publish_attributes=True)

    def test_2024_only_vector_rotate_uses_compose_on_2022(self):
        # v3.J: vector.rotate() now falls back to composeMatrix +
        # vectorProduct(op=3) pre-2024.
        from rig import set_options, vector as vec

        # Disable publishing so result is the internal vectorProduct plug,
        # not the published container output (v4.P leaf-frame rule).
        set_options(maya_version=2022, publish_attributes=False)
        try:
            v      = Node.create("transform", name="v")
            eul    = Node.create("transform", name="eul")
            result = vec.rotate(v.t, eul.r)
            # Final output comes from a vectorProduct.
            node = str(result).split(".")[0]
            self.assertEqual(cmds.nodeType(node), "vectorProduct")
            # Verify a composeMatrix was created and feeds into it.
            compose_nodes = cmds.ls(type="composeMatrix")
            self.assertGreater(
                len(compose_nodes),
                0,
                "vector.rotate() pre-2024 should create a composeMatrix",
            )
        finally:
            set_options(publish_attributes=True)