"""Tests for ``rig.bridges.nodes`` (PEP 562 lazy nodetype factories)."""

import numpy as np
from maya import cmds
from rig import Node
from rig._internal.maya_version import get_maya_version
from rig.bridges import nodes
from rig._tests._base import MayaTestCase


class TestNodesFactories(MayaTestCase):
    """v4.D: PEP 562 lazy factories for every Maya node type."""

    TEST_START_NEW_SCENE = True

    def _skip_without_logic_nodes(self):
        # ``nodes`` factories are derived from Maya's node-type registry, and
        # the ``and`` / ``or`` / ``not`` types only exist from 2024 onwards.
        if get_maya_version() < 2024:
            self.skipTest("`and` / `or` / `not` node types require Maya 2024+")

    def test_create_with_attr_kwargs(self):
        # plusMinusAverage(operation=2) creates the node AND sets .operation
        n = nodes.plusMinusAverage(operation=2)
        self.assertIsInstance(n, Node)
        self.assertEqual(cmds.getAttr(f"{n}.operation"), 2)

    def test_create_with_compound_attr_via_dsl_inject(self):
        # DSL << handles compound vectors -- Eric's raw setAttr version
        # would have to handle this manually.
        n = nodes.transform(name="cube1", translate=[1, 2, 3])
        self.assertIsInstance(n, Node)
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 3.0)

    def test_create_kwargs_are_consumed_not_setattr(self):
        # name / parent / shared / skipSelect / container should NOT be
        # interpreted as attribute setters -- they're createNode kwargs.
        n = nodes.transform(name="foo")
        self.assertEqual(str(n), "foo")

    def test_parent_kwarg_routed_to_create(self):
        root  = nodes.transform(name="root")
        child = nodes.transform(name="child", parent=root)
        self.assertIsInstance(child, Node)
        self.assertEqual(cmds.listRelatives("child", parent=True) or [], ["root"])

    def test_python_keyword_alias_and(self):
        # `and` is a Python keyword -- exposed via trailing `_`
        self._skip_without_logic_nodes()
        n = nodes.and_(name="myAnd")
        self.assertEqual(cmds.nodeType(str(n)), "and")

    def test_python_keyword_alias_or_not(self):
        self._skip_without_logic_nodes()
        o  = nodes.or_(name="myOr")
        nt = nodes.not_(name="myNot")
        self.assertEqual(cmds.nodeType(str(o)), "or")
        self.assertEqual(cmds.nodeType(str(nt)), "not")

    def test_unknown_nodetype_raises(self):
        with self.assertRaises(AttributeError):
            _ = nodes.thisNodeTypeDoesNotExist

    def test_dir_includes_aliased_keywords(self):
        attrs = dir(nodes)
        self.assertIn("transform", attrs)
        self.assertIn("plusMinusAverage", attrs)
        self._skip_without_logic_nodes()
        self.assertIn("and_", attrs)
        self.assertIn("or_",  attrs)
        self.assertIn("not_", attrs)

    def test_lazy_caching(self):
        from rig.bridges.nodes import _WRAPPER_CACHE

        # Force fresh state so this test is deterministic.
        _WRAPPER_CACHE.clear()
        _ = nodes.transform
        self.assertIn("transform", _WRAPPER_CACHE)
        self.assertIs(nodes.transform, _WRAPPER_CACHE["transform"])

    def test_container_optout(self):
        n = nodes.transform(name="standalone", container=False)
        self.assertIsInstance(n, Node)
        # Verify container=False didn't reach Maya (would have raised TypeError).

    def test_matrix_attr_via_decompose(self):
        # DSL << routes matrix sources through _decompose
        m        = np.eye(4)
        m[3, :3] = [9, 0, 0]
        n        = nodes.transform(name="m1", matrix=m.tolist())
        self.assertIsInstance(n, Node)
        self.assertAlmostEqual(cmds.getAttr("m1.tx"), 9.0, places=4)

    def test_short_form_kwargs(self):
        # `n=` should be accepted as alias for `name=`
        n = nodes.transform(n="shortName")
        self.assertEqual(str(n), "shortName")