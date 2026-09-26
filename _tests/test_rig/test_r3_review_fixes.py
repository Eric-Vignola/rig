"""Regression tests for the fixes of the round 3 review (step R3_FIX).

Each class covers one finding of the review of perf/efficiency at 29a4128; the
docstring of each test says what it pinned before the fix.
"""

import gc
from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Node, Plug
from rig.nodetypes import _base
from rig._internal.node_ops import NodeOp
from rig._tests._base import MayaTestCase


class TestSmallFixes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_world_mobject_error_names_its_api_type(self):
        # `apiTypeStr` is a property in API 2.0: calling it raised
        # TypeError "'str' object is not callable" instead of this ValueError
        root = OpenMaya.MItDag().root()
        with self.assertRaisesRegex(ValueError, r"^Invalid MObject API Type: kWorld$"):
            _base._mobject_to_str(root)

    def test_component_type_cache_skips_plug_setattr(self):
        # the lazy cache wrote through `Plug.__setattr__`
        cmds.polyCube(name="pc", constructionHistory=False)
        plug = Node("pcShape").vtx
        with mock.patch.object(Plug, "__setattr__", autospec=True, wraps=Plug.__setattr__) as spy:
            self.assertEqual(plug._component_type, "kMeshVertComponent")
        names = [call.args[1] for call in spy.call_args_list]
        self.assertNotIn("_Attribute__component_type", names)
        self.assertEqual(plug.__dict__["_Attribute__component_type"], "kMeshVertComponent")

    def test_nodeop_fall_through_keeps_no_traceback_cycle(self):
        # the NotImplementedError of a fall-through kept its traceback, whose
        # frame held the error: a cycle per fall-through
        op = NodeOp("r3fallthrough")

        @op.impl(since=2024)
        def _native(value):
            raise NotImplementedError("native cannot")

        @op.impl(since=0)
        def _legacy(value):
            return value

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        gc.set_debug(gc.DEBUG_SAVEALL)
        try:
            self.assertEqual(op(3.0), 3.0)
            gc.collect()
            leaked = [obj for obj in gc.garbage if isinstance(obj, NotImplementedError)]
        finally:
            gc.set_debug(0)
            del gc.garbage[:]
            if was_enabled:
                gc.enable()
        self.assertEqual(leaked, [])

    def test_nodeop_all_impls_failing_still_names_the_last_error(self):
        op = NodeOp("r3allfail")

        @op.impl(since=0)
        def _legacy(value):
            raise NotImplementedError("legacy cannot")

        with self.assertRaisesRegex(RuntimeError, r"raised NotImplementedError\. Last: legacy cannot$"):
            op(1.0)
