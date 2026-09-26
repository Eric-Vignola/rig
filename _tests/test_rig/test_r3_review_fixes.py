"""Regression tests for the fixes of the round 3 review (step R3_FIX).

Each class covers one finding of the review of perf/efficiency at 29a4128; the
docstring of each test says what it pinned before the fix.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Node, Plug
from rig.nodetypes import _base
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
