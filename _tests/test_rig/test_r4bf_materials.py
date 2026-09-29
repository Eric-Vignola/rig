"""Round 4b follow-up F1: making a material costs the same in any scene.

* ``TestShaderListCheck``: ``ShadingEngine.for_material`` lists a shader in
  ``defaultShaderList1`` exactly when no plug of it feeds an element of
  ``defaultShaderList1.shaders`` -- the answer of the whole-list read it
  replaced: a shader made listed, a bare one, one listed twice or through
  another plug, one in another defaultShaderList only, namespaced ones, with
  ``namespace -relativeNames`` on, the list emptied, referenced ones, and
  ``create=False``. ``defaultShaderList1`` itself can be neither deleted nor
  renamed, so the check never meets a missing list.
* ``TestNoWholeListRead``: no create, define, assignment or adopt reads
  ``defaultShaderList1.shaders`` whole (a per-material read of every shader
  of the scene: N materials cost O(N^2)).
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds

from rig import Node
from rig.bridges import nodes as rn
from rig.nodetypes import Blinn, ShadingEngine
from rig._tests._base import MayaTestCase


def _scene():
    return set(cmds.ls())


def _uuid(name):
    return cmds.ls(name, uuid=True)[0]


def _slots(shader):
    """The ``defaultShaderList1.shaders`` elements any plug of ``shader``
    feeds, in index order: the whole-list read, the tests' own oracle."""
    pairs = cmds.listConnections(
        ":defaultShaderList1.shaders", source=True, destination=False, plugs=True,
        connections=True,
    ) or []
    uuid = _uuid(shader)
    return [
        slot.lstrip(":") for slot, source in zip(pairs[0::2], pairs[1::2])
        if _uuid(source.split(".", 1)[0]) == uuid
    ]


def _message_slots(shader):
    """The elements ``shader.message`` feeds."""
    return [
        slot for slot in cmds.listConnections(
            f"{shader}.message", source=False, destination=True, plugs=True
        ) or []
        if "defaultShaderList1.shaders[" in slot
    ]


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        cmds.namespace(relativeNames=False)
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def assertAdopted(self, shader, engine, listed_before):
        """``for_material(shader)`` builds ``engine`` and leaves the shader
        listed as before, or listed once by its message when it was not."""
        before = _slots(shader)
        self.assertEqual(bool(before), listed_before)
        made = ShadingEngine.for_material(shader)
        self.assertEqual(_uuid(made.name), _uuid(engine))
        after = _slots(shader)
        if listed_before:
            self.assertEqual(after, before)
        else:
            self.assertEqual(len(after), 1)
            self.assertEqual(len(_message_slots(shader)), 1)


class TestShaderListCheck(_Case):
    """The listing a new engine adds, case by case (the answers of the
    whole-list read at b9a263d)."""

    def test_listed_bare_and_named_twice(self):
        cmds.shadingNode("blinn", asShader=True, name="listed")
        self.assertAdopted("listed", "listedSG", listed_before=True)
        cmds.createNode("blinn", name="bare")
        self.assertAdopted("bare", "bareSG", listed_before=False)
        cmds.createNode("blinn", name="twice")
        cmds.connectAttr("twice.message", "defaultShaderList1.shaders[40]")
        cmds.connectAttr("twice.message", "defaultShaderList1.shaders[41]")
        self.assertAdopted("twice", "twiceSG", listed_before=True)
        self.assertEqual(_slots("twice"), ["defaultShaderList1.shaders[40]", "defaultShaderList1.shaders[41]"])

    def test_another_plug_or_another_list(self):
        # any plug of the shader into the list counts, as the list's own read
        # counted every source node
        cmds.createNode("blinn", name="other")
        cmds.connectAttr("other.outColorR", "defaultShaderList1.shaders", nextAvailable=True)
        self.assertAdopted("other", "otherSG", listed_before=True)
        self.assertEqual(_message_slots("other"), [])
        # another defaultShaderList is not defaultShaderList1
        second = cmds.createNode("defaultShaderList")
        cmds.createNode("blinn", name="second")
        cmds.connectAttr("second.message", f"{second}.shaders", nextAvailable=True)
        self.assertAdopted("second", "secondSG", listed_before=False)

    def test_namespaces_and_relative_names(self):
        cmds.namespace(add="lib")
        cmds.createNode("blinn", name="lib:bare")
        self.assertAdopted("lib:bare", "lib:bareSG", listed_before=False)
        cmds.shadingNode("blinn", asShader=True, name="lib:listed")
        self.assertAdopted("lib:listed", "lib:listedSG", listed_before=True)
        cmds.createNode("blinn", name="relbare")
        cmds.createNode("blinn", name="lib:relbare")
        # a defaultShaderList named like the default one inside the namespace
        # is not the scene's list
        cmds.createNode("defaultShaderList", name="lib:defaultShaderList1")
        cmds.namespace(setNamespace="lib")
        cmds.namespace(relativeNames=True)
        self.assertAdopted(":relbare", ":relbareSG", listed_before=False)
        self.assertAdopted("relbare", ":lib:relbareSG", listed_before=False)
        self.assertIsNone(cmds.listConnections(":lib:defaultShaderList1.shaders"))

    def test_the_list_emptied(self):
        pairs = cmds.listConnections(
            "defaultShaderList1.shaders", source=True, destination=False, plugs=True,
            connections=True,
        )
        for slot, source in zip(pairs[0::2], pairs[1::2]):
            cmds.disconnectAttr(source, slot)
        cmds.createNode("blinn", name="lone")
        self.assertAdopted("lone", "loneSG", listed_before=False)

    def test_referenced_shaders(self):
        folder = tempfile.mkdtemp(prefix="r4bf_f1_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(cmds.file, new=True, force=True)
        cmds.shadingNode("blinn", asShader=True, name="refl")
        cmds.createNode("blinn", name="refb")
        path = os.path.join(folder, "mat.ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="lib")
        self.assertAdopted("lib:refl", "reflSG", listed_before=True)
        self.assertAdopted("lib:refb", "refbSG", listed_before=False)

    def test_create_false_writes_nothing(self):
        cmds.createNode("blinn", name="bare")
        before = _scene()
        self.assertIsNone(ShadingEngine.for_material("bare", create=False))
        self.assertEqual(_scene(), before)
        self.assertEqual(_slots("bare"), [])

    def test_the_list_cannot_go(self):
        cmds.delete("defaultShaderList1")   # Maya prints an error, deletes nothing
        self.assertTrue(cmds.objExists("defaultShaderList1"))
        with self.assertRaisesRegex(RuntimeError, "read only"):
            cmds.rename("defaultShaderList1", "gone")
        self.assertTrue(cmds.objExists("defaultShaderList1"))

    def test_the_rig_spellings(self):
        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        red  = Blinn.create(name="red")
        self.assertEqual(len(_slots("red")), 1)
        cube << rn.blinn(name="bare")
        self.assertEqual(len(_slots("bare")), 1)
        self.assertEqual(cmds.listConnections("bareSG.surfaceShader"), ["bare"])
        cube << red
        self.assertEqual(len(_slots("red")), 1)


class TestNoWholeListRead(_Case):
    """No material verb reads ``defaultShaderList1.shaders`` whole."""

    def test_the_material_verbs(self):
        cube  = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        reads = []
        real  = cmds.listConnections

        def spy(*args, **kwargs):
            reads.extend(str(arg) for arg in args)
            return real(*args, **kwargs)

        with mock.patch.object(cmds, "listConnections", spy):
            red = Blinn.create(name="red")
            Blinn.define("green")
            cube << Blinn.create(name="blue")
            cube << red
            cube << rn.blinn(name="bare")
        self.assertTrue(reads)
        self.assertEqual([arg for arg in reads if "defaultShaderList" in arg], [])
        self.assertEqual(len(_slots("bare")), 1)
