"""Tests for shader conversion: ``mat.astype(Phong)``, ``shade.convert()``
and the parking engine behind them.

Round 4b (NC7): a conversion makes a new node and RETURNS it (``red =
red.astype(Phong)``); the old node object then raises, naming the
conversion. ``Phong(mat)`` is a reference, never a conversion.
``dry_run=True`` returns the :class:`Conversion` report.

``cmds.warning`` is monkeypatched to count and capture; every refusal
asserts a zero ``cmds.ls()`` delta and an unchanged connection snapshot.
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from rig import container, lock, memoize, Node, NodeTypeError
from rig import shade
from rig.bridges import nodes as rn
from rig.shade import (
    Blinn,
    Conversion,
    Default,
    Lambert,
    Material,
    Phong,
    StandardSurface,
)
from rig._internal import shade_convert
from rig._tests._base import MayaTestCase


CONVERTED = r"'red' was converted to a {}; use the node astype\(\) returned"


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _members(engine):
    return cmds.sets(str(engine), query=True) or []


def _sources(plug):
    return cmds.listConnections(plug, source=True, destination=False, plugs=True) or []


def _destinations(plug):
    return cmds.listConnections(plug, source=False, destination=True, plugs=True) or []


def _wiring(node):
    """Every connection of a node, as a sortable snapshot."""
    pairs = cmds.listConnections(node, connections=True, plugs=True) or []
    return sorted(zip(pairs[0::2], pairs[1::2]))


def _dsl1_index(material):
    for plug in _destinations(f"{material}.message"):
        if plug.startswith("defaultShaderList1.shaders["):
            return plug
    return None


def _info(engine):
    return cmds.listConnections(f"{engine}.message", type="materialInfo")[0]


def _visible(node):
    return cmds.listAttr(node, scalar=True, multi=True, read=True, visible=True) or []


# --------------------------------------------------------------------- #
#  Before a material exists: its type is its define's class
# --------------------------------------------------------------------- #


class TestLazyRetype(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_retype_is_free_and_returns_an_equal_fresh_handle(self):
        """Historical id (v2.0.0a2): pinned Phong(spec) retyping a lazy spec for
        free; it now pins that a reference never retypes (Phong(blinn) is the
        reference's NodeTypeError naming astype, nothing written) and that a
        material's type before it exists is its define's class (round 4b NC7:
        Phong.define is the spelling)."""
        cube   = _cube("cube")
        red    = Blinn.define("red", color=[1, 0, 0])
        before = set(cmds.ls())
        with mock.patch.object(cmds, "warning") as warn:
            for call in (lambda: Phong(red), lambda: Phong("red"), lambda: Phong(Node("red"))):
                with self.assertRaisesRegex(
                    NodeTypeError, r"^'red' is a blinn, not a phong; .*Blinn\('red'\)\.astype\(Phong\) converts it$"
                ):
                    call()
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertIs(type(red), Blinn)
        self.assertTrue(red.is_valid)
        # the type is chosen when the material is made
        p = Phong.define("p", color=[1, 0, 0])
        cube << p
        self.assertEqual(cmds.nodeType("p"), "phong")
        self.assertEqual(cmds.getAttr("p.color")[0], (1.0, 0.0, 0.0))
        self.assertEqual([repr(m) for m in Material.of(cube)], ['Phong("p")'])
        self.assertEqual(Phong("p"), p)

    def test_dropped_kwarg_warns_once_and_is_pruned(self):
        """Historical id (v2.0.0a2): pinned a lazy retype pruning the kwargs the
        target lacks with one warning; it now pins that define / create check
        every attribute against their own type before anything is made (a phong
        has no eccentricity: AttributeError, nothing written), and that the
        conversion options live on astype / convert only (round 4b NC7)."""
        bump   = rn.bump2d(name="bump")
        before = set(cmds.ls())
        with mock.patch.object(cmds, "warning") as warn:
            with self.assertRaisesRegex(AttributeError, "a phong has no attribute 'eccentricity'"):
                Phong.define("k", eccentricity=0.6, normalCamera=bump.outNormal)
            with self.assertRaisesRegex(AttributeError, "a phong has no attribute 'eccentricity'"):
                Phong.create(name="k", eccentricity=0.6)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(set(cmds.ls()), before)
        # options live on astype / convert only
        with self.assertRaisesRegex(TypeError, "takes no attributes"):
            Phong("k", strict=True)
        with self.assertRaisesRegex(TypeError, "takes no attributes"):
            Material(Blinn.define("k3"), type="phong")
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "no attribute 'nope'"):
            Blinn("k3").astype(Phong, nope=1)
        self.assertEqual(set(cmds.ls()), before)
        p = Phong.define("k", normalCamera=bump.outNormal)
        self.assertEqual(_sources("k.normalCamera"), ["bump.outNormal"])
        self.assertIs(type(p), Phong)


# --------------------------------------------------------------------- #
#  The scene conversion
# --------------------------------------------------------------------- #


class TestConversion(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)
        self.cube = _cube("cube")

    def test_nothing_set_asks_no_questions_and_keeps_the_topology(self):
        mat = Blinn.define("red", color=[1, 0, 0])
        self.cube << mat
        info = _info("redSG")
        cmds.connectAttr("red.message", f"{info}.material")
        slot   = _dsl1_index("red")
        before = set(cmds.ls())
        with mock.patch.object(cmds, "warning") as warn:
            p = mat.astype(Phong)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertEqual(_sources("redSG.surfaceShader"), ["red.outColor"])
        self.assertEqual(_sources(f"{info}.material"), ["red.message"])
        self.assertEqual(_dsl1_index("red"), slot)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        # the new node is returned; the held one names the conversion
        self.assertIs(type(p), Phong)
        self.assertEqual(repr(p), 'Phong("red")')
        self.assertEqual(p, Node("red"))
        self.assertEqual(str(p.engine), "redSG")
        self.assertEqual([repr(m) for m in Material.of(self.cube)], ['Phong("red")'])
        p.cosinePower << 40
        self.assertEqual(cmds.getAttr("red.cosinePower"), 40)
        with self.assertRaisesRegex(RuntimeError, CONVERTED.format("phong")):
            str(mat)
        with self.assertRaisesRegex(TypeError, r"Phong\('red'\)\.astype\(Blinn\) converts it"):
            Blinn("red")
        self.assertEqual(cmds.ls("*__rigconvert*"), [])
        self.assertEqual(cmds.ls("*__rigold*"), [])

    def _lossy_red(self):
        """A blinn with something of every kind: a locked blinn-only value,
        a createNode-made ramp into a blinn-only attr, a locked shared
        value, an animCurve on a shared attr, wires into shared attrs at
        parent and child level, and a user attribute."""
        red = Blinn.define("red")
        self.cube        << red
        red.eccentricity << 0.66 << lock
        rn.ramp(name="ramp1")
        red.specularRollOff << Node("ramp1").outAlpha
        red.diffuse         << 0.33 << lock
        cmds.setKeyframe("red.reflectivity", v=0.9, t=1)
        rn.file(name="tex")
        red.incandescence << Node("tex").outColor
        rn.ramp(name="ramp2")
        red.transparencyG << Node("ramp2").outAlpha
        rn.ramp(name="ramp3")
        red.specularColor << Node("ramp3").outColor
        cmds.addAttr("red", longName="myNote", dataType="string")
        cmds.setAttr("red.myNote", "hi", type="string")
        return red

    def test_loss_report_with_parking(self):
        red    = self._lossy_red()
        report = shade.convert(red, "phong", dry_run=True)
        self.assertIsInstance(report, Conversion)
        self.assertTrue(report)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        with mock.patch.object(cmds, "warning") as warn:
            red = red.astype(Phong)
        self.assertEqual(warn.call_count, 1)
        self.assertEqual(warn.call_args[0][0], str(report))
        self.assertEqual(str(report).splitlines(), [
            "rig.shade: 'red' blinn -> phong parks:",
            "   wire        ramp1.outAlpha -> specularRollOff",
            "   value       eccentricity = 0.66 [locked]",
        ])
        self.assertEqual(
            [(a, round(v, 4), locked) for a, v, locked in report.lost_values],
            [("eccentricity", 0.66, True)],
        )
        self.assertEqual(report.lost_wires,     (("ramp1.outAlpha", "specularRollOff", "ramp"),))
        self.assertEqual(report.lost_animation, ())
        self.assertEqual(report.lost_outputs,   ())
        self.assertEqual(report.opaque,         ())
        self.assertEqual(report.parked,         ("eccentricity", "specularRollOff"))
        self.assertEqual(report.dynamic,        ("myNote",))
        for attr in ("diffuse", "reflectivity", "incandescence", "transparency", "specularColor"):
            self.assertIn(attr, report.carried)
        self.assertNotIn("eccentricity", report.carried)
        # what moved
        self.assertIs(type(red), Phong)
        self.assertEqual(cmds.nodeType("red"),          "phong")
        self.assertEqual(_sources("red.incandescence"), ["tex.outColor"])
        self.assertEqual(_sources("red.transparencyG"), ["ramp2.outAlpha"])
        self.assertEqual(_sources("red.specularColor"), ["ramp3.outColor"])
        self.assertEqual(_sources("red.reflectivity"),  ["red_reflectivity.output"])
        self.assertAlmostEqual(cmds.getAttr("red.diffuse"), 0.33, places=5)
        self.assertTrue(cmds.getAttr("red.diffuse", lock=True))
        self.assertEqual(cmds.getAttr("red.myNote"), "hi")
        self.assertEqual(_members("redSG"), ["cubeShape"])
        # what was parked: hidden, typed like the original, wire re-homed
        self.assertEqual(
            cmds.listAttr("red", userDefined=True),
            ["myNote", "__eccentricity__", "__specularRollOff__"],
        )
        self.assertAlmostEqual(cmds.getAttr("red.__eccentricity__"), 0.66, places=5)
        self.assertTrue(cmds.getAttr("red.__eccentricity__", lock=True))
        self.assertTrue(cmds.attributeQuery("__eccentricity__", node="red", hidden=True))
        self.assertEqual(cmds.attributeQuery("__eccentricity__", node="red", attributeType=True), "float")
        self.assertEqual(_destinations("ramp1.outAlpha"), ["red.__specularRollOff__"])
        self.assertNotIn("__eccentricity__", _visible("red"))
        self.assertEqual(cmds.listConnections("red", type="blinn"), None)

    def test_the_cascade_pin(self):
        # rig keeps a createNode-made ramp through the conversion
        red = Blinn.define("red")
        self.cube           << red
        red.specularRollOff << Node.create("ramp", name="ramp1").outAlpha
        with mock.patch.object(cmds, "warning"):
            red.astype(Phong)
        self.assertTrue(cmds.objExists("ramp1"))
        self.assertEqual(_destinations("ramp1.outAlpha"), ["red.__specularRollOff__"])
        # the pin: raw create + copyAttr + delete on a copy loses it, which
        # is why doomed wires are re-homed before the delete
        copy = cmds.createNode("blinn", name="copy")
        cmds.createNode("ramp", name="doomed")
        cmds.connectAttr("doomed.outAlpha", "copy.specularRollOff")
        new = cmds.createNode("phong", name="copy2")
        cmds.copyAttr(copy, new, values=True, attribute=["diffuse"])
        cmds.delete(copy)
        self.assertFalse(cmds.objExists("doomed"))

    def test_animation_on_a_lost_attr_is_parked_with_its_keys(self):
        red = Phong.define("red")
        self.cube << red
        cmds.setKeyframe("red.cosinePower", v=10, t=1)
        cmds.setKeyframe("red.cosinePower", v=30, t=10)
        with mock.patch.object(cmds, "warning") as warn:
            red = red.astype(Blinn)
        lines = warn.call_args[0][0].splitlines()
        self.assertEqual(lines[0], "rig.shade: 'red' phong -> blinn parks:")
        self.assertEqual(
            lines[1],
            "   animation   red_cosinePower -> cosinePower   "
            "(restored when 'red' next becomes a type with cosinePower)",
        )
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertTrue(cmds.objExists("red_cosinePower"))
        self.assertEqual(cmds.keyframe("red_cosinePower", query=True, keyframeCount=True), 2)
        self.assertEqual(_destinations("red_cosinePower.output"), ["red.__cosinePower__"])
        # through a type that lacks it: still parked
        with mock.patch.object(cmds, "warning") as warn:
            red = red.astype(Lambert)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(cmds.listAttr("red", userDefined=True), ["__cosinePower__"])
        self.assertEqual(_destinations("red_cosinePower.output"), ["red.__cosinePower__"])
        # restored at the end (read from the scene), parked attribute destroyed
        with mock.patch.object(cmds, "warning") as warn:
            red = shade.convert(red, Phong)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(_sources("red.cosinePower"), ["red_cosinePower.output"])
        self.assertEqual(cmds.keyframe("red_cosinePower", query=True, keyframeCount=True), 2)
        self.assertIsNone(cmds.listAttr("red", userDefined=True))
        self.assertIs(type(red), Phong)

    def test_round_trip_restores_everything_and_destroys_the_parked_attrs(self):
        red = self._lossy_red()
        with mock.patch.object(cmds, "warning"):
            red = red.astype(Phong)
        report = shade.convert(red, "blinn", dry_run=True)
        self.assertFalse(report)
        self.assertEqual(str(report), "")
        with mock.patch.object(cmds, "warning") as warn:
            red = shade.convert(red, "blinn")
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertAlmostEqual(cmds.getAttr("red.eccentricity"), 0.66, places=5)
        self.assertTrue(cmds.getAttr("red.eccentricity", lock=True))
        self.assertEqual(_sources("red.specularRollOff"), ["ramp1.outAlpha"])
        self.assertEqual(_sources("red.reflectivity"),    ["red_reflectivity.output"])
        self.assertEqual(_sources("red.incandescence"),   ["tex.outColor"])
        self.assertEqual(_sources("red.transparencyG"),   ["ramp2.outAlpha"])
        self.assertEqual(_sources("red.specularColor"),   ["ramp3.outColor"])
        self.assertTrue(cmds.getAttr("red.diffuse", lock=True))
        self.assertEqual(cmds.listAttr("red", userDefined=True), ["myNote"])
        self.assertEqual(cmds.getAttr("red.myNote"), "hi")
        self.assertIs(type(red), Blinn)

    def test_park_false_drops_and_disconnects(self):
        red = Blinn.define("red")
        self.cube           << red
        red.eccentricity    << 0.5
        red.specularRollOff << rn.ramp(name="ramp1").outAlpha
        with mock.patch.object(cmds, "warning") as warn:
            red = red.astype("phong", park=False)
        self.assertEqual(warn.call_args[0][0].splitlines(), [
            "rig.shade: 'red' blinn -> phong loses:",
            "   wire        ramp1.outAlpha -> specularRollOff   "
            "(ramp1 kept in the scene, disconnected; rig.cleanup() never sweeps it)",
            "   value       eccentricity = 0.5",
        ])
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertTrue(cmds.objExists("ramp1"))
        self.assertEqual(_destinations("ramp1.outAlpha"), [])
        self.assertIsNone(cmds.listAttr("red", userDefined=True))
        red.astype(Blinn)
        self.assertAlmostEqual(cmds.getAttr("red.eccentricity"), 0.3, places=5)
        self.assertEqual(_sources("red.specularRollOff"), [])

    def test_locked_connected_shared_attr_moves_with_the_lock_on_the_parent(self):
        red = Blinn.define("red")
        self.cube << red
        red.color << rn.ramp(name="ramp1").outColor << lock
        with mock.patch.object(cmds, "warning") as warn:
            red.astype(Phong)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(_sources("red.color"), ["ramp1.outColor"])
        self.assertTrue(cmds.getAttr("red.color", lock=True))
        cmds.setAttr("red.color", lock=False)
        self.assertFalse(cmds.getAttr("red.colorR", lock=True))

    def test_outgoing_wires_bind_attr_aliases_and_the_container_survive(self):
        red = Blinn.define("red")
        self.cube << red
        other = Blinn.define("other")
        _cube("cube2") << other
        other.color    << red.color
        asset = cmds.container(name="asset", addNode=["red"])
        cmds.container(asset, edit=True, publishName="tint")
        cmds.container(asset, edit=True, bindAttr=["red.color", "tint"])
        bound = cmds.container(asset, query=True, bindAttr=True)
        with container("look"):
            with mock.patch.object(cmds, "warning") as warn:
                red = red.astype(Phong)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(_sources("other.color"), ["red.color"])
        self.assertEqual(cmds.container(asset, query=True, bindAttr=True), bound)
        self.assertEqual(cmds.container(query=True, findContainer=["red"]), asset)
        self.assertEqual(cmds.container(asset, query=True, nodeList=True).count("red"), 1)
        for name in cmds.ls(type="container"):
            if name != asset:
                self.assertNotIn("red", cmds.container(name, query=True, nodeList=True) or [])
        # a lost output is disconnected, a lost alias unbound, both named
        with mock.patch.object(cmds, "warning") as warn:
            red.astype(StandardSurface)
        self.assertIn("   output      color -> other.color   (disconnected)", warn.call_args[0][0])
        self.assertIn("   output      color -> asset.tint   (disconnected)", warn.call_args[0][0])
        self.assertEqual(_sources("other.color"), [])
        self.assertIsNone(cmds.container(asset, query=True, bindAttr=True))
        self.assertEqual(cmds.container(asset, query=True, publishName=True), ["tint", "tintR", "tintG", "tintB"])
        self.assertEqual(cmds.container(query=True, findContainer=["red"]), asset)

    def test_namespace_is_kept_and_the_temp_name_never_survives(self):
        cmds.namespace(add="look")
        cmds.namespace(set="look")
        try:
            self.cube << Blinn.define("red")
        finally:
            cmds.namespace(set=":")
        self.assertTrue(cmds.objExists("look:red"))
        held = Material("look:red")
        new  = held.astype("phong")
        self.assertEqual(repr(new), 'Phong("look:red")')
        self.assertEqual(cmds.nodeType("look:red"), "phong")
        self.assertEqual(_sources("look:redSG.surfaceShader"), ["look:red.outColor"])
        self.assertEqual(cmds.ls("*__rigconvert*", recursive=True), [])
        self.assertEqual(cmds.ls("red*", recursive=True), ["look:red", "look:redSG"])
        with self.assertRaisesRegex(RuntimeError, r"'look:red' was converted to a phong"):
            str(held)

    def test_strict_refuses_with_the_warning_text(self):
        red    = self._lossy_red()
        report = shade.convert(red, "phong", dry_run=True)
        before = set(cmds.ls())
        wiring = _wiring("red")
        with mock.patch.object(cmds, "warning") as warn:
            with self.assertRaises(ValueError) as caught:
                red.astype("phong", strict=True)
        self.assertEqual(str(caught.exception), str(report))
        self.assertEqual(warn.call_count,       0)
        self.assertEqual(set(cmds.ls()),        before)
        self.assertEqual(_wiring("red"),        wiring)
        self.assertEqual(cmds.nodeType("red"),  "blinn")
        self.assertIs(type(red), Blinn)
        self.assertTrue(red.is_valid)
        # nothing lossy: strict passes
        clean = Blinn.define("clean")
        self.cube << clean
        self.assertIs(type(clean.astype("phong", strict=True)), Phong)
        self.assertEqual(cmds.nodeType("clean"), "phong")

    def test_same_type_opens_no_chunk_and_follows_the_found_material_rule(self):
        m = Phong.define("red")
        self.cube << m
        cmds.flushUndo()
        with mock.patch.object(cmds, "warning") as warn:
            self.assertIs(m.astype("phong"), m)
            self.assertIs(m.astype(Phong), m)
            self.assertEqual(shade.convert("red", Phong), m)
            self.assertEqual(shade.convert(m, "phong", dry_run=True), Conversion("red", "phong", "phong"))
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertTrue(m.is_valid)
        # attributes on a material already of that type are skipped
        self.assertIs(m.astype(Phong, cosinePower=40), m)
        self.assertEqual(cmds.getAttr("red.cosinePower"), 20)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertIs(m.astype(Phong, cosinePower=40, update=True), m)
        self.assertEqual(cmds.getAttr("red.cosinePower"), 40)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")

    def test_kwargs_on_a_conversion_write_inside_the_chunk(self):
        m = Blinn.define("red", color=[1, 0, 0])
        self.cube << m
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "no attribute 'nope'"):
            m.astype(Phong, nope=1)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertIs(type(m), Blinn)
        self.assertTrue(m.is_valid)
        p = m.astype(Phong, cosinePower=40)
        self.assertEqual(cmds.getAttr("red.cosinePower"), 40)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.shade.convert")
        cmds.undo()
        self.assertEqual(cmds.nodeType("red"), "blinn")
        cmds.redo()
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertEqual(cmds.getAttr("red.cosinePower"), 40)
        # a held node dies with its scene; a look that outlives it is a recipe
        cmds.file(new=True, force=True)
        with self.assertRaisesRegex(RuntimeError, "freed by a new scene"):
            str(p)

        def look():
            return Phong.define("red", color=[1, 0, 0], cosinePower=40)

        _cube("cube") << look()
        self.assertEqual(cmds.nodeType("red"),            "phong")
        self.assertEqual(cmds.getAttr("red.cosinePower"), 40)
        self.assertEqual(cmds.getAttr("red.color")[0],    (1.0, 0.0, 0.0))

    def test_cross_family_only_non_default_values_cross(self):
        chrome = Blinn.define("chrome")
        self.cube << chrome
        with mock.patch.object(cmds, "warning") as warn:
            chrome.astype(StandardSurface)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(cmds.nodeType("chrome"), "standardSurface")
        self.assertEqual(cmds.getAttr("chrome.specularColor")[0], (1.0, 1.0, 1.0))
        set_ = Blinn.define("set", specularColor=[0.2, 0.6, 0.8], color=[1, 0, 0], diffuse=0.5)
        _cube("cube2") << set_
        report = shade.convert(set_, "standardSurface", dry_run=True)
        self.assertEqual(report.default_shift[0][0], "specularColor")
        self.assertIn(
            "   meaning     specularColor = [0.2, 0.6, 0.8] carried; default differs "
            "(blinn 0.5 -> standardSurface 1)",
            str(report),
        )
        self.assertIn("   value       color = [1, 0, 0]", str(report))
        self.assertIn("   value       diffuse = 0.5", str(report))
        self.assertEqual(report.parked, ("diffuse", "color"))
        with mock.patch.object(cmds, "warning"):
            set_.astype(StandardSurface)
        got = cmds.getAttr("set.specularColor")[0]
        self.assertAlmostEqual(got[0], 0.2, places=5)
        self.assertAlmostEqual(got[2], 0.8, places=5)
        self.assertAlmostEqual(cmds.getAttr("set.baseColor")[0][0], 0.8, places=5)
        self.assertEqual(
            cmds.listAttr("set", userDefined=True),
            ["__diffuse__", "__color__", "__color__X", "__color__Y", "__color__Z"],
        )

    def test_type_gate_never_bypassed(self):
        rs = Blinn.define("rs", color=[1, 0, 0], reflectivity=0.9, diffuse=0.4)
        self.cube << rs
        report = shade.convert(rs, "rampShader", dry_run=True)
        self.assertEqual(report.parked, ("color", "reflectivity"))
        self.assertIn("diffuse", report.carried)
        with mock.patch.object(cmds, "warning"):
            rs = rs.astype("rampShader")
        self.assertIs(type(rs), Material)
        self.assertEqual(cmds.nodeType("rs"), "rampShader")
        self.assertAlmostEqual(cmds.getAttr("rs.diffuse"), 0.4, places=5)
        cmds.setAttr("rs.color[0].color_Color", 0, 0, 1, type="float3")
        report = shade.convert(rs, "blinn", dry_run=True)
        self.assertIn("color", report.opaque)
        self.assertIn("   opaque      color", str(report))
        with mock.patch.object(cmds, "warning"):
            rs = rs.astype(Blinn)
        self.assertEqual(cmds.nodeType("rs"), "blinn")
        self.assertEqual(cmds.getAttr("rs.color")[0], (1.0, 0.0, 0.0))
        self.assertAlmostEqual(cmds.getAttr("rs.reflectivity"), 0.9, places=5)
        # an enum with other labels is a different kind
        rs.matteOpacityMode << 0
        report = shade.convert(rs, "useBackground", dry_run=True)
        self.assertIn("   value       matteOpacityMode = 0", str(report))
        # a shared attr whose factory default differs: the meaning line
        aniso = Material.define("aniso", type="anisotropic", roughness=0.6)
        _cube("cube2") << aniso
        report = shade.convert(aniso, "phongE", dry_run=True)
        self.assertFalse(report)
        self.assertEqual(str(report).splitlines(), [
            "rig.shade: 'aniso' anisotropic -> phongE notes:",
            "   meaning     roughness = 0.6 carried; default differs (anisotropic 0.7 -> phongE 0.5)",
        ])
        # nothing lossy: strict passes, the note is still warned once
        with mock.patch.object(cmds, "warning") as warn:
            aniso.astype("phongE", strict=True)
        self.assertEqual(warn.call_count, 1)
        self.assertAlmostEqual(cmds.getAttr("aniso.roughness"), 0.6, places=5)

    def test_refusals_write_nothing(self):
        red = Blinn.define("red")
        self.cube << red
        before = set(cmds.ls())
        wiring = _wiring("red")
        with self.assertRaisesRegex(ValueError, "not a registered surface shader"):
            red.astype("aiStandardSurface")
        self.assertEqual(cmds.ls(type="unknown"), [])
        with self.assertRaisesRegex(TypeError, "not a surface shader"):
            red.astype("ramp")
        with self.assertRaisesRegex(RuntimeError, "default node"):
            Material("lambert1").astype("phong")
        with self.assertRaisesRegex(NodeTypeError, "is a shadingEngine, not a surface shader"):
            shade.convert(Default(), "phong")
        with self.assertRaisesRegex(TypeError, r"None is not a material name"):
            Phong(Material(None))
        with self.assertRaisesRegex(NodeTypeError, r"Blinn\('red'\)\.astype\(Phong\) converts it"):
            Phong(Node("red"))
        with self.assertRaisesRegex(NodeTypeError, r"Blinn\('red'\)\.astype\(Phong\) converts it"):
            Phong("red")
        with self.assertRaisesRegex(TypeError, "takes no attributes"):
            Blinn(red, type="phong")
        with self.assertRaisesRegex(NodeTypeError, "is a transform, not a surface shader"):
            shade.convert(Node("cube"), "phong")
        with self.assertRaisesRegex(ValueError, "no surface shader named 'nope'"):
            shade.convert("nope", "phong")
        with self.assertRaisesRegex(TypeError, "membership token"):
            shade.convert(-red, Phong)
        with self.assertRaisesRegex(TypeError, "conversion target"):
            red.astype(5)
        with self.assertRaisesRegex(TypeError, "names no node type"):
            red.astype(Material)
        cmds.lockNode("red", lock=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "lockNode"):
                red.astype(Phong)
        finally:
            cmds.lockNode("red", lock=False)
        self.assertEqual(set(cmds.ls()),       before)
        self.assertEqual(_wiring("red"),       wiring)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertIs(type(red), Blinn)
        self.assertTrue(red.is_valid)
        # a name converts too, and returns the new node
        p = shade.convert("red", "phong")
        self.assertIs(type(p), Phong)
        self.assertEqual(p, Node("red"))
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertFalse(red.is_valid)

    def test_referenced_shader_is_refused(self):
        folder = tempfile.mkdtemp(prefix="rig_shade_convert_ref_")
        path   = os.path.join(folder, "rig_shade_convert_ref.ma")
        try:
            cmds.file(new=True, force=True)
            cmds.shadingNode("blinn", asShader=True, name="refmat")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            before = set(cmds.ls())
            with self.assertRaisesRegex(RuntimeError, "referenced"):
                Material("ref:refmat").astype("phong")
            with self.assertRaisesRegex(RuntimeError, "referenced"):
                shade.convert("ref:refmat", Phong)
            self.assertEqual(set(cmds.ls()), before)
            self.assertEqual(cmds.nodeType("ref:refmat"), "blinn")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_one_undo_restores_everything(self):
        red   = self._lossy_red()
        asset = cmds.container(name="asset", addNode=["red"])
        info  = _info("redSG")
        cmds.connectAttr("red.message", f"{info}.material")
        uuid   = cmds.ls("red", uuid=True)[0]
        before = set(cmds.ls())
        wiring = _wiring("red")
        slot   = _dsl1_index("red")
        with mock.patch.object(cmds, "warning"):
            new = red.astype(Phong)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.shade.convert")
        cmds.undo()
        self.assertEqual(cmds.nodeType("red"),      "blinn")
        self.assertEqual(cmds.ls("red", uuid=True), [uuid])
        self.assertEqual(set(cmds.ls()),            before)
        self.assertEqual(_wiring("red"),            wiring)
        self.assertEqual(_dsl1_index("red"),        slot)
        self.assertAlmostEqual(cmds.getAttr("red.eccentricity"), 0.66, places=5)
        self.assertTrue(cmds.getAttr("red.eccentricity", lock=True))
        self.assertTrue(cmds.getAttr("red.diffuse", lock=True))
        self.assertEqual(cmds.listAttr("red", userDefined=True), ["myNote"])
        self.assertEqual(cmds.container(query=True, findContainer=["red"]), asset)
        self.assertEqual(cmds.container(asset, query=True, nodeList=True).count("red"), 1)
        self.assertEqual(_sources("redSG.surfaceShader"), ["red.outColor"])
        self.assertEqual(_sources(f"{info}.material"), ["red.message"])
        # the undo revives the old node; the new one is gone (plain message)
        self.assertTrue(red.is_valid)
        self.assertEqual(red, Blinn("red"))
        with self.assertRaisesRegex(RuntimeError, r"^red__rigconvert already deleted!$"):
            str(new)
        with mock.patch.object(cmds, "warning") as warn:
            self.assertIs(red.astype(Blinn), red)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        cmds.redo()
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertEqual(
            cmds.listAttr("red", userDefined=True),
            ["myNote", "__eccentricity__", "__specularRollOff__"],
        )
        with self.assertRaisesRegex(RuntimeError, CONVERTED.format("phong")):
            str(red)
        red = Phong("red")
        # the grammar interaction costs two entries: convert, then assign
        cmds.flushUndo()
        other = _cube("other")
        with mock.patch.object(cmds, "warning"):
            other << red.astype(Blinn)
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "otherShape"])
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        cmds.undo()
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.shade.convert")
        cmds.undo()
        self.assertEqual(cmds.nodeType("red"), "phong")
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")

    def test_abort_paths_leave_the_scene_untouched(self):
        red    = self._lossy_red()
        before = set(cmds.ls())
        wiring = _wiring("red")
        cmds.flushUndo()
        # PREPARE: the dynamic-attribute clone fails -> the new node is deleted
        with mock.patch.object(shade_convert, "_clone", side_effect=RuntimeError("boom")):
            with mock.patch.object(cmds, "warning"):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    red.astype(Phong)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_wiring("red"), wiring)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertIs(type(red), Blinn)
        self.assertTrue(red.is_valid)
        # COMMIT: the rename fails after the delete -> one guarded undo
        with mock.patch.object(cmds, "rename", side_effect=RuntimeError("boom")):
            with mock.patch.object(cmds, "warning"):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    red.astype(Phong)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_wiring("red"), wiring)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertEqual(cmds.listAttr("red", userDefined=True), ["myNote"])
        self.assertTrue(red.is_valid)
        # with the undo queue off it still rolls back, and the queue is off after
        cmds.undoInfo(state=False)
        try:
            with mock.patch.object(cmds, "rename", side_effect=RuntimeError("boom")):
                with mock.patch.object(cmds, "warning"):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        red.astype(Phong)
            self.assertFalse(cmds.undoInfo(query=True, state=True))
        finally:
            cmds.undoInfo(state=True, infinity=True)
        self.assertEqual(set(cmds.ls()),       before)
        self.assertEqual(_wiring("red"),       wiring)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        # and a conversion with the queue off works
        cmds.undoInfo(state=False)
        try:
            with mock.patch.object(cmds, "warning"):
                self.assertIs(type(red.astype(Phong)), Phong)
            self.assertEqual(cmds.nodeType("red"), "phong")
            self.assertFalse(cmds.undoInfo(query=True, state=True))
        finally:
            cmds.undoInfo(state=True, infinity=True)

    def test_stale_handles_memoize_and_dry_run(self):
        red = self._lossy_red()

        @memoize
        def wrap(node):
            return node

        stale = Node("red")
        wrap(stale)
        self.assertEqual(len(wrap._cache), 1)
        before = set(cmds.ls())
        with mock.patch.object(cmds, "warning") as warn:
            dry = shade.convert(red, "phong", dry_run=True)
        self.assertEqual(warn.call_count,      0)
        self.assertEqual(set(cmds.ls()),       before)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertIs(type(red), Blinn)
        with mock.patch.object(cmds, "warning") as warn:
            real = shade.convert(red, "phong")
        # re-pinned (round 4b NC7): the conversion returns the new node; its
        # warning is the dry run's report text
        self.assertEqual(warn.call_args[0][0], str(dry))
        self.assertEqual(real, Node("red"))
        self.assertIs(type(real), Phong)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            stale.name
        with self.assertRaisesRegex(RuntimeError, CONVERTED.format("phong")):
            stale.name
        self.assertEqual(str(Node("red")), "red")
        self.assertAlmostEqual(real.diffuse >> None, 0.33, places=5)
        self.assertEqual(len(wrap._cache), 0)
        self.assertEqual([repr(m) for m in Material.of(self.cube)], ['Phong("red")'])
        # nodes compare by class and name (the specs compared by name)
        x = Blinn.define("x")
        self.assertEqual(Blinn("x"), x)
        self.assertEqual(hash(Blinn("x")), hash(x))
        self.assertNotEqual(x, Blinn.define("y"))
        self.assertNotEqual(-x, x)
        self.assertNotEqual(x, "x")
