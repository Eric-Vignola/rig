"""``rig.shade`` is retired (user decision 2026-10-22, "Retire rig.shade"), and a
positional name given to ``create`` names both forms (user decision 2026-10-22,
"create() call shape").

* The shader classes live in ``rig.nodetypes`` (with ``Conversion``, the report
  ``mat.astype(..., dry_run=True)`` returns); ``Default()`` is in
  ``rig.membership`` (and ``rig``), beside ``Tag`` and ``Layer``;
  ``ShadingEngine.repair()`` / ``ShadingEngine.tidy()`` are the scene
  clean-ups. ``shade.convert`` / ``shade.bindings`` / ``shade.materials`` are
  gone: ``Material(name).astype(Phong)``, ``mesh.get_material_bindings()``,
  ``Material.of(x)``.
* ``Transform.create("ctl")`` says what to write: ``Transform.create(name=
  "ctl")`` makes one, ``Transform.define("ctl")`` finds or makes it.
"""

import importlib
import os
import re
import sys

from maya import cmds
import rig
from rig import List, Node
from rig._tests._base import MayaTestCase


_RIG = os.path.dirname(os.path.abspath(rig.__file__))

# what a user of the retired module would write
_RETIRED = re.compile(
    r"\brig\.shade\b|\bfrom rig import [^\n]*\bshade\b|\bshade\.(?:convert|bindings|materials|repair"
    r"|tidy|Default|Conversion)\b"
)


def _cube(name="cube"):
    return Node(cmds.polyCube(name=name, constructionHistory=False)[0])


class TestShadeRetired(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_module_gone(self):
        sys.modules.pop("rig.shade", None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("rig.shade")
        self.assertFalse(hasattr(rig, "shade"))
        self.assertNotIn("shade", rig.__all__)

    def test_shader_classes_and_conversion_in_nodetypes(self):
        from rig import nodetypes
        from rig.nodetypes import Blinn, Conversion, Phong

        for name in (
            "Material", "Lambert", "Blinn", "Phong", "PhongE", "SurfaceShader",
            "StandardSurface", "OpenPBRSurface", "Conversion",
        ):
            self.assertIn(name, nodetypes.__all__)
        red    = Blinn.define("red")
        report = red.astype(Phong, dry_run=True)
        self.assertIsInstance(report, Conversion)
        self.assertEqual((report.name, report.source, report.target), ("red", "blinn", "phong"))
        self.assertEqual(cmds.nodeType("red"), "blinn")

    def test_astype_on_a_name(self):
        """``shade.convert("red", "phong")`` is ``Material("red").astype("phong")``."""
        from rig.nodetypes import Blinn, Material, Phong

        Blinn.define("red")
        got = Material("red").astype("phong")
        self.assertIs(type(got), Phong)
        self.assertEqual(cmds.nodeType("red"), "phong")
        with self.assertRaisesRegex(TypeError, r"^Material names no node type"):
            Material("red").astype(Material)

    def test_default_in_membership(self):
        from rig import membership
        from rig.nodetypes import ShadingEngine

        self.assertIn("Default", membership.__all__)
        self.assertIn("Default", rig.__all__)
        self.assertIs(rig.Default, membership.Default)
        self.assertEqual(membership.Default(), ShadingEngine("initialShadingGroup"))
        with self.assertRaisesRegex(TypeError, r"^Default\(\) takes no name"):
            membership.Default("x")

    def test_material_of_answers_default(self):
        from rig import Default
        from rig.nodetypes import Material

        cube = _cube()
        self.assertEqual(Material.of(cube), [Default()])
        self.assertEqual(str(Default().get_material()), "standardSurface1")

    def test_shading_engine_repair(self):
        from rig.nodetypes import Blinn, ShadingEngine

        self.undo_on()
        cube = _cube()
        red  = Blinn.define("red")
        cube << red
        red.delete()
        self.assertEqual(cmds.listConnections("cubeShape", type="shadingEngine"), None)
        fixed = ShadingEngine.repair()
        self.assertIsInstance(fixed, List)
        self.assertEqual([str(x) for x in fixed], ["cubeShape"])
        self.assertEqual(cmds.listConnections("cubeShape", type="shadingEngine"), ["initialShadingGroup"])
        cmds.undo()   # one undo step
        self.assertEqual(cmds.listConnections("cubeShape", type="shadingEngine"), None)
        self.assertEqual([str(x) for x in ShadingEngine.repair(cube)], ["cubeShape"])

    def test_shading_engine_tidy(self):
        from rig.nodetypes import Blinn, ShadingEngine

        self.undo_on()
        cube = _cube()
        red  = Blinn.define("red")
        cube.f[:3] << red
        cube.f[3:] << red
        self.assertEqual(cmds.sets("redSG", query=True), ["cube.f[0:5]"])
        self.assertIsNone(ShadingEngine.tidy())
        self.assertEqual(cmds.sets("redSG", query=True), ["cubeShape"])
        cmds.undo()   # one undo step
        self.assertEqual(cmds.sets("redSG", query=True), ["cube.f[0:5]"])
        self.assertIsNone(ShadingEngine.tidy(cube))
        self.assertEqual(cmds.sets("redSG", query=True), ["cubeShape"])

    def test_no_shade_in_sources_and_docs(self):
        """No module, example, doc or test of the package writes the retired
        names (this file excepted)."""
        found = []
        for root, dirs, files in os.walk(_RIG):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if not name.endswith((".py", ".md")) or name == os.path.basename(__file__):
                    continue
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as fh:
                    for number, line in enumerate(fh, 1):
                        if _RETIRED.search(line):
                            found.append(f"{os.path.relpath(path, _RIG)}:{number}: {line.strip()}")
        self.assertEqual(found, [])

    def test_help_rig_names_the_homes(self):
        doc = rig.__doc__
        self.assertNotRegex(doc, _RETIRED)
        for text in ("ShadingEngine.repair", "rig.nodetypes"):
            self.assertIn(text, doc)


class TestCreatePositionalName(MayaTestCase):
    """create's positional is the input a class is built from; a name given
    there names the keyword form and define (user decision 2026-10-22)."""

    TEST_START_NEW_SCENE = True

    def assertNamed(self, call, message):
        before = set(cmds.ls())
        with self.assertRaises(TypeError) as caught:
            call()
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(set(cmds.ls()), before)

    def test_transform_classes(self):
        from rig.nodetypes import Joint, Transform

        self.assertNamed(
            lambda: Transform.create("ctl"),
            "Transform.create('ctl'): the name is a keyword: Transform.create(name='ctl') makes one, "
            "Transform.define('ctl') finds or makes it; a positional argument is the TransformData "
            "a node is built from",
        )
        self.assertNamed(
            lambda: Joint.create("j1"),
            "Joint.create('j1'): the name is a keyword: Joint.create(name='j1') makes one, "
            "Joint.define('j1') finds or makes it; a positional argument is the TransformData "
            "a node is built from",
        )

    def test_plain_classes(self):
        from rig.nodetypes import Choice, Follicle, ObjectSet

        for cls in (Choice, ObjectSet):
            with self.subTest(cls.__name__):
                c = cls.__name__
                self.assertNamed(
                    lambda: cls.create("x"),
                    f"{c}.create('x'): the name is a keyword: {c}.create(name='x') makes one, "
                    f"{c}.define('x') finds or makes it; it takes no positional argument",
                )
        self.assertNamed(
            lambda: Follicle.create("x"),
            "Follicle.create('x'): the name is a keyword: Follicle.create(name='x') makes one; "
            "it takes no positional argument",
        )

    def test_abstract_classes_name_their_abstraction(self):
        from rig.nodetypes import DAGNode, Geometry

        for cls in (DAGNode, Geometry):
            with self.subTest(cls.__name__):
                with self.assertRaisesRegex(TypeError, r"makes no node: .* is an abstract Maya type"):
                    cls.create("x")

    def test_shader_classes(self):
        from rig.nodetypes import Blinn, Material

        self.assertNamed(
            lambda: Blinn.create("red"),
            "Blinn.create('red'): the name is a keyword: Blinn.create(name='red') makes one, "
            "Blinn.define('red') finds or makes it; it takes no positional argument",
        )
        self.assertNamed(
            lambda: Material.create("red"),
            "Material.create('red'): the name is a keyword: Material.create(name='red', "
            "type='blinn') makes one, Material.define('red', type='blinn') finds or makes it; "
            "it takes no positional argument",
        )

    def test_data_classes(self):
        from rig.nodetypes import AnimCurveTL, Mesh, NurbsCurve, NurbsSurface

        self.assertNamed(
            lambda: Mesh.create("m"),
            "Mesh.create('m'): the name is a keyword: Mesh.create(mesh_data, name='m') makes one; "
            "a positional argument is the MeshData a mesh is built from",
        )
        self.assertNamed(
            lambda: NurbsCurve.create("c"),
            "NurbsCurve.create('c'): the name is a keyword: NurbsCurve.create(points, name='c') "
            "makes one; a positional argument is the points or BSplineData a curve is built from",
        )
        self.assertNamed(
            lambda: NurbsSurface.create("s"),
            "NurbsSurface.create('s'): the name is a keyword: NurbsSurface.create(data, name='s') "
            "makes one; a positional argument is the BSplinePatchData a surface is built from",
        )
        self.assertNamed(
            lambda: AnimCurveTL.create("k"),
            "AnimCurveTL.create('k'): the name is a keyword: AnimCurveTL.create(name='k') makes "
            "one, AnimCurveTL.define('k') finds or makes it; a positional argument is the keys a "
            "curve is built from",
        )

    def test_keyword_forms_work(self):
        from rig.nodetypes import Blinn, Transform

        self.assertEqual(str(Transform.create(name="ctl")), "ctl")
        self.assertEqual(str(Transform.define("ctl")), "ctl")
        self.assertEqual(str(Blinn.create(name="red")), "red")
