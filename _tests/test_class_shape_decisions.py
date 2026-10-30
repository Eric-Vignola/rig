"""The user's decisions of 2026-10-30 on shapes and their spaces (feat/shape-under-transform,
commit 3), one test class each:

* Q1, a surface update: data without knots in a direction (an older cgmath's, or Maya's default
  ones) says even knots there. It is refused, as other knots are, when the surface is uneven
  there at degree 2 or more (the shape would change); a degree-1 uneven direction stays taken
  (it only reparametrizes), and the surface keeps its knots.
* Q2, ``Mesh('x').load(path, world_space=...)``: anything but a bool is a TypeError, before the
  files are read, with the message of the NURBS node-form loads.
* Q3, ``Mesh.get_per_face_vertex_normals(*, world_space=False)``: object space by default.
* Q5, ``serialize_hierarchy`` skips user attributes that hold geometry, with a warning: their
  value never loads back (no file takes it; it broke every save of the hierarchy).
* Q6, ``world_space=True`` without ``transform=`` freezes the shape at its world positions, on
  ``create`` and on the class-form ``load``: the data's matrix goes in the points, a mesh's normals
  through its inverse transpose, and the new transform sits at the origin with identity channels.
  False, the default, is the new transform at the matrix, the points the shape's own.
* Q7, ``transform=`` takes any node derived from transform (the README says so).
* Q8, ``transform=`` with ``world_space=True`` refuses a singular target (its world 3x3 scales some
  direction by less than 1e-9), before anything is built; ``world_space=False`` stays taken.
* Q9, the default shape name under ``transform=`` is in the target's namespace, whatever the
  current one: a referenced target's shape is a local node in the reference's namespace, which
  survives a save and a reopen. ``name=`` is used as given.

Q4 (a locked or driven segmentScaleCompensate keeps raising) changes nothing.

Each test failed against the source files of 4c53f99 (negative control), except the guards.
"""

import inspect
import os
import re
from unittest import mock

import numpy as np
from maya import cmds

from rig import container
from rig.nodetypes import Mesh, NurbsCurve, NurbsSurface, Transform
from rig._tests.test_class_nurbs_create import CURVE_FIELDS, PATCH_FIELDS, _older_cgmath
from rig._tests.test_class_shape_space import NEW_FIELDS, NO_FIELDS, OPEN, PERIODIC, _cylinder, _surface, _uneven
from rig._tests.test_class_shape_under_transform import (
    CLASSES,
    OBJECT,
    WORLD,
    _args,
    _ball,
    _Case,
    _circle,
    _cube,
    _data_points,
    _local,
    _normals,
    _shading,
    _sheared,
    _source,
    _surface_normals,
    _world,
)

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodetypes", "README.md")

# a mesh holds float32 points, a curve and a surface doubles
ATOL = {Mesh: 1e-5, NurbsCurve: 1e-9, NurbsSurface: 1e-9}


def _quiet_serialize(shape):
    """What `shape`'s serialize gives on a cgmath without the new fields, its warning unsaid."""
    fields = {NurbsCurve: ("BSplineData", CURVE_FIELDS), NurbsSurface: ("BSplinePatchData", PATCH_FIELDS)}
    with _older_cgmath(*fields[type(shape)]), mock.patch("rig.nodetypes.nurbs.LOGGER"):
        return shape.serialize()


def _new_transform(shape):
    """The transform `shape` was made under, by its long name."""
    return shape.get_parent().get_long_name()


class TestQ1ASurfaceUpdateOfDataWithoutKnots(_Case):
    REFUSAL = (
        r"^NurbsSurface\('{name}'\)\.load: {src} holds another CV layout than '{name}' \({size} CVs of degrees "
        r"{degrees} against its {size} of degrees {degrees}, or other forms or knots\); a load updates the CVs "
        r"of the same layout: NurbsSurface\.load\({src}\) makes a new surface$"
    )

    def surface(self, name, knots_u, knots_v, degrees, forms=(OPEN, OPEN)):
        """A surface of these knots, of as many CVs as they take, with a bump on it. Its NurbsSurface."""
        counts = [
            len(knots) - degree + 1 - (degree if form == PERIODIC else 0)
            for knots, degree, form in zip((knots_u, knots_v), degrees, forms)
        ]
        grid = [[(u, v, 0.1 * ((u * v) % 3)) for v in range(counts[1])] for u in range(counts[0])]
        if forms[0] == PERIODIC:
            grid += grid[: degrees[0]]
        if forms[1] == PERIODIC:
            grid = [row + row[: degrees[1]] for row in grid]
        cvs = [cv for row in grid for cv in row]
        return NurbsSurface(_surface(name, cvs, knots_u, knots_v, degrees, forms, cmds.createNode("transform")))

    def refused(self, surface, src="the BSplinePatchData", data=None):
        fn   = surface.get_fn_set()
        size = f"{fn.numCVsInU} x {fn.numCVsInV}"
        return self.assert_nothing_made(
            ValueError,
            self.REFUSAL.format(name=surface, src=src, size=size, degrees=f"{fn.degreeInU} x {fn.degreeInV}"),
            lambda: surface.load(_quiet_serialize(surface) if data is None else data),
        )

    def test_an_uneven_direction_of_degree_2_or_more_refuses_it(self):
        """U, V, a degree of 2 or 3, open or periodic: the CVs would make another shape there."""
        periodic = np.cumsum([-3.0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1])  # the spacing repeats every 8 spans
        for label, knots_u, knots_v, degrees, forms in (
            ("degree 3 in U", (0, 0, 0, 1, 5, 5, 5), (0, 1, 2), (3, 1), (OPEN, OPEN)),
            ("degree 2 in U", (0, 0, 1, 5, 5), (0, 1, 2), (2, 1), (OPEN, OPEN)),
            ("degree 3 in V", (0, 1, 2), (0, 0, 0, 1, 5, 5, 5), (1, 3), (OPEN, OPEN)),
            ("degree 2 in V", (0, 0, 1, 2, 2), (0, 0, 1, 5, 5), (2, 2), (OPEN, OPEN)),
            ("periodic, degree 3 in U", periodic.tolist(), (0, 1, 2), (3, 1), (PERIODIC, OPEN)),
        ):
            with self.subTest(label):
                surface = self.surface(label.replace(" ", "_").replace(",", ""), knots_u, knots_v, degrees, forms)
                before  = _local(surface)
                self.refused(surface)
                np.testing.assert_array_equal(_local(surface), before)

    def test_a_degree_1_uneven_direction_stays_taken(self):
        """guard: it only reparametrizes: the CVs land, the surface keeps its knots."""
        cylinder = NurbsSurface(_cylinder(cmds.createNode("transform", name="grp")))
        data     = _quiet_serialize(cylinder)
        knots    = list(cylinder.get_fn_set().knotsInV())
        self.assertIsNone(getattr(data, "knots_v", None))
        cmds.xform(f"{cylinder}.cv[0][0]", translation=(5, 5, 5), objectSpace=True)
        moved = _local(cylinder)
        self.assertIs(cylinder.load(data), cylinder)
        self.assertGreater(np.abs(_local(cylinder) - moved).max(), 1)
        np.testing.assert_allclose(_local(cylinder), _data_points(data), rtol=0, atol=1e-12)
        self.assertEqual(list(cylinder.get_fn_set().knotsInV()), knots)

    def test_one_uneven_direction_of_degree_3_is_enough(self):
        """A degree-1 uneven direction beside it does not save it."""
        self.refused(NurbsSurface(_uneven(cmds.createNode("transform", name="grp"))))

    def test_the_file_of_an_older_cgmath(self):
        surface = NurbsSurface(_uneven(cmds.createNode("transform", name="grp")))
        path    = os.path.join(self.folder(), "patch.npz")
        _quiet_serialize(surface).save(path)
        self.refused(surface, re.escape(repr(path)), path)

    def test_the_new_cgmath_s_data_of_another_direction(self):
        """Data that holds knots in V alone (U is Maya's default there) on a surface uneven in U."""
        if not NEW_FIELDS:
            self.skipTest(NO_FIELDS)
        source = self.surface("src", (0, 0, 0, 1, 2, 2, 2), (0, 0, 0, 1, 5, 5, 5), (3, 3))
        target = self.surface("tgt", (0, 0, 0, 1, 5, 5, 5), (0, 0, 0, 1, 5, 5, 5), (3, 3))
        data   = source.serialize()
        self.assertIsNone(data.knots_u)
        self.assertIsNotNone(data.knots_v)
        self.refused(target, data=data)

    def test_even_knots_stay_taken(self):
        """guard: even knots, Maya's default ones or to another range, make the same shape."""
        for label, knots in (("default", (0, 0, 0, 1, 2, 2, 2)), ("other_range", (0, 0, 0, 3, 6, 6, 6))):
            with self.subTest(label):
                surface = self.surface(label, knots, (0, 1, 2), (3, 1))
                data    = _quiet_serialize(surface)
                self.assertIs(surface.load(data), surface)
                self.assertEqual(list(surface.get_fn_set().knotsInU()), list(knots))


class TestQ2AMeshUpdateTakesABool(_Case):
    def setUp(self):
        super().setUp()
        self.box  = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        self.path = self.box.save(os.path.join(self.folder(), "box.npz"))

    def test_anything_but_a_bool_is_refused(self):
        before = _local(self.box)
        for value in (None, 0, 1, "world", 1.0):
            with self.subTest(value=value):
                self.assert_nothing_made(
                    TypeError,
                    r"^Mesh\('boxShape'\)\.load: world_space= takes True or False \(the space the file was saved "
                    rf"in\), not {re.escape(repr(value))}$",
                    lambda: self.box.load(self.path, world_space=value),
                )
        np.testing.assert_array_equal(_local(self.box), before)

    def test_before_the_files_are_read(self):
        self.assert_nothing_made(TypeError, "takes True or False",
                                 lambda: self.box.load("no_such_file.npz", world_space=None))

    def test_the_message_of_the_nurbs_loads(self):
        curve = _circle()
        path  = curve.save(os.path.join(self.folder(), "c.npz"))
        said  = []
        for node, file in ((curve, path), (self.box, self.path)):
            with self.assertRaises(TypeError) as caught:
                node.load(file, world_space=None)
            said.append(str(caught.exception).partition(": ")[2])
        self.assertEqual(said[0], said[1])

    def test_a_bool_stays_taken(self):
        """guard: True, False and a numpy bool."""
        for value in (True, False, np.True_, np.False_):
            with self.subTest(value=value):
                self.assertIs(self.box.load(self.path, world_space=value), self.box)

    def test_the_readme_says_it_of_the_mesh_too(self):
        with open(README, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        self.assertIn("The space of a `Mesh`, `NurbsCurve` or `NurbsSurface` update is a bool", text)


class TestQ3PerFaceVertexNormals(_Case):
    def setUp(self):
        super().setUp()
        self.box = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        cmds.xform("box", rotation=(0, 0, 90), scale=(2, 1, 3))

    def read(self, **keywords):
        return np.array([list(normal) for normal in self.box.get_per_face_vertex_normals(**keywords)])

    def test_object_space_by_default(self):
        normals = self.read()
        self.assertEqual(normals.shape, (24, 3))
        np.testing.assert_allclose(normals, _normals(self.box, OBJECT), atol=1e-6)
        self.assertGreater(np.abs(normals - _normals(self.box, WORLD)).max(), 0.5)

    def test_world_space_by_keyword(self):
        np.testing.assert_allclose(self.read(world_space=True), _normals(self.box, WORLD), atol=1e-6)
        np.testing.assert_allclose(self.read(world_space=False), _normals(self.box, OBJECT), atol=1e-6)

    def test_by_keyword_only(self):
        parameter = inspect.signature(self.box.get_per_face_vertex_normals).parameters["world_space"]
        self.assertEqual((parameter.kind, parameter.default), (inspect.Parameter.KEYWORD_ONLY, False))
        with self.assertRaisesRegex(TypeError, "takes 1 positional argument but 2 were given"):
            self.box.get_per_face_vertex_normals(True)

    def test_the_readme_names_no_world_only_query(self):
        with open(README, encoding="utf-8") as fh:
            self.assertNotIn("one query is world space", fh.read())


class TestQ5GeometryAttrsAreSkipped(_Case):
    MOVED = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0)

    def setUp(self):
        super().setUp()
        self.root = cmds.createNode("transform", name="root")
        cube      = cmds.polyCube(name="box", constructionHistory=False)[0]
        curve     = cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0)])
        plane     = cmds.nurbsPlane(name="srf", constructionHistory=False)[0]
        cage      = cmds.lattice(cube, name="cage")[1]
        for name, kind, plug in (
            ("held_mesh", "mesh", "boxShape.worldMesh[0]"),
            ("held_curve", "nurbsCurve", f"{cmds.listRelatives(curve, shapes=True)[0]}.worldSpace[0]"),
            ("held_surface", "nurbsSurface", f"{plane}Shape.worldSpace[0]"),
            ("held_lattice", "lattice", f"{cage}.worldLattice"),
        ):
            cmds.addAttr(self.root, longName=name, dataType=kind)
            cmds.connectAttr(plug, f"{self.root}.{name}")
        cmds.addAttr(self.root, longName="note", dataType="string")
        cmds.setAttr(f"{self.root}.note", "kept", type="string")

    GEOMETRY = ("held_curve", "held_lattice", "held_mesh", "held_surface")

    def skipped(self, node):
        """The user attributes of `node`'s data, and the attributes its warnings name."""
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            data = Transform(node).serialize()
        names = [re.search(rf"attr {node}\.(\S+):", line).group(1) for line in logs.output]
        return data.user_defined_attributes, names

    def test_they_are_skipped_with_a_warning(self):
        kept, skipped = self.skipped(self.root)
        self.assertEqual(sorted(skipped), sorted(self.GEOMETRY))
        self.assertEqual(sorted(kept), ["note"])
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            Transform(self.root).serialize()
        self.assertEqual(
            logs.records[0].getMessage(),
            "Skipped user defined attr root.held_mesh: it holds geometry, whose value never loads back",
        )

    def test_the_hierarchy_goes_to_every_file_and_back(self):
        folder = self.folder()
        for ext in ("npz", "json", "pkl"):
            with self.subTest(ext=ext):
                with self.assertLogs("rig.nodetypes.transform", level="WARNING"):
                    path = Transform(self.root).save_hierarchy(os.path.join(folder, f"h.{ext}"))
                made = Transform.load_hierarchy(path)[0]
                self.assertEqual([str(a).partition(".")[2] for a in made.list_attr(ud=True)], ["note"])
                self.assertEqual(cmds.getAttr(f"{made}.note"), "kept")

    def test_create_hierarchy_takes_the_data(self):
        with self.assertLogs("rig.nodetypes.transform", level="WARNING"):
            data = Transform(self.root).serialize_hierarchy()
        made = Transform.create_hierarchy(data)[0]
        self.assertEqual([str(a).partition(".")[2] for a in made.list_attr(ud=True)], ["note"])

    def test_an_empty_one_is_skipped_too(self):
        """By its type: what it holds right now says nothing of what the attribute is for."""
        node = cmds.createNode("transform", name="empty")
        cmds.addAttr(node, longName="slot", dataType="mesh")
        self.assertEqual(self.skipped(node), ({}, ["slot"]))

    def test_a_multi_is_skipped_with_the_same_words(self):
        node = cmds.createNode("transform", name="multi")
        cmds.addAttr(node, longName="many", dataType="mesh", multi=True)
        cmds.connectAttr("boxShape.worldMesh[0]", f"{node}.many[2]")
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            data = Transform(node).serialize()
        self.assertEqual(data.user_defined_attributes, {})
        self.assertEqual(
            logs.records[0].getMessage(),
            "Skipped user defined attr multi.many: it holds geometry, whose value never loads back",
        )

    def test_a_compound_keeps_its_other_children(self):
        node = cmds.createNode("transform", name="grouped")
        cmds.addAttr(node, longName="group", attributeType="compound", numberOfChildren=3)
        cmds.addAttr(node, longName="first", attributeType="double",   parent="group")
        cmds.addAttr(node, longName="geo",   dataType="mesh",          parent="group")
        cmds.addAttr(node, longName="last",  attributeType="double",   parent="group")
        cmds.setAttr(f"{node}.last", 2.5)
        kept, skipped = self.skipped(node)
        self.assertEqual((sorted(kept), skipped), (["first", "group", "last"], ["geo"]))
        self.assertEqual(kept["group"]["numberOfChildren"], 2)
        data = Transform(node).serialize_hierarchy()
        made = Transform.create_hierarchy(data)[0]
        self.assertEqual(cmds.attributeQuery("last", node=str(made), listParent=True), ["group"])
        self.assertEqual(cmds.getAttr(f"{made}.last"), 2.5)

    def test_a_compound_of_nothing_but_geometry_goes_with_them(self):
        """Nested too: nothing of the parent loads back."""
        node = cmds.createNode("transform", name="grouped")
        cmds.addAttr(node, longName="outer", attributeType="compound", numberOfChildren=2)
        cmds.addAttr(node, longName="inner", attributeType="compound", numberOfChildren=1, parent="outer")
        cmds.addAttr(node, longName="plain", attributeType="double", parent="outer")
        cmds.addAttr(node, longName="geo", dataType="nurbsCurve", parent="inner")
        kept, skipped = self.skipped(node)
        self.assertEqual((sorted(kept), sorted(skipped)), (["outer", "plain"], ["geo", "inner"]))
        self.assertEqual(kept["outer"]["numberOfChildren"], 1)

    def test_a_generic_slot_stays(self):
        """guard: a typed attr of any data keeps its spec; what a connection fed it is not saved."""
        node = cmds.createNode("transform", name="slot")
        cmds.addAttr(node, longName="any", attributeType="typed")
        cmds.connectAttr("boxShape.worldMesh[0]", f"{node}.any")
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            data = Transform(node).serialize().user_defined_attributes
        self.assertEqual(data["any"]["attributeType"], "typed")
        self.assertNotIn("value", data["any"])

    def test_the_other_typed_data_stays(self):
        """guard: a string, an array and a matrix keep their values."""
        node = cmds.createNode("transform", name="typed")
        cmds.addAttr(node, longName="text", dataType="string")
        cmds.setAttr(f"{node}.text", "a", type="string")
        cmds.addAttr(node, longName="numbers", dataType="doubleArray")
        cmds.setAttr(f"{node}.numbers", [1.0, 2.0], type="doubleArray")
        cmds.addAttr(node, longName="held", dataType="matrix")
        cmds.setAttr(f"{node}.held", *self.MOVED, type="matrix")
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            data = Transform(node).serialize().user_defined_attributes
        self.assertEqual(sorted(data),             ["held", "numbers", "text"])
        self.assertEqual(data["text"]["value"],    "a")
        self.assertEqual(data["numbers"]["value"], [1.0, 2.0])
        self.assertEqual(data["held"]["value"],    list(self.MOVED))


class TestQ6FrozenAtWorldPositions(_Case):
    def assert_at_rest(self, shape):
        """The transform `shape` was made under sits at the origin with identity channels."""
        node = _new_transform(shape)
        self.assertEqual(
            [cmds.getAttr(f"{node}.{attr}")[0] for attr in ("t", "r", "s", "sh")],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)],
        )
        np.testing.assert_array_equal(np.array(cmds.getAttr(f"{node}.worldMatrix[0]")).reshape(4, 4), np.eye(4))

    def source(self, cls):
        """`cls`'s moved, rotated, scaled source and the data it serializes, in object space."""
        if cls is not Mesh and not NEW_FIELDS:
            self.skipTest(NO_FIELDS)  # object-space B-spline data holds its matrix on the new cgmath only
        source = _source(cls)
        return source, _args(source)

    def test_the_user_s_example(self):
        """vtx[0] of a polyCube moved to (0, 10, 0), the numbers the user saw."""
        cmds.polyCube(name="box")
        cmds.xform("box", translation=(0, 10, 0))
        cmds.xform(cmds.createNode("transform", name="ctl"), translation=(5, 0, 0))
        data, uvs = Mesh("box").serialize()
        path      = Mesh("box").save(os.path.join(self.folder(), "box.npz"))

        def vertex(shape, translate, own, world):
            self.assertEqual(cmds.getAttr(f"{_new_transform(shape)}.t")[0], translate)
            self.assertEqual(tuple(_local(shape)[0]), own)
            self.assertEqual(tuple(_world(shape)[0]), world)

        copy_a = Mesh.create(data, uvs, name="copyA")
        copy_b = Mesh.create(data, uvs, name="copyB", world_space=True)
        loaded = Mesh.load(path, name="copyC", world_space=True)
        under  = Mesh.create(data, uvs, transform="ctl", world_space=True)
        self.assertEqual(_new_transform(copy_a), "|copyA")
        self.assertEqual(_new_transform(copy_b), "|copyB")
        vertex(copy_a, (0.0, 10.0, 0.0), (-0.5, -0.5, 0.5), (-0.5, 9.5, 0.5))
        vertex(copy_b, (0.0, 0.0, 0.0),  (-0.5, 9.5, 0.5),  (-0.5, 9.5, 0.5))
        vertex(loaded, (0.0, 0.0, 0.0),  (-0.5, 9.5, 0.5),  (-0.5, 9.5, 0.5))
        vertex(under,  (5.0, 0.0, 0.0),  (-5.5, 9.5, 0.5),  (-0.5, 9.5, 0.5))

    def test_every_class_under_a_moved_rotated_scaled_source(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source, args = self.source(cls)
                shape = cls.create(*args, world_space=True)
                self.assert_at_rest(shape)
                np.testing.assert_allclose(_local(shape), _world(source), rtol=0, atol=ATOL[cls])
                np.testing.assert_allclose(_world(shape), _world(source), rtol=0, atol=ATOL[cls])
                self.assertGreater(np.abs(_local(shape) - _local(source)).max(), 1)

    def test_false_is_the_new_transform_at_the_matrix(self):
        """guard: the default, and False by keyword, as they were."""
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source, args = self.source(cls)
                for shape in (cls.create(*args), cls.create(*args, world_space=False)):
                    np.testing.assert_array_equal(_local(shape), _data_points(args[0]))
                    np.testing.assert_allclose(_world(shape), _world(source), rtol=0, atol=ATOL[cls])

    def test_a_mesh_s_normals_and_shading_follow(self):
        source = _cube()
        shape  = Mesh.create(*_args(source), world_space=True)
        np.testing.assert_allclose(_normals(shape), _normals(source), rtol=0, atol=1e-5)
        np.testing.assert_allclose(_normals(shape, OBJECT), _normals(source), rtol=0, atol=1e-5)
        self.assertEqual(_shading(shape)[0], _shading(source)[0])
        self.assertEqual(shape.get_uv_sets(), ("map1", "uv2"))
        self.assertEqual([str(e) for e in shape.get_shading_engines()], ["initialShadingGroup"])

    def test_normals_false_leaves_the_mesh_smooth(self):
        source = _cube()
        shape  = Mesh.create(*_args(source), world_space=True, normals=False)
        np.testing.assert_allclose(_local(shape), _world(source), rtol=0, atol=1e-5)
        self.assertEqual(_shading(shape), (0, 0))

    def test_world_space_data_has_nothing_to_freeze(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source = _source(cls)
                args   = _args(source, True)
                frozen = cls.create(*args, world_space=True)
                self.assert_at_rest(frozen)
                np.testing.assert_array_equal(_local(frozen), _local(cls.create(*args)))

    def test_data_without_a_matrix_is_the_identity(self):
        """An older cgmath's B-spline data, and points: the CVs as they are."""
        for maker, cls in ((_circle, NurbsCurve), (_ball, NurbsSurface)):
            with self.subTest(cls=cls.__name__):
                source = maker()
                data   = _quiet_serialize(source)
                frozen = cls.create(data, world_space=True)
                self.assert_at_rest(frozen)
                np.testing.assert_array_equal(_local(frozen), _local(source))
        points = [(0, 0, 0), (1, 2, 0), (2, 0, 1), (3, 1, 1), (4, 0, 0)]
        curve  = NurbsCurve.create(points, world_space=True)
        self.assert_at_rest(curve)
        np.testing.assert_array_equal(_local(curve), points)

    def test_a_matrix_of_none_is_the_identity(self):
        if not NEW_FIELDS:
            self.skipTest(NO_FIELDS)
        source      = _circle()
        data        = source.serialize()
        data.matrix = None
        frozen      = NurbsCurve.create(data, world_space=True)
        self.assert_at_rest(frozen)
        np.testing.assert_array_equal(_local(frozen), _local(source))

    def test_knots_and_forms_come_through(self):
        if not NEW_FIELDS:
            self.skipTest(NO_FIELDS)
        parent = cmds.createNode("transform", name="grp")
        cmds.xform(parent, translation=(1, 2, 3), rotation=(10, 20, 30), scale=(2, 1, 0.5))
        for shape in (_uneven(parent, "srf"), _cylinder(parent, "cyl")):
            with self.subTest(shape=shape):
                source = NurbsSurface(shape)
                frozen = NurbsSurface.create(source.serialize(), world_space=True)
                fn, own = frozen.get_fn_set(), source.get_fn_set()
                self.assertEqual(list(fn.knotsInU()),      list(own.knotsInU()))
                self.assertEqual(list(fn.knotsInV()),      list(own.knotsInV()))
                self.assertEqual((fn.formInU, fn.formInV), (own.formInU, own.formInV))
                np.testing.assert_allclose(_local(frozen), _world(source), rtol=0, atol=1e-9)
        uneven = cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)],
                            knot=[0, 0, 0, 1, 5, 5, 5], name="uneven")
        cmds.xform(uneven, translation=(1, 2, 3), rotation=(10, 20, 30), scale=(2, 1, 0.5))
        for source in (NurbsCurve(uneven), _circle()):
            with self.subTest(curve=str(source)):
                frozen = NurbsCurve.create(source.serialize(), world_space=True)
                self.assertEqual(frozen.get_fn_set().form, source.get_fn_set().form)
                self.assertEqual(list(frozen.get_fn_set().knots()), list(source.get_fn_set().knots()))
                np.testing.assert_allclose(_local(frozen), _world(source), rtol=0, atol=1e-9)

    def test_a_holed_face(self):
        from cgmath.geometry import MeshData

        square = [(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)]
        data   = MeshData(
            indices      = np.arange(7),
            counts       = np.array([7]),
            points       = np.array(square + [(1, 0, 1), (3, 0, 1), (2, 0, 3)], float),
            hole_faces   = np.array([0]),
            hole_counts  = np.array([3]),
            hole_indices = np.array([4, 5, 6]),
        )
        data.matrix = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 5, 0, 1)], float)
        shape       = Mesh.create(data, world_space=True)
        self.assertEqual(shape.get_fn_set().getHoles(), ((0, (4, 5, 6)),))
        np.testing.assert_allclose(_local(shape), data.points + (0, 5, 0), atol=1e-5)
        self.assert_at_rest(shape)

    def test_the_class_form_load(self):
        folder = self.folder()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source, _ = self.source(cls)
                path   = source.save(os.path.join(folder, f"{cls.__name__}.npz"))
                frozen = cls.load(path, world_space=True)
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.load")
                self.assert_at_rest(frozen)
                np.testing.assert_allclose(_local(frozen), _world(source), rtol=0, atol=ATOL[cls])
                kept = cls.load(path, world_space=False)
                np.testing.assert_array_equal(_local(kept), _local(source))
                np.testing.assert_allclose(_world(kept), _world(source), rtol=0, atol=ATOL[cls])

    def test_the_class_form_load_takes_the_data_too(self):
        source = _cube()
        data   = source.serialize(include_uvs=False)
        shape  = Mesh.load(data, world_space=True)
        np.testing.assert_allclose(_local(shape), _world(source), rtol=0, atol=1e-5)

    def test_the_keyword_of_the_class_form_load(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                parameter = inspect.signature(cls.load).parameters["world_space"]
                self.assertEqual((parameter.kind, parameter.default), (inspect.Parameter.KEYWORD_ONLY, False))

    def test_anything_but_a_bool(self):
        for cls in CLASSES:
            args = _args(_source(cls))
            path = _source(cls).save(os.path.join(self.folder(), f"{cls.__name__}.npz"))
            for value in (None, 1, "world"):
                with self.subTest(cls=cls.__name__, value=value):
                    said = (
                        r"world_space= takes True or False \(where the data's points land: in the world, or as the "
                        rf"shape's own\), not {re.escape(repr(value))}$"
                    )
                    self.assert_nothing_made(TypeError, rf"^{cls.__name__}\.create\(\): {said}",
                                             lambda: cls.create(*args, world_space=value))
                    self.assert_nothing_made(TypeError, rf"^{cls.__name__}\.load: {said}",
                                             lambda: cls.load(path, world_space=value))

    def test_a_bool_is_checked_before_the_file_is_read(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                self.assert_nothing_made(TypeError, "takes True or False",
                                         lambda: cls.load("no_such_file.npz", world_space=None))

    def test_an_object_s_transform_goes_in_the_points(self):
        from cgmath.render.scene import Object

        data  = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0]).serialize(include_uvs=False)
        hero  = Object(name="hero", mesh=data, translate=(5, 1, 2), rotate=(10, 45, 0), scale=(2, 1, 3), rotate_order=3,
                       rotate_axis=(0, 30, 0), visibility=False)
        shape = Mesh.create(hero, world_space=True)
        self.assert_at_rest(shape)
        self.assertEqual(cmds.getAttr(f"{_new_transform(shape)}.rotateAxis")[0], (0.0, 0.0, 0.0))
        placed = Mesh.create(hero)
        np.testing.assert_allclose(_local(shape), _world(placed), rtol=0, atol=1e-5)
        np.testing.assert_allclose(_normals(shape), _normals(placed), rtol=0, atol=1e-5)
        # what moves nothing still goes on it
        self.assertEqual(cmds.getAttr(f"{_new_transform(shape)}.rotateOrder"), 3)
        self.assertFalse(cmds.getAttr(f"{_new_transform(shape)}.visibility"))

    def test_an_object_under_a_parent(self):
        """In a Scene: its world matrix goes in the points, its rotate order on the transform, and
        no visibility (only a lone Object gives one, frozen or not)."""
        from cgmath.hierarchy import TransformData
        from cgmath.render.scene import Object, Scene

        data  = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0]).serialize(include_uvs=False)
        group = TransformData(name="grp")
        hero  = Object(name="hero", mesh=data, translate=(5, 0, 0), rotate=(0, 90, 0), rotate_order=2, visibility=False)
        scene = Scene()
        scene.append(group)
        scene.append(hero)
        hero.set_parent(group)
        shape = Mesh.create(hero, world_space=True)
        self.assert_at_rest(shape)
        self.assertEqual(cmds.getAttr(f"{_new_transform(shape)}.rotateOrder"), 2)
        self.assertTrue(cmds.getAttr(f"{_new_transform(shape)}.visibility"))
        np.testing.assert_allclose(_local(shape), _world(Mesh.create(hero)), rtol=0, atol=1e-5)

    def test_one_undo_step(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                _, args = self.source(cls)
                cmds.flushUndo()
                before = self.scene_state()
                shape  = cls.create(*args, world_space=True)
                path   = shape.get_long_name()
                uuid   = cmds.ls(path, uuid=True)[0]
                after  = self.scene_state(meshes=[path] if cls is Mesh else [])
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.create")
                self.undo_steps(1)
                self.assert_state_equal(self.scene_state(), before)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.redo_steps(1)
                self.assert_state_equal(self.scene_state(meshes=[path] if cls is Mesh else []), after)
                self.assertEqual(cmds.ls(uuid, long=True), [path])

    def test_the_selection_is_kept(self):
        cmds.createNode("transform", name="picked")
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                _, args = self.source(cls)
                cmds.select("picked", replace=True)
                cls.create(*args, world_space=True)
                self.assertEqual(cmds.ls(orderedSelection=True), ["picked"])

    def test_in_a_container(self):
        """The transform and the shape join, as without the keyword."""
        args = _args(_cube())
        with container("box"):
            shape = Mesh.create(*args, name="frozen", world_space=True)
        self.assertEqual(sorted(cmds.container("box", query=True, nodeList=True)), ["frozen", "frozenShape"])
        self.assert_at_rest(shape)

    def test_a_mirrored_matrix(self):
        """Pinned as documented, not fixed, as under a mirrored target: a frozen mesh lands, its world
        normals where they were and every one locked (Maya's own face the other way); a frozen
        surface lands inside out against the source."""
        source = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        cmds.xform("box", translation=(1, 2, 3), rotation=(10, 20, 30), scale=(-1, 1, 1))
        frozen = Mesh.create(*_args(source), world_space=True)
        np.testing.assert_allclose(_world(frozen), _world(source), rtol=0, atol=1e-5)
        np.testing.assert_allclose(_normals(frozen), _normals(source), rtol=0, atol=1e-5)
        self.assertEqual(_shading(frozen), (_shading(source)[0], frozen.get_fn_set().numNormals))
        if NEW_FIELDS:
            ball = _ball()
            cmds.xform(ball.get_parent().get_long_name(), scale=(-1, 1, 1))
            flipped = NurbsSurface.create(*_args(ball), world_space=True)
            np.testing.assert_allclose(_world(flipped), _world(ball), rtol=0, atol=1e-9)
            np.testing.assert_allclose(_surface_normals(flipped), -_surface_normals(ball), rtol=0, atol=1e-6)

    def soft_sphere(self, **placed):
        """A 32 x 32 sphere, every edge soft and no normal locked, at `placed` (xform's keywords): its Mesh."""
        t = cmds.polySphere(name="ball", subdivisionsAxis=32, subdivisionsHeight=32, constructionHistory=False)[0]
        cmds.polySoftEdge(t, angle=180, constructionHistory=False)
        cmds.xform(t, **placed)
        return Mesh(t)

    def test_a_non_uniform_scale_locks_soft_normals(self):
        """Pinned as documented, not fixed: Maya computes other normals for the frozen points than
        the inverse transpose gives the data's, so those are set, which locks them, where the
        source and a plain create lock none. They land, and normals=False locks none."""
        source = self.soft_sphere(scale=(1, 2, 3))
        args   = _args(source)
        frozen = Mesh.create(*args, world_space=True)
        self.assertEqual(_shading(source)[1], 0)
        self.assertEqual(_shading(Mesh.create(*args))[1], 0)
        self.assertGreater(_shading(frozen)[1], 0.9 * frozen.get_fn_set().numNormals)
        np.testing.assert_allclose(_normals(frozen), _normals(source), rtol=0, atol=1e-5)
        self.assertEqual(_shading(Mesh.create(*args, world_space=True, normals=False))[1], 0)

    def test_a_rigid_matrix_near_the_origin_locks_nothing(self):
        """A move, a rotation and a uniform scale: Maya computes the data's normals, there."""
        source = self.soft_sphere(translation=(1, 2, 3), rotation=(30, 40, 50), scale=(2, 2, 2))
        self.assertEqual(_shading(Mesh.create(*_args(source), world_space=True))[1], 0)

    def test_a_hard_edged_cube_locks_nothing(self):
        """A non-uniform scale and a shear: a flat face's normal is what Maya computes."""
        box = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        cmds.xform("box", translation=(1, 2, 3), rotation=(10, 20, 30), scale=(1, 2, 3), shear=(0.5, 0.2, -0.3))
        self.assertEqual(_shading(Mesh.create(*_args(box), world_space=True)), (12, 0))

    def test_the_same_goes_under_a_target_and_in_an_update(self):
        """guard: what locks is the matrix the normals go through, whichever call takes them there."""
        source = self.soft_sphere()
        args   = _args(source, True)
        target = cmds.createNode("transform", name="ctl")
        cmds.xform(target, scale=(1, 2, 3))
        under = Mesh.create(*args, transform=target, world_space=True)
        self.assertGreater(_shading(under)[1], 0.9 * under.get_fn_set().numNormals)
        cmds.xform("ball", scale=(1, 2, 3))
        source.load(*args, world_space=True)
        self.assertGreater(_shading(source)[1], 0.9 * source.get_fn_set().numNormals)

    def test_the_readme_says_what_locks(self):
        with open(README, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        self.assertIn("**`world_space=True` can lock a mesh's normals**", text)

    def test_a_matrix_with_no_inverse(self):
        """Exactly singular, which only data made by hand holds: a mesh's normals cannot follow it,
        and without them it flattens, a transform-less scale of 0."""
        data        = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0]).serialize(include_uvs=False)
        data.matrix = np.diag([0.0, 1.0, 1.0, 1.0])
        self.assert_nothing_made(ValueError, r"^MeshData: the matrix has no inverse, which the normals go through",
                                 lambda: Mesh.create(data, world_space=True))
        flat = Mesh.create(data, world_space=True, normals=False)
        self.assert_at_rest(flat)
        np.testing.assert_array_equal(_local(flat)[:, 0], np.zeros(8))

    def test_a_matrix_that_is_not_finite(self):
        data        = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0]).serialize(include_uvs=False)
        data.matrix = np.full((4, 4), np.nan)
        refusal     = r"^Mesh\.create\(\): the data's matrix must be finite \(got nan or inf\)$"
        for label, keywords in (("frozen", {}), ("under a target", {"transform": _sheared()})):
            with self.subTest(label):
                self.assert_nothing_made(ValueError, refusal, lambda: Mesh.create(data, world_space=True, **keywords))

    def test_a_matrix_that_is_not_finite_on_nurbs_data(self):
        if not NEW_FIELDS:
            self.skipTest(NO_FIELDS)
        for cls, source in ((NurbsCurve, _circle()), (NurbsSurface, _ball())):
            with self.subTest(cls=cls.__name__):
                data        = source.serialize()
                data.matrix = np.full((4, 4), np.inf)
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): the data's matrix must be finite",
                                         lambda: cls.create(data, world_space=True))


class TestQ7AnyNodeDerivedFromTransform(_Case):
    """A joint is not the only other one. The first test holds the README's line; the others are
    guards: the targets were taken at 4c53f99."""

    PLUGIN = '''
import maya.OpenMaya as om1
import maya.OpenMayaMPx as mpx

NAME, NODE, MATRIX = "rigProbeTransform", om1.MTypeId(0x7F0A1), om1.MTypeId(0x7F0A2)


class Matrix(mpx.MPxTransformationMatrix):
    pass


class Node(mpx.MPxTransform):
    def __init__(self):
        mpx.MPxTransform.__init__(self)


def initializePlugin(plugin):
    mpx.MFnPlugin(plugin).registerTransform(
        NAME, NODE, lambda: mpx.asMPxPtr(Node()), lambda: None, lambda: mpx.asMPxPtr(Matrix()), MATRIX
    )


def uninitializePlugin(plugin):
    mpx.MFnPlugin(plugin).deregisterNode(NODE)
'''

    def check(self, target):
        source = _cube()
        args   = _args(source)
        own    = Mesh.create(*args, transform=target)
        placed = Mesh.create(*args, transform=target, world_space=True)
        self.assertEqual(own.get_parent().get_long_name(), cmds.ls(target, long=True)[0])
        np.testing.assert_array_equal(_local(own), _local(source))
        np.testing.assert_allclose(_world(placed), _world(source), rtol=0, atol=1e-4)

    def test_the_readme_says_so(self):
        with open(README, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        self.assertIn(
            "`transform=` takes any node derived from transform: a transform, a joint, an `ikHandle`, a "
            "constraint, a `dagContainer`, a plug-in's `MPxTransform` control.", text,
        )

    def test_an_ik_handle(self):
        """guard."""
        start = cmds.createNode("joint", name="j1")
        end   = cmds.createNode("joint", name="j2", parent=start)
        cmds.setAttr(f"{end}.ty", 2)
        self.check(cmds.ikHandle(startJoint=start, endEffector=end, name="ik")[0])

    def test_a_constraint(self):
        """guard."""
        driver = cmds.spaceLocator(name="driver")[0]
        cmds.setAttr(f"{driver}.t", 1, 2, 3)
        self.check(cmds.parentConstraint(driver, cmds.createNode("transform", name="driven"))[0])

    def test_a_dag_container(self):
        """guard."""
        self.check(cmds.container(type="dagContainer", name="dag"))

    def test_a_plug_in_s_transform(self):
        """guard: an MPxTransform of a plug-in of this test's own."""
        path = os.path.join(self.folder(), "rig_probe_transform.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.PLUGIN)
        cmds.loadPlugin(path)

        def unload():
            cmds.file(new=True, force=True)
            cmds.flushUndo()
            cmds.unloadPlugin("rig_probe_transform.py")

        self.addCleanup(unload)
        node = cmds.createNode("rigProbeTransform", name="control")
        self.assertIn("transform", cmds.nodeType(node, inherited=True))
        cmds.xform(node, translation=(1, 2, 3), rotation=(0, 45, 0))
        self.check(node)

    def test_a_shape_stays_refused(self):
        """guard: derived from transform, not a node with a transform."""
        box = cmds.polyCube(name="box", constructionHistory=False)[0]
        with self.assertRaisesRegex(TypeError, "transform= is the transform"):
            Mesh.create(*_args(Mesh(box)), transform="boxShape")


class TestQ8ASingularTarget(_Case):
    REFUSAL = (
        r"^{method}: '{target}' is singular \(its world scale is \({scale}\){flat}\), so world_space=True has no "
        r"object points that land where the data puts them; world_space=False \(the default\) takes the data's "
        r"points as the shape's own$"
    )

    def flat(self, attr="scaleY", value=0.0, name="flat"):
        node = cmds.createNode("transform", name=name)
        cmds.setAttr(f"{node}.{attr}", value)
        return node

    def refusal(self, method, target="flat", scale="1, 1e-12, 1", flat=""):
        return self.REFUSAL.format(method=re.escape(method), target=target, scale=scale, flat=flat)

    def test_a_zero_scale_is_refused_before_anything_is_built(self):
        args = {cls: _args(_source(cls)) for cls in CLASSES}
        for attr, scale in (("scaleX", "1e-12, 1, 1"), ("scaleY", "1, 1e-12, 1"), ("scaleZ", "1, 1, 1e-12")):
            target = self.flat(attr)
            for cls in CLASSES:
                with self.subTest(attr=attr, cls=cls.__name__):
                    self.assert_nothing_made(ValueError, self.refusal(f"{cls.__name__}.create()", scale=scale),
                                             lambda: cls.create(*args[cls], transform=target, world_space=True))
            cmds.delete(target)

    def test_its_scale_is_named_x_y_z(self):
        """The axes' lengths as the channels hold them, not sorted."""
        target = cmds.createNode("transform", name="flat")
        cmds.setAttr(f"{target}.scale", 2, 0, 3)
        args = _args(_cube())
        self.assert_nothing_made(ValueError, self.refusal("Mesh.create()", scale="2, 1e-12, 3"),
                                 lambda: Mesh.create(*args, transform=target, world_space=True))

    def test_a_zero_scale_on_a_parent(self):
        flat = self.flat()
        kid  = cmds.createNode("transform", name="kid", parent=flat)
        args = _args(_cube())
        self.assert_nothing_made(ValueError, self.refusal("Mesh.create()", "kid"),
                                 lambda: Mesh.create(*args, transform=kid, world_space=True))

    def test_a_rotated_child_of_a_flat_parent(self):
        """No axis of its world 3x3 is short, and it is as flat: at 4c53f99 this cube's points landed
        1.5e5 off, through object points 4.4e12 out (a probe). The message adds the flat direction."""
        flat = self.flat()
        kid  = cmds.createNode("transform", name="kid", parent=flat)
        cmds.xform(kid, rotation=(30, 45, 60))
        axes = np.linalg.norm(np.array(cmds.getAttr(f"{kid}.worldMatrix[0]")).reshape(4, 4)[:3, :3], axis=1)
        self.assertGreater(axes.min(), 0.5)
        args  = _args(_cube())
        scale = re.escape(", ".join(f"{axis:.3g}" for axis in axes))
        flat  = ", but it scales one direction by 1e-12"
        self.assert_nothing_made(ValueError, self.refusal("Mesh.create()", "kid", scale, flat),
                                 lambda: Mesh.create(*args, transform=kid, world_space=True))

    def test_just_under_the_line(self):
        target = self.flat(value=5e-10)
        args   = _args(_circle())
        self.assert_nothing_made(ValueError, self.refusal("NurbsCurve.create()", scale="1, 5e-10, 1"),
                                 lambda: NurbsCurve.create(*args, transform=target, world_space=True))

    def test_the_class_form_load_refuses_it_before_the_file_is_read(self):
        target = self.flat()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                self.assert_nothing_made(ValueError, self.refusal(f"{cls.__name__}.load"),
                                         lambda: cls.load("no_such_file.npz", transform=target, world_space=True))

    def test_false_stays_allowed(self):
        """guard: the data's points are the shape's own, whatever the target's scale."""
        target = self.flat()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args  = _args(_source(cls))
                shape = cls.create(*args, transform=target)
                np.testing.assert_array_equal(_local(shape), _data_points(args[0]))

    def test_a_small_scale_stays_allowed(self):
        """guard: a scale of 1e-6 has an inverse, and 2e-9 is over the line: the points land."""
        for name, value in (("tiny", 1e-6), ("least", 2e-9)):
            target = cmds.createNode("transform", name=name)
            cmds.setAttr(f"{target}.scale", value, value, value)
            for cls in CLASSES:
                with self.subTest(target=name, cls=cls.__name__):
                    source = _source(cls)
                    shape  = cls.create(*_args(source, True), transform=target, world_space=True)
                    self.assertGreater(np.abs(_local(shape)).max(), 1e5)
                    atol   = 2e-4 if cls is Mesh else 1e-6
                    np.testing.assert_allclose(_world(shape), _world(source), rtol=0, atol=atol)


class TestQ9TheTargetsNamespace(_Case):
    def setUp(self):
        super().setUp()
        cmds.namespace(add="char")
        cmds.namespace(add="other")
        self.target = cmds.createNode("transform", name="char:ctl")

    def test_the_default_name_is_in_the_target_s_namespace(self):
        made = [cls.create(*_args(_source(cls)), transform=self.target) for cls in CLASSES]
        self.assertEqual(
            [shape.get_long_name() for shape in made],
            ["|char:ctl|char:ctlShape", "|char:ctl|char:ctlShape1", "|char:ctl|char:ctlShape2"],
        )

    def test_whatever_the_current_namespace(self):
        args = _args(_cube())
        for current, name in ((":other", "char:ctlShape"), (":char", "char:ctlShape1")):
            with self.subTest(current=current):
                cmds.namespace(set=current)
                self.assertEqual(Mesh.create(*args, transform=self.target).get_long_name(), f"|char:ctl|{name}")
                cmds.namespace(set=":")

    def test_a_target_at_the_root_from_another_namespace(self):
        args   = _args(_cube())
        target = cmds.createNode("transform", name="ctl")
        cmds.namespace(set=":other")
        self.assertEqual(Mesh.create(*args, transform=target).get_long_name(), "|ctl|ctlShape")

    def test_a_nested_namespace(self):
        cmds.namespace(add="a:b")
        target = cmds.createNode("transform", name="a:b:ctl")
        self.assertEqual(Mesh.create(*_args(_cube()), transform=target).get_long_name(), "|a:b:ctl|a:b:ctlShape")

    def test_undo_and_redo_keep_it(self):
        args = _args(_cube())
        cmds.flushUndo()
        shape = Mesh.create(*args, transform=self.target)
        uuid  = cmds.ls(shape.get_long_name(), uuid=True)[0]
        self.undo_steps(1)
        self.assertIsNone(cmds.listRelatives(self.target, children=True))
        self.redo_steps(1)
        self.assertEqual(cmds.ls(uuid, long=True), ["|char:ctl|char:ctlShape"])

    def test_in_a_container_the_default_name_takes_no_prefix(self):
        args = _args(_cube())
        with container("leg"):
            with container("lower"):
                shape = Mesh.create(*args, transform=self.target)
        self.assertEqual(shape.get_long_name(), "|char:ctl|char:ctlShape")
        self.assertEqual(cmds.container("leg", query=True, nodeList=True), ["char:ctlShape"])

    def test_name_is_used_as_given(self):
        """guard: in the current namespace, as Maya renames, or in the one the name spells."""
        args = _args(_cube())
        self.assertEqual(Mesh.create(*args, transform=self.target, name="petal").get_long_name(), "|char:ctl|petal")
        self.assertEqual(
            Mesh.create(*args, transform=self.target, name="char:petal").get_long_name(), "|char:ctl|char:petal"
        )
        cmds.namespace(set=":other")
        self.assertEqual(
            Mesh.create(*args, transform=self.target, name="petal").get_long_name(), "|char:ctl|other:petal"
        )

    def referenced(self):
        """A new scene that references a moved transform as ``ref:refctl``, and one with a curve of
        its own as ``ref:hoop``: the referenced file's folder."""
        folder = self.folder()
        path   = os.path.join(folder, "ref_target.ma")
        cmds.file(new=True, force=True)
        cmds.xform(cmds.createNode("transform", name="refctl"), translation=(2, 0, -1), rotation=(0, 30, 0))
        cmds.circle(name="hoop", constructionHistory=False)
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="ref")
        return folder

    def test_a_referenced_target_s_shape_is_a_local_node_in_its_namespace(self):
        self.referenced()
        source = _cube()
        args   = _args(source)
        shape  = Mesh.create(*args, transform="ref:refctl", world_space=True)
        self.assertEqual(shape.get_long_name(), "|ref:refctl|ref:refctlShape")
        self.assertFalse(cmds.referenceQuery(shape.get_long_name(), isNodeReferenced=True))
        np.testing.assert_allclose(_world(shape), _world(source), rtol=0, atol=1e-5)
        # the file's own shape takes the name: Maya numbers the new one
        curve = NurbsCurve.create([(0, 0, 0), (1, 1, 0)], 1, transform="ref:hoop")
        self.assertEqual(curve.get_long_name(), "|ref:hoop|ref:hoopShape1")

    def test_the_referenced_file_comes_to_hold_the_name(self):
        """Maya numbers the local shape at the next open, as any clash, and nothing is lost."""
        folder = self.referenced()
        shape  = Mesh.create(*_args(_cube()), transform="ref:refctl")
        uuid   = cmds.ls(shape.get_long_name(), uuid=True)[0]
        scene  = os.path.join(folder, "scene.ma")
        cmds.file(rename=scene)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(os.path.join(folder, "ref_target.ma"), open=True, force=True)
        cmds.createNode("locator", name="refctlShape", parent="refctl")
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(scene, open=True, force=True)
        self.assertEqual(cmds.ls(uuid, long=True), ["|ref:refctl|ref:refctlShape1"])
        self.assertEqual(cmds.polyEvaluate("|ref:refctl|ref:refctlShape1", face=True), 24)

    def test_it_survives_a_save_and_a_reopen(self):
        for kind, ext in (("mayaAscii", "ma"), ("mayaBinary", "mb")):
            with self.subTest(kind=kind):
                folder = self.referenced()
                source = _cube()
                shape  = Mesh.create(*_args(source), transform="ref:refctl", world_space=True)
                path   = shape.get_long_name()
                uuid   = cmds.ls(path, uuid=True)[0]
                world  = _world(shape)
                cmds.file(rename=os.path.join(folder, f"scene.{ext}"))
                cmds.file(save=True, type=kind, force=True)
                cmds.file(new=True, force=True)
                cmds.file(os.path.join(folder, f"scene.{ext}"), open=True, force=True)
                self.assertEqual(cmds.ls(uuid, long=True), [path])
                self.assertFalse(cmds.referenceQuery(path, isNodeReferenced=True))
                np.testing.assert_allclose(_world(Mesh(path)), world, rtol=0, atol=1e-5)
                # and through an unload and a reload of the reference
                node = cmds.referenceQuery("ref:refctl", referenceNode=True)
                cmds.file(unloadReference=node)
                self.assertEqual(cmds.ls(uuid, long=True), ["|refRNfosterParent1|ref:refctlShape"])
                cmds.file(loadReference=node)
                self.assertEqual(cmds.ls(uuid, long=True), [path])
                np.testing.assert_allclose(_world(Mesh(path)), world, rtol=0, atol=1e-5)
                # and an open that leaves the reference unloaded
                cmds.file(save=True, type=kind, force=True)
                cmds.file(new=True, force=True)
                cmds.file(os.path.join(folder, f"scene.{ext}"), open=True, force=True, loadReferenceDepth="none")
                self.assertEqual(cmds.ls(uuid, long=True), ["|refRNfosterParent1|ref:refctlShape"])
                cmds.file(loadReference=node)
                self.assertEqual(cmds.ls(uuid, long=True), [path])

