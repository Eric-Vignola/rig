"""Review fix (features run): save / load, and the grammar around the new methods.

* ``BlendShape('bs').load(path)`` left a live target mesh connected: its offsets replaced
  the file's at the next evaluation. A target the file replaces now gives up its live mesh.
* ``Mesh('body').load(path)`` on a skinned mesh said "delete the history", which deletes the
  skin. A deformed mesh's refusal now names the shape under its deformers
  (``Mesh('bodyShapeOrig').load(path)``), whose load the deformed shape follows; a
  deformed curve the same; construction history still names the delete.
* ``SkinCluster.load('body', skin_data)`` (the data in the geometry's place) was a
  FileNotFoundError for 'body'; it is the order TypeError, as with a node first.
* ``Mesh.load(md, world_space=False)`` named ``Mesh.create()`` and its flags; the class form
  now names the instance form for ``world_space=`` (Mesh, NurbsCurve, NurbsSurface).
* ``Joint.load_hierarchy.__qualname__`` read ``Transform.load_hierarchy``.
* ``SkinCluster('x').load(path)`` of the weights it holds made an undo step that did nothing.
* A new attribute named like a Python member of the node's class (``lock``, ``save``,
  ``parent`` ...) is made with a warning naming ``Plug(node.find_attr(name))``: the member
  hides it (``ctl.lock`` is the method).
* ``List(shapes).save(path)`` said "'List' object is not callable"; a method of each element
  now refuses with the loop to write.

Each test failed at 8286cb8 (negative control), except the guards.
"""

import logging

import numpy as np
from maya import cmds

from rig import Bool, Float, List, Node, Plug
from rig.nodetypes import BlendShape, Joint, Mesh, NurbsCurve, NurbsSurface, SkinCluster, Transform
from rig._tests.test_class_saveload import _Case


class TestABlendShapesLiveTarget(_Case):
    def test_the_load_disconnects_the_live_mesh(self):
        base = cmds.polyCube(name="base", ch=False)[0]
        smile = cmds.polyCube(name="smile", ch=False)[0]
        cmds.move(0, 1, 0, "smile.vtx[0]", relative=True)
        bs = BlendShape.create(Node(smile), Node(base), name="bs")
        cmds.move(0, 2, 0, "smile.vtx[0]", relative=True)
        bs.save(self.path("bs.npz"))
        cmds.move(0, -2, 0, "smile.vtx[0]", relative=True)
        self.assertIn("smile", cmds.listConnections("bs", source=True, destination=False, type="mesh"))
        self.ready()
        BlendShape("bs").load(self.path("bs.npz"))
        self.assertNotIn("smile", cmds.listConnections("bs", source=True, destination=False, type="mesh") or [])
        cmds.setAttr("bs.smile", 1)
        y = cmds.xform("base.vtx[0]", query=True, objectSpace=True, translation=True)[1]
        cmds.move(0, 5, 0, "smile.vtx[1]", relative=True)  # the live mesh moves on
        self.assertAlmostEqual(cmds.xform("base.vtx[0]", query=True, objectSpace=True, translation=True)[1], y)
        self.assertAlmostEqual(y, 2.5)

    def test_one_undo_step(self):
        """guard: the load and its disconnect are one undo step, which brings the live
        mesh back."""
        base = cmds.polyCube(name="base", ch=False)[0]
        smile = cmds.polyCube(name="smile", ch=False)[0]
        cmds.move(0, 1, 0, "smile.vtx[0]", relative=True)
        bs = BlendShape.create(Node(smile), Node(base), name="bs")
        bs.save(self.path("bs.npz"))
        self.ready()
        before = self.scene_state()
        BlendShape("bs").load(self.path("bs.npz"))
        self.assertEqual(self.undo_name(), "rig.BlendShape.load")
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(), before, "undo")
        self.assertIn("smile", cmds.listConnections("bs", source=True, destination=False, type="mesh"))


class TestADeformedShape(_Case):
    def test_a_skinned_mesh_names_its_orig_shape(self):
        geo = Node(cmds.polyCube(name="body", ch=False)[0])
        Mesh("bodyShape").save(self.path("body.npz"))
        j1 = Joint.create(name="j1")
        SkinCluster.create(geo, [j1])
        self.ready()
        self.assert_refused(
            ValueError, r"^Mesh\('bodyShape'\).load: 'bodyShape' is deformed \(",
            lambda: Mesh("bodyShape").load(self.path("body.npz")),
        )
        try:
            Mesh("bodyShape").load(self.path("body.npz"))
        except ValueError as e:
            text = str(e)
        self.assertIn("Mesh('bodyShapeOrig').load(path) updates the shape under them", text)
        self.assertIn("Deleting its history would delete the deformers", text)
        self.assertNotIn("delete the history", text)
        # the route it names works, and the deformed shape follows
        data = Mesh("bodyShape").serialize(include_uvs=False, world_space=False)
        data.points = np.asarray(data.points) * 2.0
        Mesh("bodyShapeOrig").load(data, world_space=False)
        box = cmds.exactWorldBoundingBox("body")
        self.assertAlmostEqual(box[3] - box[0], 2.0, places=5)

    def test_construction_history_names_the_delete(self):
        """guard: a polyCube with its history keeps the delete."""
        t = cmds.polyCube(name="live", constructionHistory=True)[0]
        Mesh(cmds.listRelatives(t, shapes=True)[0]).serialize(include_uvs=False).save(self.path("m.npz"))
        self.assert_refused(
            ValueError, r"has history \(its inMesh is connected: construction history\).*delete the history",
            lambda: Mesh("liveShape").load(self.path("m.npz")),
        )

    def test_a_deformed_curve(self):
        crv = cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        cmds.rename(cmds.listRelatives(crv, shapes=True)[0], "crvShape")
        NurbsCurve("crvShape").save(self.path("crv.npz"))
        cmds.cluster("crv.cv[0]", name="clu")
        self.ready()
        self.assert_refused(
            ValueError, r"^NurbsCurve\('crvShape'\).load: 'crvShape' is deformed \(cluHandleCluster\).*"
            r"NurbsCurve\('crvShapeOrig'\).load\(path, world_space=\.\.\.\) updates the shape under them",
            lambda: NurbsCurve("crvShape").load(self.path("crv.npz"), world_space=True),
        )


class TestTheArguments(_Case):
    def skinned(self):
        geo = Node(cmds.polyCube(name="body", ch=False)[0])
        j1 = Joint.create(name="j1")
        skin = SkinCluster.create(geo, [j1])
        return geo, skin

    def test_a_data_object_in_the_geometrys_place(self):
        _geo, skin = self.skinned()
        data = skin.serialize()
        self.assert_refused(
            TypeError, r"^SkinCluster.load\(skin, geometry\) takes the file first, then the geometry: "
            r"got 'body' first and a SkinData second",
            lambda: SkinCluster.load("body", data),
        )

    def test_a_file_that_does_not_exist_stays_a_file_error(self):
        """guard: a missing file first, a node second, is still FileNotFoundError."""
        geo, _skin = self.skinned()
        self.assert_refused(FileNotFoundError, "no file", lambda: SkinCluster.load(self.path("none.npz"), geo))

    def test_the_class_form_names_the_instance_form(self):
        t = cmds.polyCube(name="body", ch=False)[0]
        md = Mesh("bodyShape").serialize(include_uvs=False)
        crv = cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0)])
        cmds.rename(cmds.listRelatives(crv, shapes=True)[0], "crvShape")
        cd = NurbsCurve("crvShape").serialize()
        for cls, data, what in ((Mesh, md, "mesh"), (NurbsCurve, cd, "curve")):
            with self.subTest(cls=cls.__name__):
                self.assert_refused(
                    TypeError,
                    rf"^{cls.__name__}.load\(...\) creates a new {what}, and takes no world_space=; "
                    rf"{cls.__name__}\('x'\).load\(path, world_space=...\) updates an existing {what} in place$",
                    lambda: cls.load(data, world_space=False),
                )
        self.assert_refused(TypeError, r"^NurbsSurface.load\(...\) creates a new surface",
                            lambda: NurbsSurface.load(cd, world_space=True))

    def test_the_qualname_is_the_class_called(self):
        self.assertEqual(Joint.load_hierarchy.__qualname__, "Joint.load_hierarchy")
        self.assertEqual(Transform.load_hierarchy.__qualname__, "Transform.load_hierarchy")
        self.assertEqual(Mesh.load.__qualname__, "Mesh.load")
        self.assertIs(Joint.load_hierarchy.__func__, Joint.load_hierarchy.__func__)


class TestASkinLoadOfWhatItHolds(_Case):
    def test_no_undo_step(self):
        geo = Node(cmds.polyCube(name="body", ch=False, sy=4)[0])
        j1 = Joint.create(name="j1")
        j2 = Joint.create(name="j2", parent=j1)
        skin = SkinCluster.create(geo, [j1, j2])
        w = skin.get_weights()
        w[:, 0] = np.linspace(0, 1, len(w))
        w[:, 1] = 1 - w[:, 0]
        skin.set_weights(w)
        skin.save(self.path("skin.npz"))
        self.ready()
        SkinCluster(skin.get_name()).load(self.path("skin.npz"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        # weights that differ are written, in one step
        w[:, 0] = 1 - w[:, 0]
        w[:, 1] = 1 - w[:, 1]
        skin.set_weights(w)
        cmds.flushUndo()
        SkinCluster(skin.get_name()).load(self.path("skin.npz"))
        self.assertEqual(self.undo_name(), "rig.SkinCluster.load")


class _Sided(Transform):
    """A node class with class data: a Python member that hides a Maya attr of
    its name (the property ban left class data as the one such member)."""

    side = None


class TestAShadowedAttribute(_Case):
    def test_a_warning_names_find_attr(self):
        # re-pinned twice (user decisions 2026-10-16): a method name no longer
        # shadows (name clashes, see test_a_method_name_is_no_shadow), nor does
        # is_locked, a method since the property ban; class data still does
        plain = Node.create("transform", name="plain")
        logger = logging.getLogger("rig.spec._base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level=logging.WARNING):
                plain << Bool("is_locked")
        self.assertEqual(str(plain.is_locked), "plain.is_locked")  # the plug
        self.assertIs(plain.is_locked(), False)  # the method
        ctl = _Sided._wrap(Node.create("transform", name="ctl"))
        with self.assertLogs("rig.spec._base", level=logging.WARNING) as logs:
            plug = ctl << Bool("side")
        self.assertEqual(str(plug), "ctl.side")
        text = "\n".join(logs.output)
        self.assertIn("'side' is a Python member of _Sided", text)
        self.assertIn("Plug(ctl.find_attr('side'))", text)
        Plug(ctl.find_attr("side")) << 1
        self.assertTrue(cmds.getAttr("ctl.side"))

    def test_the_short_name_too(self):
        # re-pinned (user decision 2026-10-16, the property ban): class data
        ctl = _Sided._wrap(Node.create("transform", name="ctl"))
        with self.assertLogs("rig.spec._base", level=logging.WARNING) as logs:
            ctl << Float("sideWeight", sn="side")
        self.assertIn("'side' is a Python member of _Sided", "\n".join(logs.output))

    def test_no_warning_for_a_free_name_or_a_redeclaration(self):
        """guard. Re-pinned (user decision 2026-10-16, the property ban): class
        data is the shadowing member."""
        ctl = _Sided._wrap(Node.create("transform", name="ctl"))
        logger = logging.getLogger("rig.spec._base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level=logging.WARNING):
                ctl << Float("weight")
        with self.assertLogs(logger, level=logging.WARNING):
            ctl << Bool("side")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level=logging.WARNING):
                ctl << Bool("side")  # re-declared: kept, no second warning

    def test_a_method_name_is_no_shadow(self):
        """user decision 2026-10-16, name clashes: node.lock is the plug once the node
        has the attr, node.lock() the method."""
        ctl = Node.create("transform", name="ctl")
        logger = logging.getLogger("rig.spec._base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level=logging.WARNING):
                ctl << Bool("lock")
        self.assertIsInstance(ctl.lock, Plug)
        ctl.lock << 1
        self.assertTrue(ctl.lock >> None)
        self.assertIs(ctl.lock(), ctl)
        self.assertTrue(ctl.is_locked())
        ctl.unlock()


class TestAListOfMethods(_Case):
    def test_a_call_names_the_loop(self):
        a = cmds.polyCube(name="a", ch=False)[0]
        b = cmds.polyCube(name="b", ch=False)[0]
        shapes = List([Mesh("aShape"), Mesh("bShape")])
        with self.assertRaises(TypeError) as caught:
            shapes.save(self.path("x.npz"))
        self.assertIn("a method does not broadcast over a List", str(caught.exception))
        self.assertIn("[x.save(...) for x in items]", str(caught.exception))
        # the methods are still there, one per element
        methods = shapes.save
        self.assertEqual(len(methods), 2)
        self.assertEqual(methods[0](self.path("a.npz")), self.path("a.npz"))

    def test_plugs_still_broadcast(self):
        """guard."""
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        List([a, b]).tx << 3
        self.assertEqual(cmds.getAttr("b.tx"), 3.0)


class TestTheShadingCost(_Case):
    """Mesh.create's shading restore on hard-edged and locked-normal meshes (the review's
    create_ab: at 8286cb8 every edge hard cost 4.2x the create without normals, 30% hard
    2.1x, 2000 vertices of locked normals 2.3x). The edge ids now come from the face lists
    in numpy, each checked by its vertices, and the smoothing is set in one call: about
    2.6x, 1.75x and 2.1x. The guards leave room for a busy machine; the 1.5x budget of a
    mostly smooth mesh is test_class_mesh_create's."""

    def best(self, source, uvs, runs=7):
        import time

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            mesh = Mesh.create(source, uv_data=uvs)
            times.append(time.perf_counter() - start)
            cmds.delete(mesh.get_parent().get_long_name())
        return min(times)

    def ratio(self, prepare):
        t = cmds.polySphere(name="src", sx=200, sy=100, constructionHistory=False)[0]
        prepare(t)
        cmds.delete(t, constructionHistory=True)
        data, uvs = Mesh(t).serialize(world_space=False)
        cmds.delete(t)
        plain = data.copy()
        plain.normals = None
        plain.normal_indices = None
        self.best(data, uvs, 2)  # warm up
        return self.best(data, uvs) / self.best(plain, uvs)

    def test_every_edge_hard(self):
        ratio = self.ratio(lambda t: cmds.polySoftEdge(t, angle=0, ch=False))
        self.assertLess(ratio, 3.3, f"{ratio:.2f}x")

    def test_a_third_of_the_edges_hard(self):
        def prepare(t):
            cmds.polySoftEdge(t, angle=180, ch=False)
            n = cmds.polyEvaluate(t, edge=True)
            ids = sorted(np.random.default_rng(1).choice(n, int(n * 0.3), replace=False).tolist())
            cmds.polySoftEdge([f"{t}.e[{i}]" for i in ids], angle=0, ch=False)
        ratio = self.ratio(prepare)
        self.assertLess(ratio, 2.0, f"{ratio:.2f}x")

    def test_the_hard_edges_come_back(self):
        """guard: every edge the data makes hard (``_hard_edges``: its normals split at an
        end) is hard on the new mesh, and no other one, on the dense route (the ids guessed
        from the face lists, each checked)."""
        t = cmds.polySphere(name="src", sx=40, sy=20, constructionHistory=False)[0]
        n = cmds.polyEvaluate(t, edge=True)
        ids = sorted(np.random.default_rng(2).choice(n, n // 2, replace=False).tolist())
        cmds.polySoftEdge(t, angle=180, ch=False)
        cmds.polySoftEdge([f"{t}.e[{i}]" for i in ids], angle=0, ch=False)
        cmds.delete(t, constructionHistory=True)
        from rig.nodetypes import mesh as mesh_module

        data, uvs = Mesh(t).serialize(world_space=False)
        shading = mesh_module._shading_of(data, {})
        self.assertIsNotNone(shading.hard[3])  # the dense route: guessed ids
        want = {frozenset(pair) for pair in shading.hard[0].tolist()}
        self.assertGreater(len(want), 100)
        made = Mesh.create(data, uv_data=uvs)
        fn = made.get_fn_set()
        got = {frozenset(fn.getEdgeVertices(e)) for e in range(fn.numEdges) if not fn.isEdgeSmooth(e)}
        self.assertEqual(got, want)
