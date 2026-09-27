"""Invariants the performance work must keep: ``PyNode`` dispatch through the
per-type class cache picks the same class, raises the same errors and follows
new class registrations like the name-based path does. A cast that skips the
type check builds the same wrapper as the constructor, and the node a plug
caches when it is first named behaves as before. ``Attribute.set`` passes the
same ``type`` argument, and raises the same errors, when it skips the type
query. ``Attribute.data_type`` shares a fixed-kind attr's or a typed array
root's type across nodes of a type, and keeps querying every type that can
change. ``Attribute.__init__``
leaves the same instance state without going through ``Plug.__setattr__``. A
plug reuses the wrapper of the Node or parent plug it came from only where a
fresh cast would rebuild that wrapper unchanged, so it reports the same owner
and raises the same errors. ``Attribute.connect`` skips the ``isConnected``
query only for a destination it must find unconnected, with the same result,
errors and scene as the query path. DGNode's data type fallback hook reuses the
type ``Attribute.data_type`` just queried only when no code that could change it
ran in between, and queries again otherwise. The canonical wrapper check keeps
its class checks per class only until a node class or class attribute changes,
and takes a DAG wrapper's name from its own path only when that is the node's
only path. ``Attribute.full_name`` reads a plug owner's name property once
(the owner is the node itself since the round-4a class swap), with the same
names and errors."""

import contextlib
import os
import shutil
import sys
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Container, InjectionError, Node, Plug, lock
from rig.nodetypes import Choice, DAGNode, DGNode, Joint, PyNode, Transform
from rig.nodetypes import _base
from rig.nodetypes import dg_node as dg_node_module
from rig.nodetypes._base import (
    CUSTOM_TYPE_ATTR,
    _mobject_to_str,
    _pynode_legacy_tail,
    get_custom_type,
    is_valid_maya_uid,
    set_custom_type,
)
from rig._internal.plug import ComponentPlug
from rig.spec import Float, Matrix, Vector
from rig._tests._base import MayaTestCase


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _outcome(func, *args):
    """The class and name a cast returns, or the type and message it raises."""
    try:
        node = func(*args)
    except Exception as exc:
        return ("error", type(exc), str(exc))
    return ("ok", type(node), node.name)


def _holds_freed_node(plug):
    """True if the node object ``plug`` holds is a node that was freed (by a new
    scene or a reference unload), not just deleted to the undo queue. Only the
    node's API 1.0 handle is read: naming a freed node reads freed memory."""
    # re-pinned (round 4a M4, C8): the owner is the node object itself (no
    # wrapper to unwrap), so its handle is read from its __dict__ directly
    owner  = plug.__dict__["_node"]
    handle = getattr(owner, "__dict__", {}).get("_objhandle1")
    return handle is not None and not handle.isAlive()


class TestPyNodeDispatch(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        # drop any class a test registered, then the dispatch it cached
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def _sweep_scene(self):
        """About 30 node types plus namespaces, duplicate names and an instance."""
        names = []
        for node_type in (
            "multiplyDivide",
            "plusMinusAverage",
            "condition",
            "reverse",
            "blendColors",
            "clamp",
            "choice",
            "network",
            "objectSet",
            "displayLayer",
            "transform",
            "joint",
            "locator",
            "mesh",
            "nurbsCurve",
            "nurbsSurface",
            "follicle",
            "decomposeMatrix",
            "composeMatrix",
            "multMatrix",
            "remapValue",
            "setRange",
            "distanceBetween",
            "animCurveTL",
        ):
            names.append(cmds.createNode(node_type, name=f"n_{node_type}"))
        cube = cmds.polyCube(name="cube")[0]
        cmds.select(clear=True)
        jnt = cmds.joint(name="jnt")
        cmds.select(clear=True)
        names += [cube, cmds.listRelatives(cube, shapes=True)[0], jnt]
        names += cmds.lattice(cube, name="ffd")
        names += cmds.skinCluster(jnt, cube, name="skin")
        names += cmds.blendShape(cube, name="bs")
        names.append(cmds.polyCube(name="cube", constructionHistory=True)[1])
        cmds.namespace(add="ns")
        names.append(cmds.createNode("transform", name="ns:inNs"))
        names.append(cmds.createNode("transform", name="dup"))
        names.append(cmds.createNode("transform", name="dup", parent=cube))
        top = cmds.createNode("transform", name="T1")
        cmds.createNode("locator", name="S", parent=top)
        cmds.instance(top, name="T2")
        names += ["|T1|S", "T2|S", "time1", "lambert1", "initialShadingGroup"]
        return [cmds.ls(name, long=True)[0] for name in names]

    def test_alias_named_custom_type_attr_takes_legacy_dispatch(self):
        # attributeQuery(exists) matches the alias, so the legacy cast reads
        # the aliased string attr as the custom type and picks Transform
        jnt = cmds.createNode("joint", name="aliased")
        cmds.addAttr(jnt, longName="foo", dataType="string")
        cmds.setAttr(f"{jnt}.foo", "transform", type="string")
        cmds.aliasAttr(CUSTOM_TYPE_ATTR, f"{jnt}.foo")
        self.assertIs(type(PyNode(jnt)), Transform)
        self.assertEqual(get_custom_type(jnt), "transform")

        expected = type(_pynode_legacy_tail(PyNode, jnt))
        self.assertIs(expected, Transform)
        for obj in (jnt, _mobject(jnt), OpenMaya.MDagPath.getAPathTo(_mobject(jnt))):
            with mock.patch.object(
                _base, "_pynode_legacy_tail", wraps=_pynode_legacy_tail
            ) as legacy:
                self.assertIs(type(PyNode(obj)), expected)
            self.assertEqual(legacy.call_count, 1)

        plain = cmds.createNode("joint", name="plain")
        self.assertIs(type(PyNode(plain)), Joint)
        self.assertIsNone(get_custom_type(plain))

    def test_new_node_class_invalidates_dispatch_cache(self):
        node = cmds.createNode("multiplyDivide", name="md1")
        self.assertIs(type(PyNode(node)), DGNode)
        self.assertIs(type(PyNode(_mobject(node))), DGNode)
        self.assertTrue(PyNode._CLASS_BY_TYPE)

        class _MultiplyDivide(DGNode):
            NATIVE_NODE_TYPE = "multiplyDivide"

        self.assertFalse(PyNode._CLASS_BY_TYPE)
        self.assertFalse(PyNode._CASTABLE_TYPES)
        self.assertIs(type(PyNode(node)), _MultiplyDivide)
        self.assertIs(type(PyNode(_mobject(node))), _MultiplyDivide)

    def test_set_custom_type_after_wrap_redispatches(self):
        class _CustomTransform(Transform):
            CUSTOM_NODE_TYPE = "perfInvariantsCustom"

        xform = cmds.createNode("transform", name="xform")
        self.assertIs(type(PyNode(xform)), Transform)
        self.assertIs(type(PyNode(_mobject(xform))), Transform)

        set_custom_type(xform, "perfInvariantsCustom")
        self.assertIs(type(PyNode(xform)), _CustomTransform)
        self.assertIs(type(PyNode(_mobject(xform))), _CustomTransform)
        self.assertIs(type(PyNode(cmds.ls(xform, uuid=True)[0])), _CustomTransform)

        other = cmds.createNode("transform", name="other")
        self.assertIs(type(PyNode(other)), Transform)

    def test_dispatch_equivalence_sweep(self):
        names = self._sweep_scene()
        inputs = []
        for name in names:
            mobj = _mobject(name)
            inputs.append((name, name))
            inputs.append((mobj, _mobject_to_str(mobj)))
            if mobj.hasFn(OpenMaya.MFn.kDagNode):
                for path in OpenMaya.MDagPath.getAllPathsTo(mobj):
                    inputs.append((path, path.partialPathName()))
        inputs.append(("dup", "dup"))
        inputs.append(("n_transform", "n_transform"))

        for label in ("cold", "warm"):
            if label == "cold":
                PyNode._CLASS_BY_TYPE.clear()
            for obj, legacy_name in inputs:
                with self.subTest(cache=label, obj=legacy_name, kind=type(obj).__name__):
                    self.assertEqual(
                        _outcome(PyNode, obj),
                        _outcome(_pynode_legacy_tail, PyNode, legacy_name),
                    )

    def test_warm_dispatch_makes_no_name_queries(self):
        node = cmds.createNode("multiplyDivide", name="md1")
        PyNode(node)
        probes = [
            mock.patch.object(cmds, name, wraps=getattr(cmds, name))
            for name in ("ls", "nodeType", "attributeQuery")
        ]
        mocks = [probe.start() for probe in probes]
        try:
            self.assertIs(type(PyNode(node)), DGNode)
            self.assertIs(type(PyNode(_mobject(node))), DGNode)
        finally:
            for probe in probes:
                probe.stop()
        self.assertEqual([m.call_count for m in mocks], [0, 0, 0])

    def test_pynode_error_messages_unchanged(self):
        cmds.createNode("transform", name="dup")
        cmds.createNode("transform", name="dup", parent=cmds.createNode("transform"))
        for name in ("nope", "dup", "", "|", "du*"):
            with self.subTest(name=name):
                expected = _outcome(_pynode_legacy_tail, PyNode, name)
                self.assertEqual(expected[0], "error")
                self.assertEqual(_outcome(PyNode, name), expected)

        for unknown in (
            "DEADBEEF-0000-4000-8000-000000000000",
            "abcdefabcdefabcdefabcdefabcdefab",
        ):
            with self.subTest(uid=unknown):
                self.assertTrue(is_valid_maya_uid(unknown))
                with self.assertRaises(TypeError) as ctx:
                    PyNode(unknown)
                self.assertEqual(
                    str(ctx.exception), f"No object matches uuid: {unknown}."
                )

    def test_deleted_node_mobject_casts_by_name(self):
        cmds.undoInfo(state=True, infinity=True)
        PyNode(_mobject(cmds.createNode("multiplyDivide")))
        held = _mobject(cmds.createNode("multiplyDivide", name="foo"))
        cmds.delete("foo")
        self.assertEqual(
            _outcome(PyNode, held),
            ("error", TypeError, "No object matches name: foo"),
        )
        cmds.createNode("transform", name="foo")
        self.assertEqual(_outcome(PyNode, held), ("ok", Transform, "foo"))

        # a deleted node's type never caches the class of the node that took
        # its name
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        held = _mobject(cmds.createNode("multiplyDivide", name="bar"))
        cmds.delete("bar")
        cmds.createNode("transform", name="bar")
        self.assertEqual(_outcome(PyNode, held), ("ok", Transform, "bar"))
        fresh = cmds.createNode("multiplyDivide", name="fresh")
        self.assertEqual(_outcome(PyNode, fresh), ("ok", DGNode, "fresh"))
        self.assertEqual(_outcome(PyNode, _mobject(fresh)), ("ok", DGNode, "fresh"))

    def test_undone_node_dag_path_casts_by_name(self):
        cmds.undoInfo(state=True, infinity=True)
        PyNode(cmds.createNode("transform"))
        cmds.undoInfo(openChunk=True)
        cmds.createNode("transform", name="undoneT")
        cmds.undoInfo(closeChunk=True)
        sel = OpenMaya.MSelectionList()
        sel.add("undoneT")
        path = sel.getDagPath(0)
        held = sel.getDependNode(0)
        cmds.undo()
        for obj in (path, held):
            self.assertEqual(
                _outcome(PyNode, obj),
                ("error", TypeError, "No object matches name: undoneT"),
            )

    def test_deleted_joint_mobject_casts_the_node_that_took_its_name(self):
        cmds.undoInfo(state=True, infinity=True)
        PyNode(_mobject(cmds.createNode("joint")))
        jnt  = cmds.createNode("joint", name="foo")
        held = _mobject(jnt)
        plug = OpenMaya.MFnDependencyNode(held).findPlug("translateX", False)
        cmds.delete(jnt)
        cmds.createNode("transform", name="foo")
        self.assertEqual(_outcome(PyNode, held), ("ok", Transform, "foo"))
        # re-pinned (round 3b review): a plug built from an MPlug of a node
        # deleted to the undo queue can take no handle of it (a deleted node is
        # not found by name), so it raises the deleted node's "already
        # deleted!" when it is built; it read the node that took the name
        # ("foo.translateX"), or another one once the deleted node was freed
        with self.assertRaises(RuntimeError) as ctx:
            Plug(plug)
        self.assertEqual(str(ctx.exception), "foo already deleted!")


def _wrapper_state(node):
    """Everything a node wrapper holds, in the order its constructor set it."""
    state = {
        "class": type(node),
        "keys":  list(vars(node)),
        "fn":    type(node._fn_set),
        "name":  node.name,
        "api1":  (node._fn_set1.name(), node._objhandle1.isValid()),
        "attrs": node._attr_dict,
    }
    if "_mdagpath" in vars(node):
        state["path"] = node._mdagpath.fullPathName()
    return state


class TestCheckedTypeConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_checked_type_cast_matches_constructor(self):
        names = [
            cmds.createNode(node_type, name=f"n_{node_type}")
            for node_type in (
                "multiplyDivide",
                "network",
                "choice",
                "objectSet",
                "displayLayer",
                "transform",
                "joint",
                "locator",
                "follicle",
            )
        ]
        cube = cmds.polyCube(name="cube")[0]
        names += [cube, cmds.listRelatives(cube, shapes=True)[0]]
        cmds.namespace(add="ns")
        names.append(cmds.createNode("transform", name="ns:inNs"))
        names.append(cmds.createNode("transform", name="dup"))
        names.append(cmds.createNode("transform", name="dup", parent=cube))
        top = cmds.createNode("transform", name="T1")
        cmds.createNode("locator", name="S", parent=top)
        cmds.instance(top, name="T2")
        names += ["|T1|S", "T2|S"]
        for name in cmds.ls(names, long=True):
            with self.subTest(node=name):
                mobj  = _mobject(name)
                first = PyNode(mobj)
                again = PyNode(mobj)
                built = type(first)(_mobject_to_str(mobj))
                self.assertEqual(_wrapper_state(again), _wrapper_state(built))
                self.assertEqual(_wrapper_state(again), _wrapper_state(first))
                self.assertTrue(again._mobject == built._mobject)

    def test_checked_type_cast_skips_type_queries(self):
        for node_type, cls in (("transform", Transform), ("joint", Joint)):
            with self.subTest(node_type=node_type):
                first = cmds.createNode(node_type)
                other = cmds.createNode(node_type)
                PyNode(_mobject(first))
                probes = [
                    mock.patch.object(cmds, name, wraps=getattr(cmds, name))
                    for name in ("nodeType", "objectType", "attributeQuery")
                ]
                mocks = [probe.start() for probe in probes]
                try:
                    self.assertIs(type(PyNode(_mobject(other))), cls)
                finally:
                    for probe in probes:
                        probe.stop()
                self.assertEqual([m.call_count for m in mocks], [0, 0, 0])

    def test_first_cast_of_a_type_runs_the_constructor(self):
        xform = cmds.createNode("transform")
        PyNode(_mobject(xform))
        PyNode._CASTABLE_TYPES.clear()
        with mock.patch.object(cmds, "nodeType", wraps=cmds.nodeType) as node_type:
            self.assertIs(type(PyNode(_mobject(xform))), Transform)
        self.assertGreater(node_type.call_count, 0)
        with mock.patch.object(cmds, "nodeType", wraps=cmds.nodeType) as node_type:
            self.assertIs(type(PyNode(_mobject(xform))), Transform)
        self.assertEqual(node_type.call_count, 0)

    def test_class_with_its_own_constructor_or_type_check_runs_them(self):
        class _Network(DGNode):
            NATIVE_NODE_TYPE = "network"
            inits            = 0

            def __init__(self, node):
                type(self).inits += 1
                super().__init__(node)

        class _Condition(DGNode):
            NATIVE_NODE_TYPE = "condition"
            checks           = 0

            @classmethod
            def is_type(cls, node_name, failfast=False, **kwargs):
                cls.checks += 1
                return super().is_type(node_name, failfast=failfast, **kwargs)

        for node_type, cls, counter in (
            ("network", _Network, "inits"),
            ("condition", _Condition, "checks"),
        ):
            with self.subTest(node_type=node_type):
                nodes = [cmds.createNode(node_type) for _ in range(2)]
                for name in nodes + nodes:
                    self.assertIs(type(PyNode(_mobject(name))), cls)
                self.assertEqual(getattr(cls, counter), 4)

    def test_plug_name_tracks_rename_reparent_namespace_instance(self):
        a    = cmds.createNode("transform", name="A")
        b    = cmds.createNode("transform", name="B")
        c    = cmds.createNode("transform", name="C", parent=a)
        plug = Node(c).tx
        held = Plug(plug.plug)
        self.assertEqual(str(plug), "C.translateX")

        steps = (
            (lambda: cmds.rename(c, "C2"), "C2.translateX"),
            (lambda: cmds.parent("A|C2", b), "C2.translateX"),
            (
                lambda: cmds.createNode("transform", name="C2", parent=a),
                "B|C2.translateX",
            ),
            (lambda: cmds.namespace(add="ns"), "B|C2.translateX"),
            (lambda: cmds.rename("B|C2", "ns:C3"), "ns:C3.translateX"),
            (lambda: cmds.instance(b), "B|ns:C3.translateX"),
        )
        for change, expected in steps:
            change()
            with self.subTest(expected=expected):
                self.assertEqual(str(plug), expected)
                self.assertEqual(str(held), expected)
                self.assertEqual(str(Plug(plug.plug)), expected)

        top = cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="S", parent=top)
        other = cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", other, add=True, relative=True)
        self.assertEqual(str(Node("|T2|S").visibility), "T2|S.visibility")
        self.assertEqual(Node("|T2|S").visibility.node._dg_node.long_name, "|T2|S")

    def test_named_plug_keeps_its_node_after_delete(self):
        cmds.undoInfo(state=True, infinity=True)
        for node_type, attr in (("multiplyDivide", "input1X"), ("transform", "tx")):
            with self.subTest(node_type=node_type):
                PyNode(_mobject(cmds.createNode(node_type)))
                name  = cmds.createNode(node_type, name=f"gone_{node_type}")
                named = Plug(f"{name}.{attr}")
                fresh = Plug(named.plug)
                str(named)
                owner = named.node
                cmds.delete(name)

                with self.assertRaises(RuntimeError) as ctx:
                    str(named)
                self.assertEqual(str(ctx.exception), f"{name} already deleted!")
                self.assertIs(named.node, owner)
                self.assertFalse(owner.is_valid)
                # re-pinned (round 3b): a plug built from an MPlug, never asked
                # for its node, checks the handle of its node it took and raises
                # the deleted node's "already deleted!" (it raised a TypeError or
                # a ValueError casting the deleted node)
                with self.assertRaises(RuntimeError) as ctx:
                    str(fresh)
                self.assertEqual(str(ctx.exception), f"{name} already deleted!")

                cmds.undo()
                self.assertTrue(str(named).startswith(f"{name}."))
                self.assertIs(named.node, owner)

    def test_uncastable_owner_still_raises(self):
        cube  = cmds.polyCube(name="cube")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        empty = cmds.createNode("mesh", name="emptyShape")
        for _ in range(2):
            self.assertEqual(str(Plug(f"{shape}.visibility")), f"{shape}.visibility")
            with self.assertRaises(ValueError) as ctx:
                str(Plug(f"{empty}.visibility"))
            self.assertEqual(
                str(ctx.exception), "object is incompatible with MFnMesh constructor"
            )

    def test_plug_hash_value_unchanged(self):
        # Historical id: the value is no longer v2.0.0a2's. Round 3, decision
        # D-B: a plug hashes by its Maya plug, so a rename keeps the key (it was
        # the node's name plus the alias). Step S2 used the node's API 1.0
        # MObjectHandle hashCode, which Maya hands to a node made after a freed
        # one, so a plug of the freed node and one of the new node shared a key
        # (the review of 29a4128); the node part is now a serial that is never
        # reused (see `_node_serial`). The value is still pinned: every spelling
        # of one plug hashes the same.
        from rig.nodetypes._base import _node_serial

        def hash_code(node_name):
            return _node_serial(PyNode(node_name))

        md  = cmds.createNode("multiplyDivide", name="md1")
        grp = cmds.createNode("transform", name="grp")
        cmds.createNode("transform", name="dup")
        cmds.createNode("transform", name="dup", parent=grp)
        for plug, node_name, alias in (
            (Plug(f"{md}.input1X"), "md1", "input1X"),
            (Node("grp|dup").tx, "grp|dup", "translateX"),
            (Plug(Node("grp|dup").ty.plug), "grp|dup", "translateY"),
        ):
            with self.subTest(plug=node_name):
                self.assertEqual(hash(plug), hash((hash_code(node_name), alias)))


def _type_queries(probe):
    """The ``cmds.getAttr(..., type=True)`` calls a spy recorded."""
    return [call for call in probe.call_args_list if call.kwargs.get("type")]


class TestSetTypeArgument(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _set_kwargs(self, attr, *args):
        """The keyword arguments ``attr.set(*args)`` hands to cmds.setAttr()."""
        with mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as probe:
            attr.set(*args)
        return probe.call_args.kwargs

    def test_set_on_array_element_child_keeps_legacy_error(self):
        for node_type, attr in (
            ("timeEditorClip", "layer[0].layerWeight"),
            ("timeEditorTracks", "crossfade[0].crossfadeMode"),
        ):
            with self.subTest(node_type=node_type):
                try:
                    name = f"{cmds.createNode(node_type)}.{attr}"
                except RuntimeError:
                    self.skipTest(f"{node_type} is unavailable")
                try:
                    cmds.getAttr(name, type=True)
                except RuntimeError as exc:
                    message = str(exc)
                else:
                    self.skipTest(f"getAttr(type=True) resolves {name}")

                plug = Plug(name)
                with self.assertRaises(InjectionError) as ctx:
                    plug << 0.5
                self.assertEqual(str(ctx.exception), f"Cannot set {name!r}: {message}")
                with self.assertRaises(RuntimeError) as ctx:
                    plug.set(0.25)
                self.assertNotIsInstance(ctx.exception, InjectionError)
                self.assertEqual(str(ctx.exception), message)
                # A modifier swallows that error, as v2.0.0a2 does: it logs it
                # and leaves the plug unlocked, although the plug can be locked.
                with self.assertLogs("rig.spec._base", level="DEBUG") as logs:
                    self.assertIs(plug << lock, plug)
                self.assertEqual(
                    logs.output,
                    [f"DEBUG:rig.spec._base:set modifier on {name} failed: {message}"],
                )
                self.assertFalse(cmds.getAttr(name, lock=True))
                cmds.setAttr(name, lock=True)
                self.assertTrue(cmds.getAttr(name, lock=True))

    def test_lock_modifier_on_array_element_child(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="items", at="compound", nc=1, multi=True)
        cmds.addAttr(net, ln="weight", at="double", p="items")
        plug = Plug("net.items[3].weight")
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            self.assertIs(plug << lock, plug)
        self.assertEqual(len(_type_queries(probe)), 1)
        self.assertTrue(cmds.getAttr("net.items[3].weight", lock=True))
        self.assertEqual(cmds.getAttr("net.items", multiIndices=True), [3])

    def test_scalar_set_makes_no_type_query(self):
        loc = Node.create("transform", name="loc")
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="dbl", at="double")
        cmds.addAttr(net, ln="enm", at="enum", en="a:b")
        cmds.addAttr(net, ln="cmp", at="double3")
        for axis in "XYZ":
            cmds.addAttr(net, ln=f"cmp{axis}", at="double", p="cmp")
        net = Node(net)
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            loc.tx.set(1.0)
            loc.ry.set(45.0)
            loc.v.set(False)
            net.dbl.set(2.5)
            net.enm.set(1)
            net.cmpY.set(3.0)
            net.cmp.set(1.0, 2.0, 3.0)
            loc.tz << lock
        self.assertEqual(_type_queries(probe), [])
        self.assertEqual(cmds.getAttr("loc.tx"), 1.0)
        self.assertEqual(cmds.getAttr("net.cmp"), [(1.0, 2.0, 3.0)])
        self.assertEqual(cmds.getAttr("net.dbl"), 2.5)
        self.assertTrue(cmds.getAttr("loc.tz", lock=True))

    def test_matrix_set_still_passes_type(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="dtm", dt="matrix")
        cmds.addAttr(net, ln="atm", at="matrix")
        cmds.addAttr(net, ln="flm", at="fltMatrix")
        cmds.addAttr(net, ln="str", dt="string")
        cmds.addAttr(net, ln="dbl", at="double")
        net      = Node(net)
        identity = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1]
        for attr, args, expected in (
            (net.dtm, identity, {"type": "matrix"}),
            (net.atm, identity, {"type": "matrix"}),
            (net.flm, identity, {"type": "matrix"}),
            (net.str, ["text"], {"type": "string"}),
            (net.dbl, [1.5], {}),
        ):
            with self.subTest(attr=str(attr)):
                self.assertEqual(self._set_kwargs(attr, *args), expected)
        self.assertEqual(cmds.getAttr("net.dtm")[12:15], [2.0, 3.0, 4.0])

    def test_set_on_a_readded_matrix_attr_passes_type(self):
        matrix = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1]
        net    = Node(cmds.createNode("network", name="net"))
        for spec in (Float("foo"), Vector("foo")):
            net << spec
            plug = net.foo
            net << Matrix("foo")
            self.assertEqual(self._set_kwargs(plug, *matrix), {"type": "matrix"})
            self.assertEqual(cmds.getAttr("net.foo"), matrix)
            cmds.deleteAttr("net.foo")

        for flags in ({"dt": "matrix"}, {"at": "matrix"}, {"at": "fltMatrix"}):
            cmds.addAttr("net", ln="foo", at="double")
            plug = net.foo
            cmds.deleteAttr("net.foo")
            cmds.addAttr("net", ln="foo", **flags)
            plug.set(matrix)
            self.assertEqual(cmds.getAttr("net.foo"), matrix)
            cmds.deleteAttr("net.foo")

    def test_set_under_array_element_keeps_type_query(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="items", at="compound", nc=2, multi=True)
        cmds.addAttr(net, ln="weight", at="double", p="items")
        cmds.addAttr(net, ln="pos", at="double3", p="items")
        for axis in "XYZ":
            cmds.addAttr(net, ln=f"pos{axis}", at="double", p="pos")
        cmds.addAttr(net, ln="dbls", at="double", multi=True)
        for name in ("net.items[0].weight", "net.items[0].pos.posX", "net.dbls[2]"):
            plug = Plug(name)
            with self.subTest(plug=name):
                with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
                    plug.set(0.5)
                self.assertEqual(len(_type_queries(probe)), 1)
                self.assertEqual(cmds.getAttr(name), 0.5)

    def test_set_errors_match_the_type_query_path(self):
        def outcome(action):
            try:
                action()
            except Exception as exc:
                return (type(exc), str(exc))
            return None

        def network(name):
            node = cmds.createNode("network", name=name)
            cmds.addAttr(node, ln="dbl", at="double")
            cmds.addAttr(node, ln="cmp", at="double3")
            for axis in "XYZ":
                cmds.addAttr(node, ln=f"cmp{axis}", at="double", p="cmp")
            return node

        cmds.undoInfo(state=True, infinity=True)
        network("net")
        loc = Node.create("transform", name="loc")
        cmds.connectAttr("net.dbl", "loc.tx")
        # The deleted-node case gets a node of its own: deleting net would
        # also disconnect loc.tx and leave the other cases no node to set.
        gone = Plug(f"{network('gone')}.cmpY")
        str(gone)
        cmds.delete("gone")
        cases = [
            (lambda: loc.tx.set(1.0), True),  # connected destination
            (lambda: Plug("net.dbl").set("text"), True),
            (lambda: Plug("net.cmp").set(1.0), True),
            (lambda: Plug("net.cmpX").set(1.0, 2.0), False),  # Maya drops the extra value
            (lambda: Plug("net.dbl").set(), False),  # no value: Maya sets nothing
            (lambda: gone.set(1.0), True),
        ]
        self.assertTrue(cmds.isConnected("net.dbl", "loc.tx"))
        for index, (action, raises) in enumerate(cases):
            with self.subTest(case=index):
                fast = outcome(action)
                with mock.patch.object(
                    _base.Attribute, "_is_fixed_kind_outside_array", return_value=False
                ):
                    legacy = outcome(action)
                self.assertEqual(fast, legacy)
                self.assertEqual(fast is not None, raises)
        self.assertTrue(cmds.isConnected("net.dbl", "loc.tx"))
        self.assertEqual(cmds.getAttr("net.cmp"), [(1.0, 0.0, 0.0)])

    def test_set_on_a_connected_destination_keeps_the_connection(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="dbl", at="double")
        cmds.setAttr("net.dbl", 7.0)
        loc = Node.create("transform", name="loc")
        cmds.connectAttr("net.dbl", "loc.tx")
        message = "setAttr: The attribute 'loc.translateX' is locked or connected and cannot be modified.\n"
        for lookup, plug in (
            ("loc.tx", loc.tx),
            ("Plug('loc.tx')", Plug("loc.tx")),
            ("Plug('loc.translateX')", Plug("loc.translateX")),
        ):
            with self.subTest(lookup=lookup):
                with self.assertRaises(RuntimeError) as ctx:
                    plug.set(1.0)
                self.assertEqual(str(ctx.exception), message)
                self.assertTrue(cmds.isConnected("net.dbl", "loc.tx"))
                self.assertEqual(cmds.getAttr("loc.tx"), 7.0)
        # ``<<`` breaks the incoming connection to assign the value
        self.assertEqual(str(loc.tx << 4.0), "loc.translateX")
        self.assertFalse(cmds.isConnected("net.dbl", "loc.tx"))
        self.assertEqual(cmds.getAttr("loc.tx"), 4.0)
        self.assertEqual(cmds.getAttr("net.dbl"), 7.0)


class TestStaticDataTypeCache(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _base._STATIC_DATA_TYPE.clear()

    def _data_type(self, name):
        """The data type of a fresh ``Plug(name)`` and the type queries it made."""
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            typ = Plug(name).data_type
        return typ, len(_type_queries(probe))

    def test_static_data_type_cache_hit(self):
        first  = cmds.createNode("multiplyDivide")
        second = cmds.createNode("multiplyDivide")
        mult_a = cmds.createNode("multMatrix")
        mult_b = cmds.createNode("multMatrix")
        for attr in ("input1X", "input1", "operation", "message"):
            with self.subTest(attr=attr):
                expected = cmds.getAttr(f"{second}.{attr}", type=True)
                self.assertEqual(self._data_type(f"{first}.{attr}"), (expected, 1))
                self.assertEqual(self._data_type(f"{second}.{attr}"), (expected, 0))
        self.assertEqual(self._data_type(f"{mult_a}.matrixIn"), ("compound", 1))
        self.assertEqual(self._data_type(f"{mult_b}.matrixIn"), ("compound", 0))
        self.assertEqual(self._data_type(f"{mult_a}.matrixSum"), ("matrix", 1))
        self.assertEqual(self._data_type(f"{mult_b}.matrixSum"), ("matrix", 0))
        self.assertEqual(self._data_type(f"{mult_b}.matrixIn[0]"), ("matrix", 1))
        self.assertEqual(self._data_type(f"{mult_b}.matrixIn[0]"), ("matrix", 1))

    def test_choice_output_type_never_cached(self):
        pick = Node.create("choice", name="pick")
        loc  = Node.create("transform", name="loc")
        cmds.connectAttr("loc.translate", "pick.input[0]")
        cmds.connectAttr("loc.worldMatrix[0]", "pick.input[1]")
        self.assertIsNone(pick.output._static_type_key())
        for selector, expected in ((0, "double3"), (1, "matrix"), (0, "double3")):
            with self.subTest(selector=selector):
                pick.selector.set(selector)
                self.assertEqual(pick.output.data_type, expected)
                typ, queries = self._data_type("pick.output")
                self.assertEqual(typ, expected)
                self.assertGreaterEqual(queries, 1)
        cmds.disconnectAttr("loc.translate", "pick.input[0]")
        cmds.connectAttr("loc.worldMatrix[0]", "pick.input[0]", force=True)
        pick.selector.set(0)
        self.assertEqual(pick.output.data_type, "matrix")
        self.assertEqual(loc.worldMatrix.data_type, "matrix")

    def test_dynamic_attr_readd_new_type(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="foo", at="double")
        held = Plug("net.foo")
        self.assertIsNone(held._static_type_key())
        self.assertEqual(held.data_type, "double")
        cmds.deleteAttr("net.foo")
        cmds.addAttr(net, ln="foo", dt="string")
        self.assertEqual(self._data_type("net.foo"), ("string", 1))
        self.assertEqual(self._data_type("net.foo"), ("string", 1))

    def test_extension_attr_readd_new_type(self):
        def delete_extension():
            cmds.deleteExtension(
                nodeType="network", attribute="perfExt", forceDelete=True
            )

        try:
            cmds.addExtension(nodeType="network", longName="perfExt", at="double")
            cmds.createNode("network", name="netA")
            self.assertIsNone(Plug("netA.perfExt")._static_type_key())
            self.assertEqual(Plug("netA.perfExt").data_type, "double")
            delete_extension()
            cmds.addExtension(nodeType="network", longName="perfExt", at="enum", en="a:b")
            cmds.createNode("network", name="netB")
            self.assertEqual(self._data_type("netB.perfExt"), ("enum", 1))
            self.assertEqual(self._data_type("netA.perfExt"), ("enum", 1))
        finally:
            if cmds.attributeQuery("perfExt", type="network", exists=True):
                delete_extension()
        md = cmds.createNode("multiplyDivide")
        self.assertIsNotNone(Plug(f"{md}.input1X")._static_type_key())

    def test_element_subtree_not_cached(self):
        pma = cmds.createNode("plusMinusAverage", name="pma")
        cmds.setAttr("pma.input3D[2]", 1, 2, 3)
        for name in ("pma.input3D[2]", "pma.input3D[2].input3Dx", "pma.input1D[0]"):
            with self.subTest(plug=name):
                self.assertIsNone(Plug(name)._static_type_key())
                expected = cmds.getAttr(name, type=True)
                for _ in range(2):
                    self.assertEqual(self._data_type(name), (expected, 1))
        self.assertIsNotNone(Plug(f"{pma}.input3D")._static_type_key())
        self.assertEqual(self._data_type("pma.input3D"), ("compound", 1))
        self.assertEqual(self._data_type("pma.input3D"), ("compound", 0))

    def test_deleted_plug_raises_despite_cache(self):
        def outcome(plug):
            try:
                typ = plug.data_type
            except Exception as exc:
                return (type(exc), str(exc))
            return typ

        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("multiplyDivide", name="warm")
        self.assertEqual(self._data_type("warm.input1X"), ("float", 1))
        name  = cmds.createNode("multiplyDivide", name="gone")
        named = Plug(f"{name}.input1X")
        fresh = Plug(named.plug)
        str(named)
        cmds.delete(name)
        for plug in (named, fresh):
            with self.subTest(plug=plug is named):
                cached = outcome(plug)
                with mock.patch.object(
                    _base.Attribute, "_static_type_key", return_value=None
                ):
                    legacy = outcome(plug)
                self.assertEqual(cached, legacy)
                self.assertIsInstance(cached, tuple)
        self.assertEqual(outcome(named), (RuntimeError, f"{name} already deleted!"))

    def test_plugin_load_and_unload_clear_the_cache(self):
        for plugin in ("invertShape", "curveWarp", "quatNodes"):
            if not cmds.pluginInfo(plugin, query=True, loaded=True):
                break
        else:
            self.skipTest("no unloaded plug-in to load")
        cmds.createNode("multiplyDivide", name="md")
        try:
            Plug("md.input1X").data_type
            self.assertTrue(_base._STATIC_DATA_TYPE)
            cmds.loadPlugin(plugin, quiet=True)
            self.assertEqual(_base._STATIC_DATA_TYPE, {})
            Plug("md.input1X").data_type
            self.assertTrue(_base._STATIC_DATA_TYPE)
        finally:
            cmds.unloadPlugin(plugin)
        self.assertEqual(_base._STATIC_DATA_TYPE, {})


def _data_type_outcome(plug):
    """The data type of `plug`, or the type and message it raises."""
    try:
        return plug.data_type
    except Exception as exc:
        return (type(exc), str(exc))


def _legacy_data_type_outcome(plug):
    """`_data_type_outcome` with the static data type cache bypassed."""
    with mock.patch.object(_base.Attribute, "_static_type_key", return_value=None):
        return _data_type_outcome(plug)


# A plug-in locator (DAG) node with world space matrix array roots: `wsMatIn` can
# be connected into, `wsMatOut` cannot.
_WS_MATRIX_PLUGIN = '''
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui


def maya_useNewAPI():
    pass


class PerfWsMatrixLoc(omui.MPxLocatorNode):
    kName = "perfWsMatrixLoc"
    kId   = om.MTypeId(0x0007F2A1)

    @staticmethod
    def creator():
        return PerfWsMatrixLoc()

    @staticmethod
    def initialize():
        for long, short, writable in (
            ("wsMatIn", "wmi", True),
            ("wsMatOut", "wmo", False),
        ):
            fn            = om.MFnTypedAttribute()
            attr          = fn.create(long, short, om.MFnData.kMatrix)
            fn.array      = True
            fn.worldSpace = True
            fn.writable   = writable
            PerfWsMatrixLoc.addAttribute(attr)


def initializePlugin(obj):
    om.MFnPlugin(obj).registerNode(
        PerfWsMatrixLoc.kName,
        PerfWsMatrixLoc.kId,
        PerfWsMatrixLoc.creator,
        PerfWsMatrixLoc.initialize,
        om.MPxNode.kLocatorNode,
    )


def uninitializePlugin(obj):
    om.MFnPlugin(obj).deregisterNode(PerfWsMatrixLoc.kId)
'''


class TestStaticTypedRootCache(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _base._STATIC_DATA_TYPE.clear()

    def _data_type(self, plug):
        """The data type of `plug` and the type queries it made."""
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            typ = plug.data_type
        return typ, len(_type_queries(probe))

    def test_typed_array_root_cache_hit(self):
        first  = Node.create("choice", name="first")
        second = Node.create("choice", name="second")
        self.assertEqual(self._data_type(first.input), ("compound", 1))
        self.assertEqual(self._data_type(second.input), ("compound", 0))
        self.assertEqual(self._data_type(Plug("second.input")), ("compound", 0))
        one = Node.create("transform", name="one")
        two = Node.create("transform", name="two")
        for attr in (
            "worldMatrix",
            "worldInverseMatrix",
            "parentMatrix",
            "parentInverseMatrix",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(getattr(one, attr).plug.isArray)
                self.assertEqual(self._data_type(getattr(one, attr)), ("matrix", 1))
                self.assertEqual(self._data_type(getattr(two, attr)), ("matrix", 0))
                # an element is still queried every time
                element = getattr(two, attr)[0]
                self.assertIsNone(element._static_type_key())
                self.assertEqual(self._data_type(element), ("matrix", 1))
                self.assertEqual(self._data_type(element), ("matrix", 1))

    def test_typed_attr_outside_a_root_follows_its_data(self):
        loc  = Node.create("transform", name="loc")
        pick = Node.create("choice", name="pick")
        dist = [Node.create("distanceBetween", name=f"dist{i}") for i in range(2)]
        info = [Node.create("curveInfo", name=f"info{i}") for i in range(2)]
        cmds.connectAttr("loc.translate", "pick.input[0]")
        cmds.connectAttr("loc.worldMatrix[0]", "pick.input[1]")
        cmds.connectAttr("pick.output", "dist0.inMatrix1")
        cmds.connectAttr("pick.output", "info0.inputCurve")
        plugs = [node.inMatrix1 for node in dist] + [node.inputCurve for node in info]
        for plug in plugs:
            self.assertIsNone(plug._static_type_key())
        self.assertEqual(self._data_type(dist[1].inMatrix1), ("matrix", 1))
        self.assertEqual(self._data_type(info[1].inputCurve), ("nurbsCurve", 1))
        for selector, expected in ((0, "double3"), (1, "matrix"), (0, "double3")):
            with self.subTest(selector=selector):
                pick.selector.set(selector)
                self.assertEqual(self._data_type(dist[0].inMatrix1), (expected, 1))
                self.assertEqual(self._data_type(dist[1].inMatrix1), ("matrix", 1))
        pick.selector.set(1)
        self.assertEqual(self._data_type(info[0].inputCurve), ("matrix", 1))
        self.assertEqual(self._data_type(info[1].inputCurve), ("nurbsCurve", 1))
        self.assertEqual(loc.worldMatrix.data_type, "matrix")

    def test_roots_that_report_their_data_are_not_shared(self):
        cmds.polyCube(name="cube")
        cmds.createNode("mesh", name="emptyShape")
        cluster = cmds.cluster("cube")[0]
        shape   = Node("cubeShape")
        # a world space root reports its first element's data
        self.assertIsNone(shape.worldMesh._static_type_key())
        self.assertIsNotNone(shape.worldMatrix._static_type_key())
        # an internal root, and a root with no declared data type
        for name in ("cubeShape.face", "emptyShape.face", f"{cluster}.outputGeometry"):
            with self.subTest(plug=name):
                plug = Plug(name)
                self.assertIsNone(plug._static_type_key())
                expected = _legacy_data_type_outcome(plug)
                for _ in range(2):
                    self.assertEqual(_data_type_outcome(Plug(name)), expected)
        # the face root's query fails, so nothing is stored for it
        self.assertIsInstance(_data_type_outcome(Plug("emptyShape.face")), tuple)
        self.assertIsInstance(_data_type_outcome(Plug("cubeShape.face")), tuple)

    def test_choice_input_root_across_connections(self):
        src  = Node.create("transform", name="src")
        pick = Node.create("choice", name="pick")
        cmds.addAttr("src", ln="label", dt="string")
        cmds.polyCube(name="cube")
        cmds.circle(name="crv")
        sources = (
            "src.message",
            "src.translate",
            "src.translateX",
            "src.worldMatrix[0]",
            "src.label",
            "cubeShape.outMesh",
            "crvShape.worldSpace[0]",
        )
        self.assertEqual(self._data_type(pick.input), ("compound", 1))
        for index, source in enumerate(sources):
            with self.subTest(source=source):
                cmds.connectAttr(source, f"pick.input[{index}]")
                pick.selector.set(index)
                self.assertEqual(self._data_type(pick.input), ("compound", 0))
                self.assertEqual(self._data_type(Plug("pick.input")), ("compound", 0))
                self.assertEqual(cmds.getAttr("pick.input", type=True), "TdataCompound")
                element = pick.input[index]
                self.assertIsNone(element._static_type_key())
                self.assertEqual(
                    _data_type_outcome(element), _legacy_data_type_outcome(element)
                )
                self.assertEqual(
                    _data_type_outcome(pick.output),
                    _legacy_data_type_outcome(pick.output),
                )

    def test_held_typed_root_across_delete_rename_and_reuse(self):
        cmds.undoInfo(state=True, infinity=True)
        warm = Node.create("joint", name="warm")
        self.assertEqual(warm.worldMatrix.data_type, "matrix")
        warm_pick = Node.create("choice", name="warmPick")
        self.assertEqual(warm_pick.input.data_type, "compound")
        joint = Node.create("joint", name="jnt")
        pick  = Node.create("choice", name="pick")
        held  = [joint.worldMatrix, joint.worldInverseMatrix, pick.input]
        fresh = [Plug(plug.plug) for plug in held]
        for plug in held:
            str(plug)

        def rename():
            cmds.rename("jnt", "jnt2")
            cmds.rename("pick", "pick2")

        def reuse():
            cmds.delete("jnt2", "pick2")
            cmds.createNode("choice", name="jnt2")
            cmds.createNode("joint", name="pick2")

        steps = (
            ("live", lambda: None),
            ("deleted", lambda: cmds.delete("jnt", "pick")),
            ("undone", cmds.undo),
            ("renamed", rename),
            ("reused", reuse),
        )
        seen = {}
        for label, step in steps:
            step()
            for index, plug in enumerate(held + fresh):
                with self.subTest(step=label, plug=index):
                    cached = _data_type_outcome(plug)
                    self.assertEqual(cached, _legacy_data_type_outcome(plug))
                    seen[label, index] = cached
        self.assertEqual(
            [seen["live", index] for index in range(3)],
            ["matrix", "matrix", "compound"],
        )
        self.assertEqual(seen["deleted", 0], (RuntimeError, "jnt already deleted!"))
        self.assertEqual(seen["deleted", 2], (RuntimeError, "pick already deleted!"))
        self.assertEqual(
            [seen["undone", index] for index in range(3)],
            ["matrix", "matrix", "compound"],
        )
        self.assertEqual(seen["reused", 0], (RuntimeError, "jnt2 already deleted!"))

    def test_dynamic_and_extension_typed_roots_not_cached(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="mats", dt="matrix", multi=True)
        cmds.addAttr(net, ln="labels", dt="string", multi=True)
        for name in ("net.mats", "net.labels"):
            with self.subTest(plug=name):
                self.assertIsNone(Plug(name)._static_type_key())
                for _ in range(2):
                    self.assertEqual(self._data_type(Plug(name)), ("compound", 1))
        cmds.deleteAttr("net.mats")
        cmds.addAttr(net, ln="mats", at="double")
        self.assertEqual(self._data_type(Plug("net.mats")), ("double", 1))

        def delete_extension():
            cmds.deleteExtension(
                nodeType="choice", attribute="perfExt", forceDelete=True
            )

        try:
            cmds.addExtension(
                nodeType="choice", longName="perfExt", dt="matrix", multi=True
            )
            cmds.createNode("choice", name="pickA")
            self.assertIsNone(Plug("pickA.perfExt")._static_type_key())
            self.assertEqual(self._data_type(Plug("pickA.perfExt")), ("compound", 1))
            cmds.delete("pickA")
            delete_extension()
            cmds.addExtension(nodeType="choice", longName="perfExt", at="double")
            cmds.createNode("choice", name="pickB")
            self.assertEqual(self._data_type(Plug("pickB.perfExt")), ("double", 1))
        finally:
            if cmds.attributeQuery("perfExt", type="choice", exists=True):
                delete_extension()

    def test_connectable_world_space_matrix_root_not_shared(self):
        # a world space root reports its first element, which a generic source
        # connected into it can hand another type
        folder = tempfile.mkdtemp(prefix="rig_ws_matrix_")
        path   = os.path.join(folder, "perfWsMatrixLoc.py")
        with open(path, "w") as fh:
            fh.write(_WS_MATRIX_PLUGIN)
        try:
            cmds.loadPlugin(path, quiet=True)
            for wired_first in (False, True):
                with self.subTest(wired_first=wired_first):
                    cmds.file(new=True, force=True)
                    _base._STATIC_DATA_TYPE.clear()
                    for name in ("locA", "locB"):
                        cmds.createNode("perfWsMatrixLoc", name=name)
                    cmds.createNode("transform", name="src")
                    cmds.createNode("choice", name="pick")
                    cmds.connectAttr("src.translate", "pick.input[0]")
                    cmds.connectAttr("pick.output", "locB.wsMatIn[0]")
                    names = ["locB", "locA"] if wired_first else ["locA", "locB"]
                    plugs = [Node(name).wsMatIn for name in names + names]
                    self.assertEqual(
                        [_data_type_outcome(plug) for plug in plugs],
                        [_legacy_data_type_outcome(plug) for plug in plugs],
                    )
                    for plug in plugs:
                        self.assertIsNone(plug._static_type_key())
                    self.assertEqual(Node("locA").wsMatIn.data_type, "matrix")
                    self.assertEqual(Node("locB").wsMatIn.data_type, "double3")
                    self.assertEqual(
                        cmds.getAttr("locB.wsMatIn", type=True), "double3"
                    )
                    # a root nothing can connect into is still shared
                    self.assertIsNotNone(Node("locA").wsMatOut._static_type_key())
                    for name, queries in (("locA", 1), ("locB", 0)):
                        self.assertEqual(
                            self._data_type(Node(name).wsMatOut), ("matrix", queries)
                        )
        finally:
            cmds.file(new=True, force=True)
            cmds.flushUndo()
            if cmds.pluginInfo("perfWsMatrixLoc", query=True, loaded=True):
                cmds.unloadPlugin("perfWsMatrixLoc")
            shutil.rmtree(folder, ignore_errors=True)


# The instance state ``Attribute.__init__`` leaves, in the order it is stored.
_ATTRIBUTE_STATE_KEYS = (
    "_mplug",
    "_mobject",
    "_fn_set",
    "_node",
    "_Attribute__child_name_dict",
    "_Attribute__child_id_dict",
    "_Attribute__component_type",
    "_geometry_attr_cache",
    "_polymorphic_owner_cache",
    "_static_key_cache",
    # re-pinned (round 3b): an attr with no owner keeps an API 1.0 handle of
    # its node, so it raises "already deleted!" once the node is deleted or freed
    "_handle1",
    # re-pinned (round 3b review): a dynamic attr keeps an API 1.0 handle of its
    # attribute, which a delete frees once it leaves the undo queue (None here)
    "_attr1",
)
_COMPONENT_STATE_KEYS = ("_comp_node", "_comp_alias", "_comp_ndims", "_comp_coords")


class TestAttributeInitState(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _assert_fresh_state(self, attr, mplug):
        state = vars(attr)
        self.assertEqual(tuple(state), _ATTRIBUTE_STATE_KEYS)
        self.assertIs(state["_mplug"], mplug)
        for key in _ATTRIBUTE_STATE_KEYS[1:4] + _ATTRIBUTE_STATE_KEYS[6:9]:
            self.assertIsNone(state[key])
        self.assertIs(state["_static_key_cache"], _base._STATIC_KEY_UNSET)
        self.assertEqual(state["_Attribute__child_name_dict"], {})
        self.assertEqual(state["_Attribute__child_id_dict"], {})
        self.assertIsNot(
            state["_Attribute__child_name_dict"], state["_Attribute__child_id_dict"]
        )

    def test_attribute_init_state_keys(self):
        md    = cmds.createNode("multiplyDivide", name="md")
        mplug = Plug(f"{md}.input1").plug
        for cls in (_base.Attribute, Plug):
            with self.subTest(cls=cls.__name__):
                self._assert_fresh_state(cls(mplug), mplug)
                named = cls(f"{md}.input1")
                self._assert_fresh_state(named, vars(named)["_mplug"])
                self.assertEqual(named.plug, mplug)
                self.assertEqual(named.full_name, "md.input1")
                # every instance gets its own child caches
                self.assertIsNot(
                    vars(cls(mplug))["_Attribute__child_name_dict"],
                    vars(cls(mplug))["_Attribute__child_name_dict"],
                )
        surface = cmds.sphere(name="nurbsSphere1", constructionHistory=False)[0]
        handle  = Node(surface).cv
        element = handle[1, 2]
        for plug in (handle, element):
            self.assertIsInstance(plug, ComponentPlug)
            self.assertEqual(
                tuple(vars(plug)), _ATTRIBUTE_STATE_KEYS + _COMPONENT_STATE_KEYS
            )
        self.assertIsNone(handle._comp_coords)
        self.assertEqual(element._comp_coords, (1, 2))
        self.assertEqual(str(element), "nurbsSphere1Shape.cv[1][2]")

    def test_plug_init_makes_no_setattr_calls(self):
        cmds.createNode("multiplyDivide", name="md")
        names    = []
        original = Plug.__setattr__

        def counting(self, name, value):
            names.append(name)
            original(self, name, value)

        with mock.patch.object(Plug, "__setattr__", counting):
            plug = Plug("md.input1X")
            Plug(plug.plug)
        self.assertEqual(names, [])
        # later state writes still go through ``Plug.__setattr__``
        with mock.patch.object(Plug, "__setattr__", counting):
            plug.node
        self.assertIn("_node", names)

    def test_bad_source_error_unchanged(self):
        for source in (5, None, 1.5):
            with self.subTest(source=source):
                with self.assertRaises(ValueError) as ctx:
                    _base.Attribute(source)
                self.assertEqual(
                    str(ctx.exception), f"{source} is not a string or MPlug."
                )


def _owner(plug):
    """The name, owner classes and owner path a plug reports, or what it raises."""
    try:
        name = str(plug)
        node = plug.node._dg_node
        path = node._mdagpath.fullPathName() if "_mdagpath" in vars(node) else None
    except Exception as exc:
        return ("error", type(exc), str(exc))
    return ("ok", name, type(plug.node), type(node), node.long_name, path)


class TestPlugNodeReuse(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def _casts(self, func):
        """The ``PyNode`` casts ``Attribute.node`` (and the ``Node`` factory) make
        while ``func`` runs."""
        # re-pinned (round 4a M4, C8): Plug.node is gone, the lazy cast is
        # Attribute.node's, through the nodetypes._base global
        cast = mock.Mock(side_effect=PyNode)
        with mock.patch.object(_base, "PyNode", cast):
            func()
        return cast.call_count

    def _known_type(self, node_type):
        """A new node of a type ``PyNode`` already cast from an MObject."""
        PyNode(_mobject(cmds.createNode(node_type)))
        return cmds.createNode(node_type)

    def test_lookup_child_and_element_plugs_reuse_the_wrapper(self):
        node = Node(self._known_type("transform"))
        pma  = Node(self._known_type("plusMinusAverage"))
        for plug, expected in (
            (node.tx, "transform2.translateX"),
            (node.t[0], "transform2.translateX"),
            (node.t[-1], "transform2.translateZ"),
            (node.t.translateY, "transform2.translateY"),
            (node.t.ty, "transform2.translateY"),
            (node.t[:][2], "transform2.translateZ"),
            (node.translate.child(1), "transform2.translateY"),
            (pma.input1D[3], "plusMinusAverage2.input1D[3]"),
            (pma.input1D[0:2][1], "plusMinusAverage2.input1D[1]"),
            (pma.input3D[1].input3Dx, "plusMinusAverage2.input3D[1].input3Dx"),
        ):
            with self.subTest(plug=expected):
                self.assertEqual(self._casts(lambda: str(plug)), 0)
                self.assertEqual(str(plug), expected)
                # re-pinned (round 4a M4, C8): the owner is the typed node
                self.assertIsInstance(plug.node, Node)
                self.assertIs(plug.node, pma if expected.startswith("plus") else node)
                fresh = Plug(plug.plug)
                self.assertEqual(self._casts(lambda: str(fresh)), 1)
                self.assertEqual(_owner(plug), _owner(fresh))
        # a parent that holds no wrapper hands none on and is not resolved
        held = Plug(node.rotate.plug)
        self.assertIsNone(held.child(0)._node)
        self.assertIsNone(held.rotateY._node)
        self.assertIsNone(held._node)
        str(held)
        self.assertEqual(self._casts(lambda: str(held.rotateX)), 0)
        self.assertEqual(self._casts(lambda: str(held[2])), 0)

    def test_first_plug_on_a_type_still_casts(self):
        # Historical id: under the owner rule (round 3, D-A) the first plug on
        # a type casts nothing; the body pins 0 casts.
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        node = Node(cmds.createNode("multiplyDivide"))
        plug = node.input1X
        # the plug holds the node it was read from, so naming it casts nothing
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        self.assertEqual(self._casts(lambda: str(plug)), 0)
        self.assertEqual(self._casts(lambda: str(node.input1Y)), 0)

    def test_instanced_shape_plug_not_seeded(self):
        top = cmds.createNode("transform", name="T1")
        PyNode(_mobject(cmds.createNode("transform", name="S", parent=top)))
        other = cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", other, add=True, relative=True)
        # Owner rule (C3): a plug holds the node it was read from, so building
        # and naming it casts nothing, and it is named through that path.
        # re-pinned (round 4a M4, C8): the Node factory casts through the same
        # PyNode global, so the node is built outside the counted call
        held_t2 = Node("|T2|S")
        for attr, lookup in (
            ("visibility", lambda: held_t2.visibility),
            ("translateX", lambda: held_t2.t[0]),
        ):
            with self.subTest(plug=attr):
                names = []
                self.assertEqual(self._casts(lambda: names.append(str(lookup()))), 0)
                self.assertEqual(names, [f"T2|S.{attr}"])
                plug = lookup()
                self.assertEqual(plug.name, attr)
                self.assertEqual(str(plug), f"T2|S.{attr}")
                self.assertEqual(plug.node._dg_node.long_name, "|T2|S")
        held_t1 = Node("|T1|S")
        self.assertEqual(self._casts(lambda: str(held_t1.visibility)), 0)

    def test_instanced_plug_names_match_v2_0_0a2(self):
        # Historical id: the |T2|S subTests are named through T2 under the owner
        # rule (round 3, D-A), no longer as on v2.0.0a2; the |T1|S ones and the
        # Plug-from-string ones still match v2.0.0a2.
        top = cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="S", parent=top)
        other = cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", other, add=True, relative=True)
        for path in ("|T1|S", "|T2|S"):
            for lookup, get, attr in (
                ("visibility", lambda n: n.visibility, "visibility"),
                ("v", lambda n: n.v, "visibility"),
                ("t", lambda n: n.t, "translate"),
                ("tx", lambda n: n.tx, "translateX"),
                ("t[0]", lambda n: n.t[0], "translateX"),
                ("t.translateY", lambda n: n.t.translateY, "translateY"),
                ("t[:][2]", lambda n: n.t[:][2], "translateZ"),
                ("translate.child(1)", lambda n: n.translate.child(1), "translateY"),
            ):
                with self.subTest(path=path, lookup=lookup):
                    # Owner rule (C3): a plug is named through the path of the
                    # node it was read from ('|T1|S' as in v2.0.0a2, '|T2|S'
                    # through the second instance)
                    plug = get(Node(path))
                    self.assertEqual(str(plug), f"{path[1:]}.{attr}")
                    self.assertEqual(plug.full_name, f"{path[1:]}.{attr}")
                    # re-pinned (round 4a M4, C8): the owner is the typed node
                    self.assertIsInstance(plug.node, Node)
                    self.assertIs(type(plug.node), Transform)
                    self.assertEqual(plug.node.long_name, path)
        for name in ("|T2|S.visibility", "T1|S.visibility"):
            with self.subTest(name=name):
                self.assertEqual(str(Plug(name)), "T1|S.visibility")
        self.assertEqual(str(Node("|T2|S")), "T2|S")
        # the same Maya plug read through either path is one key
        first, second = Node("|T1|S").visibility, Node("|T2|S").visibility
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(len({first: 1, second: 2}), 1)

    def test_user_chosen_class_not_seeded(self):
        plug = Node(Transform(self._known_type("joint"))).tx
        self.assertEqual(self._casts(lambda: str(plug)), 0)
        self.assertIs(type(plug.node._dg_node), Transform)

    def test_shape_attr_via_transform_gets_shape_node(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        curve  = cmds.curve(point=points, name="crv")
        shape  = cmds.listRelatives(curve, shapes=True)[0]
        PyNode(_mobject(curve))
        PyNode(_mobject(shape))
        plug = Node(curve).controlPoints
        self.assertEqual(str(plug), f"{shape}.controlPoints")
        self.assertEqual(plug.node.name, shape)
        self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))

    def test_child_plug_tracks_rename(self):
        node     = Node(self._known_type("transform"))
        children = (node.t[0], node.t.translateY, node.t[:][2])
        cmds.rename(str(node), "y")
        for child, attr in zip(children, ("translateX", "translateY", "translateZ")):
            with self.subTest(attr=attr):
                self.assertEqual(self._casts(lambda: str(child)), 0)
                self.assertEqual(str(child), f"y.{attr}")

    def test_container_plug_node_is_plain_node(self):
        # Historical id: under the owner rule (round 3, D-A) the plug's node is
        # the Container it was read from, not a plain Node.
        PyNode(_mobject(cmds.container(name="box0")))
        ctn  = Container(cmds.container(name="box"))
        plug = ctn.blackBox
        self.assertEqual(self._casts(lambda: str(plug)), 0)
        self.assertIs(plug.node, ctn)
        self.assertEqual(plug.node.name, "box")

    def test_element_plug_shares_node(self):
        base   = cmds.polyCube(name="base")[0]
        target = cmds.polyCube(name="target")[0]
        bs     = cmds.blendShape(target, base, name="bs")[0]
        PyNode(_mobject(bs))
        for plug in (Node(bs).weight[0], Node(bs).w[0], Node(bs).weight[0:1][0]):
            with self.subTest(plug=plug.name):
                self.assertEqual(self._casts(lambda: str(plug)), 0)
                self.assertEqual(plug.node.name, Plug(f"{bs}.weight[0]").node.name)
                self.assertEqual(_owner(plug), _owner(Plug(f"{bs}.weight[0]")))

    def test_changes_after_lookup_match_a_fresh_plug(self):
        a    = cmds.createNode("transform", name="A")
        b    = cmds.createNode("transform", name="B")
        node = Node(cmds.parent(self._known_type("transform"), a)[0])
        steps = (
            lambda: cmds.rename("A|transform2", "C"),
            lambda: cmds.parent("A|C", b),
            lambda: cmds.createNode("transform", name="C", parent=a),
            lambda: cmds.instance(b),
            lambda: cmds.rename("B|C", "C2"),
            lambda: cmds.parent("B|C2", world=True),
        )
        for index, step in enumerate(steps):
            held = (node.tx, node.t[1], node.t.tz)
            step()
            for plug in held:
                with self.subTest(step=index, plug=plug.name):
                    self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))
                    self.assertEqual(_owner(plug)[0], "ok")

    def test_custom_type_set_after_lookup(self):
        class _Probe(Transform):
            CUSTOM_NODE_TYPE = "perfReuseProbe"

        node = Node(self._known_type("transform"))
        held = (node.tx, node.t[0])
        set_custom_type(str(node), "perfReuseProbe")
        for plug in held:
            with self.subTest(plug=plug.name):
                self.assertEqual(self._casts(lambda: str(plug)), 0)
                self.assertIs(type(plug.node._dg_node), Transform)
                # a fresh cast sees the custom type
                self.assertNotEqual(_owner(plug), _owner(Plug(plug.plug)))

    def test_deleted_node_error_unchanged(self):
        # Historical id: the error changed under the owner rule (round 3, D-A):
        # a held plug of a deleted node raises '<name> already deleted!', also
        # once a new node takes the name, instead of retargeting to it.
        cmds.undoInfo(state=True, infinity=True)
        for node_type, attr, parent in (
            ("multiplyDivide", "input1X", "input1"),
            ("transform", "tx", "translate"),
        ):
            with self.subTest(node_type=node_type):
                name  = self._known_type(node_type)
                node  = Node(name)
                plugs = (getattr(node, attr), getattr(node, parent)[1])
                deleted = ("error", RuntimeError, f"{name} already deleted!")
                cmds.delete(name)
                # a held plug's owner is the deleted node, not a new cast
                for plug in plugs:
                    self.assertEqual(_owner(plug), deleted)
                    # re-pinned (round 3b review): a plug built from the MPlug
                    # of a deleted node raises when it is built (it can take no
                    # handle of the node), no longer when it is named
                    self.assertEqual(_outcome(Plug, plug.plug), deleted)
                cmds.undo()
                for plug in plugs:
                    self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))
                    self.assertEqual(_owner(plug)[0], "ok")

                # the cast resolves the name, so it finds a newer node of that
                # name; a held plug does not retarget to it
                plugs = (getattr(node, attr), getattr(node, parent)[1])
                cmds.delete(name)
                cmds.createNode(node_type, name=name)
                for plug in plugs:
                    self.assertEqual(_owner(plug), deleted)
                    # re-pinned (round 3b review): it named the node that took
                    # the name ("ok"); a plug built from the MPlug of a deleted
                    # node now raises when it is built, as a held plug does
                    self.assertEqual(_outcome(Plug, plug.plug), deleted)

    def test_plug_node_wrapper_is_its_own(self):
        # Historical id: under the owner rule (round 3, D-A) the plug's node is
        # the user's own wrapper (the body pins assertIs(owner, wrapper)).
        node    = Node(self._known_type("transform"))
        wrapper = node >> None
        cached  = wrapper.find_attr("tx")
        vars(wrapper)["user_tag"] = "set on the user's wrapper"
        for plug in (node.tx, node.t[0], node.t.translateY, node.rotate.child(0)):
            owner = plug.node >> None
            self.assertIs(type(owner), Transform)
            self.assertIs(owner, wrapper)
            self.assertIs(owner.mdagpath, wrapper.mdagpath)
            self.assertIs(owner.mobject, wrapper.mobject)
            self.assertIs(owner.fn_set, wrapper.fn_set)
            self.assertTrue(hasattr(owner, "user_tag"))
            self.assertIs(owner.find_attr("tx"), cached)
        # find_attr filters a lookup the user's wrapper already cached
        self.assertIsNone(node.tx.node.find_attr("translateX", data_type="string"))

        cube  = cmds.polyCube(name="gc")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        PyNode(_mobject(shape))
        mesh  = Node(shape)
        local = (mesh >> None).local_shape_attr
        self.assertIs((mesh.outMesh.node >> None).local_shape_attr, local)

    def test_plug_node_finds_a_readded_extension_attr(self):
        node = Node(self._known_type("transform"))
        try:
            cmds.addExtension(nodeType="transform", longName="perfExt", at="double")
            node.perfExt
            cmds.deleteExtension(
                nodeType="transform", attribute="perfExt", forceDelete=True
            )
            cmds.addExtension(nodeType="transform", longName="perfExt", dataType="string")
            plug = node.tx.node.perfExt
            self.assertEqual(str(plug), f"{node}.perfExt")
            self.assertEqual(plug.data_type, "string")
        finally:
            if cmds.attributeQuery("perfExt", type="transform", exists=True):
                cmds.deleteExtension(
                    nodeType="transform", attribute="perfExt", forceDelete=True
                )


class TestCachedAttributeOwner(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_cached_attr_keeps_the_node_it_was_read_from(self):
        cmds.undoInfo(state=True, infinity=True)
        for node_type, attr in (("multiplyDivide", "input1X"), ("transform", "tx")):
            with self.subTest(node_type=node_type):
                PyNode(_mobject(cmds.createNode(node_type)))
                name = cmds.createNode(node_type, name=f"held_{node_type}")
                node = Node(name)
                getattr(node, attr)
                cached = node.find_attr(attr)
                owner  = cached._node
                # the node it was read from, whose attr cache holds it
                self.assertIs(owner, node._dg_node)
                self.assertIsNone(owner.find_attr(attr, data_type="string"))

                cmds.delete(name)
                cmds.createNode(node_type, name=name)
                for func in (str, repr, lambda a: a.get(), lambda a: a.set(3.0)):
                    with self.assertRaises(RuntimeError) as ctx:
                        func(cached)
                    self.assertEqual(str(ctx.exception), f"{name} already deleted!")
                self.assertFalse(cached.node.is_valid)
                self.assertEqual(cmds.getAttr(f"{name}.{attr}"), 0.0)

    def test_cached_attr_of_an_instance_names_the_first_path(self):
        # Historical id: under the owner rule (round 3, D-A) the cached attr is
        # named through the path the node was read from (T2), not the first.
        top = cmds.createNode("transform", name="T1")
        PyNode(_mobject(cmds.createNode("locator", name="S", parent=top)))
        cmds.instance(top, name="T2")
        node = Node("|T2|S")
        node.visibility
        cached = node.find_attr("visibility")
        # named through the path the node was read from
        self.assertEqual(cached._node._mdagpath.fullPathName(), "|T2|S")
        self.assertEqual(str(cached), "T2|S.visibility")


def _named(plug):
    """``plug`` once it has been named, as a plug in use would be."""
    str(plug)
    return plug


def _readded(node, source=None):
    """A plug held on ``node.dd`` across the attr being deleted and re-added,
    connected from ``source`` afterwards if given."""
    held = _named(Plug(f"{node}.dd"))
    cmds.deleteAttr(f"{node}.dd")
    cmds.addAttr(node, ln="dd", at="double")
    if source:
        cmds.connectAttr(source, f"{node}.dd")
    return held


def _undone(node):
    """A plug on a dynamic attr whose creation was undone."""
    cmds.addAttr(node, ln="undone", at="double")
    held = _named(Plug(f"{node}.undone"))
    cmds.undo()
    return held


def _deleted(node):
    """A plug on ``node.tx`` named before the node was deleted."""
    held = _named(Node(node).tx)
    cmds.delete(node)
    return held


def _renamed():
    held = _named(Node("b").tx)
    cmds.rename("b", "renamed")
    return held


def _delete_undone():
    held = _named(Node("b").tx)
    cmds.delete("b")
    cmds.undo()
    return held


def _component():
    """A NURBS surface CV, named after its component, not its MPlug."""
    cmds.nurbsPlane(name="plane")
    return Node("planeShape").cv[1][1]


def _unindexed_child():
    """An Attribute on ``pma.input3D.input3Dx`` without an element index."""
    mobject = _mobject("pma")
    fn      = OpenMaya.MFnDependencyNode(mobject)
    return _base.Attribute(OpenMaya.MPlug(mobject, fn.attribute("input3Dx")))


def _delete_extension():
    cmds.deleteExtension(
        nodeType="plusMinusAverage", attribute="perfConnect", forceDelete=True
    )


def _extension(delete):
    """A plug on the ``perfConnect`` extension attr, deleted if ``delete``."""
    cmds.addExtension(nodeType="plusMinusAverage", longName="perfConnect", at="double")
    held = _named(Node("pma").perfConnect)
    if delete:
        _delete_extension()
    return held


def _instance_scene():
    """``loc`` under ``t1``, instanced under ``t2``, and an ``other`` locator."""
    cmds.createNode("transform", name="t1")
    cmds.createNode("transform", name="t2")
    cmds.createNode("locator", name="loc", parent="t1")
    cmds.parent("t1|loc", "t2", add=True, shape=True)
    cmds.createNode("locator", name="other")
    # a bare ".attr" name resolves on the selection
    cmds.select(clear=True)


def _stale_instance(remove):
    """A plug named through ``t1|loc`` held across that instance being removed
    (``remove``) or its transform deleted: the shape lives on under ``t2``."""
    _instance_scene()
    held = _named(Plug("t1|loc.localPositionX"))
    if remove:
        cmds.parent("t1|loc", removeObject=True, shape=True)
    else:
        cmds.delete("t1")
    return held


def _instance_plug(name):
    """A factory of the plug ``name`` that sets up the instanced scene first."""

    def plug():
        _instance_scene()
        return Plug(name)

    return plug


def _wired_instance_element():
    """``instObjGroups[1]`` of the second instance, named ``t1|loc.instObjGroups``
    (its first instance's element), with that name connected from ``other``."""
    held = _named(Plug("t2|loc.instObjGroups[1]"))
    cmds.connectAttr("other.instObjGroups[0]", str(held))
    return held


def _plug(name):
    return lambda: Plug(name)


# (case, connections made first, source, destination, force, skips the query):
# the source and destination factories run in that order once the scene is set up.
_CONNECT_CASES = (
    ("plain", (), lambda: Node("a").tx, lambda: Node("b").tx, False, True),
    ("attribute", (), _plug("a.tx"), lambda: _base.Attribute("b.tx"), False, True),
    ("connected", [("a.tx", "b.tx")], _plug("a.tx"), _plug("b.tx"), False, False),
    ("connected_forced", [("a.tx", "b.tx")], _plug("a.tx"), _plug("b.tx"), True, False),
    ("occupied", [("a.ty", "b.tx")], _plug("a.tx"), _plug("b.tx"), False, False),
    ("occupied_forced", [("a.ty", "b.tx")], _plug("a.tx"), _plug("b.tx"), True, False),
    ("parent_connected", [("a.t", "b.t")], _plug("a.tx"), _plug("b.tx"), False, True),
    ("child_connected", [("a.tx", "b.tx")], _plug("a.t"), _plug("b.t"), False, True),
    ("reverse", [("b.tx", "a.tx")], _plug("a.tx"), _plug("b.tx"), False, True),
    ("unit_conversion", (), _plug("a.tx"), _plug("b.rx"), False, True),
    ("unit_again", [("a.tx", "b.rx")], _plug("a.tx"), _plug("b.rx"), True, False),
    ("new_element", (), _plug("a.tx"), lambda: Node("pma").input1D[5], False, True),
    ("new_child", (), _plug("a.tx"), _plug("pma.input3D[2].input3Dx"), False, True),
    ("element", [("a.tx", "pma.input1D[0]")], _plug("a.tx"), _plug("pma.input1D[0]"),
     False, False),
    ("array_root", (), _plug("a.tx"), _plug("pma.input1D"), False, True),
    ("alias", (), _plug("a.tx"), _plug("b.bar"), False, True),
    ("alias_wired", [("a.tx", "b.foo")], _plug("a.tx"), _plug("b.bar"), False, False),
    ("alias_source", (), _plug("b.bar"), _plug("a.tx"), False, True),
    ("type_mismatch", (), _plug("a.worldMatrix[0]"), _plug("b.tx"), False, True),
    ("locked", (), _plug("a.tx"), _plug("b.ty"), False, True),
    ("same_plug", (), _plug("a.tx"), _plug("a.tx"), False, True),
    ("string", (), _plug("a.tx"), lambda: "b.tx", False, False),
    ("string_missing", (), _plug("a.tx"), lambda: "b.nope", False, False),
    ("readded", (), _plug("a.tx"), lambda: _readded("b"), False, False),
    ("readded_wired", (), _plug("a.tx"), lambda: _readded("b", "a.tx"), False, False),
    ("readded_source", (), lambda: _readded("b"), _plug("a.tx"), False, False),
    ("undone", (), _plug("a.tx"), lambda: _undone("b"), False, False),
    ("undone_source", (), lambda: _undone("b"), _plug("a.tx"), False, False),
    ("deleted", (), _plug("a.tx"), lambda: _deleted("b"), False, False),
    ("deleted_source", (), lambda: _deleted("b"), _plug("a.tx"), False, False),
    ("renamed", (), _plug("a.tx"), _renamed, False, True),
    ("delete_undone", (), _plug("a.tx"), _delete_undone, False, True),
    ("component", (), _plug("a.t"), _component, False, False),
    ("component_source", (), _component, _plug("a.t"), False, False),
    ("unindexed_child", (), _plug("a.tx"), _unindexed_child, False, False),
    ("extension", (), _plug("a.tx"), lambda: _extension(False), False, True),
    ("extension_gone", (), _plug("a.tx"), lambda: _extension(True), False, False),
    ("instance", (), _plug("a.tx"), _instance_plug("t2|loc.lpx"), False, True),
    ("stale_instance", (), _plug("a.tx"), lambda: _stale_instance(False), False,
     True),
    ("stale_instance_source", (), lambda: _stale_instance(False), _plug("b.tx"),
     False, True),
    ("removed_instance", (), _plug("a.tx"), lambda: _stale_instance(True), False,
     True),
    ("removed_instance_source", (), lambda: _stale_instance(True), _plug("b.tx"),
     False, True),
    ("world_source", (), _plug("a.worldMatrix[0]"), _plug("b.offsetParentMatrix"),
     False, True),
    ("instance_world_source", (), _instance_plug("t2|loc.worldMatrix[1]"),
     _plug("b.offsetParentMatrix"), False, False),
    ("instance_element", (), _instance_plug("other.instObjGroups[0]"),
     _plug("t1|loc.instObjGroups[0]"), False, True),
    ("other_instance_element", (), _instance_plug("other.instObjGroups[0]"),
     _plug("t2|loc.instObjGroups[1]"), False, False),
    ("wired_instance_element", (), _instance_plug("other.instObjGroups[0]"),
     _wired_instance_element, False, False),
)


class TestConnectQuery(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        if cmds.attributeQuery("perfConnect", type="plusMinusAverage", exists=True):
            _delete_extension()
        super().tearDown()

    def _scene(self):
        cmds.file(new=True, force=True)
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.addAttr("b", ln="foo", at="double")
        cmds.aliasAttr("bar", "b.foo")
        cmds.addAttr("b", ln="dd", at="double")
        cmds.setAttr("b.ty", lock=True)

    def _connect(self, case, legacy):
        """What ``source.connect(destination)`` returns or raises, the scene it
        leaves and the isConnected queries it makes, from a fresh scene."""
        _, connections, source, destination, force, _ = case
        self._scene()
        for pair in connections:
            cmds.connectAttr(*pair)
        source      = source()
        destination = destination()
        query       = mock.patch.object(_base, "_names_own_plug", return_value=False)
        with mock.patch.object(cmds, "isConnected", wraps=cmds.isConnected) as probe:
            with query if legacy else contextlib.nullcontext():
                try:
                    outcome = source.connect(destination, force=force)
                except Exception as exc:
                    outcome = (type(exc), str(exc))
        wired = cmds.listConnections(
            cmds.ls(), connections=True, plugs=True, source=False
        ) or []
        state = (
            outcome,
            sorted(cmds.ls()),
            sorted(zip(wired[::2], wired[1::2])),
            cmds.getAttr("pma.input1D", multiIndices=True),
            cmds.getAttr("pma.input3D", multiIndices=True),
        )
        if cmds.attributeQuery("perfConnect", type="plusMinusAverage", exists=True):
            _delete_extension()
        return state, probe.call_count

    def test_connect_matches_the_query_path(self):
        for case in _CONNECT_CASES:
            with self.subTest(case=case[0]):
                fast, fast_queries     = self._connect(case, legacy=False)
                legacy, legacy_queries = self._connect(case, legacy=True)
                self.assertEqual(fast, legacy)
                self.assertEqual(fast_queries, 0 if case[5] else legacy_queries)

    def test_connect_outcomes(self):
        cases   = {case[0]: case for case in _CONNECT_CASES}
        missing = (ValueError, "No object matches name: b.undone")
        # a held path that went stale re-resolves to the surviving instance
        for name, expected in (
            ("plain", None),
            ("connected", None),
            ("readded_wired", None),
            ("undone", missing),
            ("undone_source", missing),
            ("stale_instance", None),
            ("stale_instance_source", None),
            ("removed_instance", None),
            ("removed_instance_source", None),
            ("wired_instance_element", None),
        ):
            with self.subTest(case=name):
                state, _ = self._connect(cases[name], legacy=False)
                self.assertEqual(state[0], expected)
        state, _ = self._connect(cases["plain"], legacy=False)
        self.assertIn(("a.translateX", "b.translateX"), state[2])

    def test_connect_names_each_plug_once(self):
        self._scene()
        names    = []
        original = _base.Attribute.full_name

        def full_name(attr):
            names.append(original.fget(attr))
            return names[-1]

        source      = Node("a").tx
        destination = Node("b").tx
        with mock.patch.object(_base.Attribute, "full_name", property(full_name)):
            with mock.patch.object(cmds, "connectAttr") as connect:
                source.connect(destination)
                self.assertEqual(names, ["b.translateX", "a.translateX"])
                connect.assert_called_once_with(
                    "a.translateX", "b.translateX", force=False
                )
                del names[:]
                source.connect("b.ty", force=True)
                self.assertEqual(names, ["a.translateX"])

    def test_names_own_plug(self):
        self._scene()
        # re-pinned (round 3b review): the extension plug is made first. Its
        # addExtension flushes the undo queue, which freed the attributes of the
        # re-added and undone plugs made before it, and a plug of a freed
        # attribute now raises "already deleted!" when it is named: the
        # subtests are labelled by the str buffer, and a plug is named inside
        # its subtest, since the deleted extension attr's attribute is freed too
        # (naming it read freed memory)
        extension = _extension(True)
        for plug, expected in (
            (Node("a").tx, True),
            (_base.Attribute("pma.input3D[2].input3Dx"), True),
            (Plug("b.bar"), True),
            (Node("pma").input1D, True),
            ("a.tx", False),
            (_component(), False),
            (_unindexed_child(), False),
            (_readded("b"), False),
            (_undone("a"), False),
            (extension, False),
        ):
            with self.subTest(plug=str.__str__(plug)):
                try:
                    str(plug)  # named, as a plug in use would be
                except RuntimeError as exc:
                    self.assertFalse(expected)
                    self.assertTrue(str(exc).endswith("already deleted!"))
                self.assertIs(_base._names_own_plug(plug), expected)

    def test_names_own_plug_on_instances(self):
        self._scene()
        _instance_scene()
        for plug, expected in (
            (Plug("t2|loc.lpx"), True),
            (Node("t2|loc").lpx, True),
            (Plug("a.worldMatrix[0]"), True),
            (Plug("t1|loc.worldMatrix[0]"), True),
            (Plug("t1|loc.instObjGroups[0]"), True),
            (Plug("t2|loc.worldMatrix[1]"), False),
            (Plug("t2|loc.instObjGroups[1]"), False),
            (Plug("a.instObjGroups[3]"), False),
        ):
            with self.subTest(plug=plug.plug.name()):
                self.assertIs(_base._names_own_plug(_named(plug)), expected)
        for remove in (False, True):
            with self.subTest(remove=remove):
                self._scene()
                self.assertIs(_base._names_own_plug(_stale_instance(remove)), False)


def _requery_outcome(plug):
    """`_data_type_outcome` with DGNode's fallback hook querying the type again."""
    with mock.patch.object(dg_node_module, "_keeps_query", return_value=False):
        return _data_type_outcome(plug)


def _connecting_hook(hook):
    """`hook` behind a step that connects src.translate into the attr first."""

    def connecting(self, attr):
        name = attr.full_name
        if not cmds.listConnections(name, source=True, destination=False):
            cmds.connectAttr("src.translate", name, force=True)
        return hook(self, attr)

    return connecting


def _attr_type_queries(probe):
    """`_type_queries` on plugs other than a ``selector``."""
    return [
        call for call in _type_queries(probe) if not call.args[0].endswith(".selector")
    ]


class TestFallbackQueryReuse(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        # drop any class a test registered, then the dispatch it cached
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def _data_type(self, plug):
        """The data type outcome of `plug` and the type queries it made, leaving
        out the ones a choice's ``selector.get()`` makes."""
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            outcome = _data_type_outcome(plug)
        self.assertIsNone(_base._FALLBACK_QUERY)
        return outcome, len(_attr_type_queries(probe))

    def _requery(self, plug):
        """`_data_type` with DGNode's fallback hook querying the type again."""
        with mock.patch.object(dg_node_module, "_keeps_query", return_value=False):
            return self._data_type(plug)

    def test_unresolved_choice_queries_once(self):
        Node.create("transform", name="src")
        pick = Node.create("choice", name="pick")
        cmds.connectAttr("src.translate", "pick.input[1]")
        for selector in (0, 3, 1):
            pick.selector.set(selector)
            for plug in (
                pick.output,
                pick.input[0],
                pick.input[4],
                Plug("pick.input[2]"),
                _base.Attribute("pick.output"),
                _base.Attribute("pick.input[5]"),
            ):
                with self.subTest(selector=selector, plug=str(plug)):
                    resolved = selector == 1 and plug.name == "output"
                    requery  = self._requery(plug)
                    if resolved:
                        self.assertEqual(requery[0], "double3")
                        self.assertEqual(self._data_type(plug), requery)
                    else:
                        self.assertEqual(requery, ("Tdata", 2))
                        self.assertEqual(self._data_type(plug), ("Tdata", 1))
        self.assertEqual(self._data_type(pick.input[1]), self._requery(pick.input[1]))
        self.assertEqual(pick.input[1].data_type, "double3")

    def test_generic_attr_on_plain_node_queries_once(self):
        Node.create("transform", name="src")
        xform = Node.create("transform", name="xform")
        cmds.createNode("network", name="net")
        cmds.addAttr("net", ln="generic", at="typed")
        for plug in (
            xform.specifiedManipLocation,
            _base.Attribute("xform.specifiedManipLocation"),
            Plug("net.generic"),
            _base.Attribute("net.generic"),
        ):
            with self.subTest(plug=str(plug)):
                self.assertEqual(self._requery(plug), ("Tdata", 2))
                self.assertEqual(self._data_type(plug), ("Tdata", 1))
        # a source's type is the answer of the first query, with no hook
        cmds.connectAttr("src.worldMatrix[0]", "net.generic")
        self.assertEqual(self._data_type(Plug("net.generic")), ("matrix", 1))
        self.assertEqual(self._requery(Plug("net.generic")), ("matrix", 1))

    def test_nested_choice_chain(self):
        picks = [Node.create("choice", name=f"pick{i}") for i in range(3)]
        for index in range(2):
            cmds.connectAttr(f"pick{index}.output", f"pick{index + 1}.input[0]")
        # each output queries once; only the innermost reaches DGNode's hook
        self.assertEqual(self._data_type(picks[2].output), ("Tdata", 3))
        self.assertEqual(self._requery(picks[2].output), ("Tdata", 4))
        self.assertEqual(self._data_type(picks[2].input[0]), ("Tdata", 3))
        self.assertEqual(self._requery(picks[2].input[0]), ("Tdata", 4))
        Node.create("transform", name="src")
        cmds.connectAttr("src.worldMatrix[0]", "pick0.input[0]")
        self.assertEqual(
            self._data_type(picks[2].output), self._requery(picks[2].output)
        )
        self.assertEqual(picks[2].output.data_type, "matrix")

    def test_answer_is_only_for_the_running_call(self):
        pick = Node.create("choice", name="pick")
        attr = _base.Attribute("pick.output")
        node = pick._dg_node
        self.assertIsNone(_base._queried_data_type(attr, node))
        with mock.patch.object(_base, "_FALLBACK_QUERY", (attr, "Tdata", node)):
            self.assertEqual(_base._queried_data_type(attr, node), "Tdata")
            self.assertIsNone(
                _base._queried_data_type(_base.Attribute("pick.output"), node)
            )
            self.assertIsNone(_base._queried_data_type(pick.output, node))
            # nor for a hook running on another node than the call's
            other = Node.create("choice", name="other")._dg_node
            self.assertIsNone(_base._queried_data_type(attr, other))
            self.assertIsNone(_base._queried_data_type(attr, None))
        with mock.patch.object(_base, "_FALLBACK_QUERY", (attr, "Tdata", None)):
            self.assertIsNone(_base._queried_data_type(attr, node))
        # the hook called outside data_type queries the type itself
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            self.assertEqual(PyNode("pick")._attr_data_type_fallback(attr), "Tdata")
        self.assertEqual(len(_attr_type_queries(probe)), 1)

    def test_hook_error_restores_state(self):
        pick = Node.create("choice", name="pick")
        with mock.patch.object(
            Choice, "_attr_data_type_fallback", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(self._data_type(pick.output), ((RuntimeError, "boom"), 1))
        self.assertEqual(self._data_type(pick.output), ("Tdata", 1))

    def test_overriding_hooks_query_again(self):
        Node.create("transform", name="src")

        class _ConnectingChoice(Choice):
            CUSTOM_NODE_TYPE = "perfFallbackConnecting"

            # changes the attr, then skips Choice's source lookup
            _attr_data_type_fallback = _connecting_hook(
                DGNode._attr_data_type_fallback
            )

        class _PlainChoice(Choice):
            CUSTOM_NODE_TYPE = "perfFallbackPlain"

        for name, custom in (
            ("connecting", "perfFallbackConnecting"),
            ("plain", "perfFallbackPlain"),
        ):
            cmds.createNode("choice", name=name)
            set_custom_type(name, custom)
        self.assertIs(type(PyNode("connecting")), _ConnectingChoice)
        self.assertIs(type(PyNode("plain")), _PlainChoice)
        self.assertEqual(self._data_type(Plug("connecting.input[2]")), ("double3", 2))
        self.assertEqual(self._data_type(Plug("plain.input[2]")), ("Tdata", 2))

        # a class-level patch of Choice's hook, or of DGNode's under it
        cmds.createNode("choice", name="pick")
        choice_hook = Choice.__dict__["_attr_data_type_fallback"]
        base_hook   = DGNode.__dict__["_attr_data_type_fallback"]

        def forwarding(self, attr):
            return choice_hook(self, attr)

        with mock.patch.object(Choice, "_attr_data_type_fallback", forwarding):
            self.assertEqual(self._data_type(Plug("pick.input[4]")), ("Tdata", 2))
        connecting = _connecting_hook(base_hook)
        with mock.patch.object(DGNode, "_attr_data_type_fallback", connecting):
            self.assertEqual(self._data_type(Plug("pick.input[5]")), ("double3", 2))
        self.assertEqual(self._data_type(Plug("pick.input[6]")), ("Tdata", 1))

    def test_wrapper_and_instance_hooks_query_again(self):
        Node.create("transform", name="src")
        for index in range(6):
            cmds.createNode("network", name=f"net{index}")
            cmds.addAttr(f"net{index}", ln="generic", at="typed")
        base_hook = DGNode._attr_data_type_fallback
        # the hook that runs first changes the attr, then reaches DGNode's hook
        connected = ("double3", 2)

        # re-pinned (round 4a M4, C8): the Node wrapper and its forwarding hook
        # are gone, so its two blocks patch and subclass the node class itself.
        # A class-level patch of DGNode's hook
        patched = mock.patch.object(
            DGNode, "_attr_data_type_fallback", _connecting_hook(base_hook)
        )
        with patched:
            self.assertEqual(self._data_type(Plug("net0.generic")), connected)

        # a DGNode subclass overriding the hook
        class _ConnectingNode(DGNode):
            _attr_data_type_fallback = _connecting_hook(base_hook)

        plug = Plug("net1.generic")
        plug.__dict__["_node"] = _ConnectingNode("net1")
        self.assertEqual(self._data_type(plug), connected)

        # a hook set on one node instance, owned directly or through a Node
        attr = _base.Attribute("net2.generic")
        attr._node = PyNode("net2")
        plug = _named(Plug("net3.generic"))
        for owned, owner in ((attr, attr._node), (plug, plug.node._dg_node)):
            with self.subTest(owner=type(owned).__name__):
                hook     = _connecting_hook(base_hook)
                instance = mock.patch.object(
                    owner,
                    "_attr_data_type_fallback",
                    lambda attr, owner=owner: hook(owner, attr),
                    create=True,
                )
                with instance:
                    self.assertEqual(self._data_type(owned), connected)
                self.assertNotIn("_attr_data_type_fallback", vars(owner))

        # unpatched, both kinds of owner reuse the query again
        attr = _base.Attribute("net4.generic")
        attr._node = PyNode("net4")
        self.assertEqual(self._data_type(attr), ("Tdata", 1))
        self.assertEqual(self._data_type(Plug("net5.generic")), ("Tdata", 1))

    def test_held_plugs_across_delete_undo_rename_reuse(self):
        cmds.undoInfo(state=True, infinity=True)
        Node.create("transform", name="src")
        pick = Node.create("choice", name="pick")
        cmds.connectAttr("src.translate", "pick.input[1]")
        held = [
            pick.output,
            pick.input[0],
            _base.Attribute("pick.input[3]"),
            Plug("pick.input[1]"),
        ]
        for plug in held:
            str(plug)

        def rename():
            cmds.rename("pick", "pick2")

        def reuse():
            cmds.delete("pick2")
            cmds.createNode("transform", name="pick2")
            cmds.createNode("choice", name="pick")

        steps = (
            ("live", lambda: None),
            ("deleted", lambda: cmds.delete("pick")),
            ("undone", cmds.undo),
            ("renamed", rename),
            ("reused", reuse),
        )
        seen = {}
        for label, step in steps:
            step()
            for index, plug in enumerate(held):
                with self.subTest(step=label, plug=index):
                    # a freed node (new scene) is named from freed memory
                    self.assertFalse(_holds_freed_node(plug))
                    outcome = _data_type_outcome(plug)
                    self.assertEqual(outcome, _requery_outcome(plug))
                    self.assertIsNone(_base._FALLBACK_QUERY)
                    seen[label, index] = outcome
        live = ["Tdata", "Tdata", "Tdata", "double3"]
        for label in ("live", "undone", "renamed"):
            self.assertEqual([seen[label, index] for index in range(4)], live)
        self.assertEqual(seen["deleted", 0], (RuntimeError, "pick already deleted!"))
        self.assertEqual(seen["reused", 2], (RuntimeError, "pick2 already deleted!"))
        cmds.file(new=True, force=True)
        cmds.createNode("choice", name="pick")
        self.assertEqual(self._data_type(Plug("pick.input[0]")), ("Tdata", 1))


def _canonical_calls(wrapper, mobject):
    """``_wrapper_is_canonical``'s answer, and the fn sets and cast names it builds."""
    fn_sets = mock.Mock(side_effect=OpenMaya.MFnDependencyNode)
    names   = mock.Mock(side_effect=_mobject_to_str)
    with mock.patch.object(_base.OpenMaya, "MFnDependencyNode", fn_sets):
        with mock.patch.object(_base, "_mobject_to_str", names):
            result = _base._wrapper_is_canonical(wrapper, mobject)
    return result, fn_sets.call_count, names.call_count


def _dag_path(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDagPath(0)


class TestCanonicalWrapperCheck(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        _base._CANONICAL_KIND.clear()
        super().tearDown()

    def _wrapper(self, name):
        """A wrapper cast from the MObject of ``name``, and that MObject."""
        mobject = _mobject(name)
        return PyNode(mobject), mobject

    def test_wrappers_use_their_own_fn_set_and_path(self):
        top    = cmds.createNode("transform", name="top")
        points = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)]
        curve  = cmds.curve(point=points, name="crv")
        for name in (
            cmds.createNode("multiplyDivide"),
            cmds.createNode("choice"),
            cmds.createNode("transform", name="leaf", parent=top),
            cmds.createNode("joint"),
            cmds.createNode("locator", parent=top),
            cmds.listRelatives(curve, shapes=True, fullPath=True)[0],
        ):
            wrapper, mobject = self._wrapper(name)
            with self.subTest(node=wrapper.name, cls=type(wrapper).__name__):
                fresh = _mobject(name)
                self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))
                self.assertEqual(_canonical_calls(wrapper, fresh), (True, 0, 0))
                self.assertFalse(_base._wrapper_is_canonical(wrapper, _mobject(top)))
                self.assertFalse(
                    _base._wrapper_is_canonical(wrapper, OpenMaya.MObject())
                )

    def test_dag_hierarchy_edits_keep_the_cast_name(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="P")
        cmds.createNode("transform", name="Q")
        wrapper, mobject = self._wrapper(
            cmds.createNode("transform", name="T", parent="P")
        )
        def same_short_name():
            cmds.createNode("transform", name="T", parent="Q")

        for label, step, expected in (
            ("rename parent", lambda: cmds.rename("P", "P2"), "T"),
            ("reparent", lambda: cmds.parent("P2|T", "Q"), "T"),
            ("to world", lambda: cmds.parent("Q|T", world=True), "T"),
            ("same short name", same_short_name, "|T"),
            ("rename", lambda: cmds.rename("|T", "T3"), "T3"),
            ("reparent under the other T", lambda: cmds.parent("T3", "Q|T"), "T3"),
        ):
            step()
            with self.subTest(step=label):
                self.assertEqual(wrapper.name, expected)
                self.assertEqual(_mobject_to_str(mobject), expected)
                self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))

        cmds.rename("T3", "abcdefabcdefabcdefabcdefabcdefab")
        self.assertEqual(_canonical_calls(wrapper, mobject), (False, 0, 0))
        cmds.undo()
        cmds.delete("P2")
        self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))
        cmds.delete("Q")
        self.assertEqual(_canonical_calls(wrapper, mobject), (False, 0, 0))
        cmds.undo()
        self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))
        self.assertEqual(wrapper.name, "T3")

    def test_instanced_and_underworld_nodes_compare_the_cast_name(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        PyNode(_mobject(cmds.createNode("locator", name="S", parent="T1")))
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        first   = PyNode(_dag_path("|T1|S"))
        second  = PyNode(_dag_path("|T2|S"))
        mobject = _mobject("|T1|S")
        self.assertEqual(_canonical_calls(first, mobject), (True, 0, 1))
        self.assertEqual(_canonical_calls(second, mobject), (False, 0, 1))

        # the path of the removed instance is invalid, the other is the only one:
        # naming the wrapper re-resolves its path to it, so it names that path
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(second.name, "S")
        self.assertEqual(_canonical_calls(second, mobject), (True, 0, 0))
        self.assertEqual(_canonical_calls(first, mobject), (True, 0, 0))
        # undoing the removal makes the path the wrapper was taken through valid
        # again, and naming the wrapper takes it again (as at d6ad8b2)
        cmds.undo()
        self.assertEqual(_canonical_calls(second, mobject), (False, 0, 1))
        self.assertEqual(second.long_name, "|T2|S")
        self.assertEqual(_canonical_calls(first, mobject), (True, 0, 1))

        plane = cmds.nurbsPlane(name="plane")[0]
        cmds.curveOnSurface(plane, uv=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.2)])
        surface    = cmds.listRelatives(plane, shapes=True, fullPath=True)[0]
        underworld = [
            name
            for name in cmds.listRelatives(surface, allDescendents=True, fullPath=True)
            if cmds.nodeType(name) == "nurbsCurve"
        ]
        self.assertEqual(len(underworld), 1)
        wrapper, mobject = self._wrapper(underworld[0])
        result, _, names = _canonical_calls(wrapper, mobject)
        self.assertEqual(names, 1)
        self.assertEqual(result, wrapper.name == _mobject_to_str(mobject))

    def test_deleted_and_freed_nodes(self):
        cmds.undoInfo(state=True, infinity=True)
        for node_type in ("multiplyDivide", "transform"):
            with self.subTest(node_type=node_type):
                name             = cmds.createNode(node_type)
                wrapper, mobject = self._wrapper(name)
                cmds.delete(name)
                self.assertEqual(_canonical_calls(wrapper, mobject), (False, 0, 0))
                cmds.undo()
                self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))
                cmds.delete(name)
                cmds.createNode(node_type, name=name)
                self.assertFalse(_base._wrapper_is_canonical(wrapper, mobject))
                self.assertFalse(_base._wrapper_is_canonical(wrapper, _mobject(name)))
        wrappers = [PyNode(_mobject(cmds.createNode(t))) for t in ("choice", "joint")]
        cmds.file(new=True, force=True)
        for wrapper in wrappers:
            self.assertFalse(_base._wrapper_is_canonical(wrapper, wrapper._mobject))

    def test_custom_type_attr_or_alias_added_later(self):
        for node_type, attr in (("multiplyDivide", "input1X"), ("transform", "tx")):
            with self.subTest(node_type=node_type):
                custom           = cmds.createNode(node_type)
                wrapper, mobject = self._wrapper(custom)
                cmds.addAttr(custom, longName=CUSTOM_TYPE_ATTR, dataType="string")
                self.assertEqual(_canonical_calls(wrapper, mobject), (False, 0, 0))
                aliased          = cmds.createNode(node_type)
                wrapper, mobject = self._wrapper(aliased)
                cmds.aliasAttr(CUSTOM_TYPE_ATTR, f"{aliased}.{attr}")
                self.assertEqual(_canonical_calls(wrapper, mobject), (False, 0, 0))
                cmds.aliasAttr(f"{aliased}.{CUSTOM_TYPE_ATTR}", remove=True)
                self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))

    def test_class_checks_run_once_per_class(self):
        wrappers = [
            self._wrapper(cmds.createNode(node_type))
            for node_type in (
                "multiplyDivide",
                "plusMinusAverage",
                "transform",
                "transform",
            )
        ]
        _base._CANONICAL_KIND.clear()
        checks = mock.Mock(side_effect=_base._canonical_kind)
        with mock.patch.object(_base, "_canonical_kind", checks):
            for _ in range(3):
                for wrapper, mobject in wrappers:
                    self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))
        self.assertEqual(checks.call_count, 2)
        self.assertEqual(
            _base._CANONICAL_KIND,
            {DGNode: _base._NAMED_DG, Transform: _base._NAMED_DAG_PATH},
        )

    def test_class_attribute_changes_are_seen(self):
        class _Probe(Transform):
            NATIVE_NODE_TYPE = "transform"

        wrapper, mobject = self._wrapper(cmds.createNode("transform"))
        self.assertIs(type(wrapper), _Probe)
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))

        _Probe.CUSTOM_NODE_TYPE = "perfCanonicalProbe"
        self.assertFalse(_base._wrapper_is_canonical(wrapper, mobject))
        del _Probe.CUSTOM_NODE_TYPE
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))

        passing = classmethod(lambda cls, *args, **kwargs: True)
        with mock.patch.object(_Probe, "is_type", passing):
            self.assertFalse(_base._wrapper_is_canonical(wrapper, mobject))
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))
        with mock.patch.object(DGNode, "_cache_api1_objects", lambda self, name: None):
            self.assertFalse(_base._wrapper_is_canonical(wrapper, mobject))
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))

        # a fn_set of its own makes the check compare the cast's name
        same_fn_set = property(lambda self: self._fn_set)
        with mock.patch.object(_Probe, "fn_set", same_fn_set):
            self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 1))
            self.assertEqual(_base._CANONICAL_KIND[_Probe], _base._NAMED_DAG)
        self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))
        with mock.patch.object(_Probe, "name", DGNode.__dict__["name"]):
            self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 1))
            self.assertEqual(_base._CANONICAL_KIND[_Probe], _base._NAMED_DG)
        self.assertEqual(_canonical_calls(wrapper, mobject), (True, 0, 0))

        # so does a new class, which may change the dispatch
        self.assertIn(_Probe, _base._CANONICAL_KIND)

        class _Other(DGNode):
            NATIVE_NODE_TYPE = "perfCanonicalOther"

        self.assertEqual(_base._CANONICAL_KIND, {})

    def test_class_with_a_plain_base_is_checked_every_time(self):
        class _Plain:
            pass

        class _Mixed(_Plain, Transform):
            NATIVE_NODE_TYPE = "transform"

        wrapper, mobject = self._wrapper(cmds.createNode("transform"))
        self.assertIs(type(wrapper), _Mixed)
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))
        self.assertNotIn(_Mixed, _base._CANONICAL_KIND)
        _Plain.is_type = classmethod(lambda cls, *args, **kwargs: True)
        try:
            self.assertFalse(_base._wrapper_is_canonical(wrapper, mobject))
        finally:
            del _Plain.is_type
        self.assertTrue(_base._wrapper_is_canonical(wrapper, mobject))

    def test_fn_set_override_compares_the_cast_name(self):
        class _OwnFnSet(Transform):
            NATIVE_NODE_TYPE = "transform"

            @property
            def fn_set(self):
                self.ensure_valid()
                path = OpenMaya.MDagPath.getAPathTo(self._mobject)
                return OpenMaya.MFnDagNode(path)

        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        PyNode(_mobject(cmds.createNode("transform", name="S", parent="T1")))
        cmds.parent("T1|S", "T2", add=True, relative=True)
        # the name comes from the first path, the fn set from the second
        wrapper = PyNode(_dag_path("|T2|S"))
        self.assertIs(type(wrapper), _OwnFnSet)
        self.assertEqual(wrapper.name, "T1|S")
        self.assertEqual(_canonical_calls(wrapper, _mobject("|T1|S")), (True, 0, 1))
        self.assertEqual(_base._CANONICAL_KIND[_OwnFnSet], _base._NAMED_DAG)

    def test_other_objects_are_not_canonical(self):
        name    = cmds.createNode("multiplyDivide")
        mobject = _mobject(name)
        PyNode(mobject)
        # re-pinned (round 4a M4, C8): Node(name) is the canonical typed node
        # now (the cast itself), so it left the tuple
        for other in (None, name, 3, Plug(f"{name}.input1X")):
            with self.subTest(other=type(other).__name__):
                self.assertFalse(_base._wrapper_is_canonical(other, mobject))
                self.assertNotIn(type(other), _base._CANONICAL_KIND)


def _result(func):
    """What ``func()`` returns, or the type and message it raises."""
    try:
        return ("ok", func())
    except Exception as exc:
        return (type(exc), str(exc))


def _name_forwards(func):
    """What ``func()`` returns or raises, and the reads of the ``DGNode`` /
    ``DAGNode`` ``name`` property it makes."""
    # re-pinned (round 4a M4, C8): there is no Node.__getattr__ forwarding to
    # count any more; a plug's owner is the node, whose name property is read.
    # The constructor parts the casts compare the property with are bound first,
    # so they never capture the counting ones
    _base._construct_checked_type(DGNode, "")
    _base._canonical_kind(DGNode)
    reads   = []
    patches = []
    for cls in (DGNode, DAGNode):
        prop = cls.__dict__["name"]

        def counting(node, prop=prop):
            # a constructor's type check names the node it builds (the owner's
            # cast), which is not a read of the plug's name
            if sys._getframe(1).f_code.co_name != "__init__":
                reads.append(node)
            return prop.fget(node)

        patches.append(mock.patch.object(cls, "name", property(counting)))
    with patches[0], patches[1]:
        result = _result(func)
    return result, len(reads)


def _name_raises():
    """A plug whose Node wraps a DGNode whose ``name`` property raises, so the
    lookup falls through to the node's ``name`` Maya attr instead."""

    class _NameRaises(Transform):
        NATIVE_NODE_TYPE = "perfFullNameProbe"

        @property
        def name(self):
            raise AttributeError("no name")

    cmds.createNode("transform", name="w")
    cmds.addAttr("w", longName="name", dataType="string")
    wrapper = object.__new__(_NameRaises)
    wrapper.__dict__.update(PyNode(_mobject("w")).__dict__)
    plug = Plug("w.tx")
    plug.__dict__["_node"] = Node(wrapper)
    return plug


def _container_owner():
    ctr  = cmds.container(name="ctr")
    plug = Plug(f"{ctr}.blackBox")
    plug.__dict__["_node"] = Container(ctr)
    return plug


def _reused():
    held = _named(Node("b").tx)
    cmds.delete("b")
    cmds.createNode("transform", name="b")
    return held


def _underworld():
    plane = cmds.nurbsPlane(name="plane")[0]
    cmds.curveOnSurface(plane, uv=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.2)])
    surface = cmds.listRelatives(plane, shapes=True, fullPath=True)[0]
    curve   = [
        name
        for name in cmds.listRelatives(surface, allDescendents=True, fullPath=True)
        if cmds.nodeType(name) == "nurbsCurve"
    ]
    return Node(curve[0]).visibility


def _instanced():
    cmds.createNode("transform", name="T1")
    cmds.createNode("transform", name="T2")
    cmds.createNode("locator", name="S", parent="T1")
    cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
    return Node("|T2|S").visibility


def _no_wrapped_node():
    """A plug whose node object was never constructed."""
    # re-pinned (round 4a M4, C8): no wrapper; a half-built DGNode instead
    plug = Node("b").tx
    plug.__dict__["_node"] = DGNode.__new__(DGNode)
    return plug


_DELETED_B = (RuntimeError, "b already deleted!")

# (case, plug factory, full_name or (error type, message), the reads of the
# DGNode / DAGNode name property each full_name makes, and whether the fn-set
# oracle (`_fn_set_full_name`) names the plug too). No case holds a freed node:
# its "already deleted" message would be read from freed memory.
# Re-pinned (round 4a M4, C8): the rows counted the name lookups the Node
# wrapper forwarded (0 on the fast path, 1 through the forwarding); with the
# wrapper gone every owner is the node itself and full_name reads its name
# property once. A component is named after its component (not the oracle's
# MPlug name); a node whose name property raises is named by its fn set.
_FULL_NAME_CASES = (
    ("node_attr", lambda: Node("a").tx, "a.translateX", 1, True),
    ("compound", lambda: Node("a").t, "a.translate", 1, True),
    ("child", lambda: Node("a").t.tx, "a.translateX", 1, True),
    ("child_index", lambda: Node("a").t[1], "a.translateY", 1, True),
    ("string", _plug("a.tx"), "a.translateX", 1, True),
    ("mplug_child", _unindexed_child, "pma.input3D[-1].input3Dx", 1, True),
    ("new_element", lambda: Node("pma").input1D[3], "pma.input1D[3]", 1, True),
    ("element_child", _plug("pma.input3D[2].input3Dx"), "pma.input3D[2].input3Dx", 1,
     True),
    ("array_root", lambda: Node("pma").input1D, "pma.input1D", 1, True),
    ("alias", _plug("b.bar"), "b.bar", 1, True),
    ("alias_long", lambda: Node("b").foo, "b.bar", 1, True),
    ("world_matrix", lambda: Node("a").worldMatrix[0], "a.worldMatrix", 1, True),
    ("same_short_name", lambda: Node("|g1|dup").tx, "g1|dup.translateX", 1, True),
    ("namespace", lambda: Node("ns:n").tx, "ns:n.translateX", 1, True),
    ("shape", lambda: Node("locShape").localPositionX, "locShape.localPositionX", 1,
     True),
    ("choice", lambda: Node("pick").input[0], "pick.input[0]", 1, True),
    ("joint", lambda: Node("jnt").jointOrientX, "jnt.jointOrientX", 1, True),
    ("underworld", _underworld, "planeShape->curveShape1.visibility", 1, True),
    ("instanced", _instanced, "T2|S.visibility", 1, True),
    ("renamed", _renamed, "renamed.translateX", 1, True),
    ("delete_undone", _delete_undone, "b.translateX", 1, True),
    ("readded", lambda: _readded("b"), "b.dd", 1, True),
    ("undone", lambda: _undone("b"), "b.undone", 1, True),
    ("extension", lambda: _extension(False), "pma.perfConnect", 1, True),
    # re-pinned (round 3b review): deleteExtension frees the attribute, which the
    # name was read from ("pma."); the plug keeps a handle of it and raises
    ("extension_gone", lambda: _extension(True),
     (RuntimeError, "pma.perfConnect already deleted!"), 1, False),
    ("deleted", lambda: _deleted("b"), _DELETED_B, 1, False),
    ("reused", _reused, _DELETED_B, 1, False),
    ("component", _component, "planeShape.cv[1][1]", 1, False),
    ("attribute", lambda: _base.Attribute("b.tx"), "b.translateX", 1, True),
    ("container_owner", _container_owner, "ctr.blackBox", 1, True),
    ("name_raises", _name_raises, "w.translateX", 0, True),
    # the half-built node's name property raises AttributeError, which Python
    # retries through DGNode.__getattr__ ("name"), and full_name's through
    # Plug.__getattr__, whose container query names the node once more
    ("no_wrapped_node", _no_wrapped_node, (AttributeError, "name"), 2, False),
)


def _fn_set_full_name(plug):
    """The name of ``plug`` read without ``full_name``, ``name`` or ``alias``: its
    owner's fn set name (the cast of its MPlug's node when it holds none) and the
    MPlug's partial name."""
    owner = plug.__dict__["_node"] or PyNode(plug.__dict__["_mplug"].node())
    fn    = owner.__dict__["_fn_set"]
    name  = fn.partialPathName() if isinstance(fn, OpenMaya.MFnDagNode) else fn.name()
    attr  = plug.__dict__["_mplug"].partialName(False, False, False, True, False, True)
    return f"{name}.{attr}"


class TestFullNameReadsTheWrappedNode(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        if cmds.attributeQuery("perfConnect", type="plusMinusAverage", exists=True):
            _delete_extension()
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        _base._CANONICAL_KIND.clear()
        super().tearDown()

    def _scene(self):
        cmds.file(new=True, force=True)
        cmds.undoInfo(state=True, infinity=True)
        if cmds.attributeQuery("perfConnect", type="plusMinusAverage", exists=True):
            _delete_extension()
        for name in ("a", "b", "g1", "g2"):
            cmds.createNode("transform", name=name)
        cmds.createNode("transform", name="dup", parent="g1")
        cmds.createNode("transform", name="dup", parent="g2")
        cmds.createNode("locator", name="locShape", parent="a")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("choice", name="pick")
        cmds.createNode("joint", name="jnt")
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:n")
        cmds.addAttr("b", ln="foo", at="double")
        cmds.aliasAttr("bar", "b.foo")
        cmds.addAttr("b", ln="dd", at="double")

    def _full_name(self, factory):
        """What ``full_name`` and ``str`` give twice on the plug ``factory`` builds
        in a fresh scene, the class of the owner the plug then holds and the name
        property reads made."""
        self._scene()
        plug = factory()
        results, reads = _name_forwards(
            lambda: [
                _result(lambda: plug.full_name),
                _result(lambda: str(plug)),
                _result(lambda: plug.full_name),
                _result(lambda: str(plug)),
            ]
        )
        return results, type(plug.__dict__["_node"]), reads

    def test_node_class_is_registered(self):
        # re-pinned (round 4a M4, C8): no wrapper class is registered any more;
        # the DSL Node is the root of the node classes and its factory returns
        # the typed node PyNode casts
        self.assertTrue(issubclass(DGNode, Node))
        self.assertFalse(hasattr(_base, "_NODE_WRAPPER_CLASS"))
        cmds.createNode("transform", name="t")
        self.assertIs(type(Node("t")), type(PyNode("t")))

    def test_full_name_matches_the_forwarding(self):
        # re-pinned (round 4a M4, C8): there is no forwarding to compare with;
        # full_name reads the owner's name property once, and the oracle naming
        # the plug from the owner's fn set and the MPlug agrees with it
        for case, factory, expected, reads, oracle in _FULL_NAME_CASES:
            with self.subTest(case=case):
                results, owner_cls, count = self._full_name(factory)
                if not isinstance(expected, tuple):
                    expected = ("ok", expected)
                self.assertEqual(results, ("ok", [expected] * 4))
                self.assertTrue(issubclass(owner_cls, Node))
                self.assertEqual(count, 4 * reads)
                if oracle:
                    self._scene()
                    self.assertEqual(_fn_set_full_name(factory()), expected[1])

    def test_cases_hold_no_freed_node(self):
        # a freed node's message is arbitrary and naming it can crash Maya, so a
        # case may delete its node, to the undo queue, but not free it
        for case, factory, *_ in _FULL_NAME_CASES:
            with self.subTest(case=case):
                self._scene()
                self.assertFalse(_holds_freed_node(factory()))

    def test_full_name_matches_the_previous_formula(self):
        for case, factory, expected, _, _ in _FULL_NAME_CASES:
            if not isinstance(expected, str):
                continue
            with self.subTest(case=case):
                self._scene()
                plug = factory()
                # a node whose name property raises is named by its fn set, not
                # by its Maya attr of that name
                if type(plug) is not ComponentPlug and case != "name_raises":
                    self.assertEqual(plug.full_name, f"{plug.node.name}.{plug.alias}")
                self.assertEqual(plug.full_name, expected)

    def test_referenced_node(self):
        folder = tempfile.mkdtemp(prefix="rig_full_name_ref_")
        path   = os.path.join(folder, "rig_full_name_ref.ma")
        try:
            cmds.file(new=True, force=True)
            cmds.createNode("transform", name="refT")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            # re-pinned (round 4a M4, C8): the forwarded variant is gone; the
            # fn-set oracle names the plug without reading the name property
            for oracle in (False, True):
                with self.subTest(oracle=oracle):
                    cmds.file(new=True, force=True)
                    cmds.file(path, reference=True, namespace="ref")
                    plug = _named(Node("ref:refT").tx)
                    if oracle:
                        loaded = _name_forwards(lambda: _fn_set_full_name(plug))
                    else:
                        loaded = _name_forwards(lambda: plug.full_name)
                    self.assertEqual(
                        loaded, (("ok", "ref:refT.translateX"), int(not oracle))
                    )
                    # unloading frees the node, whose name is then read from freed
                    # memory, so the plug is not named again
                    reference = cmds.referenceQuery(path, referenceNode=True)
                    cmds.file(unloadReference=reference)
                    self.assertTrue(_holds_freed_node(plug))
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_plug_keeps_the_attribute_property(self):
        # `_names_own_plug` and the tests that wrap `Attribute.full_name` still see
        # the one property on Plug; ComponentPlug names itself
        self.assertIs(Plug.full_name, _base.Attribute.full_name)
        self.assertIsNot(ComponentPlug.full_name, _base.Attribute.full_name)
