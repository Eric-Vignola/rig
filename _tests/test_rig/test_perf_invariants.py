"""Invariants the performance work must keep: ``PyNode`` dispatch through the
per-type class cache picks the same class, raises the same errors and follows
new class registrations like the name-based path does."""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes import DGNode, Joint, PyNode, Transform
from rig.nodetypes import _base
from rig.nodetypes._base import (
    CUSTOM_TYPE_ATTR,
    _mobject_to_str,
    _pynode_legacy_tail,
    get_custom_type,
    is_valid_maya_uid,
    set_custom_type,
)
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
