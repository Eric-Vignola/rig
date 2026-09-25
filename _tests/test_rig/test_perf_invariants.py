"""Invariants the performance work must keep: ``PyNode`` dispatch through the
per-type class cache picks the same class, raises the same errors and follows
new class registrations like the name-based path does. A cast that skips the
type check builds the same wrapper as the constructor, and the node a plug
caches when it is first named behaves as before. ``Attribute.set`` passes the
same ``type`` argument, and raises the same errors, when it skips the type
query."""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import InjectionError, Node, Plug, lock
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
        self.assertEqual(str(Node("|T2|S").visibility), "T1|S.visibility")
        self.assertEqual(Node("|T2|S").visibility.node._dg_node.long_name, "|T1|S")

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
                with self.assertRaises((TypeError, ValueError)):
                    str(fresh)

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
                self.assertEqual(hash(plug), hash((hash(node_name), alias)))


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
                for action in (lambda: plug.set(0.25), lambda: plug << lock):
                    with self.assertRaises(RuntimeError) as ctx:
                        action()
                    self.assertNotIsInstance(ctx.exception, InjectionError)
                    self.assertEqual(str(ctx.exception), message)

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

        cmds.undoInfo(state=True, infinity=True)
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="dbl", at="double")
        cmds.addAttr(net, ln="cmp", at="double3")
        for axis in "XYZ":
            cmds.addAttr(net, ln=f"cmp{axis}", at="double", p="cmp")
        loc = Node.create("transform", name="loc")
        cmds.connectAttr("net.dbl", "loc.tx")
        cases = [
            (lambda: loc.tx.set(1.0), True),
            (lambda: Plug("net.dbl").set("text"), True),
            (lambda: Plug("net.cmp").set(1.0), True),
            (lambda: Plug("net.cmpX").set(1.0, 2.0), True),
            (lambda: Plug("net.dbl").set(), True),
        ]
        gone = Plug("net.cmpY")
        str(gone)
        cmds.delete(net)
        cases.append((lambda: gone.set(1.0), True))
        for index, (action, raises) in enumerate(cases):
            with self.subTest(case=index):
                fast = outcome(action)
                with mock.patch.object(
                    _base.Attribute, "_is_fixed_kind_outside_array", return_value=False
                ):
                    legacy = outcome(action)
                self.assertEqual(fast, legacy)
                self.assertEqual(fast is not None, raises)
