"""Invariants the performance work must keep: ``PyNode`` dispatch through the
per-type class cache picks the same class, raises the same errors and follows
new class registrations like the name-based path does. A cast that skips the
type check builds the same wrapper as the constructor, and the node a plug
caches when it is first named behaves as before. ``Attribute.set`` passes the
same ``type`` argument, and raises the same errors, when it skips the type
query. ``Attribute.data_type`` shares a fixed-kind attr's type across nodes of
a type, and keeps querying every type that can change. ``Attribute.__init__``
leaves the same instance state without going through ``Plug.__setattr__``. A
plug reuses the wrapper of the Node or parent plug it came from only where a
fresh cast would rebuild that wrapper unchanged, so it reports the same owner
and raises the same errors."""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Container, InjectionError, Node, Plug, lock
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
from rig._internal import plug as plug_module
from rig._internal.plug import ComponentPlug
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
        """The ``PyNode`` casts ``Plug.node`` makes while ``func`` runs."""
        cast = mock.Mock(side_effect=PyNode)
        with mock.patch.object(plug_module, "PyNode", cast):
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
                self.assertIs(type(plug.node), Node)
                self.assertIsNot(plug.node, node)
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
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        node = Node(cmds.createNode("multiplyDivide"))
        plug = node.input1X
        # the lookup casts the owner of the attr it caches, which warms the type
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        self.assertEqual(self._casts(lambda: str(plug)), 1)
        self.assertEqual(self._casts(lambda: str(node.input1Y)), 0)

    def test_instanced_shape_plug_not_seeded(self):
        top = cmds.createNode("transform", name="T1")
        PyNode(_mobject(cmds.createNode("transform", name="S", parent=top)))
        other = cmds.createNode("transform", name="T2")
        cmds.parent("T1|S", other, add=True, relative=True)
        for plug in (Node("|T2|S").visibility, Node("|T2|S").t[0]):
            with self.subTest(plug=plug.name):
                self.assertEqual(self._casts(lambda: str(plug)), 1)
                self.assertTrue(str(plug).startswith("T1|S."))
                self.assertEqual(plug.node._dg_node.long_name, "|T1|S")
        self.assertEqual(self._casts(lambda: str(Node("|T1|S").visibility)), 0)

    def test_user_chosen_class_not_seeded(self):
        plug = Node(Transform(self._known_type("joint"))).tx
        self.assertEqual(self._casts(lambda: str(plug)), 1)
        self.assertIs(type(plug.node._dg_node), Joint)

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
        PyNode(_mobject(cmds.container(name="box0")))
        ctn  = Container(cmds.container(name="box"))
        plug = ctn.blackBox
        self.assertEqual(self._casts(lambda: str(plug)), 0)
        self.assertIs(type(plug.node), Node)
        self.assertIsNot(plug.node, ctn)
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
                self.assertEqual(self._casts(lambda: str(plug)), 1)
                self.assertIs(type(plug.node._dg_node), _Probe)
                self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))

    def test_deleted_node_error_unchanged(self):
        cmds.undoInfo(state=True, infinity=True)
        for node_type, attr, parent in (
            ("multiplyDivide", "input1X", "input1"),
            ("transform", "tx", "translate"),
        ):
            with self.subTest(node_type=node_type):
                name  = self._known_type(node_type)
                node  = Node(name)
                plugs = (getattr(node, attr), getattr(node, parent)[1])
                cmds.delete(name)
                for plug in plugs:
                    self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))
                    self.assertEqual(_owner(plug)[0], "error")
                cmds.undo()
                for plug in plugs:
                    self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))
                    self.assertEqual(_owner(plug)[0], "ok")

                # the cast resolves the name, so it finds a newer node of that name
                plugs = (getattr(node, attr), getattr(node, parent)[1])
                cmds.delete(name)
                cmds.createNode(node_type, name=name)
                for plug in plugs:
                    self.assertEqual(_owner(plug), _owner(Plug(plug.plug)))
                    self.assertEqual(_owner(plug)[0], "ok")


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
                self.assertEqual(
                    _wrapper_state(owner), _wrapper_state(PyNode(_mobject(name)))
                )
                self.assertIsNot(owner, node._dg_node)
                self.assertIsNot(owner.mobject, node._dg_node.mobject)
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
        top = cmds.createNode("transform", name="T1")
        PyNode(_mobject(cmds.createNode("locator", name="S", parent=top)))
        cmds.instance(top, name="T2")
        node = Node("|T2|S")
        node.visibility
        cached = node.find_attr("visibility")
        self.assertEqual(cached._node._mdagpath.fullPathName(), "|T1|S")
        self.assertEqual(str(cached), "T1|S.visibility")
