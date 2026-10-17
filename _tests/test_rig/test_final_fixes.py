"""The fix round after the features steps (2026-10-16): one class per finding, each
failing on e69bae0 (the negative control).

* `TestFreedPlugWalk`: walking every attribute of a plug of a node a new scene freed
  never reads its MPlug (the private properties raised nothing and read freed memory:
  an access violation).
* `TestUnderscoreProbeOnADeletedNode`: a '_' probe (hasattr, numpy) on a compound plug
  or a bare Attribute of a node deleted to the undo queue is an AttributeError (user
  decision 2026-10-16, must-fix); any other name keeps "already deleted!".
* `TestLateClassAssignment`: only a mock is kept as it is; any other callable that is no
  function follows the class-body rule, as it does in a class body.
* `TestPlugRuleInTheMetaclass`: a plug subclass cannot switch the class-body rule off.
* `TestDecoratedMethods`: a method-making decorator is refused with its own message.
* `TestBarePredicateCompare`: ``node.is_valid == True`` raises as ``bool()`` does; a
  bare predicate copies.
* `TestAliasOfAnElementChild`: an alias of a child below an element names that child at
  plug level too.
* `TestSetNamespaceRefusal`: the locked-node refusal names set_namespace.
* `TestHandleOldNames`: the scene handles' old member names name the method now.
* `TestTokenVerbsArePrivate`: a token's inject / query / contains are private (one way
  only: the operators).
* `TestRemovalDetails`: user decisions "Removal methods details": Tag.remove takes
  members only; DisplayLayer / ShadingEngine refuse the raw set verbs.
* `TestBareAttributeStrMethods`: a bare Attribute's str methods follow the name rule.
* `TestPlugDir`: dir(plug) lists no str method names (a Plug's are lookups).
* `TestNoNoiseAtExit`: no "Exception ignored" from the held-element table at exit.
* `TestClassRouteMirrors`: the class-route mirrors (`_base._MIRRORED`) are always the
  class's member, and rig's internals read them; a rail_spine build makes few class
  accesses of a plug-method descriptor and no slow isinstance on its hot paths.
"""

import copy
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from unittest import mock

from maya import cmds

import rig
from rig import Layer, List, Node, Tag, get_options, set_options
from rig._internal.plug import Plug
from rig.nodetypes import _base
from rig.nodetypes import DisplayLayer, ObjectSet, ShadingEngine, Transform
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes._base import Attribute
from rig.shade import Blinn
from rig._tests._base import MayaTestCase


def _in_own_mayapy(code):
    """(exit code, stdout, stderr) of `code` run in a new mayapy on this rig."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
    env["MAYA_DISABLE_CER"] = "1"
    cwd = tempfile.mkdtemp(prefix="rig_final_fixes_")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, timeout=600,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return (proc.returncode, proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


def _result(test, out, err):
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    test.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
    result = json.loads(lines[0][len("RESULT "):])
    here = os.path.normcase(os.path.dirname(rig.__file__))
    test.assertEqual(os.path.normcase(os.path.dirname(result.pop("file"))), here)
    return result


_WALK = r"""
import json, os, sys
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    import rig
    from rig import Node
    cmds.file(new=True, force=True)
    t = Node(cmds.createNode("transform", name="T"))
    cmds.addAttr("T", ln="pair", at="compound", nc=2)
    cmds.addAttr("T", ln="aa", at="double", p="pair")
    cmds.addAttr("T", ln="get", at="double", p="pair")
    cmds.addAttr("T", ln="serialize", at="bool")
    objs = {"pair": t.pair, "tx": t.tx, "ser": t.serialize, "attr": t.find_attr("pair"),
            "child": t.pair.aa, "wm": t.worldMatrix[0]}
    names = {k: dir(o) for k, o in objs.items()}
    cmds.file(new=True, force=True)
    walked, private = {}, {}
    for key, o in objs.items():
        for n in names[key]:
            try:
                getattr(o, n)
            except Exception:
                pass
        walked[key] = len(names[key])
        # the private readers that read the MPlug: "already deleted!" first
        for label, fn in (
            ("_owner_is_polymorphic", lambda: o._owner_is_polymorphic),
            ("_is_geometry_typed_attr", lambda: o._is_geometry_typed_attr),
            ("_get_geometry_value", lambda: o._get_geometry_value()),
            ("_static_type_key", lambda: o._static_type_key()),
        ):
            try:
                fn()
                private[key + " " + label] = "returned"
            except RuntimeError as error:
                private[key + " " + label] = "already deleted!" in str(error)
    print("RESULT " + json.dumps({"file": rig.__file__, "walked": walked, "private": private}))
    sys.stdout.flush()
finally:
    try:
        maya.standalone.uninitialize()
    except BaseException:
        pass
    sys.stdout.flush()
    os._exit(0)
"""


class TestFreedPlugWalk(MayaTestCase):
    def test_every_attribute_of_a_freed_plug_reads_no_mplug(self):
        # a new mayapy: an access violation fails this test instead of the run
        code, out, err = _in_own_mayapy(_WALK)
        result = _result(self, out, err)
        self.assertEqual(sorted(result["walked"]), ["attr", "child", "pair", "ser", "tx", "wm"])
        for key, value in sorted(result["private"].items()):
            with self.subTest(key=key):
                self.assertIs(value, True)


class TestUnderscoreProbeOnADeletedNode(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_probe_is_an_attribute_error(self):
        cmds.undoInfo(state=True)
        t = Node(cmds.createNode("transform", name="T"))
        cmds.addAttr("T", ln="pair", at="compound", nc=2)
        cmds.addAttr("T", ln="cmpA", at="double", p="pair")
        cmds.addAttr("T", ln="cmpB", at="double", p="pair")
        cached = t.pair
        cached.cmpA  # builds the child cache
        plugs = {
            "compound": t.translate, "dynamic compound": t.pair, "child cache": cached,
            "bare Attribute": t.find_attr("translate"), "scalar": t.tx, "array": t.worldMatrix,
        }
        cmds.delete("T")
        for label, plug in plugs.items():
            for name in ("_zz", "__array__", "__array_interface__"):
                with self.subTest(plug=label, name=name):
                    self.assertIs(hasattr(plug, name), False)
        # any other name on a deleted node raises its "already deleted!", as before
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            plugs["bare Attribute"].nope
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            plugs["compound"].nope


class TestLateClassAssignment(MayaTestCase):
    def test_a_callable_that_is_no_function_follows_the_class_body_rule(self):
        class LateNode(Transform):
            pass

        class LatePlug(Plug):
            pass

        for cls in (LateNode, LatePlug):
            for value in (functools.partial(len), len, type("Call", (), {"__call__": lambda s: 1})()):
                with self.subTest(cls=cls.__name__, value=type(value).__name__):
                    with self.assertRaisesRegex(TypeError, r"has no public data"):
                        setattr(cls, "helper", value)
                    self.assertNotIn("helper", vars(cls))

    def test_a_mock_is_kept_and_restored(self):
        original = Transform.get_parent
        with mock.patch.object(Transform, "get_parent", return_value="parent") as patched:
            self.assertIs(Transform.__dict__["get_parent"], patched)
        self.assertNotIn("get_parent", Transform.__dict__)
        self.assertIs(Transform.get_parent, original)
        with mock.patch.object(Plug, "get_node", return_value="node") as patched:
            self.assertIs(Plug.__dict__["get_node"], patched)
        self.assertIs(Plug.get_node, Attribute.get_node)


class TestPlugRuleInTheMetaclass(MayaTestCase):
    def test_an_init_subclass_without_super_cannot_skip_it(self):
        class MyPlug(Plug):
            def __init_subclass__(cls, **kwargs):
                pass

        with self.assertRaisesRegex(TypeError, r"Sub\.bad: a plug class has no public property"):
            class Sub(MyPlug):  # noqa: F841
                @property
                def bad(self):
                    return 1

        class Fine(MyPlug):
            def helper(self):
                return 2

        self.assertIsInstance(vars(Fine)["helper"], _base._ChildOrMethod)
        self.assertIn(Fine, _base._PLUG_TYPES)


class TestDecoratedMethods(MayaTestCase):
    def test_a_decorated_method_is_refused_by_its_own_message(self):
        def f(self):
            return 1

        decorated = (
            functools.lru_cache(maxsize=None)(f),
            functools.cache(f),
            functools.partialmethod(f),
            functools.singledispatchmethod(f),
        )
        for base in (Transform, Plug):
            for value in decorated:
                with self.subTest(base=base.__name__, value=type(value).__name__):
                    with self.assertRaisesRegex(
                        TypeError, r"X\.helper: a (node|plug) class takes a public method as a plain "
                        r"function .* a decorated method \(\w+\) is not one: decorate a private helper"
                    ):
                        type("X", (base,), {"helper": value})


class TestBarePredicateCompare(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_comparison_raises_as_bool_does(self):
        t = Node(cmds.createNode("transform", name="t"))
        for label, fn in (
            ("node ==", lambda: t.is_valid == True),  # noqa: E712
            ("node !=", lambda: t.is_valid != False),  # noqa: E712
            ("plug ==", lambda: t.tx.is_locked == False),  # noqa: E712
            ("plug !=", lambda: t.tx.is_multi != True),  # noqa: E712
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(TypeError, r"is a method: call it \(\w+\(\)\)"):
                    fn()
        self.assertIs(t.is_valid(), True)
        self.assertIsInstance(hash(t.is_valid), int)

    def test_a_bare_predicate_copies(self):
        t = Node(cmds.createNode("transform", name="t"))
        for pred in (t.is_valid, t.tx.is_locked):
            with self.subTest(pred=repr(pred)):
                twin = copy.copy(pred)
                self.assertIsInstance(twin, _base._BarePredicate)
                self.assertIs(twin(), pred())


class TestAliasOfAnElementChild(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_element_names_its_child(self):
        t = Node(cmds.createNode("transform", name="T"))
        cmds.addAttr("T", ln="items", at="compound", nc=1, multi=True)
        cmds.addAttr("T", ln="val", at="double", p="items")
        cmds.setAttr("T.items[0].val", 1)
        cmds.setAttr("T.items[1].val", 1)
        cmds.aliasAttr("connect", "T.items[0].val")
        self.assertIsInstance(t.connect, Plug)
        plug = t.items[0].connect
        self.assertIsInstance(plug, Plug)
        self.assertEqual(plug._mplug.partialName(False, True, True, False, True, True), "items[0].val")
        plug << 3.0
        self.assertEqual(cmds.getAttr("T.items[0].val"), 3.0)
        # another element's child is not the alias: the method
        self.assertNotIsInstance(t.items[1].connect, Plug)
        self.assertTrue(callable(t.items[1].connect))


class TestSetNamespaceRefusal(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_refusal_names_set_namespace(self):
        a = Node(cmds.createNode("transform", name="a"))
        a.lock()
        with self.assertRaisesRegex(Exception, r"^cannot set the namespace of 'a': it is locked"):
            a.set_namespace("ns1")
        self.assertFalse(cmds.namespace(exists="ns1"))
        a.unlock()


class TestHandleOldNames(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_each_old_name_names_its_method(self):
        from rig.nodetypes.mesh import ColorSet

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        faces = cube.f
        cset  = cube.get_shape().add_color_set("cs1", ColorSet.Representation.RGBA)
        rows = (
            (lambda: faces.count, r"'Components' object has no attribute 'count'; c\.count is now c\.get_count\(\)"),
            (lambda: faces.indices, r"c\.indices is now c\.get_indices\(\)"),
            (lambda: faces.names, r"c\.names is now c\.get_names\(\)"),
            (lambda: faces.kind, r"c\.kind is now c\.get_kind\(\)"),
            (lambda: faces.shape, r"c\.shape is now c\.get_shape\(\)"),
            (lambda: faces.geometry, r"c\.geometry is now c\.get_shape\(\)\.get_node_type\(\)"),
            (lambda: cset.name, r"cset\.name is now cset\.get_name\(\)"),
            (lambda: cset.mesh, r"cset\.mesh is now cset\.get_mesh\(\)"),
            (lambda: cset.representation, r"cset\.representation is now cset\.get_representation\(\)"),
            (lambda: cset.data, r"cset\.data is now cset\.get_data\(\) / cset\.set_data\(colours\)"),
            (lambda: setattr(cset, "data", []), r"cset\.data is now cset\.get_data\(\) / cset\.set_data"),
            (lambda: rig.container.stack, r"rig\.container\.stack is now rig\.container\.get_stack\(\)"),
            (lambda: rig.container.containers, r"rig\.container\.containers is gone: rig\.container\.get_stack\(\)"),
        )
        for fn, pattern in rows:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(AttributeError, pattern):
                    fn()
        # other names: Python's text only; probes still answer
        with self.assertRaisesRegex(AttributeError, r"^'ColorSet' object has no attribute 'nmae'$"):
            cset.nmae = 1
        self.assertIs(hasattr(faces, "__array__"), False)
        self.assertEqual(copy.copy(faces), faces)
        self.assertEqual(faces.get_count(), 6)


class TestTokenVerbsArePrivate(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_operators_are_the_only_spelling(self):
        cube  = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        layer = Layer.define("L")
        red   = Blinn.define("red")
        for token in (-layer, Layer(), -red, Blinn()):
            for verb in ("inject", "query", "contains"):
                with self.subTest(token=repr(token), verb=verb):
                    with self.assertRaisesRegex(AttributeError, r"a token has no methods: x << token"):
                        getattr(token, verb)
        for verb in ("inject", "query", "contains"):
            with self.subTest(tag=verb):
                self.assertIs(hasattr(Tag("cap"), verb), False)
        cube << layer
        self.assertIn(cube, layer)
        layer.remove(cube)
        self.assertNotIn(cube, layer)


class TestRemovalDetails(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        self.sph  = Node(cmds.polySphere(name="sph", constructionHistory=False)[0])
        self.sph.vtx[:4] << Tag("cap")

    def test_tag_remove_takes_members_only(self):
        before = (sorted(cmds.ls()), (self.sph >> Tag("cap")).tolist())
        for members in (self.sph, [self.sph.vtx[0], self.sph], List([self.sph])):
            with self.subTest(members=repr(members)):
                with self.assertRaisesRegex(
                    TypeError, r"^Tag\('cap'\)\.remove takes members .*write Tag\('cap'\)\.delete\("
                ):
                    Tag("cap").remove(members)
                self.assertEqual((sorted(cmds.ls()), (self.sph >> Tag("cap")).tolist()), before)
        members = self.sph.vtx[:2]
        self.assertIs(Tag("cap").remove(members), members)
        self.assertEqual((self.sph >> Tag("cap")).tolist(), [2, 3])
        # the operator still deletes the tag (the user decides later about -c)
        self.sph << -Tag("cap")
        self.assertEqual(Tag.of(self.sph), [])

    def test_a_layer_and_an_engine_refuse_the_raw_set_verbs(self):
        layer = Layer.define("L")
        sg    = Node("initialShadingGroup")
        before = (cmds.sets("initialShadingGroup", query=True), DisplayLayer.for_node(self.cube))
        rows = (
            (lambda: layer.add_members(self.cube), r"layer\.add_members is not rig's form: write cube << "),
            (lambda: layer.remove_members(self.cube), r"layer\.remove_members is not rig's form: write .*\.remove\(cube\)"),
            (lambda: sg.add_members(self.cube), r"sg\.add_members is not rig's form: write cube\.f\[0:2\] << "),
            (lambda: sg.remove_members(self.cube), r"sg\.remove_members is not rig's form: write .*\.remove\("),
            (lambda: sg.force_elements(self.cube), r"sg\.force_elements is not rig's form: write cube\.f\[0:2\] << "),
        )
        for fn, pattern in rows:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(TypeError, pattern):
                    fn()
                self.assertEqual(
                    (cmds.sets("initialShadingGroup", query=True), DisplayLayer.for_node(self.cube)), before
                )
        self.assertIsInstance(sg, ShadingEngine)
        # the rig forms, and a layer's own clear / delete
        self.cube << layer
        layer.clear()
        self.assertNotIn(self.cube, layer)
        self.cube << layer
        layer.delete()
        self.assertIsNone(DisplayLayer.for_node(self.cube))
        # a plain object set keeps them
        group = ObjectSet.define("group")
        group.add_members(self.cube)
        self.assertEqual(cmds.sets("group", query=True), ["cube"])
        group.remove_members(self.cube)
        group.force_elements(self.cube)
        self.assertEqual(cmds.sets("group", query=True), ["cube"])


class TestBareAttributeStrMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_child_named_like_a_str_method(self):
        t = Node(cmds.createNode("transform", name="t"))
        cmds.addAttr("t", ln="cmp", at="compound", nc=2)
        cmds.addAttr("t", ln="upper", at="double", p="cmp")
        cmds.addAttr("t", ln="translate2", at="double", p="cmp")
        attr = t.find_attr("cmp")
        self.assertIs(type(attr), Attribute)
        child = attr.upper
        self.assertIsInstance(child, Attribute)
        self.assertEqual(str(child), "t.upper")
        self.assertEqual(attr.upper(), "T.CMP")              # with brackets: str's method
        self.assertEqual(child(), "T.CMP")                   # a method plug's call runs it
        # no such child: str's method, bare or called
        self.assertEqual(attr.lower(), "t.cmp")
        self.assertEqual(attr.split(".")[1], "cmp")
        self.assertEqual(t.find_attr("tx").upper(), "T.TRANSLATEX")
        self.assertEqual(Attribute.upper, str.upper)
        # a Plug's str names are attribute lookups, as before
        self.assertEqual(str(t.cmp.upper), "t.upper")


class TestPlugDir(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dir_lists_no_str_method(self):
        t = Node(cmds.createNode("transform", name="t"))
        names = dir(t.tx)
        for name in ("upper", "count", "index", "split", "translate"):
            with self.subTest(name=name):
                self.assertNotIn(name, names)
        for name in ("get", "set", "get_node", "is_locked", "connect"):
            with self.subTest(name=name):
                self.assertIn(name, names)
        self.assertFalse([n for n in names if n.startswith("_raw_")])
        self.assertFalse([n for n in dir(t) if n.startswith("_raw_")])


_EXIT = r"""
import os, sys
import maya.standalone
maya.standalone.initialize(name="python")
import rig
from rig import Node
from maya import cmds
t = Node(cmds.createNode("transform", name="t"))
cmds.addAttr("t", ln="w", at="double", multi=True)
held = [t.w[0], t.w[1], t.find_attr("w")[2]]
print("RESULT {\"file\": %s}" % repr(rig.__file__).replace("'", '"').replace("\\\\", "/"))
sys.stdout.flush()
maya.standalone.uninitialize()
"""


class TestNoNoiseAtExit(MayaTestCase):
    def test_held_elements_print_nothing_at_exit(self):
        code, out, err = _in_own_mayapy(_EXIT)
        self.assertEqual(code, 0, out[-2000:] + err[-2000:])
        self.assertIn("RESULT ", out)
        self.assertNotIn("Exception ignored", err)
        self.assertNotIn("_drop_held", err)


class TestClassRouteMirrors(MayaTestCase):
    TEST_START_NEW_SCENE = True

    @staticmethod
    def _classes():
        node_classes = set(_base._NODE_CLASS_DICT.values()) | {DGNode, Transform}
        return sorted(node_classes, key=lambda c: c.__qualname__) + sorted(
            _base._PLUG_TYPES, key=lambda c: c.__qualname__
        )

    def _assert_mirrors(self, cls):
        for name in sorted(_base._MIRRORED):
            member = getattr(cls, name, _base._MISSING)
            if member is _base._MISSING:
                continue
            mirror = getattr(cls, "_raw_" + name)
            if name.isupper():
                self.assertIs(mirror, member, (cls, name))
            elif hasattr(member, "__func__") and isinstance(getattr(member, "__self__", None), type):
                self.assertIs(mirror.__func__, member.__func__, (cls, name))  # a classmethod
            else:
                self.assertIs(mirror, member, (cls, name))

    def test_every_class_mirrors_its_members(self):
        for cls in self._classes():
            with self.subTest(cls=cls.__qualname__):
                self._assert_mirrors(cls)

    def test_a_patch_and_its_restore_are_mirrored(self):
        original = DGNode.ensure_valid
        with mock.patch.object(DGNode, "ensure_valid") as patched:
            self.assertIs(Transform._raw_ensure_valid, patched)
            t = Node(cmds.createNode("transform", name="t"))
            t.get_fn_set()           # a hot reader runs the patched method
            self.assertTrue(patched.called)
        self.assertIs(Transform._raw_ensure_valid, original)
        # a subclass that inherits the member: the restore deletes its mirror
        with mock.patch.object(Transform, "get_uuid", return_value="uuid") as patched:
            self.assertIs(Transform._raw_get_uuid, patched)
        self.assertNotIn("_raw_get_uuid", Transform.__dict__)
        self.assertIs(Transform._raw_get_uuid, DGNode.get_uuid)
        with mock.patch.object(Plug, "get_full_name", return_value="x.y"):
            self.assertEqual(str(Node("t").tx), "x.y")
        self.assertEqual(str(Node("t").tx), "t.translateX")
        # a late assignment of a function, and its deletion
        def get_uuid(self):
            return "late"

        Transform.get_uuid = get_uuid
        try:
            self.assertIs(Transform._raw_get_uuid, get_uuid)
        finally:
            del Transform.get_uuid
        self.assertNotIn("_raw_get_uuid", Transform.__dict__)
        self._assert_mirrors(Transform)

    def test_a_user_subclass_mirrors_its_override(self):
        class Mine(Transform):
            def get_uuid(self):
                return "mine"

        class MinePlug(Plug):
            def is_locked(self):
                return "mine"

        self.assertEqual(Mine._raw_get_uuid(None), "mine")
        self.assertEqual(MinePlug._raw_is_locked(None), "mine")
        self._assert_mirrors(Mine)
        self._assert_mirrors(MinePlug)

    def test_rig_reads_the_mirrors(self):
        # no class route of a mirrored name is left in rig's source: a
        # ``type(x).<name>`` read runs the plug-method descriptor
        import ast

        root  = os.path.dirname(rig.__file__)
        found = []
        used  = set()
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("_tests", "__pycache__")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(base, fname)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Attribute):
                        continue
                    if node.attr.startswith("_raw_") and node.attr != "_raw_function":
                        used.add(node.attr[5:])
                    value = node.value
                    if (
                        node.attr in _base._MIRRORED
                        and isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "type"
                    ):
                        found.append(f"{os.path.relpath(path, root)}:{node.lineno} type(..).{node.attr}")
        self.assertEqual(found, [])
        self.assertEqual(used - _base._MIRRORED, set())

    def test_a_rail_spine_build_makes_few_class_accesses(self):
        # measured 2026-10-16: 152,761 descriptor class accesses per build on
        # e69bae0, 4,548 (the cast's class data) before its mirror, about 300 now
        from rig.examples import rail_spine as rs

        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))
        set_options(create_containers=True)
        pcs = List()
        for i in range(5):
            pcs.append(Node(cmds.spaceLocator()[0]))
            pcs[i].ty << i * 5
        counts = Counter()
        state  = {"on": False}
        for cls in (_base._PlugMethod, _base._PlugDescriptor, _base._PlugClassMethod,
                    _base._PlugStaticMethod, _base._ChildOrMethod, _base._ClassData):
            original = cls.__dict__["__get__"]

            def counting(self, inst, owner=None, _original=original, _cls=cls):
                if inst is None and state["on"]:
                    counts[_cls.__name__] += 1
                return _original(self, inst, owner)

            type.__setattr__(cls, "__get__", counting)
            self.addCleanup(type.__setattr__, cls, "__get__", original)
        instance_checks = Counter()
        original_check  = type(Attribute).__instancecheck__

        def counting_check(cls, inst):
            if state["on"]:
                instance_checks[cls.__name__] += 1
            return original_check(cls, inst)

        type.__setattr__(type(Attribute), "__instancecheck__", counting_check)
        self.addCleanup(type.__delattr__, type(Attribute), "__instancecheck__")
        u = [x / 19 for x in range(20)]
        state["on"] = True
        rs.create_rail(pcs, u, scale_controls=pcs, orient_controls=pcs, degree=3, periodic=False,
                       aim_axis=1, up_axis=0, debug=True)
        state["on"] = False
        self.assertLessEqual(sum(counts.values()), 1000, counts)
        # isinstance(x, Attribute) through the metaclass: the hot paths read
        # `_PLUG_TYPES` instead (measured 2026-10-16: 152,425 per build on e69bae0)
        self.assertLessEqual(sum(instance_checks.values()), 30000, instance_checks)
