"""Round 4b, step NC1: one error family, one lookup rule.

* ``TestErrorFamily``: ``NodeLookupError(LookupError, TypeError, ValueError)``
  with ``NodeNotFoundError`` and ``AmbiguousNodeError`` under it, and
  ``NodeTypeError(TypeError, ValueError)``. Every ``except TypeError`` /
  ``except ValueError`` written for the errors before catches them; each keeps
  its facts and builds its message when first printed, with hints that are
  dropped (never raised) when the scene cannot be read.
* ``TestLookupRule``: ``Node(x)`` and the node classes' constructors: a path
  or ``ns:name`` as written; a bare name at the root namespace; two DAG nodes
  ambiguous (an instanced node is one); a pattern refused; a uuid miss named;
  a real constructor error never masked; every refusal writes nothing; a hit
  at the root namespace makes no command call.
* ``TestNamespaceLookup``: while ``char`` is current a bare name is looked up
  at ``:x`` and ``:char:x``: one of them is the node, both are ambiguous,
  with ``namespace -relativeNames`` off and on.
* ``TestFindNodeSharesTheRule``: the membership lookups (``Layer``, the
  materials) follow the same rule, and an ambiguity refuses before any write;
  ``define`` (which replaced ``get_or_create`` in NC4) reads its key.
"""

import pickle
from unittest import mock

from maya import cmds

import rig
import rig.nodetypes
from rig import (
    AmbiguousNodeError,
    Layer,
    Node,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
    container,
)
from rig.nodetypes import (
    DAGNode,
    DGNode,
    DisplayLayer,
    Joint,
    Mesh,
    ObjectSet,
    Transform,
    _base,
    errors as _errors,
)
from rig.bridges import commands as rc
from rig.nodetypes._base import _lookup, set_custom_type
from rig.shade import Blinn
from rig._internal.members import _find_node
from rig._tests._base import MayaTestCase


def _build_scene():
    """``spine_01``; ``|g1|a`` and ``|g2|a``; ``:x`` and ``char:x``;
    ``char:root``, ``char:only``, ``:solo``; the cube ``box`` instanced as
    ``box_inst``. The current namespace is the root."""
    cmds.joint(name="spine_01")
    cmds.select(clear=True)
    for group in ("g1", "g2"):
        cmds.createNode("transform", name=group)
        cmds.createNode("transform", name="a", parent=group)
    cmds.createNode("transform", name="x")
    cmds.createNode("transform", name="solo")
    cmds.namespace(add="char")
    cmds.createNode("transform", name="char:x")
    cmds.createNode("joint", name="char:root")
    cmds.createNode("transform", name="char:only")
    cmds.polyCube(name="box", ch=False)
    cmds.instance("box", name="box_inst")
    cmds.select(clear=True)


def _uuid(name):
    return cmds.ls(name, uuid=True)[0]


class TestErrorFamily(MayaTestCase):
    """The family: its classes, what catches them, their texts and hints."""

    TEST_START_NEW_SCENE = True

    def test_the_classes(self):
        for cls in (NodeLookupError, NodeNotFoundError, AmbiguousNodeError):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, LookupError))
                self.assertTrue(issubclass(cls, TypeError))
                self.assertTrue(issubclass(cls, ValueError))
                self.assertTrue(issubclass(cls, NodeLookupError))
        self.assertTrue(issubclass(NodeTypeError, TypeError))
        self.assertTrue(issubclass(NodeTypeError, ValueError))
        self.assertFalse(issubclass(NodeTypeError, LookupError))
        self.assertFalse(issubclass(NodeTypeError, NodeLookupError))
        self.assertFalse(issubclass(NodeNotFoundError, AmbiguousNodeError))

    def test_exported_from_rig_and_rig_nodetypes(self):
        for name in ("NodeLookupError", "NodeNotFoundError", "AmbiguousNodeError", "NodeTypeError"):
            with self.subTest(name=name):
                self.assertIs(getattr(rig, name), getattr(_errors, name))
                self.assertIs(getattr(rig.nodetypes, name), getattr(_errors, name))
                self.assertIn(name, rig.__all__)
                self.assertIn(name, _errors.__all__)

    def test_every_old_except_clause_catches_them(self):
        errors = (
            NodeLookupError("red*"),
            NodeNotFoundError("nosuch"),
            AmbiguousNodeError("a", ["|g1|a", "|g2|a"]),
            NodeTypeError("grp", "joint", "transform"),
        )
        for error in errors:
            for clause in (TypeError, ValueError, Exception):
                with self.subTest(error=type(error).__name__, clause=clause.__name__):
                    try:
                        raise error
                    except clause as caught:
                        self.assertIs(caught, error)
            if isinstance(error, NodeLookupError):
                with self.subTest(error=type(error).__name__, clause="LookupError"):
                    with self.assertRaises(LookupError):
                        raise error

    def test_facts(self):
        error = NodeNotFoundError("spnie_01", "joint")
        self.assertEqual((error.name, error.label, error.uuid), ("spnie_01", "joint", False))
        self.assertEqual(error.args, ("spnie_01",))
        error = AmbiguousNodeError("a", ("|g1|a", "|g2|a"))
        self.assertEqual((error.name, error.label, error.candidates, error.namespaces),
                         ("a", "node", ["|g1|a", "|g2|a"], False))
        error = NodeTypeError("grp", "joint", "transform", "hint")
        self.assertEqual((error.name, error.label, error.node_type, error.hint),
                         ("grp", "joint", "transform", "hint"))

    def test_texts(self):
        cases = (
            (NodeNotFoundError("nosuch"), "no node named 'nosuch'"),
            (NodeNotFoundError("nosuch", "joint"), "no joint named 'nosuch'"),
            (NodeNotFoundError("DEADBEEF-0000-4000-8000-000000000000", uuid=True),
             "no node has the uuid 'DEADBEEF-0000-4000-8000-000000000000'"),
            (NodeNotFoundError(""), "no node named ''"),
            (AmbiguousNodeError("a", ["|g1|a", "|g2|a"]),
             "'a' is ambiguous: it names 2 nodes: |g1|a, |g2|a; use a path"),
            (AmbiguousNodeError("x", [":x", ":char:x"], namespaces=True),
             "'x' is ambiguous: it names :x and :char:x; spell the namespace"),
            (NodeLookupError("red*"),
             "'red*' is a pattern, not a node name; a pattern is a search: cmds.ls('red*'), "
             "or Transform.find_all() for the nodes of a class"),
            (NodeLookupError("red*", "joint"),
             "'red*' is a pattern, not a joint name; a pattern is a search: cmds.ls('red*'), "
             "or Transform.find_all() for the nodes of a class"),
            (NodeLookupError("x"), "'x' names no single node"),
            (NodeTypeError("grp", "joint", "transform"), "'grp' is a transform, not a joint"),
            (NodeTypeError("s", "objectSet", "transform", 'Node("s") is Transform("s")'),
             "'s' is a transform, not an objectSet; Node(\"s\") is Transform(\"s\")"),
        )
        for error, text in cases:
            with self.subTest(text=text):
                self.assertEqual(str(error), text)

    def test_a_long_candidate_list_is_cut(self):
        paths = [f"|g{i}|a" for i in range(12)]
        text = str(AmbiguousNodeError("a", paths))
        self.assertIn("it names 12 nodes: |g0|a, ", text)
        self.assertIn("|g9|a, ...; use a path", text)
        self.assertNotIn("|g10|a", text)

    def test_did_you_mean(self):
        cmds.joint(name="spine_01")
        cmds.select(clear=True)
        self.assertEqual(
            str(NodeNotFoundError("spnie_01", "joint")),
            "no joint named 'spnie_01' (did you mean 'spine_01'?)",
        )
        # nothing close enough: no hint
        self.assertEqual(str(NodeNotFoundError("zzz")), "no node named 'zzz'")

    def test_another_namespace(self):
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        self.assertEqual(str(NodeNotFoundError("root", "joint")),
                         "no joint named 'root' ('char:root' exists)")
        # a typo in the namespace names the node of that leaf
        self.assertEqual(str(NodeNotFoundError("chr:root")),
                         "no node named 'chr:root' ('char:root' exists)")
        cmds.namespace(add="rig")
        cmds.createNode("joint", name="rig:root")
        self.assertEqual(str(NodeNotFoundError("root")),
                         "no node named 'root' ('char:root', 'rig:root' exist)")

    def test_the_scope_prefix(self):
        """A flattened scope prefixes a created name; the miss names the
        prefixed node, also once the scope is gone (the prefix is read at the
        raise, the scene when the message is printed)."""
        with container("outer"):
            with container("inner"):
                Node.create("transform", name="k")
                inside = NodeNotFoundError("k")
        self.assertEqual(inside.scope_name, "inner_k")
        self.assertEqual(str(inside), "no node named 'k' ('inner_k' exists (the scope prefix))")
        # outside a scope: no scope name, no hint
        self.assertIsNone(NodeNotFoundError("k").scope_name)
        self.assertEqual(str(NodeNotFoundError("k")), "no node named 'k'")
        # a namespaced name keeps the prefix on its leaf
        with container("outer"):
            with container("inner"):
                self.assertEqual(NodeNotFoundError("ns:k").scope_name, "ns:inner_k")
                self.assertIsNone(NodeNotFoundError("|g|k").scope_name)

    def test_the_message_is_built_once(self):
        cmds.joint(name="spine_01")
        error = NodeNotFoundError("spnie_01")
        with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls:
            first = str(error)
            calls = ls.call_count
            self.assertGreater(calls, 0)
            self.assertIs(str(error), first)
            self.assertEqual(ls.call_count, calls)

    def test_no_hint_is_read_before_the_print(self):
        with container("outer"):
            with container("inner"):
                with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls, \
                        mock.patch.object(cmds, "objExists", wraps=cmds.objExists) as exists:
                    scoped = NodeNotFoundError("k")
                    error  = NodeNotFoundError("nosuch")
        self.assertEqual((ls.call_count, exists.call_count), (0, 0))
        self.assertEqual(scoped.scope_name, "inner_k")
        self.assertEqual(error.args, ("nosuch",))

    def test_str_never_raises(self):
        """A hint that cannot be read (the scene was closed or replaced) is
        left out; the message is the facts."""
        cmds.joint(name="spine_01")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:spnie_01")
        with container("outer"):
            with container("inner"):
                held = NodeNotFoundError("spnie_01", "joint")
        cmds.file(new=True, force=True)
        # printed after a new scene: the hints read the new scene
        self.assertEqual(str(held), "no joint named 'spnie_01'")

        failing = RuntimeError("the scene is gone")
        for patched in ("ls", "objExists"):
            with self.subTest(failing=patched):
                cmds.joint(name="spine_01")
                with container("outer"):
                    with container("inner"):
                        error = NodeNotFoundError("spnie_01", "joint")
                with mock.patch.object(cmds, patched, side_effect=failing):
                    text = str(error)
                self.assertTrue(text.startswith("no joint named 'spnie_01'"))
                self.new_scene()

        # a scope hook that raises: no scope name, no failure
        with mock.patch.object(_errors, "_SCOPE_HINT_HOOK", side_effect=failing):
            error = NodeNotFoundError("k")
        self.assertIsNone(error.scope_name)
        self.assertEqual(str(error), "no node named 'k'")

        # the scene closes mid-hint: that hint is dropped, the next reads the
        # new scene
        cmds.joint(name="spine_01")
        error   = NodeNotFoundError("spnie_01")
        real_ls = cmds.ls
        closed  = []

        def closing_ls(*args, **kwargs):
            if not closed:
                closed.append(True)
                cmds.file(new=True, force=True)
                raise RuntimeError("closed")
            return real_ls(*args, **kwargs)

        with mock.patch.object(cmds, "ls", side_effect=closing_ls):
            self.assertEqual(str(error), "no node named 'spnie_01'")
        self.assertEqual(closed, [True])

    def test_pickle_keeps_the_facts(self):
        for error in (
            NodeNotFoundError("nosuch", "joint"),
            AmbiguousNodeError("a", ["|g1|a", "|g2|a"]),
            NodeTypeError("grp", "joint", "transform"),
        ):
            with self.subTest(error=type(error).__name__):
                copy = pickle.loads(pickle.dumps(error))
                self.assertIs(type(copy), type(error))
                self.assertEqual(str(copy), str(error))

    def test_node_raises_the_family(self):
        """What ``Node(x)`` raised before (Maya's TypeError, a RuntimeError for
        a pattern) is the family now, still caught by the old clauses."""
        _build_scene()
        cases = (
            ("nosuch", NodeNotFoundError),
            ("a", AmbiguousNodeError),
            ("x*", NodeLookupError),
            ("DEADBEEF-0000-4000-8000-000000000000", NodeNotFoundError),
        )
        for name, cls in cases:
            for clause in (TypeError, ValueError, LookupError):
                with self.subTest(name=name, clause=clause.__name__):
                    with self.assertRaises(clause) as ctx:
                        Node(name)
                    self.assertIs(type(ctx.exception), cls)
                    self.assertEqual(ctx.exception.name, name)

    def test_node_wrap_builds_no_message(self):
        """``Node.wrap`` passes a str that names no node through; the error it
        discards never reads a hint."""
        _build_scene()
        with mock.patch.object(NodeNotFoundError, "_hints") as hints:
            self.assertEqual(Node.wrap("some text"), "some text")
            self.assertEqual(Node.wrap("spnie_01"), "spnie_01")
            self.assertEqual(Node.wrap(["x", "nosuch"])[1], "nosuch")
        hints.assert_not_called()

    def test_the_hints_of_a_node_miss(self):
        _build_scene()
        cases = (
            (lambda: Node("spnie_01"), "no node named 'spnie_01' (did you mean 'spine_01'?)"),
            (lambda: Joint("spnie_01"), "no joint named 'spnie_01' (did you mean 'spine_01'?)"),
            (lambda: Node("root"), "no node named 'root' ('char:root' exists)"),
            (lambda: Joint("root"), "no joint named 'root' ('char:root' exists)"),
            (lambda: Node("chr:root"), "no node named 'chr:root' ('char:root' exists)"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(NodeNotFoundError) as ctx:
                    call()
                self.assertEqual(str(ctx.exception), text)
        with container("outer"):
            with container("inner"):
                Node.create("transform", name="k")
                with self.assertRaises(NodeNotFoundError) as ctx:
                    Node("k")
        self.assertEqual(str(ctx.exception), "no node named 'k' ('inner_k' exists (the scope prefix))")


class TestLookupRule(MayaTestCase):
    """``Node(x)`` and the node classes' constructors follow one lookup rule."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _build_scene()

    def tearDown(self):
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def _refused(self, call, cls, text=None):
        """``call()`` raises exactly ``cls`` (with ``text`` in its message)
        and writes nothing."""
        before = set(cmds.ls())
        with self.assertRaises(cls) as ctx:
            call()
        self.assertIs(type(ctx.exception), cls)
        if text is not None:
            self.assertIn(text, str(ctx.exception))
        self.assertEqual(set(cmds.ls()), before)
        return ctx.exception

    def test_a_qualified_name_is_looked_up_as_written(self):
        cases = (
            ("|g1|a", "|g1|a"),
            ("g2|a", "|g2|a"),
            ("|x", "|x"),
            ("char:x", "|char:x"),
            (":char:x", "|char:x"),
            ("char:root", "|char:root"),
        )
        for current in (":", ":char"):
            cmds.namespace(setNamespace=current)
            for name, long_name in cases:
                with self.subTest(current=current, name=name):
                    self.assertEqual(Node(name).long_name, long_name)
        cmds.namespace(setNamespace=":")
        # a qualified miss is not retried anywhere else
        self._refused(lambda: Node("chr:x"), NodeNotFoundError, "no node named 'chr:x'")
        self._refused(lambda: Node("g3|a"), NodeNotFoundError, "no node named 'g3|a'")
        self._refused(lambda: Node("root:x"), NodeNotFoundError)

    def test_a_bare_name_at_the_root_namespace(self):
        self.assertEqual(Node("x").long_name, "|x")
        self.assertEqual(Node("spine_01").long_name, "|spine_01")
        # char:root is not looked up while the root namespace is current
        error = self._refused(lambda: Node("root"), NodeNotFoundError)
        self.assertEqual((error.name, error.label), ("root", "node"))

    def test_several_dag_nodes_are_ambiguous(self):
        for call, label in ((lambda: Node("a"), "node"), (lambda: Transform("a"), "transform")):
            with self.subTest(label=label):
                error = self._refused(call, AmbiguousNodeError)
                self.assertEqual(sorted(error.candidates), ["|g1|a", "|g2|a"])
                self.assertEqual(error.label, label)
                self.assertFalse(error.namespaces)
                self.assertRegex(
                    str(error), r"^'a' is ambiguous: it names 2 nodes: \|g\d\|a, \|g\d\|a; use a path$"
                )

    def test_an_instanced_node_is_one_node(self):
        shape = Node("boxShape")
        self.assertIsInstance(shape, Mesh)
        self.assertEqual(str(shape), "box|boxShape")
        self.assertEqual(str(Node("box_inst|boxShape")), "box_inst|boxShape")
        self.assertEqual(_lookup("boxShape"), "|box|boxShape")

    def test_a_pattern_is_refused_before_any_command(self):
        for name in ("x*", "?", "x[0]", "*", "char:*", "|g1|*", "x*.tx", "sp?ne_01"):
            with self.subTest(name=name):
                with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls:
                    with self.assertRaises(NodeLookupError) as ctx:
                        Node(name)
                self.assertEqual(ls.call_count, 0)
                self.assertIs(type(ctx.exception), NodeLookupError)
                self.assertIn("is a pattern", str(ctx.exception))
                self.assertEqual(ctx.exception.name, name.split(".")[0])
        for name in ("x*", "zz*"):
            with self.subTest(lookup=name):
                self._refused(lambda: _lookup(name), NodeLookupError, f"cmds.ls({name!r})")
        # a typed constructor refuses a pattern that names no node
        self._refused(lambda: Transform("zz*"), NodeLookupError, "not a transform name")

    def test_names_no_node_can_have(self):
        for name in ("", "|", ":", ".tx", "1bad", "a b", "a-b"):
            with self.subTest(name=name):
                self._refused(lambda: Node(name), NodeNotFoundError)

    def test_a_uuid(self):
        for uid in ("DEADBEEF-0000-4000-8000-000000000000", "abcdefabcdefabcdefabcdefabcdefab"):
            for current in (":", ":char"):
                with self.subTest(uid=uid, current=current):
                    cmds.namespace(setNamespace=current)
                    error = self._refused(lambda: Node(uid), NodeNotFoundError)
                    self.assertTrue(error.uuid)
                    self.assertEqual(str(error), f"no node has the uuid {uid!r}")
                    self.assertEqual(Node(_uuid("|g1|a")).long_name, "|g1|a")
                    self.assertEqual(Node(_uuid("char:x")).long_name, "|char:x")
            cmds.namespace(setNamespace=":")

    def test_a_constructor_error_is_not_masked(self):
        """The lookup finds the one node the cast failed on: the cast's own
        error is raised, not a lookup error."""

        class _Broken(DGNode):
            CUSTOM_NODE_TYPE = "r4bLookupBroken"

            def __init__(self, node):
                raise RuntimeError("the broken constructor")

        def forget():
            _base._NODE_CLASS_DICT.pop("r4bLookupBroken", None)
            _base._CLASS_BY_TYPE.clear()
            _base._CASTABLE_TYPES.clear()

        self.addCleanup(forget)
        broken = cmds.createNode("multiplyDivide", name="brk")
        set_custom_type(broken, "r4bLookupBroken")
        for name in ("brk", ":brk", "brk.input1X"):
            with self.subTest(name=name):
                self._refused(lambda: Node(name), RuntimeError, "the broken constructor")

    def test_the_typed_constructors_follow_the_rule(self):
        cases = (
            (lambda: Joint("spnie_01"), NodeNotFoundError,
             "no joint named 'spnie_01' (did you mean 'spine_01'?)"),
            (lambda: Transform("nosuch"), NodeNotFoundError, "no transform named 'nosuch'"),
            (lambda: DGNode("nosuch"), NodeNotFoundError, "no node named 'nosuch'"),
            (lambda: DAGNode("nosuch"), NodeNotFoundError, "no DAG node named 'nosuch'"),
            (lambda: Transform("a"), AmbiguousNodeError, "use a path"),
            (lambda: Joint("root"), NodeNotFoundError, "('char:root' exists)"),
        )
        for call, cls, text in cases:
            with self.subTest(text=text):
                self._refused(call, cls, text)
        # a name the constructor resolves is unchanged: the lookup runs on a
        # miss only
        self.assertEqual(Joint("spine_01").name, "spine_01")
        self.assertEqual(Transform("|g1|a").long_name, "|g1|a")

    def test_a_hit_at_the_root_makes_no_command_call(self):
        names = ("x", "spine_01", "|g1|a", "char:x", "boxShape", "x.tx", "char:x.tx")
        for name in names:
            Node(name)  # warm the per-type class cache
        # name lookups only (a DAG class's constructor runs its type check,
        # cmds.nodeType, as before)
        counted = ("ls", "objExists", "namespaceInfo")
        patches = [mock.patch.object(cmds, n, wraps=getattr(cmds, n)) for n in counted]
        mocks   = [p.start() for p in patches]
        try:
            for name in names:
                Node(name)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual({n: m.call_count for n, m in zip(counted, mocks)}, dict.fromkeys(counted, 0))


class TestNamespaceLookup(MayaTestCase):
    """While a namespace other than the root is current, a bare name is looked
    up at the root namespace and in the current one, both spelled absolutely:
    ``namespace -relativeNames`` does not change the answer."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _build_scene()

    def tearDown(self):
        cmds.namespace(relativeNames=False)
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def _modes(self):
        """Each relativeNames mode, with ``char`` current."""
        for relative in (False, True):
            cmds.namespace(relativeNames=relative)
            cmds.namespace(setNamespace=":char")
            yield relative
            cmds.namespace(setNamespace=":")
            cmds.namespace(relativeNames=False)

    def test_only_at_the_root(self):
        for relative in self._modes():
            with self.subTest(relative=relative):
                self.assertEqual(Node("solo").uuid, _uuid(":solo"))
                self.assertEqual(Node("spine_01").uuid, _uuid(":spine_01"))

    def test_only_in_the_current_namespace(self):
        for relative in self._modes():
            with self.subTest(relative=relative):
                self.assertEqual(Node("only").uuid, _uuid(":char:only"))
                root = Joint("root")
                self.assertIsInstance(root, Joint)
                self.assertEqual(root.uuid, _uuid(":char:root"))
                self.assertEqual(Node("root").uuid, root.uuid)

    def test_at_both_is_ambiguous(self):
        for relative in self._modes():
            with self.subTest(relative=relative):
                before = set(cmds.ls())
                with self.assertRaises(AmbiguousNodeError) as ctx:
                    Node("x")
                error = ctx.exception
                self.assertTrue(error.namespaces)
                self.assertEqual(error.candidates, [":x", ":char:x"])
                self.assertEqual(
                    str(error), "'x' is ambiguous: it names :x and :char:x; spell the namespace"
                )
                with self.assertRaises(AmbiguousNodeError):
                    Node("x.tx")
                self.assertEqual(set(cmds.ls()), before)
                # spelled, each is found
                self.assertEqual(Node(":x").uuid, _uuid(":x"))
                self.assertEqual(Node(":char:x").uuid, _uuid(":char:x"))

    def test_the_root_namespace_looks_nowhere_else(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                cmds.namespace(relativeNames=relative)
                self.assertEqual(Node("x").uuid, _uuid(":x"))
                with self.assertRaises(NodeNotFoundError):
                    Node("only")
        cmds.namespace(relativeNames=False)

    def test_a_nested_current_namespace(self):
        cmds.namespace(add="sub", parent=":char")
        cmds.namespace(setNamespace=":char:sub")
        # char:x is in neither the root nor char:sub
        self.assertEqual(Node("x").uuid, _uuid(":x"))
        cmds.createNode("transform", name=":char:sub:x")
        with self.assertRaises(AmbiguousNodeError) as ctx:
            Node("x")
        self.assertEqual(ctx.exception.candidates, [":x", ":char:sub:x"])

    def test_a_dag_duplicate_in_one_namespace(self):
        cmds.namespace(setNamespace=":char")
        with self.assertRaises(AmbiguousNodeError) as ctx:
            Node("a")
        self.assertEqual(sorted(ctx.exception.candidates), ["|g1|a", "|g2|a"])

    def test_a_node_made_in_the_namespace_is_found(self):
        cmds.namespace(setNamespace=":char")
        made = cmds.createNode("transform", name="made")
        self.assertEqual(made, "char:made")
        self.assertEqual(Node(made).uuid, _uuid(":char:made"))
        self.assertEqual(Node("made").uuid, _uuid(":char:made"))

    def test_names_maya_returns_stay_exact(self):
        """``container.createNode``, the ``rc`` results and ``Node.wrap`` cast the
        name Maya returned as Maya resolves it (``_cast_node``): a returned
        ``x`` is never ambiguous because another namespace is current, in either
        relativeNames mode (where Maya returns the relative ``x`` for char:x)."""
        for name in ("x1", "x2", "md1", "md2"):
            node_type = "multiplyDivide" if name.startswith("md") else "transform"
            cmds.createNode(node_type, name=f":{name}")

        def in_char(node):
            return node.uuid in cmds.ls(":char:*", uuid=True)

        for relative in self._modes():
            with self.subTest(relative=relative):
                self.assertEqual(rc.ls(":x")[0].uuid, _uuid(":x"))
                self.assertEqual(rc.ls(":char:x")[0].uuid, _uuid(":char:x"))
                self.assertEqual(Node.wrap(cmds.ls(":x"))[0].uuid, _uuid(":x"))
                self.assertEqual(Node.wrap(cmds.ls(":char:x"))[0].uuid, _uuid(":char:x"))
                self.assertTrue(in_char(container.createNode("multiplyDivide", name="md1")))
                self.assertTrue(in_char(container.createNode("transform", name="x")))
                self.assertTrue(in_char(rc.createNode("transform", name="x")))
                with container("scope"):
                    self.assertTrue(in_char(Node.create("multiplyDivide", name="md1")))

    def test_one_lookup_call_off_the_root(self):
        """A bare name while another namespace is current costs one
        ``cmds.ls`` (both spellings in one call); a qualified one none."""
        Node("char:only")
        cmds.namespace(setNamespace=":char")
        Node("only")
        with mock.patch.object(cmds, "ls", wraps=cmds.ls) as ls:
            Node("only")
            self.assertEqual(ls.call_count, 1)
            Node("char:only")
            self.assertEqual(ls.call_count, 1)


class TestFindNodeSharesTheRule(MayaTestCase):
    """The membership lookups (``_find_node``: ``Layer``, the materials) follow
    ``Node(x)``'s rule; ``define`` (``get_or_create``'s replacement) reads its
    key."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        cmds.namespace(add="char")
        for name in ("x", "char:x"):
            cmds.createDisplayLayer(empty=True, name=name)
        for name in ("red", "char:red"):
            cmds.shadingNode("blinn", asShader=True, name=name)
        cmds.select(clear=True)

    def tearDown(self):
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def _layer_of_cube(self):
        return cmds.listConnections("cube.drawOverride", source=True, destination=False) or []

    def test_membership_refuses_an_ambiguity_before_any_write(self):
        cmds.namespace(setNamespace=":char")
        engines = cmds.listConnections("cubeShape", type="shadingEngine")
        before  = set(cmds.ls())
        for label, call in (
            ("cube << Layer('x')", lambda: self.cube << Layer("x")),
            ("cube >> Layer('x')", lambda: self.cube >> Layer("x")),
            ("cube << -Layer('x')", lambda: self.cube << -Layer("x")),
            ("cube << Blinn('red')", lambda: self.cube << Blinn("red")),
            ("cube >> Blinn('red')", lambda: self.cube >> Blinn("red")),
        ):
            with self.subTest(call=label):
                with self.assertRaises(AmbiguousNodeError) as ctx:
                    call()
                named = ":x and :char:x" if "Layer" in label else ":red and :char:red"
                self.assertIn(named, str(ctx.exception))
                self.assertEqual(set(cmds.ls()), before)
                self.assertEqual(self._layer_of_cube(), [])
                self.assertEqual(cmds.listConnections("cubeShape", type="shadingEngine"), engines)
        # spelled, it assigns
        self.cube << Layer("char:x")
        self.assertEqual(self._layer_of_cube(), ["char:x"])

    def test_from_the_root_namespace_the_bare_name_is_the_root_node(self):
        self.cube << Layer("x")
        self.assertEqual(self._layer_of_cube(), ["x"])

    def test_find_node(self):
        self.assertEqual(_find_node("cube"), "|cube")
        self.assertEqual(_find_node("x"), "x")
        self.assertEqual(_find_node("char:x"), "char:x")
        self.assertIsNone(_find_node("nosuch"))
        self.assertIsNone(_find_node("only"))
        cmds.namespace(setNamespace=":char")
        with self.assertRaises(AmbiguousNodeError):
            _find_node("x")
        cmds.createDisplayLayer(empty=True, name="only")
        self.assertEqual(_find_node("only"), "char:only")
        with self.assertRaises(NodeLookupError):
            _find_node("red*")

    def test_a_dag_ambiguity_keeps_the_old_words(self):
        """``assertRaisesRegex(ValueError, "ambiguous")`` (test_shade) and "use a
        path" keep matching."""
        group = cmds.group(empty=True, name="grp")
        cmds.createNode("transform", name="cube", parent=group)
        with self.assertRaisesRegex(ValueError, "ambiguous.*use a path"):
            _find_node("cube")

    def test_get_or_create(self):
        """Re-pinned in round 4b NC4 (get_or_create removed): ``define`` reads its
        key, the node create would make (``:char:x`` while ``char`` is current),
        not the lookup rule; a name the rule reads elsewhere (``:cube``) is
        refused rather than forked."""
        cmds.namespace(setNamespace=":char")
        before = set(cmds.ls())
        # the key: the current namespace's node, found
        self.assertEqual(str(DisplayLayer.define("x")), "char:x")
        with self.assertRaises(NodeTypeError):
            ObjectSet.define("red")  # :char:red is a blinn
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(DisplayLayer.define("char:x")), "char:x")
        made = ObjectSet.define("fresh")
        self.assertEqual(str(made), "char:fresh")
        self.assertEqual(ObjectSet.define("fresh"), made)
        before = set(cmds.ls())
        with self.assertRaisesRegex(AmbiguousNodeError, r"'cube' exists as :cube; this define's key is char:cube"):
            DisplayLayer.define("cube")
        self.assertEqual(set(cmds.ls()), before)
        # from the root namespace the bare name's key is the root node
        cmds.namespace(setNamespace=":")
        self.assertEqual(str(DisplayLayer.define("x")), "x")
