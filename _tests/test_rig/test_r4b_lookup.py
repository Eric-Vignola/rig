"""Round 4b, step NC1: one error family, one lookup rule.

* ``TestErrorFamily``: ``NodeLookupError(LookupError, TypeError, ValueError)``
  with ``NodeNotFoundError`` and ``AmbiguousNodeError`` under it, and
  ``NodeTypeError(TypeError, ValueError)``. Every ``except TypeError`` /
  ``except ValueError`` written for the errors before catches them; each keeps
  its facts and builds its message when first printed, with hints that are
  dropped (never raised) when the scene cannot be read.
"""

import pickle
from unittest import mock

from maya import cmds

import rig
import rig.nodetypes
from rig import (
    AmbiguousNodeError,
    Node,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
    container,
)
from rig.nodetypes import errors as _errors
from rig._tests._base import MayaTestCase


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
