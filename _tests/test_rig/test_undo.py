"""Tests for ``rig._internal.undo`` -- the per-inject undo chunk -- and the public
``rig.undo_chunk``.

The public-helper tests check the scene (``rig._tests._undo``), never the return
value of ``cmds.undo()``: Maya swallows an exception raised inside an undo.
"""

import numpy as np
from maya import cmds

import rig
from rig._internal import undo as undo_module
from rig._internal.undo import _undo_chunk
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, mesh_state
from rig.nodetypes import Mesh


class TestUndoChunk(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # mayapy starts with the undo queue off; the chunk is meaningless
        # without it.
        cmds.undoInfo(state=True, infinity=True)

    def test_two_commands_undo_as_one_step(self):
        with _undo_chunk("rig.test"):
            cmds.createNode("transform", name="first")
            cmds.createNode("transform", name="second")
        self.assertTrue(cmds.objExists("first"))
        self.assertTrue(cmds.objExists("second"))
        cmds.undo()
        self.assertFalse(cmds.objExists("first"))
        self.assertFalse(cmds.objExists("second"))
        cmds.redo()
        self.assertTrue(cmds.objExists("first"))
        self.assertTrue(cmds.objExists("second"))

    def test_chunk_closes_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with _undo_chunk("rig.test"):
                cmds.createNode("transform", name="inside")
                raise RuntimeError("boom")
        cmds.createNode("transform", name="after")
        # ``after`` is its own undo step because the chunk was closed by
        # the ``finally``; a still-open chunk would swallow it.
        cmds.undo()
        self.assertFalse(cmds.objExists("after"))
        self.assertTrue(cmds.objExists("inside"))
        cmds.undo()
        self.assertFalse(cmds.objExists("inside"))


# --- rig.undo_chunk ---------------------------------------------------------------------


@rig.undo_chunk
def build_pair(prefix):
    """Two transforms named after ``prefix``."""
    return cmds.createNode("transform", name=f"{prefix}_a"), cmds.createNode("transform", name=f"{prefix}_b")


@rig.undo_chunk("rig.test.named")
def build_named(prefix, *, suffix="n"):
    """One transform."""
    return cmds.createNode("transform", name=f"{prefix}_{suffix}")


@rig.undo_chunk()
def build_empty_parens(prefix):
    return cmds.createNode("transform", name=prefix)


@rig.undo_chunk
def build_chain(depth, made=None):
    """One transform per level, ``depth`` levels deep (recursive)."""
    made = [] if made is None else made
    made.append(cmds.createNode("transform", name=f"level{depth}"))
    if depth > 1:
        build_chain(depth - 1, made)
    return made


@rig.undo_chunk
def build_then_fail(name):
    cmds.createNode("transform", name=name)
    raise ValueError("build failed (test)")


class Builder:
    @rig.undo_chunk
    def build(self, name):
        return cmds.createNode("transform", name=name)

    @staticmethod
    @rig.undo_chunk
    def build_static(name):
        return cmds.createNode("transform", name=name)


class TestPublicUndoChunk(UndoWalk, MayaTestCase):
    """``rig.undo_chunk``: a context manager and a decorator (bare, or with a name)."""

    TEST_START_NEW_SCENE = True

    def assert_one_step(self, name, nodes):
        """The next undo is ``name`` and removes ``nodes`` in one step; the redo brings
        them back."""
        self.assertEqual(self.undo_name(), name)
        for node in nodes:
            self.assertTrue(cmds.objExists(node), node)
        self.undo_steps(1)
        for node in nodes:
            self.assertFalse(cmds.objExists(node), node)
        self.redo_steps(1)
        for node in nodes:
            self.assertTrue(cmds.objExists(node), node)

    # --- the public names

    def test_public_names(self):
        self.assertIn("undo_chunk", rig.__all__)
        self.assertIs(rig.undo_chunk, undo_module.undo_chunk)
        # rig's own chunks are the same function (one place for round 5's barrier)
        self.assertIs(_undo_chunk, rig.undo_chunk)

    # --- the context manager

    def test_context_manager_two_commands_one_named_step(self):
        with rig.undo_chunk("rig.test.public") as value:
            cmds.createNode("transform", name="first")
            cmds.createNode("transform", name="second")
        self.assertIsNone(value)
        self.assert_one_step("rig.test.public", ["first", "second"])
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_context_manager_without_a_name(self):
        with rig.undo_chunk():
            cmds.createNode("transform", name="first")
            cmds.createNode("transform", name="second")
        self.assert_one_step("rig.undo_chunk", ["first", "second"])

    def test_context_manager_closes_when_the_block_raises(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with rig.undo_chunk("rig.test.raises"):
                cmds.createNode("transform", name="inside")
                raise RuntimeError("boom")
        cmds.createNode("transform", name="after")
        # ``after`` is its own step: the chunk was closed on the way out
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("after"))
        self.assertTrue(cmds.objExists("inside"))
        self.assertEqual(self.undo_name(), "rig.test.raises")
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("inside"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_nested_chunks_are_one_step_named_by_the_outermost(self):
        with rig.undo_chunk("rig.test.outer"):
            cmds.createNode("transform", name="first")
            with rig.undo_chunk("rig.test.inner"):
                cmds.createNode("transform", name="second")
            with _undo_chunk("rig.create"):  # a rig operation inside the user's chunk
                cmds.createNode("transform", name="third")
        self.assert_one_step("rig.test.outer", ["first", "second", "third"])
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_one_chunk_object_entered_twice(self):
        chunk = rig.undo_chunk("rig.test.again")
        with chunk:
            cmds.createNode("transform", name="first")
            with chunk:  # reentrant: it holds only its name
                cmds.createNode("transform", name="second")
        with chunk:
            cmds.createNode("transform", name="third")
        self.assert_one_step("rig.test.again", ["third"])
        self.undo_steps(1)
        self.assert_one_step("rig.test.again", ["first", "second"])

    # --- the decorator

    def test_bare_decorator_names_the_step_after_the_function(self):
        result = build_pair("p")
        self.assertEqual(result, ("p_a", "p_b"))  # the return value
        self.assert_one_step("build_pair", ["p_a", "p_b"])
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_decorator_keeps_the_function_metadata(self):
        for wrapper, name, doc in (
            (build_pair, "build_pair", "Two transforms named after ``prefix``."),
            (build_named, "build_named", "One transform."),
            (build_empty_parens, "build_empty_parens", None),
        ):
            with self.subTest(name):
                self.assertEqual(wrapper.__name__, name)
                self.assertEqual(wrapper.__qualname__, name)
                self.assertEqual(wrapper.__doc__, doc)
                self.assertEqual(wrapper.__module__, __name__)
                self.assertTrue(callable(wrapper.__wrapped__))
                self.assertIsNot(wrapper.__wrapped__, wrapper)
                self.assertEqual(wrapper.__wrapped__.__qualname__, name)
        # the wrapped function itself opens no chunk: its two commands are two steps
        self.assertEqual(build_pair.__wrapped__("w"), ("w_a", "w_b"))
        self.undo_steps(1)
        self.assertTrue(cmds.objExists("w_a"))
        self.assertFalse(cmds.objExists("w_b"))
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("w_a"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_named_decorator(self):
        self.assertEqual(build_named("q", suffix="x"), "q_x")
        self.assert_one_step("rig.test.named", ["q_x"])

    def test_empty_parens_decorator_names_the_step_after_the_function(self):
        self.assertEqual(build_empty_parens("e"), "e")
        self.assert_one_step("build_empty_parens", ["e"])

    def test_method_and_static_method_are_named_by_qualname(self):
        self.assertEqual(Builder().build("m"), "m")
        self.assert_one_step("Builder.build", ["m"])
        self.assertEqual(Builder.build_static("s"), "s")
        self.assert_one_step("Builder.build_static", ["s"])

    def test_decorated_function_called_twice_is_two_steps(self):
        build_pair("one")
        build_pair("two")
        self.assert_one_step("build_pair", ["two_a", "two_b"])
        self.undo_steps(1)
        self.assertTrue(cmds.objExists("one_a"))
        self.assert_one_step("build_pair", ["one_a", "one_b"])
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_recursion_is_one_step_and_every_chunk_closes(self):
        made = build_chain(4)
        self.assertEqual(made, ["level4", "level3", "level2", "level1"])
        cmds.createNode("transform", name="after")  # its own step: every chunk closed
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("after"))
        self.assert_one_step("build_chain", made)
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_decorated_function_that_raises_closes_its_chunk(self):
        with self.assertRaisesRegex(ValueError, "build failed"):
            build_then_fail("inside")
        cmds.createNode("transform", name="after")
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("after"))
        self.assertEqual(self.undo_name(), "build_then_fail")
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("inside"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_nested_function_is_named_by_its_qualname(self):
        @rig.undo_chunk
        def local():
            cmds.createNode("transform", name="loc")

        local()
        self.assert_one_step(
            "TestPublicUndoChunk.test_a_nested_function_is_named_by_its_qualname.<locals>.local",
            ["loc"],
        )

    def test_refusals(self):
        def generator():
            yield cmds.createNode("transform")

        async def coroutine():
            pass

        cases = [
            (TypeError, lambda: rig.undo_chunk(5), "chunk name"),
            (TypeError, lambda: rig.undo_chunk(b"x"), "chunk name"),
            (TypeError, lambda: rig.undo_chunk(Builder), "not the class"),
            (TypeError, lambda: rig.undo_chunk(staticmethod(build_pair.__wrapped__)), "below @staticmethod"),
            (TypeError, lambda: rig.undo_chunk(classmethod(build_pair.__wrapped__)), "below @classmethod"),
            (TypeError, lambda: rig.undo_chunk(property(build_pair.__wrapped__)), "below @property"),
            (TypeError, lambda: rig.undo_chunk(generator), "generator or a coroutine"),
            (TypeError, lambda: rig.undo_chunk("x")(generator), "generator or a coroutine"),
            (TypeError, lambda: rig.undo_chunk(coroutine), "generator or a coroutine"),
            (TypeError, lambda: rig.undo_chunk("x")(5), "chunk name"),
            (ValueError, lambda: rig.undo_chunk(""), "empty"),
        ]
        for error, call, text in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(error, text):
                    call()
        # nothing was opened: the next command is its own step
        cmds.createNode("transform", name="after")
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_rig_edits_inside_a_user_chunk(self):
        # a rig API edit (rigUndoableAPICommand, no chunk of its own) joins the user's step
        mesh = Mesh(cmds.polyCube(name="c", constructionHistory=False)[0])
        cmds.flushUndo()
        before = mesh_state("c")
        with rig.undo_chunk("rig.test.user"):
            mesh.set_points(np.array(mesh.get_points(world_space=False))[:, :3] * 2.0, world_space=False)
            cmds.createNode("transform", name="extra")
        after = mesh_state("c")
        self.assertNotEqual(after["points"], before["points"])
        self.assertEqual(self.undo_name(), "rig.test.user")
        self.undo_steps(1)
        self.assertEqual(mesh_state("c"), before)
        self.assertFalse(cmds.objExists("extra"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertEqual(mesh_state("c"), after)
        self.assertTrue(cmds.objExists("extra"))
