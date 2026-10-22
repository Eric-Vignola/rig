"""Round 6 S2 (user decision 1A): the member add of a node ``container.createNode``
made is deferred and sent once per statement.

``container._add(node, made=True)`` keeps its frame bookkeeping and its locked-leaf
refusal, and puts ``(leaf, node)`` on ``journal._S.members``; `container._flush_members`
sends one ``container -edit -addNode -force`` per run of consecutive entries with the
same leaf (through S0's `_add_members`), the names read then, an invalid node dropped.

* ``TestOneAddPerStatement``: k nodes, one call, the same member order.
* ``TestFlushPoints``: every flush point (the statement's end, normal or raising; an
  undo chunk's close; every rig container call; the non-made add; ``outside`` / a user
  call; a ``cmds_edit`` / ``mel_eval`` not in ``NO_MEMBERSHIP``) and the commands that
  do not flush.
* ``TestEditsOfAPendingNode``: a rename, a delete or a lock of a pending node.
* ``TestErrors``: a statement that raises keeps today's membership; a flush failure is
  a note of the statement's own exception.
* ``TestUndoInsideAContainer``: the one-undo-step checks run INSIDE ``with
  container()`` (the critic's fix: the flush at an undo chunk's close).
* ``TestContainerRoute``: the lint that every rig ``cmds.container`` /
  ``cmds_edit.container`` call goes through ``journal.container_cmd``.
"""

import ast
import os
from unittest import mock

from maya import cmds

import rig
from rig import container, Node
from rig._internal import container as container_module
from rig._internal import journal
from rig._internal.journal import cmds_edit, container_cmd, mel_eval, statement, user_call
from rig.nodetypes import DisplayLayer, Transform
from rig._tests._base import MayaTestCase


def _members(ctn):
    return cmds.container(ctn, query=True, nodeList=True) or []


def _scene():
    return set(cmds.ls())


@statement
def _make(k, node_type="multiplyDivide", then=None):
    """One statement: ``k`` nodes through the scope's createNode, then ``then(nodes)``."""
    nodes = [container.createNode(node_type) for _ in range(k)]
    if then is not None:
        then(nodes)
    return nodes


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        journal._S.members = []
        cmds.undoInfo(state=True)
        super().tearDown()


class TestOneAddPerStatement(_Case):

    def test_k_nodes_are_one_add_and_keep_their_order(self):
        sent = mock.patch.object(container_module, "_send_members", wraps=container_module._send_members)
        with container("ctn"):
            with sent as spy:
                nodes = _make(5)
        names = [str(n) for n in nodes]
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], names)
        # today's order: the same nodes added one call each
        cmds.container(name="ref")
        ref = [cmds.createNode("multiplyDivide", skipSelect=True) for _ in range(5)]
        for name in ref:
            cmds.container("ref", edit=True, addNode=[name], force=True)
        order = {name: i for i, name in enumerate(names)}
        ref_order = {name: i for i, name in enumerate(ref)}
        self.assertEqual([order[n] for n in _members("ctn")], [ref_order[n] for n in _members("ref")])

    def test_nothing_is_pending_between_statements(self):
        with container("ctn"):
            _make(3)
            self.assertEqual(journal._S.members, [])
            self.assertEqual(len(_members("ctn")), 3)

    def test_a_run_per_leaf(self):
        # nested real containers: the outer's nodes, the inner's, the outer's again
        sent = mock.patch.object(container_module, "_send_members", wraps=container_module._send_members)
        with container("outer"):
            with container("inner", preserve=True) as inner:
                @statement
                def build():
                    a = container.createNode("multiplyDivide")
                    b = container.createNode("multiplyDivide")
                    return a, b

                with sent as spy:
                    a, b = build()
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(sorted(_members(str(inner))), sorted([str(a), str(b)]))
            # the nested real container keeps its parent link
            self.assertEqual(cmds.container(str(inner), query=True, parentContainer=True), ["outer"])
            self.assertIn(str(inner), _members("outer"))


class TestFlushPoints(_Case):

    def _pending_then(self, check, extra=()):
        """One statement inside ``with container('ctn')``: two nodes, then ``check(nodes)``;
        the container then holds the two nodes (and ``extra``)."""
        seen = {}

        def then(nodes):
            seen["pending"] = len(journal._S.members)
            seen.update(check(nodes) or {})

        with container("ctn"):
            nodes = _make(2, then=then)
        self.assertEqual(seen.pop("pending"), 2)
        self.assertEqual(sorted(_members("ctn")), sorted([str(n) for n in nodes] + list(extra)))
        return nodes, seen

    def test_a_container_query_sees_the_members(self):
        _, seen = self._pending_then(lambda nodes: {"list": container_cmd("ctn", query=True, nodeList=True),
                                                    "find": container_cmd(query=True, findContainer=[str(nodes[0])])})
        self.assertEqual(len(seen["list"]), 2)
        self.assertEqual(seen["find"], "ctn")

    def test_a_publish_sees_the_members(self):
        def publish(nodes):
            name = container_cmd("ctn", edit=True, publishName="out")
            container_cmd("ctn", edit=True, bindAttr=(f"{nodes[0]}.outputX", name))
            return {"bound": container_cmd("ctn", query=True, bindAttr=True)}

        _, seen = self._pending_then(publish)
        self.assertTrue(seen["bound"])

    def test_the_non_made_add(self):
        def add(nodes):
            container.add(cmds.createNode("transform", name="held", skipSelect=True))
            return {"left": len(journal._S.members)}

        _, seen = self._pending_then(add, extra=["held"])
        self.assertEqual(seen["left"], 0)

    def test_a_user_call_sees_the_members(self):
        def user(nodes):
            return {"user": user_call(lambda: cmds.container("ctn", query=True, nodeList=True))}

        _, seen = self._pending_then(user)
        self.assertEqual(len(seen["user"]), 2)

    def test_outside_before_user_code(self):
        def out(nodes):
            with journal.outside():
                return {"raw": cmds.container("ctn", query=True, nodeList=True)}

        _, seen = self._pending_then(out)
        self.assertEqual(len(seen["raw"]), 2)

    def test_a_cmds_edit_of_a_membership_command_flushes(self):
        for command in ("lockNode", "rename", "delete", "undoInfo", "mel_eval"):
            with self.subTest(command):
                cmds.file(new=True, force=True)

                def edit(nodes, command=command):
                    if command == "lockNode":
                        cmds_edit.lockNode(str(nodes[1]), lock=False)
                    elif command == "rename":
                        cmds_edit.rename(str(nodes[1]), "renamed")
                    elif command == "delete":
                        cmds_edit.delete(cmds.createNode("transform", skipSelect=True))
                    elif command == "undoInfo":
                        cmds_edit.undoInfo(stateWithoutFlush=True)
                    else:
                        mel_eval("int $rig_r6_s2 = 1;")
                    return {"left": len(journal._S.members)}

                _, seen = self._pending_then(edit)
                self.assertEqual(seen["left"], 0)

    def test_the_no_membership_commands_do_not_flush(self):
        def edits(nodes):
            a, b = str(nodes[0]), str(nodes[1])
            cmds_edit.setAttr(f"{a}.input1X", 2)
            cmds_edit.addAttr(a, longName="extra", attributeType="double")
            cmds_edit.aliasAttr("ex", f"{a}.extra")
            cmds_edit.aliasAttr(f"{a}.ex", remove=True)
            cmds_edit.deleteAttr(f"{a}.extra")
            cmds_edit.connectAttr(f"{a}.outputX", f"{b}.input1X")
            cmds_edit.disconnectAttr(f"{a}.outputX", f"{b}.input1X")
            return {"left": len(journal._S.members)}

        _, seen = self._pending_then(edits)
        self.assertEqual(seen["left"], 2)
        self.assertEqual(journal.NO_MEMBERSHIP, frozenset({
            "setAttr", "addAttr", "deleteAttr", "connectAttr", "disconnectAttr", "aliasAttr",
            "parent", "xform",
        }))

    def test_an_undo_chunk_close(self):
        with container("ctn"):
            @statement
            def build():
                with rig.undo_chunk("inner"):
                    container.createNode("multiplyDivide", name="in_chunk")
                left = len(journal._S.members)
                return left

            self.assertEqual(build(), 0)
        self.assertIn("in_chunk", _members("ctn"))

    def test_outside_a_statement_it_is_sent_at_once(self):
        with container("ctn"):
            node = cmds.createNode("multiplyDivide", skipSelect=True)
            leaf = container._leaf_real_container()
            journal.defer_member(leaf, Node(node))
            self.assertEqual(journal._S.members, [])
            self.assertIn(node, _members("ctn"))

    def test_a_new_scene_drops_the_pending_adds(self):
        journal._S.members = [(object(), object())]
        journal._discard()
        self.assertEqual(journal._S.members, [])


class TestEditsOfAPendingNode(_Case):

    def test_a_raw_rename_is_read_at_the_flush(self):
        with container("ctn"):
            _make(1, then=lambda nodes: cmds.rename(str(nodes[0]), "late_name"))
        self.assertEqual(_members("ctn"), ["late_name"])

    def test_a_deleted_pending_node_is_dropped(self):
        with container("ctn"):
            nodes = _make(3, then=lambda nodes: cmds.delete(str(nodes[1])))
        self.assertEqual(sorted(_members("ctn")), sorted([str(nodes[0]), str(nodes[2])]))

    def test_a_lock_of_a_pending_node(self):
        with container("ctn"):
            nodes = _make(2, then=lambda nodes: cmds_edit.lockNode(str(nodes[0]), lock=True))
        self.assertEqual(len(_members("ctn")), 2)
        self.assertTrue(cmds.lockNode(str(nodes[0]), query=True)[0])
        cmds.lockNode(str(nodes[0]), lock=False)

    def test_the_locked_leaf_refusal_is_at_the_create(self):
        with container("ctn"):
            def lock(nodes):
                cmds_edit.lockNode("ctn", lock=True)
                container.createNode("multiplyDivide")

            with self.assertRaisesRegex(Exception, "locked"):
                _make(1, then=lock)
            cmds.lockNode("ctn", lock=False)
        self.assertEqual(len(_members("ctn")), 1)


class TestErrors(_Case):

    def test_a_statement_that_raises_keeps_the_membership(self):
        def fail(nodes):
            raise ValueError("typo")

        with container("ctn"):
            with self.assertRaisesRegex(ValueError, "typo"):
                _make(3, then=fail)
            self.assertEqual(journal._S.members, [])
            self.assertEqual(len(_members("ctn")), 3)

    def test_a_flush_failure_is_a_note(self):
        def fail(nodes):
            raise ValueError("typo")

        with container("ctn"):
            with mock.patch.object(container_module, "_flush_members", side_effect=RuntimeError("boom")):
                with self.assertRaises(ValueError) as caught:
                    _make(1, then=fail)
        self.assertIn("boom", "\n".join(getattr(caught.exception, "__notes__", [])))

    def test_undo_and_redo_of_a_statement_that_raised(self):
        def fail(nodes):
            nodes[0].input1X << nodes[1].outputX
            raise ValueError("typo")

        with container("ctn"):
            before = _scene()
            with self.assertRaises(ValueError):
                _make(2, then=fail)
            after, members = _scene(), sorted(_members("ctn"))
            while _scene() != before:
                cmds.undo()
            while _scene() != after:
                cmds.redo()
            self.assertEqual(sorted(_members("ctn")), members)


class TestUndoInsideAContainer(_Case):
    """The one-undo-step checks of test_r4b_fix / test_r4b_define, inside a container."""

    def test_a_create_with_attributes_is_one_undo_step(self):
        for label, call in {
            "Transform.create": lambda: Transform.create(name="t", tx=1, ty=2, rotateOrder="xzy"),
            "Node.create typed": lambda: Node.create("transform", name="t", tx=1, ty=2),
            "Node.create DG":    lambda: Node.create("multiplyDivide", name="md", input1X=3, operation="divide"),
            "DisplayLayer.create": lambda: DisplayLayer.create(name="L", displayType=2, visibility=False),
        }.items():
            with self.subTest(label):
                cmds.file(new=True, force=True)
                cmds.flushUndo()
                with container("ctn"):
                    before, members = _scene(), _members("ctn")
                    node = call()
                    self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.create")
                    joined = _members("ctn")
                    if label != "DisplayLayer.create":  # (a layer is not a member)
                        self.assertIn(str(node), joined)
                    cmds.undo()
                    self.assertEqual(_scene(), before)
                    self.assertEqual(_members("ctn"), members)
                    cmds.redo()
                    self.assertEqual(_members("ctn"), joined)

    def test_a_made_define_is_one_undo_step(self):
        with container("ctn"):
            before = _scene()
            Transform.define("ctl", tx=1, rotateOrder="zxy")
            self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.define")
            self.assertIn("ctl", _members("ctn"))
            cmds.undo()
            self.assertEqual(_scene(), before)

    def test_a_statement_in_an_undo_chunk_is_one_step(self):
        with container("ctn"):
            before = _scene()
            with rig.undo_chunk("build"):
                _make(3)
                _make(2)
            self.assertEqual(len(_members("ctn")), 5)
            self.assertEqual(cmds.undoInfo(query=True, undoName=True), "build")
            cmds.undo()
            self.assertEqual(_scene(), before)
            cmds.redo()
            self.assertEqual(len(_members("ctn")), 5)


# --------------------------------------------------------------------------------------- #
#  The container route lint
# --------------------------------------------------------------------------------------- #

# the names a rig module binds maya.cmds or the journal's proxy to
_CMDS = frozenset({"cmds", "_cmds", "mc", "_mc", "cmds_edit", "_cmds_edit"})

# (file, enclosing function): the raw calls allowed: the wrapper's own query, and the
# deferred adds' send (`_send_members`, made=True), which `container_cmd`'s flush runs
_ALLOWED = frozenset({
    ("_internal/journal.py", "container_cmd"),
    ("_internal/container.py", "_send_members"),
})


def container_route_violations(rel, source):
    """``[(line, message), ...]``: the ``<cmds>.container(...)`` calls and the
    ``getattr(<cmds>, "container")`` lookups of ``source`` outside `_ALLOWED`."""
    tree    = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def where(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return ""

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "container" \
                and isinstance(node.value, ast.Name) and node.value.id in _CMDS:
            if (rel, where(node)) not in _ALLOWED:
                out.append((node.lineno, f"{node.value.id}.container in {where(node)}: use container_cmd"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in _CMDS \
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "container":
            out.append((node.lineno, f"getattr({node.args[0].id}, 'container') in {where(node)}"))
    return out


class TestContainerRoute(MayaTestCase):

    def test_every_rig_container_call_goes_through_the_wrapper(self):
        root = os.path.dirname(os.path.abspath(rig.__file__))
        violations, read = [], []
        for folder, dirs, files in os.walk(root):
            rel_dir = os.path.relpath(folder, root).replace("\\", "/")
            if rel_dir.split("/")[0] in ("_tests", "examples") or "__pycache__" in rel_dir:
                dirs[:] = []
                continue
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(folder, name)
                    rel  = os.path.relpath(path, root).replace("\\", "/")
                    read.append(rel)
                    with open(path, encoding="utf-8") as handle:
                        violations += [f"{rel}:{line}: {msg}" for line, msg in
                                       container_route_violations(rel, handle.read())]
        self.assertIn("_internal/journal.py", read)
        self.assertIn("_internal/shade_convert.py", read)
        self.assertEqual(violations, [])

    def test_the_lint_sees_a_raw_call(self):
        for source in (
            "def f():\n    cmds.container('c', query=True, nodeList=True)\n",
            "def f():\n    cmds_edit.container('c', edit=True, publishName='x')\n",
            "def f():\n    _cmds_edit.container('c', edit=True, addNode=['n'])\n",
            "def f():\n    g = cmds.container\n",
            "def f():\n    getattr(cmds, 'container')('c', q=True, nodeList=True)\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(len(container_route_violations("_internal/plug.py", source)), 1)
        self.assertEqual(container_route_violations(
            "_internal/container.py",
            "def _send_members(leaf, names, made):\n    cmds.container(leaf, edit=True, addNode=names)\n"), [])
        self.assertEqual(container_route_violations(
            "_internal/plug.py", "def f():\n    container_cmd(query=True, findContainer=['n'])\n"), [])
