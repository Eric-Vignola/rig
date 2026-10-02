"""Round 5, step A3: the kind memo (H1) and the own-plug test it answers (H9).

``_fixed_attr_kind`` keeps the kind it computes for a normal attribute with no
array element on the plug's path on the plug, as ``_kind`` (the epoch token of
``_KIND_EPOCH`` and the kind), and answers from it while the plug's node is
valid: such an attribute is never deleted, renamed or retyped while its node
type is registered, and a plug's MPlug never changes, so the computation would
give the kind again. A deleted or freed node computes again (a deleted one
gives the kind, a freed one None, as before), and so does a memo of an older
epoch: every plug-in load and unload starts a new one. ``_names_own_plug``
answers a plug of a valid DG node that keeps a memo of this epoch from its
parents alone (an array root parent fails it), which is what its loop finds for
such a plug.

* ``TestTheMemoAnswersAsTheComputation``: over a matrix of plugs (numeric,
  unit, angle, enum, bool, message, compound, compound child, compound output,
  multi roots, typed, matrix, typed multi root, the choice node's generic
  slots; DG and DAG nodes, a shape; owned plugs and plugs with no owner) and
  ten states (live, a delete with undo on, its undo, its redo, a rename, a
  flushed delete, a new scene, a file open, a reference unload, a same-name
  re-create after a flush), the first call and the repeat give what bb1fb69's
  computation gives (``_old_fixed_attr_kind``, a copy of it); which plugs keep
  a memo; a kept kind is answered without a computation; an instanced shape's
  paths.
* ``TestWhatIsNeverKept``: elements (existing, missing, generic), a child of
  an element, an unindexed child of a multi, dynamic attributes (a double, a
  compound and its child; also deleted, undone and re-added under the name)
  and an extension attribute keep no memo.
* ``TestDeletedAndFreedNodes``: a deleted node's plug computes the kind (as
  before) and its undo answers from the memo; a freed one (new scene) answers
  None, its memo unread.
* ``TestTheEpoch``: ``_clear_static_data_type`` (the plug-in load and unload
  callback) starts a new epoch, after which a kept kind is computed again; a
  plug-in load and unload start one each; a plug of a plug-in node type across
  its node's delete, the unload and the reload.
* ``TestTheMemoWaitsForThePluginCallbacks``: no memo is kept before the
  plug-in callbacks are registered (rig imported before Maya was initialised).
* ``TestNamesOwnPlug``: over the plug matrix in every state a named plug can
  be in, ``_names_own_plug`` answers as bb1fb69's loop (``_old_names_own_plug``);
  the memo answers DG plugs only (no fn set is made for them; a DAG plug, a
  deleted node, an older epoch or no memo take the loop).
* ``TestTheKindIsComputedOnce``: ``md.input1X << 1.0`` computes the kind of
  ``input1X`` once (bb1fb69: twice), a connect once per plug, a plug set twice
  once; each Plug object keeps its own memo (plugs are never cached).
"""

import contextlib
import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node, Plug
from rig._internal import callbacks
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

BASE = _base.__name__


# --------------------------------------------------------------------------- #
#  bb1fb69's computations, the oracles
# --------------------------------------------------------------------------- #


def _old_fixed_attr_kind(attr):
    """``_fixed_attr_kind`` as bb1fb69 computed it, on every call."""
    try:
        mplug = attr._mplug
        fn    = _base._plug_node_fn_set(attr, mplug)
        if fn is None:
            return None
        mobject = _base._attr_mobject(attr)
        if fn.attributeClass(mobject) == OpenMaya.MFnDependencyNode.kInvalidAttr:
            return None
        elements = []
        plug     = mplug
        while True:
            if plug.isElement:
                if plug.logicalIndex() < 0:
                    return None
                elements.append(plug)
                plug = plug.array()
            elif plug.isChild:
                plug = plug.parent()
            else:
                break
        for element in reversed(elements):
            indices = element.array().getExistingArrayAttributeIndices()
            if element.logicalIndex() not in indices:
                return None
        return mobject.apiType()
    except Exception:
        return None


def _old_names_own_plug(attr):
    """``_names_own_plug`` as bb1fb69 answered it: the loop, for every plug."""
    cls = type(attr)
    if (
        not isinstance(attr, Attribute)
        or cls.__str__ is not Attribute.__str__
        or cls.full_name is not Attribute.full_name
        or cls.alias is not Attribute.alias
    ):
        return False
    handle = attr.__dict__.get("_attrh")
    if handle is not None and not handle.isAlive():
        return False
    try:
        mplug   = attr._mplug
        mobject = mplug.node()
        fn      = OpenMaya.MFnDependencyNode(mobject)
        invalid = OpenMaya.MFnDependencyNode.kInvalidAttr
        if fn.attributeClass(mplug.attribute()) == invalid:
            return False
        path = None
        if mobject.hasFn(OpenMaya.MFn.kDagNode):
            path = _base._naming_dag_path(attr)
            if path is None or not path.isValid():
                return False
        plug = mplug
        while True:
            if plug.isElement:
                index = plug.logicalIndex()
                if index < 0:
                    return False
                name = plug.partialName(False, False, False, False, False, False)
                if not name.endswith("]") and (
                    path is None or index != path.instanceNumber()
                ):
                    return False
                plug = plug.array()
            elif plug.isChild:
                plug = plug.parent()
                if plug.isArray:
                    return False
            else:
                return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  The plug matrix and the states
# --------------------------------------------------------------------------- #

# the matrix's nodes, deleted / renamed by the frees (``cube`` is pc's mesh)
_NODES = ("md", "pma", "ch", "t", "cube", "pc")


def _scene():
    """The nodes of the plug matrix, in the current scene."""
    cmds.createNode("multiplyDivide", name="md")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.createNode("choice", name="ch")
    cmds.polyCube(name="cube", constructionHistory=True)
    cmds.rename(cmds.listConnections("cubeShape.inMesh")[0], "pc")
    cmds.createNode("transform", name="t")
    cmds.createNode("locator", name="locShape", parent="t")
    cmds.addAttr("md", ln="dynf", at="double")
    cmds.addAttr("md", ln="dync", at="double3")
    for axis in "XYZ":
        cmds.addAttr("md", ln="dync" + axis, at="double", parent="dync")
    cmds.setAttr("pma.input1D[0]", 1.0)
    cmds.setAttr("pma.input3D[0].input3Dx", 1.0)
    cmds.connectAttr("t.tx", "ch.input[0]")


def _unindexed_child(name):
    """An Attribute of ``<name>.input3Dx`` without an element index."""
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    mobject = sel.getDependNode(0)
    fn      = OpenMaya.MFnDependencyNode(mobject)
    return Attribute(OpenMaya.MPlug(mobject, fn.attribute("input3Dx")))


def _plugs(p=""):
    """[(label, kept, dag, plug)] of the matrix's nodes (their names prefixed by
    ``p``): whether the plug keeps its kind, and whether its node is a DAG node."""
    md, pma, ch, pc, t = (Node(p + n) for n in ("md", "pma", "ch", "pc", "t"))
    loc = Node(p + "locShape")
    return [
        ("md.input1X numeric", True, False, md.input1X),
        ("md.input1 compound", True, False, md.input1),
        ("md.input1.input1X compound child", True, False, md.input1.input1X),
        ("md.operation enum", True, False, md.operation),
        ("md.message message", True, False, md.message),
        ("md.dynf dynamic", False, False, md.dynf),
        ("md.dync dynamic compound", False, False, md.dync),
        ("md.dync.dyncX dynamic child", False, False, md.dync.dyncX),
        ("pma.input1D multi root", True, False, pma.input1D),
        ("pma.input3D compound multi root", True, False, pma.input3D),
        ("pma.input3D.input3Dx unindexed child", False, False, _unindexed_child(p + "pma")),
        ("pma.input1D[0] element", False, False, pma.input1D[0]),
        ("pma.input1D[5] missing element", False, False, pma.input1D[5]),
        ("pma.input3D[0].input3Dx child of an element", False, False, pma.input3D[0].input3Dx),
        ("pma.output3D compound output", True, False, pma.output3D),
        ("ch.output generic", True, False, ch.output),
        ("ch.input generic multi root", True, False, ch.input),
        ("ch.input[0] generic element", False, False, ch.input[0]),
        ("ch.selector numeric", True, False, ch.selector),
        ("pc.output typed", True, False, pc.output),
        ("t.tx unit", True, True, t.tx),
        ("t.rx angle", True, True, t.rx),
        ("t.t compound", True, True, t.t),
        ("t.rotateOrder enum", True, True, t.rotateOrder),
        ("t.visibility bool", True, True, t.visibility),
        ("t.worldMatrix typed multi root", True, True, t.worldMatrix),
        ("t.worldMatrix[0] element", False, True, t.worldMatrix[0]),
        ("t.matrix typed", True, True, t.matrix),
        ("t.offsetParentMatrix matrix", True, True, t.offsetParentMatrix),
        ("locShape.localPositionX shape child", True, True, loc.localPositionX),
        ("Plug(md.input1X) no owner", True, False, Plug(p + "md.input1X")),
        ("Attribute(md.input1Y) no owner", True, False, Attribute(p + "md.input1Y")),
        ("Plug(t.tx) no owner", True, True, Plug(p + "t.tx")),
    ]


def _churn():
    """Nodes made after a free, so freed memory is reused."""
    for _ in range(50):
        cmds.createNode("multiplyDivide")


# the states a held plug is read in; the last ones free its node
_LIVE_STATES  = ("live", "delete", "undo", "redo", "rename")
_FREED_STATES = ("flush", "new", "open", "unload", "recreate")


class _States:
    """Builds the matrix in a fresh scene, holds its plugs, computes every
    plug's kind once (the memo the build would leave), then puts the scene in a
    state. Files go to a temporary folder."""

    def __init__(self, case):
        self.folder = tempfile.mkdtemp(prefix="rig_r5_kind_")
        case.addCleanup(shutil.rmtree, self.folder, True)

    def held(self, state):
        """[(label, kept, dag, plug)] after ``state``, and the kinds computed
        before it {label: kind}."""
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        prefix = ""
        if state == "unload":
            _scene()
            path = os.path.join(self.folder, "matrix.ma").replace(os.sep, "/")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            prefix = "ref:"
        else:
            _scene()
        if state == "open":
            path = os.path.join(self.folder, "scene.ma").replace(os.sep, "/")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
        held   = _plugs(prefix)
        before = {label: _base._fixed_attr_kind(plug) for label, _, _, plug in held}
        if state in ("delete", "undo", "redo", "flush", "recreate"):
            cmds.delete(*(prefix + n for n in _NODES))
        if state in ("undo", "redo"):
            cmds.undo()
        if state == "redo":
            cmds.redo()
        if state == "rename":
            for name in _NODES:
                cmds.rename(name, name + "_renamed")
        if state in ("flush", "recreate"):
            cmds.flushUndo()
            _churn()
        if state == "recreate":
            _scene()
        if state == "new":
            cmds.file(new=True, force=True)
            _churn()
        if state == "open":
            cmds.file(path, open=True, force=True)
            _churn()
        if state == "unload":
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            _churn()
        return held, before


def _computations():
    """A spy on the kind computation: `_plug_node_fn_set` is called by it alone."""
    return mock.patch.object(_base, "_plug_node_fn_set", wraps=_base._plug_node_fn_set)


class _CountingFn(OpenMaya.MFnDependencyNode):
    """``MFnDependencyNode``, counting the fn sets made."""

    made = 0

    def __init__(self, *args):
        _CountingFn.made += 1
        super().__init__(*args)


@contextlib.contextmanager
def _fn_sets_made():
    """The block makes its ``MFnDependencyNode``s as ``_CountingFn`` (its
    ``made`` counts them)."""
    _CountingFn.made = 0
    with mock.patch.object(OpenMaya, "MFnDependencyNode", _CountingFn):
        yield _CountingFn


# --------------------------------------------------------------------------- #
#  The tests
# --------------------------------------------------------------------------- #


class TestTheMemoAnswersAsTheComputation(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_kinds_in_every_state(self):
        states = _States(self)
        for state in _LIVE_STATES + _FREED_STATES:
            held, before = states.held(state)
            for label, kept, _, plug in held:
                with self.subTest(state=state, plug=label):
                    expected = _old_fixed_attr_kind(plug)
                    self.assertEqual(_base._fixed_attr_kind(plug), expected)
                    # the repeat: the memo, where one is kept
                    self.assertEqual(_base._fixed_attr_kind(plug), expected)
                    if state in ("live", "undo", "rename"):
                        self.assertEqual(before[label], expected)
                    if state in _FREED_STATES:
                        self.assertIsNone(expected)
        cmds.file(new=True, force=True)

    def test_which_plugs_keep_their_kind(self):
        _scene()
        for label, kept, _, plug in _plugs():
            with self.subTest(plug=label):
                self.assertNotIn("_kind", vars(plug))  # not in the initial state
                kind = _base._fixed_attr_kind(plug)
                if kept:
                    self.assertIsNotNone(kind)
                    self.assertEqual(vars(plug)["_kind"], (_base._KIND_EPOCH[0], kind))
                    self.assertEqual(list(vars(plug))[-1], "_kind")
                else:
                    self.assertNotIn("_kind", vars(plug))

    def test_a_kept_kind_is_not_computed_again(self):
        _scene()
        for label, kept, _, plug in _plugs():
            with self.subTest(plug=label):
                kind = _base._fixed_attr_kind(plug)
                with _computations() as computed:
                    self.assertEqual(_base._fixed_attr_kind(plug), kind)
                    self.assertEqual(_base._fixed_attr_kind(plug), kind)
                self.assertEqual(computed.call_count, 0 if kept else 2)

    def test_an_instanced_shape(self):
        cmds.createNode("transform", name="t1")
        cmds.createNode("transform", name="t2")
        cmds.createNode("locator", name="loc", parent="t1")
        cmds.parent("t1|loc", "t2", add=True, shape=True)
        for name in ("t1|loc", "t2|loc"):
            for plug in (Node(name).lpx, Plug(f"{name}.lpx"), Node(name).worldMatrix,
                         Plug(f"{name}.worldMatrix[1]")):
                with self.subTest(plug=str.__str__(plug)):
                    expected = _old_fixed_attr_kind(plug)
                    self.assertEqual(_base._fixed_attr_kind(plug), expected)
                    self.assertEqual(_base._fixed_attr_kind(plug), expected)
                    self.assertIs("_kind" in vars(plug), not plug.plug.isElement)


class TestWhatIsNeverKept(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _assert_not_kept(self, plug, expected):
        self.assertEqual(_base._fixed_attr_kind(plug), expected)
        self.assertEqual(_old_fixed_attr_kind(plug), expected)
        self.assertNotIn("_kind", vars(plug))

    def test_a_dynamic_attribute_deleted_undone_and_added_again(self):
        md   = Node(cmds.createNode("multiplyDivide", name="md"))
        cmds.addAttr("md", ln="dd", at="double")
        plug = md.dd
        str(plug)
        numeric = OpenMaya.MFn.kNumericAttribute
        self._assert_not_kept(plug, numeric)
        cmds.deleteAttr("md.dd")
        self._assert_not_kept(plug, None)
        cmds.undo()
        self._assert_not_kept(plug, numeric)
        cmds.redo()
        cmds.addAttr("md", ln="dd", at="double3")
        cmds.addAttr("md", ln="ddX", at="double", parent="dd")
        cmds.addAttr("md", ln="ddY", at="double", parent="dd")
        cmds.addAttr("md", ln="ddZ", at="double", parent="dd")
        # the held plug's attribute is not the new one of its name
        self._assert_not_kept(plug, None)
        self._assert_not_kept(md.dd, OpenMaya.MFn.kAttribute3Double)

    def test_an_extension_attribute(self):
        # addExtension flushes the undo queue
        cmds.addExtension(nodeType="plusMinusAverage", longName="rigA3Ext",
                          attributeType="double")
        self.addCleanup(cmds.deleteExtension, nodeType="plusMinusAverage",
                        attribute="rigA3Ext", forceDelete=True)
        self.addCleanup(cmds.file, new=True, force=True)
        pma  = Node(cmds.createNode("plusMinusAverage", name="pma"))
        plug = pma.rigA3Ext
        str(plug)
        self._assert_not_kept(plug, OpenMaya.MFn.kNumericAttribute)
        self._assert_not_kept(Plug("pma.rigA3Ext"), OpenMaya.MFn.kNumericAttribute)

    def test_elements_and_their_children(self):
        _scene()
        for label, kept, _, plug in _plugs():
            if "element" in label or "unindexed" in label:
                with self.subTest(plug=label):
                    self.assertFalse(kept)
                    self._assert_not_kept(plug, _old_fixed_attr_kind(plug))
        # an element made by its own set keeps none either, and once it is
        # removed it has no kind
        element = Node("pma").input1D[7]
        element << 2.0
        self._assert_not_kept(element, OpenMaya.MFn.kNumericAttribute)
        cmds.removeMultiInstance("pma.input1D[7]", b=True)
        self._assert_not_kept(element, None)
        child = Node("pma").input3D[0].input3Dx
        self._assert_not_kept(child, OpenMaya.MFn.kNumericAttribute)
        cmds.removeMultiInstance("pma.input3D[0]", b=True)
        self._assert_not_kept(child, None)


class TestDeletedAndFreedNodes(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_deleted_node_computes_the_kind(self):
        md = Node(cmds.createNode("multiplyDivide", name="md"))
        t  = Node(cmds.createNode("transform", name="t"))
        for node, plug, kind in (
            ("md", md.input1X, OpenMaya.MFn.kNumericAttribute),
            ("md", Plug("md.input1Y"), OpenMaya.MFn.kNumericAttribute),
            ("t", t.tx, OpenMaya.MFn.kDoubleLinearAttribute),
        ):
            with self.subTest(plug=str.__str__(plug)):
                self.assertEqual(_base._fixed_attr_kind(plug), kind)
                cmds.delete(node)
                # computed on the deleted node, as before: the kind
                with _computations() as computed:
                    self.assertEqual(_base._fixed_attr_kind(plug), kind)
                self.assertEqual(computed.call_count, 1)
                self.assertEqual(_old_fixed_attr_kind(plug), kind)
                cmds.undo()
                # valid again: the memo answers
                with _computations() as computed:
                    self.assertEqual(_base._fixed_attr_kind(plug), kind)
                self.assertEqual(computed.call_count, 0)

    def test_a_freed_node_answers_none(self):
        md    = Node(cmds.createNode("multiplyDivide", name="md"))
        t     = Node(cmds.createNode("transform", name="t"))
        plugs = [md.input1X, Plug("md.input1Y"), t.tx, Plug("t.ty")]
        memos = []
        for plug in plugs:
            self.assertIsNotNone(_base._fixed_attr_kind(plug))
            memos.append(vars(plug)["_kind"])
        cmds.file(new=True, force=True)
        _churn()
        for plug, memo in zip(plugs, memos):
            with self.subTest(plug=str.__str__(plug)):
                # not the memo: the computation, which reads only the handle
                with _computations() as computed:
                    self.assertIsNone(_base._fixed_attr_kind(plug))
                self.assertEqual(computed.call_count, 1)
                self.assertIs(vars(plug)["_kind"], memo)
                self.assertIsNone(_old_fixed_attr_kind(plug))


def _unloaded_plugin():
    """(plug-in, a node type it defines, a static attribute of it) for a plug-in
    that is not loaded, or None."""
    for plugin, node_type in (("invertShape", "offsetDeformer"), ("curveWarp", "curveWarp")):
        try:
            if not cmds.pluginInfo(plugin, query=True, loaded=True):
                return plugin, node_type, "envelope"
        except RuntimeError:
            continue
    return None


def _unload(plugin):
    cmds.file(new=True, force=True)
    cmds.flushUndo()
    if cmds.pluginInfo(plugin, query=True, loaded=True):
        cmds.unloadPlugin(plugin)


class TestTheEpoch(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_new_epoch_computes_again(self):
        plug  = Node(cmds.createNode("multiplyDivide", name="md")).input1X
        kind  = _base._fixed_attr_kind(plug)
        token = vars(plug)["_kind"][0]
        self.assertIs(token, _base._KIND_EPOCH[0])
        _base._clear_static_data_type()
        self.assertIsNot(_base._KIND_EPOCH[0], token)
        with _computations() as computed:
            self.assertEqual(_base._fixed_attr_kind(plug), kind)
            self.assertEqual(_base._fixed_attr_kind(plug), kind)
        self.assertEqual(computed.call_count, 1)
        self.assertEqual(vars(plug)["_kind"], (_base._KIND_EPOCH[0], kind))

    def test_a_plugin_load_and_unload_start_new_epochs(self):
        found = _unloaded_plugin()
        if found is None:
            self.skipTest("no unloaded test plug-in")
        plugin = found[0]
        self.addCleanup(_unload, plugin)
        plug   = Node(cmds.createNode("multiplyDivide", name="md")).input1X
        kind   = _base._fixed_attr_kind(plug)
        epochs = [_base._KIND_EPOCH[0]]
        cmds.loadPlugin(plugin, quiet=True)
        epochs.append(_base._KIND_EPOCH[0])
        cmds.unloadPlugin(plugin)
        epochs.append(_base._KIND_EPOCH[0])
        self.assertEqual(len({id(token) for token in epochs}), 3)
        with _computations() as computed:
            self.assertEqual(_base._fixed_attr_kind(plug), kind)
        self.assertEqual(computed.call_count, 1)

    def test_a_plugin_node_type_across_unload_and_reload(self):
        found = _unloaded_plugin()
        if found is None:
            self.skipTest("no unloaded test plug-in")
        plugin, node_type, attr = found
        self.addCleanup(_unload, plugin)
        cmds.loadPlugin(plugin, quiet=True)
        held = getattr(Node(cmds.createNode(node_type, name="pnode")), attr)
        kind = _base._fixed_attr_kind(held)
        self.assertEqual(kind, _old_fixed_attr_kind(held))
        self.assertIn("_kind", vars(held))
        cmds.delete("pnode")
        cmds.flushUndo()
        cmds.unloadPlugin(plugin)
        cmds.loadPlugin(plugin, quiet=True)
        _churn()
        # the held plug's node was freed
        self.assertIsNone(_base._fixed_attr_kind(held))
        self.assertIsNone(_old_fixed_attr_kind(held))
        # the same build again: a new node and plug, a memo of this epoch
        again = getattr(Node(cmds.createNode(node_type, name="pnode")), attr)
        self.assertEqual(_base._fixed_attr_kind(again), kind)
        self.assertEqual(vars(again)["_kind"], (_base._KIND_EPOCH[0], kind))


class TestTheMemoWaitsForThePluginCallbacks(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_no_memo_before_the_plugin_callbacks(self):
        entry = callbacks.registry().pop(BASE)
        callbacks._remove(entry["ids"])

        def restore():
            if BASE not in callbacks.registry():
                _base._register_plugin_callbacks()

        self.addCleanup(restore)
        cmds.createNode("multiplyDivide", name="md")
        kind = OpenMaya.MFn.kNumericAttribute
        with mock.patch.object(callbacks, "maya_is_initialised", return_value=False):
            self.assertFalse(_base._register_plugin_callbacks())
            plug = Plug("md.input1X")
            # answered, but not kept, while no plug-in callback can start an epoch
            self.assertEqual(_base._fixed_attr_kind(plug), kind)
            self.assertNotIn("_kind", vars(plug))
        self.assertEqual(_base._fixed_attr_kind(plug), kind)
        self.assertTrue(_base._PLUGIN_CALLBACKS_READY)
        self.assertEqual(len(callbacks.registered_ids(BASE)), 2)
        self.assertEqual(vars(plug)["_kind"], (_base._KIND_EPOCH[0], kind))


class TestNamesOwnPlug(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_answers_are_the_loops(self):
        states = _States(self)
        for state in _LIVE_STATES:
            held, _ = states.held(state)
            for label, kept, dag, plug in held:
                with self.subTest(state=state, plug=label):
                    try:
                        str(plug)  # named, as `connect` names a plug first
                    except RuntimeError as exc:
                        self.assertTrue(str(exc).endswith("already deleted!"))
                    self.assertIs(_base._names_own_plug(plug), _old_names_own_plug(plug))
        cmds.file(new=True, force=True)

    def test_the_memo_answers_dg_plugs_only(self):
        _scene()
        for label, kept, dag, plug in _plugs():
            with self.subTest(plug=label):
                str(plug)
                _base._fixed_attr_kind(plug)
                expected = _old_names_own_plug(plug)
                with _fn_sets_made() as fn:
                    self.assertIs(_base._names_own_plug(plug), expected)
                # the loop makes one fn set of the plug's node
                self.assertEqual(fn.made, 0 if kept and not dag else 1)

    def test_a_forged_memo_shows_the_path(self):
        # a deleted dynamic attribute is not on its node: the loop answers
        # False, the parents alone True, so a forged memo tells which path a
        # plug takes
        for node, node_type in (("md", "multiplyDivide"), ("t", "transform"),
                                ("md2", "multiplyDivide")):
            cmds.createNode(node_type, name=node)
            cmds.addAttr(node, ln="gone", at="double")
        plugs = {"dg": Node("md").gone, "dag": Node("t").gone, "deleted": Node("md2").gone}
        for plug in plugs.values():
            str(plug)
        for node in ("md", "t", "md2"):
            cmds.deleteAttr(f"{node}.gone")
        cmds.delete("md2")
        kind = OpenMaya.MFn.kNumericAttribute
        for label, plug in plugs.items():
            with self.subTest(plug=label):
                self.assertTrue(vars(plug)["_attrh"].isAlive())  # on the undo queue
                self.assertFalse(_old_names_own_plug(plug))
                vars(plug)["_kind"] = (_base._KIND_EPOCH[0], kind)
                # a DG plug of a valid node is answered from its parents
                self.assertIs(_base._names_own_plug(plug), label == "dg")
                vars(plug)["_kind"] = (object(), kind)  # an older epoch: the loop
                self.assertFalse(_base._names_own_plug(plug))
                del vars(plug)["_kind"]
                self.assertFalse(_base._names_own_plug(plug))

    def test_a_child_of_an_array_root(self):
        # a child of the multi root, not of one of its elements: its parent is
        # an element with no index, so it keeps no kind and takes the loop
        pma  = Node(cmds.createNode("plusMinusAverage", name="pma"))
        plug = _unindexed_child("pma")
        self.assertTrue(plug.plug.isChild)
        parent = plug.plug.parent()
        self.assertTrue(parent.isElement and parent.logicalIndex() < 0)
        self.assertIsNone(_base._fixed_attr_kind(plug))
        self.assertNotIn("_kind", vars(plug))
        with _fn_sets_made() as fn:
            self.assertFalse(_base._names_own_plug(plug))
        self.assertEqual(fn.made, 1)
        self.assertFalse(_old_names_own_plug(plug))
        # its multi root keeps one, and is answered from it
        root = pma.input3D
        _base._fixed_attr_kind(root)
        self.assertIn("_kind", vars(root))
        with _fn_sets_made() as fn:
            self.assertTrue(_base._names_own_plug(root))
        self.assertEqual(fn.made, 0)


class TestTheKindIsComputedOnce(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _computed(self, op):
        with _computations() as computed:
            op()
        return [call.args[0] for call in computed.call_args_list]

    def test_a_set(self):
        md   = Node(cmds.createNode("multiplyDivide", name="md"))
        plug = md.input1X
        computed = self._computed(lambda: plug << 1.0)
        self.assertEqual(len(computed), 1)  # bb1fb69: 2
        self.assertIs(computed[0], plug)
        self.assertEqual(cmds.getAttr("md.input1X"), 1.0)
        # the plug set again computes nothing
        self.assertEqual(self._computed(lambda: plug << 2.0), [])
        self.assertEqual(cmds.getAttr("md.input1X"), 2.0)

    def test_a_connect(self):
        a = Node(cmds.createNode("multiplyDivide", name="a"))
        b = Node(cmds.createNode("multiplyDivide", name="b"))
        dst, src = a.input1X, b.outputX
        computed = self._computed(lambda: dst << src)
        # once per plug (bb1fb69: twice per plug)
        self.assertEqual(len(computed), 2)
        self.assertEqual({id(x) for x in computed}, {id(dst), id(src)})
        self.assertEqual(cmds.listConnections("a.input1X", plugs=True, source=True),
                         ["b.outputX"])

    def test_each_plug_object_keeps_its_own(self):
        md     = Node(cmds.createNode("multiplyDivide", name="md"))
        first  = md.input1X
        second = md.input1X
        self.assertIsNot(first, second)  # plugs are never cached
        computed = self._computed(lambda: [_base._fixed_attr_kind(p) for p in
                                           (first, second, first, second)])
        self.assertEqual([id(p) for p in computed], [id(first), id(second)])
