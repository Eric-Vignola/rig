"""Round 6, user decision (replaces step S9): the in-session cleanup registry.

rig no longer stamps the ``__rig__`` tag at build time by default. The nodes
``_gc_tag`` tagged (``container.createNode`` of a GC-eligible type, a typed create
of one) join an in-memory registry of ``MObjectHandle`` s instead: no node and no
attribute is written. ``cleanup()``'s candidates are the valid registry nodes plus
the nodes that carry a ``__rig__`` tag already (older files stay cleanable); the
scope filter, the fixpoint, the empty-container promotion and the single delete
are unchanged. The registry is cleared on a new or opened scene with the memo
caches, and checked for validity when used: an undone create does not count, the
same node redone does again. ``rig.set_options(cleanup_tags=True)`` (off by
default) stamps the tag at creation as before, for cleanup in a later session.
"""

import os
import shutil
import sys
import tempfile

from maya import cmds

from rig import cleanup, container, get_options, Node, set_options
from rig._internal import container as container_module
from rig._internal.container import _RIG_TAG, ContainerOptions, _is_rig_owned
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _tagged_nodes():
    return sorted(cmds.ls(f"*.{_RIG_TAG}", objectsOnly=True, recursive=True) or [])


def _orphans():
    """Two transforms, a used network, orphans in a container and at the root;
    returns the orphan-producing plugs' node names."""
    a = Node.create("transform", name="a")
    b = Node.create("transform", name="b")
    with container("rig1"):
        b.ty << a.tx * 2 + 1
        a.tx * 3  # an orphan
        (a.tx + 1) * 4  # an orphan chain
    a.ty * 5  # an orphan at the scene root
    b.tz << a.ty - 1


def _report(done):
    return {key: done[key] for key in ("by_owner", "deleted_containers")}


class TestRegistryCleanup(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._folder = tempfile.mkdtemp(prefix="rig_r6_registry_")
        set_options(cleanup_tags=False)

    def tearDown(self):
        set_options(cleanup_tags=False)
        cmds.file(new=True, force=True)
        shutil.rmtree(self._folder, ignore_errors=True)
        super().tearDown()

    def _save(self, name):
        path = os.path.join(self._folder, name).replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, force=True, type="mayaAscii")
        return path

    def _reopen(self, path):
        cmds.file(new=True, force=True)
        cmds.file(path, open=True, force=True)

    # -- the option

    def test_the_option_is_off_by_default(self):
        self.assertIs(ContainerOptions.cleanup_tags, False)
        self.assertIs(get_options()["cleanup_tags"], False)
        self.assertIs(container_module._GC_REGISTRY, getattr(sys, "_rig_gc_registry"))

    # -- registry cleanup, the same as the tagged route

    def test_default_build_carries_no_tag_and_cleans_as_the_tag_did(self):
        results = []
        for tags in (False, True):
            cmds.file(new=True, force=True)
            set_options(cleanup_tags=tags)
            _orphans()
            set_options(cleanup_tags=False)
            tagged = _tagged_nodes()
            dry    = _report(cleanup(dry_run=True))
            scoped = _report(cleanup(container="rig1", dry_run=True))
            done   = _report(cleanup())
            results.append((dry, scoped, done, sorted(cmds.ls()), tagged))
        default, with_tags = results
        self.assertEqual(default[4], [])  # no __rig__ attribute by default
        self.assertTrue(with_tags[4])
        self.assertTrue(default[0]["by_owner"].get("rig1"))
        self.assertTrue(default[0]["by_owner"].get("(scene root)"))
        self.assertEqual(list(default[1]["by_owner"]), ["rig1"])
        self.assertEqual(default[:4], with_tags[:4])

    def test_a_saved_default_file_has_no_tag_line(self):
        _orphans()
        path = self._save("default.ma")
        with open(path) as fh:
            self.assertNotIn(_RIG_TAG, fh.read())

    def test_user_nodes_and_unowned_types_stay(self):
        cmds.createNode("multiplyDivide", name="user_mul")
        with container("gc"):
            Node.create("transform", name="keep")
            Node.create("multiplyDivide", name="dead")
        self.assertFalse(_is_rig_owned("user_mul"))
        self.assertFalse(_is_rig_owned("keep"))
        self.assertTrue(_is_rig_owned("dead"))
        done = cleanup()
        self.assertEqual(done["by_owner"], {"gc": ["dead"]})
        self.assertTrue(cmds.objExists("user_mul") and cmds.objExists("keep"))

    def test_cleanup_on_exit_uses_the_registry(self):
        a = Node.create("transform", name="a")
        with container("tmp", cleanup_on_exit=True):
            orphan = (a.tx * 7).get_node()
            name = str(orphan)
            self.assertTrue(_is_rig_owned(name))
        self.assertFalse(cmds.objExists(name))

    # -- undo / redo

    def _orphan_in_one_step(self):
        """An orphan multiply, made as one undo step; its name."""
        a = Node.create("transform", name="a")
        cmds.flushUndo()
        cmds.undoInfo(openChunk=True)
        try:
            return str((a.tx * 3).get_node())
        finally:
            cmds.undoInfo(closeChunk=True)

    def test_undo_and_redo_of_a_create(self):
        name = self._orphan_in_one_step()
        self.assertTrue(_is_rig_owned(name))
        self.undo_steps(1)
        self.assertFalse(cmds.objExists(name))
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {})
        self.redo_steps(1)
        self.assertTrue(_is_rig_owned(name))
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {"(scene root)": [name]})

    def test_undo_of_a_cleanup_and_of_a_delete(self):
        name = self._orphan_in_one_step()
        self.assertEqual(cleanup()["by_owner"], {"(scene root)": [name]})
        self.assertFalse(cmds.objExists(name))
        self.undo_steps(1)
        self.assertTrue(_is_rig_owned(name))
        cmds.delete(name)
        self.assertFalse(cmds.objExists(name))
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {})
        self.undo_steps(1)
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {"(scene root)": [name]})

    def test_a_renamed_node_counts_by_its_new_name(self):
        a = Node.create("transform", name="a")
        name = str((a.tx * 3).get_node())
        cmds.rename(name, "renamed_orphan")
        self.assertTrue(_is_rig_owned("renamed_orphan"))
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {"(scene root)": ["renamed_orphan"]})

    # -- new / open

    def test_a_new_scene_clears_it(self):
        _orphans()
        self.assertTrue(container_module._GC_REGISTRY)
        cmds.file(new=True, force=True)
        self.assertEqual(container_module._GC_REGISTRY, {})

    def test_an_opened_file_clears_it_and_untagged_nodes_stay(self):
        _orphans()
        path = self._save("untagged.ma")
        self.assertTrue(container_module._GC_REGISTRY)
        self._reopen(path)
        self.assertEqual(container_module._GC_REGISTRY, {})
        # by design: the registry is not saved, so nothing is a candidate
        self.assertEqual(cleanup(dry_run=True)["by_owner"], {})

    def test_the_option_on_stamps_tags_and_a_reopened_file_cleans(self):
        set_options(cleanup_tags=True)
        _orphans()
        set_options(cleanup_tags=False)
        tagged = _tagged_nodes()
        self.assertTrue(tagged)
        before = _report(cleanup(dry_run=True))
        self._reopen(self._save("tagged.ma"))
        self.assertEqual(_tagged_nodes(), tagged)
        self.assertEqual(_report(cleanup(dry_run=True)), before)
        self.assertEqual(_report(cleanup()), before)

    def test_an_old_tagged_file_cleans(self):
        # a file an older rig built: the tag on every math node, through cmds
        cmds.createNode("transform", name="a")
        for name in ("used", "orphan", "untagged"):
            cmds.createNode("multiplyDivide", name=name)
        cmds.connectAttr("a.tx", "used.input1X")
        cmds.connectAttr("used.outputX", "a.ty")
        cmds.connectAttr("a.tx", "orphan.input1X")
        cmds.connectAttr("a.tx", "untagged.input1X")
        for name in ("used", "orphan"):
            cmds.addAttr(name, longName=_RIG_TAG, attributeType="bool", hidden=True)
            cmds.setAttr(f"{name}.{_RIG_TAG}", True, lock=True)
        self._reopen(self._save("old.ma"))
        self.assertEqual(cleanup()["by_owner"], {"(scene root)": ["orphan"]})
        self.assertEqual([cmds.objExists(n) for n in ("used", "orphan", "untagged")], [True, False, True])
