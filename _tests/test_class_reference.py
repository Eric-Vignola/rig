import tempfile
from pathlib import Path

from maya import cmds
from rig.nodetypes import PyNode, Reference
from rig._tests._base import MayaTestCase


class TestReference(MayaTestCase):
    """
    Reference node unit tests.
    """

    TEST_CASE_START_NEW_SCENE = True

    def test_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cmds.polyCube(ch=False, name="cube")
            path = Path(f"{temp_dir}/test.ma").as_posix()
            cmds.file(rename=path)
            cmds.file(f=True, type="mayaAscii", save=True)
            self.new_scene()

            ref1 = Reference.create(path, namespace="ns1")
            ref2 = Reference.create(path, namespace="ns2")

            self.assertEqual(ref1.file_path, path)
            self.assertEqual(ref2.file_path, path)
            self.assertEqual(ref1.file_path_with_copy_number, path)
            self.assertEqual(ref2.file_path_with_copy_number, path + "{1}")
            self.assertEqual(ref1.namespace, "ns1")
            self.assertEqual(ref2.namespace, "ns2")

            self.assertTrue(PyNode("ns1:cube") in ref1.get_nodes())
            self.assertTrue(PyNode("ns2:cube") in ref2.get_nodes())

            self.assertEqual(Reference.find_by_path(path), [ref1, ref2])

            ref1.delete()
            self.assertFalse(cmds.objExists("ns1:cube"))
            self.assertEqual(Reference.find_by_path(path), [ref2])