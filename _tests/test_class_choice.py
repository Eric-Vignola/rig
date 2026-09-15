from maya import cmds
from rig.maya.nodetypes import PyNode
from rig._tests._base import MayaTestCase

TEST_TYPES   = ["doubleLinear", "message", "matrix"]
DEFAULT_TYPE = "Tdata"


class TestChoice(MayaTestCase):
    """
    Test the Choice class
    """

    TEST_START_NEW_SCENE = True

    def _setup_scene(self, connect: bool = False):
        self.xform  = PyNode(cmds.createNode("transform"))
        self.choice = PyNode(cmds.createNode("choice"))
        if connect:
            self.xform.tx      >> self.choice.input[0]
            self.xform.message >> self.choice.input[1]
            self.xform.matrix  >> self.choice.input[2]

    def test_type_resolving_connected(self):
        self._setup_scene(connect=True)

        # inputs should have consistent types
        for i, t in enumerate(TEST_TYPES):
            self.assertEqual(self.choice.input[i].data_type, t)

        # output type depends on the selected input
        for i, t in enumerate(TEST_TYPES):
            self.choice.selector.set(i)
            self.assertEqual(self.choice.output.data_type, t)

        # unconnected input resolve to Tdata
        self.assertEqual(self.choice.input[3].data_type, DEFAULT_TYPE)

    def test_type_resolving_not_connected(self):
        # if no connections, inputs and outputs should all resolve to Tdata
        self._setup_scene(connect=False)
        for i, _ in enumerate(TEST_TYPES):
            self.assertEqual(self.choice.input[i].data_type, DEFAULT_TYPE)
        self.assertEqual(self.choice.output.data_type, DEFAULT_TYPE)