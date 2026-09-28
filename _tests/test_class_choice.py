from maya import cmds
from rig.nodetypes import Node
from rig._tests._base import MayaTestCase

TEST_TYPES   = ["doubleLinear", "message", "matrix"]
DEFAULT_TYPE = "Tdata"


class TestChoice(MayaTestCase):
    """
    Test the Choice class
    """

    TEST_START_NEW_SCENE = True

    def _setup_scene(self, connect: bool = False):
        self.xform  = Node(cmds.createNode("transform"))
        self.choice = Node(cmds.createNode("choice"))
        if connect:
            # typed Attributes connect with ``>>`` (a typed node's dotted
            # access gives a DSL Plug, whose ``>>`` clones)
            inputs = self.choice.find_attr("input")
            self.xform.find_attr("tx")      >> inputs[0]
            self.xform.find_attr("message") >> inputs[1]
            self.xform.find_attr("matrix")  >> inputs[2]

    def test_type_resolving_connected(self):
        self._setup_scene(connect=True)

        # inputs should have consistent types
        for i, t in enumerate(TEST_TYPES):
            self.assertEqual(self.choice.find_attr("input")[i].data_type, t)

        # output type depends on the selected input
        for i, t in enumerate(TEST_TYPES):
            self.choice.find_attr("selector").set(i)
            self.assertEqual(self.choice.find_attr("output").data_type, t)

        # unconnected input resolve to Tdata
        self.assertEqual(self.choice.find_attr("input")[3].data_type, DEFAULT_TYPE)

    def test_type_resolving_not_connected(self):
        # if no connections, inputs and outputs should all resolve to Tdata
        self._setup_scene(connect=False)
        for i, _ in enumerate(TEST_TYPES):
            self.assertEqual(self.choice.input[i].data_type, DEFAULT_TYPE)
        self.assertEqual(self.choice.output.data_type, DEFAULT_TYPE)