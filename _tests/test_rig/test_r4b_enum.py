"""Round 4b, step B1: the forms ``Enum(..., en=...)`` takes (user decision).

Pinned (they worked before this round, untested): a colon string
(``'red:green:blue'``, ``'red=1:green=5'``, used as written), a list and a
tuple of names (joined with a trailing ``:``, byte for byte as before; the
attribute clone passes such a list, sparse ``'b=10'`` pieces included).

Added: a dict of name -> value (``{'red': 1, 'green': 5}``) and a list of
``(name, value)`` pairs (``[('red', 1), ('green', 5)]``), both
``'red=1:green=5'``; names and pairs mix (``['red', ('green', 5)]`` is
``'red:green=5'``). ``dv=`` by field name works with every form.

Refused with a TypeError when the spec is made, so before any write: a set or
a frozenset (no order, so the field values would be arbitrary; it was joined in
hash order before), an empty list or dict, any other type, and an element that
is neither a name nor a ``(name, value)`` pair (a name is a non-empty str
without ``:`` or ``=``, a value an int that is not a bool).
"""

from maya import cmds

from rig import Node
from rig.spec import Enum
from rig._tests._base import MayaTestCase


class TestEnumNameForms(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.node = Node.create("transform", name="en_t")

    def _fields(self, attr):
        """The fields of ``en_t.<attr>`` as ``plug.enums`` and as
        ``cmds.attributeQuery(listEnum=True)`` read them."""
        plug = getattr(self.node, attr)
        return plug.enums, cmds.attributeQuery(attr, node="en_t", listEnum=True)

    # -- the four forms that worked before this round (pins) -- #

    def test_existing_forms_are_unchanged(self):
        cases = (
            ("e0", "red:green:blue",          "red:green:blue",  ["red", "green", "blue"], 1),
            ("e1", ["red", "green", "blue"],  "red:green:blue:", ["red", "green", "blue"], 1),
            ("e2", ("red", "green", "blue"),  "red:green:blue:", ["red", "green", "blue"], 1),
            ("e3", "red=1:green=5",           "red=1:green=5",   ["red=1", "green=5"],     5),
        )
        for attr, en, kargs_en, enums, green in cases:
            with self.subTest(en=en):
                spec = Enum(attr, en=en)
                self.assertEqual(spec.kargs["en"], kargs_en)
                plug = self.node << spec
                self.assertEqual(self._fields(attr), (enums, [":".join(enums)]))
                plug << "green"
                self.assertEqual(cmds.getAttr(f"en_t.{attr}"), green)

    def test_enum_name_alias_and_default_are_unchanged(self):
        self.assertEqual(Enum("e", enumName=["a", "b"]).kargs["en"], "a:b:")
        self.assertEqual(Enum("e").kargs["en"], "False:True:")
        self.node << Enum("toggle")
        self.assertEqual(self._fields("toggle"), (["False", "True"], ["False:True"]))

    def test_a_list_of_names_keeps_its_pieces_verbatim(self):
        # the pieces the attribute clone passes (plug.enums of a sparse enum)
        self.assertEqual(Enum("e", en=["a", "b=10", "c"]).kargs["en"], "a:b=10:c:")
        self.assertEqual(Enum("e", en=["X", "-X"]).kargs["en"], "X:-X:")

    # -- the dict and the (name, value) pairs (added) -- #

    def test_dict_and_pairs_declare_valued_fields(self):
        cases = (
            ("d0", {"red": 1, "green": 5}),
            ("d1", [("red", 1), ("green", 5)]),
            ("d2", (("red", 1), ("green", 5))),
            ("d3", [["red", 1], ["green", 5]]),
        )
        for attr, en in cases:
            with self.subTest(en=en):
                spec = Enum(attr, en=en)
                self.assertEqual(spec.kargs["en"], "red=1:green=5")
                plug = self.node << spec
                self.assertEqual(self._fields(attr), (["red=1", "green=5"], ["red=1:green=5"]))
                plug << "green"
                self.assertEqual(cmds.getAttr(f"en_t.{attr}"), 5)
                plug << "red"
                self.assertEqual(cmds.getAttr(f"en_t.{attr}"), 1)

    def test_a_dict_keeps_its_insertion_order(self):
        self.assertEqual(Enum("e", en={"b": 5, "a": 1}).kargs["en"], "b=5:a=1")
        self.assertEqual(Enum("e", en={"neg": -1, "zero": 0}).kargs["en"], "neg=-1:zero=0")

    def test_dv_by_field_name_with_every_form(self):
        cases = (
            ("v0", "red:green:blue", 1),
            ("v1", ["red", "green", "blue"], 1),
            ("v2", "red=1:green=5", 5),
            ("v3", {"red": 1, "green": 5}, 5),
            ("v4", [("red", 1), ("green", 5)], 5),
            ("v5", ["red", ("green", 5)], 5),
        )
        for attr, en, value in cases:
            with self.subTest(en=en):
                spec = Enum(attr, en=en, dv="green")
                self.assertEqual(spec.kargs["dv"], value)
                self.node << spec
                self.assertEqual(cmds.getAttr(f"en_t.{attr}"), value)
                self.assertEqual(
                    cmds.attributeQuery(attr, node="en_t", listDefault=True), [float(value)]
                )

    def test_names_and_pairs_mix(self):
        spec = Enum("mix", en=["red", ("green", 5), "blue"])
        self.assertEqual(spec.kargs["en"], "red:green=5:blue")
        plug = self.node << spec
        self.assertEqual(
            self._fields("mix"), (["red", "green=5", "blue"], ["red:green=5:blue"])
        )
        plug << "blue"
        self.assertEqual(cmds.getAttr("en_t.mix"), 6)
        self.assertEqual(Enum("e", en=["red", ("green", 5)]).kargs["en"], "red:green=5")

    # -- refusals: TypeError when the spec is made, nothing written -- #

    def test_a_set_is_refused(self):
        message = (
            r"^Enum 'e': en= takes an ordered list \(\['red', 'green'\]\), a dict "
            r"\(\{'red': 1, 'green': 5\}\) or a 'red:green' string; a set has no order, "
            r"so the field indices would be arbitrary$"
        )
        for en in ({"red", "green"}, frozenset({"red", "green"}), set()):
            with self.subTest(en=en):
                before = set(cmds.ls())
                with self.assertRaisesRegex(TypeError, message):
                    self.node << Enum("e", en=en)
                self.assertFalse(cmds.attributeQuery("e", node="en_t", exists=True))
                self.assertEqual(set(cmds.ls()), before)

    def test_bad_forms_are_refused_naming_the_element(self):
        cases = (
            ([("a", "1")],        r"element \('a', '1'\): the value of field 'a' is an int"),
            ([(1, 2)],            r"element \(1, 2\): a field name is a non-empty str"),
            ((1, 2),              r"element 1 is neither a field name \(a str\) nor a \(name, value\) pair"),
            ([("a", True)],       r"element \('a', True\): the value of field 'a' is an int \(not a bool\)"),
            ([("a:b", 1)],        r"element \('a:b', 1\): a field name is a non-empty str without ':' or '='"),
            ([("a=b", 1)],        r"element \('a=b', 1\): a field name"),
            ([("", 1)],           r"element \('', 1\): a field name is a non-empty str"),
            ([("a", 1, 2)],       r"element \('a', 1, 2\) is neither a field name"),
            (["a", None],         r"element None is neither a field name"),
            ({"a": 1.5},          r"element \('a', 1.5\): the value of field 'a' is an int"),
            ({1: 2},              r"element \(1, 2\): a field name"),
            ({},                  r"\{\} names no field; name at least one$"),
            ([],                  r"\[\] names no field; name at least one$"),
            (5,                   r"takes a 'red:green' string, .*; got int 5$"),
            (None,                r"takes a 'red:green' string, .*; got NoneType None$"),
        )
        for en, message in cases:
            with self.subTest(en=en):
                before = set(cmds.ls())
                with self.assertRaisesRegex(TypeError, r"^Enum 'e': en= " + message):
                    self.node << Enum("e", en=en)
                self.assertFalse(cmds.attributeQuery("e", node="en_t", exists=True))
                self.assertEqual(set(cmds.ls()), before)

    def test_refused_through_the_enum_name_alias(self):
        with self.assertRaisesRegex(TypeError, r"^Enum 'e': en= takes an ordered list"):
            Enum("e", enumName={"a", "b"})

    # -- the attribute clone -- #

    def test_a_sparse_enum_clone_keeps_its_fields(self):
        for attr, en in (("sp0", {"red": 1, "green": 5}), ("sp1", "a:b=10:c")):
            with self.subTest(en=en):
                plug  = self.node << Enum(attr, en=en)
                clone = plug >> f"{attr}_copy"
                self.assertEqual(self._fields(f"{attr}_copy"), self._fields(attr))
                clone << plug.enums[-1].split("=")[0]
                self.assertEqual(
                    cmds.getAttr(f"en_t.{attr}_copy"),
                    {"sp0": 5, "sp1": 11}[attr],
                )
