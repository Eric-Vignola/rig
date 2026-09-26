"""A plain str is not a DSL operand (decision D-C).

``Plug`` subclasses ``str``, so ``t.tx == "cube.ty"`` used to reach the op:
it built an ``equal`` node, failed to set the string on it and raised
``InjectionError``, and the node stayed in the scene. Every Plug / PlugList
operator and every public math function now raises ``TypeError`` for a plain
str operand (a str that is not an Attribute, alone or inside a list, tuple or
numpy array) before it creates any node or container. Config strings (the
``"<"`` of a condition op, ``side=``, ``name=``, ``dtype=``, ``axis=``,
``method=``, ``rotate_order``) are not operands and are not checked.
"""

import inspect

import numpy as np
from maya import cmds

import rig
from rig import (
    condition,
    constant,
    container,
    euler as E,
    force_nodes,
    functions as F,
    interpolate as I,
    InjectionError,
    matrix as M,
    Node,
    Plug,
    PlugList,
    quaternion as Q,
    random as R,
    set_options,
    trigonometry as T,
    tween as TW,
    vector as V,
)
from rig import _dispatch as D
from rig._internal import operands as operands_module
from rig._internal.math_nodes import _condition_op, _constant
from rig._internal.operands import _plain_str, operands
from rig.nodetypes import PyNode
from rig._tests._base import MayaTestCase


S = "cube.ty"  # a plain str naming a real plug

_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "**": lambda a, b: a ** b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

# every module whose public functions build nodes from operands
_MODULES = (F, V, M, Q, E, T, I, R, TW)
# public functions that take no argument
_NO_ARGUMENTS = {"frame", "pi", "inf"}


def _scene():
    """Every node and container in the scene, by long name."""
    return set(cmds.ls(long=True))


class _OperandCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("quatNodes", quiet=True)
        cmds.loadPlugin("matrixNodes", quiet=True)
        for name in ("cube", "t", "u", "w"):
            cmds.createNode("transform", name=name)
        self.t, self.u, self.w = Node("t"), Node("u"), Node("w")
        self.q1 = E.to_quaternion(self.t.r)
        self.q2 = E.to_quaternion(self.u.r)

    def tearDown(self):
        set_options(constant_folding=True, flatten_containers=True)
        super().tearDown()

    def assertRejects(self, call, *fragments):
        """``call()`` raises a TypeError (not an InjectionError) whose message
        holds every fragment, and leaves the scene as it was."""
        before = _scene()
        with self.assertRaises(TypeError) as ctx:
            call()
        self.assertNotIsInstance(ctx.exception, InjectionError)
        self.assertEqual(sorted(_scene() - before), [])
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))
        return ctx.exception


class TestPlugOperators(_OperandCase):
    def _kinds(self):
        return {
            "scalar": self.t.tx,
            "vector": self.t.t,
            "matrix": self.t.worldMatrix[0],
            "quaternion": self.q1,
        }

    def test_every_binary_operator_rejects_a_plain_str_either_side(self):
        for kind, plug in self._kinds().items():
            for symbol, op in _BINARY.items():
                with self.subTest(kind=kind, op=symbol, side="right"):
                    self.assertRejects(lambda: op(plug, S), f"{S!r} is a plain str", "Plug('cube.ty')")
                with self.subTest(kind=kind, op=symbol, side="left"):
                    # a str left of % is a format string (see the next tests)
                    expected = "does not format text" if symbol == "%" else f"{S!r} is a plain str"
                    self.assertRejects(lambda: op(S, plug), expected)

    def test_message_renders_the_expression(self):
        err = self.assertRejects(lambda: self.t.tx == S)
        self.assertEqual(
            str(err),
            "t.translateX == 'cube.ty': 'cube.ty' is a plain str, and a DSL operand is "
            "a Plug, a number or a sequence of them. Write Plug('cube.ty') for the plug "
            "of that name.",
        )
        err = self.assertRejects(lambda: S - self.t.tx)
        self.assertTrue(str(err).startswith("'cube.ty' - t.translateX: "), str(err))

    def test_text_formatting_with_a_plug_on_the_right(self):
        err = self.assertRejects(lambda: "%s" % self.t.tx, "str(plug) or an f-string")
        self.assertTrue(str(err).startswith("'%s' % t.translateX: '%' with a Plug"), str(err))
        self.assertNotIn("Plug('%s')", str(err))
        self.assertRejects(lambda: "%s" % self.t.worldMatrix[0], "does not format text")
        self.assertRejects(lambda: S % self.t.tx, "Plug('cube.ty') % plug")

    def test_concatenation_hints_at_text(self):
        err = self.assertRejects(lambda: self.t.worldMatrix + "[0]", "For text, write str(plug)")
        self.assertNotIn("Plug('[0]')", str(err))
        self.assertIn("Wrap a plug name with Plug('node.attr')", str(err))

    def test_a_str_inside_a_sequence_operand(self):
        np_str = np.array(["a", "b", "c"])
        for label, call in (
            ("list", lambda: self.t.t + [1, S, 2]),
            ("tuple", lambda: self.t.t * (S, 1, 1)),
            ("reflected list", lambda: [1, S, 2] - self.t.t),
            ("nested", lambda: self.t.worldMatrix[0] * [[1, 0, 0, 0], [0, S, 0, 0]]),
            ("numpy str array", lambda: self.t.t + np_str),
            ("numpy str_", lambda: self.t.tx + np.str_(S)),
            ("object array", lambda: self.t.t + np.array([1, S, 2], dtype=object)),
        ):
            with self.subTest(label):
                self.assertRejects(call)

    def test_list_containment_with_a_str(self):
        self.assertRejects(lambda: self.t.tx in [S])
        self.assertRejects(lambda: S in [self.t.tx])
        self.assertRejects(lambda: [self.t.tx].index(S))

    def test_in_a_container_and_without_folding(self):
        modes = {
            "container": lambda: container("box"),
            "force_nodes": force_nodes,
        }
        for mode, scope in modes.items():
            with self.subTest(mode=mode):
                with scope():
                    self.assertRejects(lambda: self.t.tx == S)
                    self.assertRejects(lambda: self.t.t % S)
                    self.assertRejects(lambda: self.t.worldMatrix[0] + S)
        members = cmds.container("box", query=True, nodeList=True) or []
        self.assertEqual(members, [])
        set_options(constant_folding=False, flatten_containers=False)
        with container("box2"):
            self.assertRejects(lambda: self.t.t // S)
            self.assertRejects(lambda: self.q1 ** S)
        self.assertEqual(cmds.container("box2", query=True, nodeList=True) or [], [])

    def test_plugs_and_typed_attributes_are_operands(self):
        self.assertEqual(cmds.nodeType((self.t.tx == Plug(S)).node), "equal")
        self.assertEqual(cmds.nodeType((self.t.tx + PyNode("cube").find_attr("ty")).node), "sum")
        self.assertEqual(cmds.nodeType((self.t.t + [1, self.u.tx, 2]).node), "plusMinusAverage")
        self.assertEqual(cmds.nodeType((PlugList([self.t.tx]) + [Plug(S)])[0].node), "sum")

    def test_text_through_str_and_fstrings(self):
        plug = self.t.tx
        before = _scene()
        self.assertEqual(f"{plug}.x", "t.translateX.x")
        self.assertEqual(str(plug) + "[0]", "t.translateX[0]")
        self.assertEqual("%s" % (plug,), "t.translateX")
        self.assertEqual("%s" % str(plug), "t.translateX")
        self.assertEqual("{}".format(plug), "t.translateX")
        self.assertEqual("-".join([plug, "x"]), "t.translateX-x")
        self.assertEqual(_scene(), before)

    def test_a_freed_plug_still_reports_the_free(self):
        plug = self.t.tx
        cmds.file(new=True, force=True)
        with self.assertRaises(RuntimeError) as ctx:
            plug + S
        self.assertIn("already deleted", str(ctx.exception))


class TestPlugListOperators(_OperandCase):
    def test_every_operator_checks_every_row_first(self):
        for symbol, op in _BINARY.items():
            for label, call in (
                ("list + str", lambda: op(PlugList([self.t.tx, self.t.ty]), S)),
                ("list + [1, str]", lambda: op(PlugList([self.t.tx, self.t.ty]), [1, S])),
                ("str + list", lambda: op(S, PlugList([self.t.tx, self.t.ty]))),
                ("[1, str] + list", lambda: op([1, S], PlugList([self.t.tx, self.t.ty]))),
            ):
                if symbol == "%" and label == "str + list":
                    continue  # str.__mod__ runs first: text formatting (next test)
                with self.subTest(op=symbol, case=label):
                    self.assertRejects(call, "is a plain str")

    def test_str_left_of_percent_is_text_formatting(self):
        # PlugList is no str subclass, so str.__mod__ formats it as a mapping
        before = _scene()
        self.assertEqual(S % PlugList([self.t.tx]), S)
        self.assertEqual(_scene(), before)

    def test_message_names_the_row(self):
        err = self.assertRejects(lambda: PlugList([self.t.tx, self.t.ty]) + [1, S])
        self.assertTrue(str(err).startswith("PlugList row 1, t.translateY + 'cube.ty': "), str(err))

    def test_a_raw_str_element_is_checked_too(self):
        items = PlugList([self.t.tx])
        list.append(items, S)  # bypasses the lifting of PlugList()
        self.assertRejects(lambda: items + self.u.tx)
        self.assertRejects(lambda: self.u.tx + items)

    def test_a_freed_plug_still_reports_the_free(self):
        items = PlugList([self.t.tx, self.t.ty])
        cmds.file(new=True, force=True)
        for call in (lambda: items + S, lambda: S + items, lambda: items == [1, S]):
            with self.assertRaises(RuntimeError) as ctx:
                call()
            self.assertIn("already deleted", str(ctx.exception))

    def test_rows_without_a_plug_are_left_alone(self):
        nodes = PlugList([Node("t"), Node("u")])
        before = _scene()
        self.assertEqual(list(nodes == ["t", "x"]), [Node("t") == "t", Node("u") == "x"])
        self.assertEqual(list(nodes != "t"), [Node("t") != "t", Node("u") != "t"])
        self.assertTrue("t.translateX" in PlugList([self.t.tx]))
        self.assertEqual(PlugList([self.t.tx]).index("t.translateX"), 0)
        self.assertEqual(PlugList([self.t.tx]).count("t.translateX"), 1)
        self.assertEqual(_scene(), before)
        # PlugList() lifts a "node.attr" str to a Plug
        self.assertEqual(cmds.nodeType((PlugList([S]) + 1)[0].node), "sum")


class TestPublicFunctions(_OperandCase):
    def _table(self):
        """(label, function, args, kwargs, data parameters). A parameter is a
        positional index, a keyword, or (index or keyword, element) for an
        element of a list argument."""
        t, u, w, q1, q2 = self.t, self.u, self.w, self.q1, self.q2
        wm, um = t.worldMatrix[0], u.worldMatrix[0]
        table = [
            ("functions.clamp", F.clamp, [t.tx, 0, 1], {}, [0, 1, 2]),
            ("functions.round", F.round, [t.tx], {"digits": 1}, [0, "digits"]),
            ("functions.sum", F.sum, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.avg", F.avg, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.max", F.max, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.min", F.min, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.pow", F.pow, [t.tx, 2], {}, [0, 1]),
            ("functions.choice", F.choice, [[t.tx, t.ty]], {"selector": t.tz}, [(0, 1), "selector"]),
            ("functions.searchsorted", F.searchsorted, [[0, 1, 2], t.tx], {}, [(0, 1), 1]),
            ("functions.all", F.all, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.any", F.any, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.argmin", F.argmin, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.argmax", F.argmax, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.diff", F.diff, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.cumsum", F.cumsum, [[t.tx, t.ty]], {}, [(0, 1)]),
            ("functions.log", F.log, [t.tx], {"base": 2}, [0, "base"]),
            ("functions.equal", F.equal, [t.tx, t.ty], {"eps": 0.1}, [0, 1, "eps"]),
            ("vector.triple_product", V.triple_product, [t.t, u.t, w.t], {}, [0, 1, 2]),
            ("vector.dot", V.dot, [t.t, u.t], {"normalize": True}, [0, 1, "normalize"]),
            ("vector.rotate", V.rotate, [t.t, u.r], {"rotate_order": 1}, [0, 1]),
            ("vector.lerp", V.lerp, [t.t, u.t], {"weight": t.tx}, [0, 1, "weight"]),
            ("vector.lerp[literal]", V.lerp, [t.t, [1, 2, 3]], {}, [(1, 1)]),
            ("matrix.decompose", M.decompose, [wm], {}, [0]),
            ("matrix.compose", M.compose, [], {"scale": t.s, "rotate": t.r, "translate": t.t,
                                               "shear": t.shear}, ["scale", "rotate", "translate", "shear"]),
            ("matrix.fourbyfour", M.fourbyfour, [], {"x": t.t, "position": u.t}, ["x", "position"]),
            ("matrix.aim", M.aim, [t.t], {"up_vector": u.t, "aim_axis": (1, 0, 0)},
             [0, "up_vector", "aim_axis"]),
            ("matrix.multiply", M.multiply, [wm, um], {}, [0, 1]),
            ("matrix.add[weights]", M.add, [wm, um], {"weights": [t.tx, 0.5]}, [1, ("weights", 0)]),
            ("matrix.blend", M.blend, [wm, um], {"weight": t.tx}, [0, 1, "weight"]),
            ("matrix.axis", M.axis, [wm], {"axis": "y"}, [0]),
            ("matrix.row", M.row, [wm], {"index": 1}, [0, "index"]),
            ("matrix.transform_point", M.transform_point, [t.t, um], {}, [0, 1]),
            ("quaternion.slerp", Q.slerp, [q1, q2], {"weight": t.tx}, [0, 1, "weight"]),
            ("quaternion.to_axis_angle", Q.to_axis_angle, [q1], {}, [0]),
            ("quaternion.from_axis_angle", Q.from_axis_angle, [t.t, t.tx], {}, [0, 1]),
            ("euler.reorder", E.reorder, [t.r, 0, 1], {}, [0]),
            ("euler.slerp", E.slerp, [t.r, u.r], {"weight": t.tx}, [0, 1, "weight"]),
            ("interpolate.sequence", I.sequence, [t.tx, [0, 1, 2], [0, 10, 20]], {}, [0, (1, 1), (2, 1)]),
            ("interpolate.smoothstep", I.smoothstep, [t.tx, t.ty], {"weight": t.tz}, [0, 1, "weight"]),
            ("interpolate.inverse_lerp", I.inverse_lerp, [t.tx, t.ty, t.tz], {}, [0, 1, 2]),
            ("random.value", R.value, [], {"trigger": t.tx, "seed": 3}, ["trigger", "seed"]),
            ("random.uniform", R.uniform, [0, 1], {"trigger": t.tx, "seed": 3}, [0, 1, "trigger"]),
            ("random.randint3D", R.randint3D, [0, 9], {"trigger": t.tx, "seed": 3}, [0, 1, "trigger"]),
            ("rig.dist", D.dist, [t.t, u.t], {}, [0, 1]),
            ("rig.lerp", D.lerp, [t.tx, t.ty], {"weight": t.tz}, [0, 1, "weight"]),
            ("rig.blend", D.blend, [wm, um], {"weight": t.tz}, [1, "weight"]),
            ("rig.to_euler", D.to_euler, [q1], {}, [0]),
            ("rig.condition", condition, [t.tx > 0, t.ty, t.tz], {}, [0, 1, 2]),
            ("rig.constant", constant, [[1, 2, 3]], {}, [0, (0, 1)]),
        ]
        # every one-operand function of these modules takes a scalar plug
        for module in (F, V, M, Q, T, TW):
            for name in module.__all__:
                fn = getattr(module, name)
                if not callable(fn):
                    continue  # vector.X / Y / Z
                params = [
                    p for p in inspect.signature(fn).parameters.values()
                    if p.default is p.empty and p.kind == p.POSITIONAL_OR_KEYWORD
                ]
                if name in _NO_ARGUMENTS or len(params) != 1:
                    continue
                sample = {V: t.t, M: wm, Q: q1}.get(module, t.tx)
                table.append((f"{module.__name__}.{name}", fn, [sample], {}, [0]))
        for name in ("atan2", "atan2d"):
            table.append((f"trigonometry.{name}", getattr(T, name), [t.tx, t.ty], {}, [0, 1]))
        for name in ("add", "multiply", "subtract", "angle"):
            table.append((f"quaternion.{name}", getattr(Q, name), [q1, q2], {}, [0, 1]))
        return table

    @staticmethod
    def _with_str(args, kwargs, param):
        args, kwargs = list(args), dict(kwargs)
        if isinstance(param, tuple):
            where, index = param
            holder = kwargs if isinstance(where, str) else args
            seq = list(holder[where])
            seq[index] = S
            holder[where] = seq
        elif isinstance(param, str):
            kwargs[param] = S
        else:
            args[param] = S
        return args, kwargs

    def test_every_data_argument_rejects_a_plain_str(self):
        for label, fn, args, kwargs, params in self._table():
            for param in params:
                a, k = self._with_str(args, kwargs, param)
                with self.subTest(label, param=param):
                    self.assertRejects(lambda: fn(*a, **k), f"{S!r} is a plain str", "() argument '")

    def test_message_names_the_function_and_argument(self):
        err = self.assertRejects(lambda: V.lerp(self.t.t, S))
        self.assertTrue(str(err).startswith("rig.vector.lerp() argument 'input2': "), str(err))
        err = self.assertRejects(lambda: rig.lerp(self.t.tx, self.t.ty, weight=S))
        self.assertTrue(str(err).startswith("rig.lerp() argument 'weight': "), str(err))
        err = self.assertRejects(lambda: M.multiply(self.t.worldMatrix[0], S))
        self.assertTrue(str(err).startswith("rig.matrix.multiply() argument 'tokens[1]': "), str(err))
        err = self.assertRejects(lambda: F.sum([self.t.tx, S]))
        self.assertTrue(str(err).startswith("rig.functions.sum() argument 'tokens': "), str(err))

    def test_a_broadcast_is_checked_before_its_first_row_builds(self):
        rows = PlugList([self.t.tx, self.u.tx])
        self.assertRejects(lambda: V.lerp(rows, [1, S]))
        self.assertRejects(lambda: F.clamp(rows, [0, S], 1))
        self.assertRejects(lambda: Q.slerp(PlugList([self.q1, self.q2]), [self.q2, S]))
        self.assertRejects(lambda: rig.lerp(rows, 1, weight=[0.5, S]))
        mixed = PlugList([self.t.tx])
        list.append(mixed, S)
        self.assertRejects(lambda: F.abs(mixed))

    def test_in_a_container_and_under_force_nodes(self):
        set_options(flatten_containers=False)
        with container("box"):
            self.assertRejects(lambda: F.searchsorted([0, S, 2], self.t.tx))
            self.assertRejects(lambda: M.compose(rotate=S))
            self.assertRejects(lambda: R.value(trigger=S, seed=1))
        self.assertEqual(cmds.container("box", query=True, nodeList=True) or [], [])
        with force_nodes():
            self.assertRejects(lambda: F.abs(S))
            self.assertRejects(lambda: T.sin(S))
            self.assertRejects(lambda: I.smoothstep(0, 1, S))

    def test_every_public_function_is_checked(self):
        missing = []
        for module in _MODULES:
            for name in module.__all__:
                fn = getattr(module, name)
                if callable(fn) and name not in _NO_ARGUMENTS and not hasattr(fn, "_operand_config"):
                    missing.append(f"{module.__name__}.{name}")
        for name in D.__all__ + ["condition", "constant"]:
            if not hasattr(getattr(rig, name), "_operand_config"):
                missing.append(f"rig.{name}")
        self.assertEqual(missing, [])

    def test_a_rejected_call_leaves_the_memo_caches_as_they_were(self):
        test = self.t.tx > 0
        sizes = (len(F.abs._cache), len(V.lerp._cache), len(condition._cache))
        self.assertRejects(lambda: F.abs(S))
        self.assertRejects(lambda: V.lerp(self.t.t, S))
        self.assertRejects(lambda: condition(test, S, 1))
        self.assertEqual((len(F.abs._cache), len(V.lerp._cache), len(condition._cache)), sizes)
        first = F.abs(self.t.tx)
        self.assertIs(F.abs(self.t.tx), first)
        self.assertEqual(cmds.nodeType(first.node), "absolute")


class TestConfigStrings(_OperandCase):
    """A str that is a choice, not an operand, is passed through as before."""

    def test_config_parameters(self):
        t = self.t
        self.assertEqual(cmds.nodeType(_condition_op(t.tx, "<", t.ty).node), "lessThan")
        self.assertEqual(cmds.nodeType(M.axis(t.worldMatrix[0], axis="z").node), "axisFromMatrix")
        with self.assertRaisesRegex(ValueError, "axis must be 0/1/2"):
            M.axis(t.worldMatrix[0], axis="w")
        self.assertEqual(str(constant([1, 2], name="cfg1", dtype="long")), "cfg1.value")
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            _constant(1, dtype="bad")
        self.assertIsInstance(F.searchsorted([0, 1, 2], t.tx, side="right"), Plug)
        with self.assertRaisesRegex(ValueError, "side must be 'left' or 'right'"):
            F.searchsorted([0, 1, 2], t.tx, side="middle")
        self.assertIsInstance(F.searchsorted([0, 1, 2], t.tx, return_index="yes"), Plug)
        self.assertIsInstance(I.sequence(t.tx, [0, 1], [0, 1], method=rig.lerp), Plug)
        self.assertIsInstance(I.smoothstep(t.tx, t.ty, 0.5, normalize="yes"), Plug)

    def test_a_numeric_condition_picks_any_value_in_python(self):
        before = _scene()
        self.assertEqual((condition(1, "yes", "no"), condition(0, "yes", "no")), ("yes", "no"))
        self.assertEqual(condition(True, ["a", "b"], "c"), ["a", "b"])
        self.assertEqual(list(condition(PlugList([1, 0]), "yes", "no")), ["yes", "no"])
        with force_nodes():
            self.assertEqual(condition(1, "yes", "no"), "yes")
        self.assertEqual(_scene(), before)
        test = self.t.tx > 0
        self.assertRejects(lambda: condition(test, "yes", "no"), "argument 'if_true'")
        self.assertRejects(lambda: condition(test, 1, ["a", 2]), "argument 'if_false'")
        # a list test builds a condition node, so its branches are operands
        self.assertRejects(lambda: condition([1, 0], "yes", "no"))
        self.assertRejects(lambda: condition(PlugList([1, test]), "yes", "no"))

    def test_rotate_order_is_not_checked_as_an_operand(self):
        # A rotate-order name never worked (d6ad8b2 and v2.0.0a2 build the node,
        # then fail to set the enum); rotate_order is exempt pending a decision.
        with self.assertRaises(InjectionError):
            M.decompose(self.t.worldMatrix[0], rotate_order="xyz")
        with self.assertRaises(InjectionError):
            E.reorder(self.t.r, "xyz", "zxy")
        self.assertEqual(M.decompose._operand_config, frozenset({"rotate_order"}))

    def test_string_attributes_still_take_strings(self):
        cmds.addAttr("t", longName="note", dataType="string")
        self.t.note << "hello"
        self.assertEqual(cmds.getAttr("t.note"), "hello")


class TestOperandHelpers(MayaTestCase):
    def test_plain_str(self):
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="a")
        self.assertIsNone(_plain_str(Plug("a.tx")))
        self.assertIsNone(_plain_str(PyNode("a").find_attr("tx")))
        self.assertIsNone(_plain_str([1, Plug("a.tx"), (2, 3.0), None, Node("a")]))
        self.assertIsNone(_plain_str(np.eye(4)))
        self.assertIsNone(_plain_str(np.array([], dtype=str)))
        self.assertEqual(_plain_str("x"), "x")
        self.assertEqual(_plain_str([[1, 2], (3, ["x"])]), "x")
        self.assertEqual(_plain_str(np.array([["p", "q"]])), "p")
        self.assertEqual(_plain_str(np.array([1, "y"], dtype=object)), "y")
        self.assertIsNone(_plain_str(b"bytes"))
        self.assertIsNone(_plain_str({"x": 1}))

    def test_decorator_contract(self):
        @operands(config=("mode",))
        def f(a, b=0, *rest, mode="x"):
            return (a, b, rest, mode)

        self.assertEqual(f(1, 2, 3, mode="y"), (1, 2, (3,), "y"))
        with self.assertRaisesRegex(TypeError, r"\.f\(\) argument 'rest\[1\]'"):
            f(1, 2, 3, "x")
        with self.assertRaisesRegex(TypeError, "has no parameter"):
            operands(lambda a: a, config=("nope",))
        self.assertEqual(list(inspect.signature(V.lerp).parameters), ["input1", "input2", "weight"])
        self.assertEqual(V.lerp.__name__, "lerp")
        self.assertIs(V.lerp._cache, V.lerp.__wrapped__._cache)
        self.assertEqual(operands_module.REFLECTED["__radd__"], "__add__")
        self.assertEqual(operands_module.REFLECTED["__lt__"], "__gt__")


class TestEdgeCases(_OperandCase):
    def test_a_deleted_node_reports_the_delete_first(self):
        held = self.t.tx
        cmds.delete("t")  # the undo queue is off: the node is freed
        cmds.createNode("transform", name="t")  # a new node takes the name
        before = _scene()
        for call in (lambda: held + S, lambda: held == S, lambda: S % held):
            with self.assertRaisesRegex(RuntimeError, "already deleted"):
                call()
        self.assertEqual(_scene(), before)
        self.assertRejects(lambda: F.abs(S))

    def test_a_deleted_node_in_the_undo_queue(self):
        cmds.undoInfo(state=True, infinity=True)
        try:
            held = self.t.tx
            cmds.delete("t")
            before = _scene()
            for call in (lambda: held + S, lambda: S + held, lambda: "%s" % held):
                with self.assertRaisesRegex(RuntimeError, "^t already deleted!$"):
                    call()
            self.assertEqual(_scene(), before)
            cmds.undo()
            self.assertRejects(lambda: held + S, "t.translateX + 'cube.ty': ")
            self.assertEqual(cmds.nodeType((held + 1).node), "sum")
        finally:
            cmds.undoInfo(state=False)

    def test_renamed_and_namespaced_nodes(self):
        cmds.rename("u", "u2")
        self.assertRejects(lambda: self.u.tx - S)
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:cube")
        self.assertRejects(lambda: self.u.tx * "ns:cube.ty", "Write Plug('ns:cube.ty')")
        self.assertEqual(cmds.nodeType((self.u.tx * Plug("ns:cube.ty")).node), "multiply")

    def test_a_rejected_call_records_no_undo_step(self):
        cmds.undoInfo(state=True, infinity=True)
        try:
            cmds.setAttr("w.tx", 5)
            self.assertRejects(lambda: self.w.tx + S)
            self.assertRejects(lambda: V.lerp(self.w.t, [1, S, 0]))
            cmds.undo()
            self.assertEqual(cmds.getAttr("w.tx"), 0.0)
        finally:
            cmds.undoInfo(state=False)

    def test_an_extension_attribute_plug_is_an_operand(self):
        cmds.addExtension(nodeType="transform", longName="extKnob", attributeType="double")
        try:
            result = self.t.extKnob + self.u.tx
            self.assertEqual(cmds.nodeType(result.node), "sum")
            self.assertRejects(lambda: self.t.extKnob + "t.extKnob")
        finally:
            cmds.file(new=True, force=True)
            cmds.deleteExtension(nodeType="transform", attribute="extKnob", forceDelete=True)
