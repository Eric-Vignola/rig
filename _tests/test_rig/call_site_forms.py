"""The call-site check of rig (rig/nodetypes/_base.py), extracted with no Maya and no rig
import and run over the call forms on this interpreter: the 18 forms of the user decision
(2026-10-16) plus super(), comprehensions, generators, settrace / sys.monitoring and
EXTENDED_ARG. ``test_call_site_forms`` runs it on this mayapy and, when it is installed, on
Maya 2027's (Python 3.13), which cannot import rig yet. Not a test module itself (no
``test_`` prefix): a script.

    mayapy -I call_site_forms.py <path to rig/nodetypes/_base.py>

Prints one line per form and ``all exact: True``; exits 1 if any form is wrong.
"""
import ast
import dis
import functools
import sys
from types import MethodType
from typing import Any

SRC = open(sys.argv[1], encoding="utf-8").read()
TREE = ast.parse(SRC)
WANT_FUNCS = {"_method_loads", "_called", "_node_has_attr", "_alias_target", "_is_predicate_name"}
WANT_CLASSES = {"_PlugMethod", "_PlugDescriptor", "_PlugClassMethod", "_PlugStaticMethod", "_BarePredicate"}
WANT_ASSIGNS = {"_METHOD_LOAD", "_GETFRAME"}
parts = []
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
        parts.append(node)
    elif isinstance(node, ast.ClassDef) and node.name in WANT_CLASSES:
        parts.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANT_ASSIGNS for t in node.targets):
        parts.append(node)
found = {getattr(p, "name", None) or p.targets[0].id for p in parts}
assert found >= WANT_FUNCS | WANT_CLASSES | WANT_ASSIGNS, found
NS = {"sys": sys, "_dis": dis, "_MethodType": MethodType, "Any": Any, "_MISSING": object(),
      "_METHOD_PLUG_HOOK": lambda inst, name, bound: ("PLUG", bound), "__name__": "rig_extract"}
exec(compile(ast.Module(body=parts, type_ignores=[]), sys.argv[1], "exec"), NS)
print(sys.version)
print("method-load table:", {dis.opname[i]: k for i, k in enumerate(NS["_METHOD_LOAD"]) if k})


class Handle:
    def isValid(self):  # noqa: N802
        return True


class CountingFn:
    def __init__(self):
        self.checks = 0

    def hasAttribute(self, name):  # noqa: N802
        self.checks += 1
        return False

    def getAliasList(self):  # noqa: N802
        return []


def stand_in(cls):
    node = object.__new__(cls)
    vars(node).update(_attr_dict={}, _objhandle=Handle(), _fn_set=CountingFn())
    return node


PM = NS["_PlugMethod"]


class N:
    def m(self, *args, **kwargs):
        return "ran"

    m = PM(m, "m")


class Sub(N):
    def m(self, *args, **kwargs):
        return super().m(*args, **kwargs) if args else super().m()

    m = PM(m, "m")


n, s = stand_in(N), stand_in(Sub)
ns = {}
exec("def big(n):\n" + "".join("    a%d = n.x%d if 0 else 0\n" % (i, i) for i in range(300)) + "    return n.m()\n", ns)
exec("def big2(n):\n" + "".join("    a%d = n.x%d if 0 else 0\n" % (i, i) for i in range(300)) + "    return n.m\n", ns)


def gen():
    yield n.m()


def traced():
    def tracer(frame, event, arg):
        return tracer
    sys.settrace(tracer)
    try:
        return n.m()
    finally:
        sys.settrace(None)


def monitored():
    if sys.version_info < (3, 12):
        return n.m()
    mon = sys.monitoring
    tool = 4
    mon.use_tool_id(tool, "rig-sniff")
    mon.register_callback(tool, mon.events.INSTRUCTION, lambda code, off: None)
    mon.set_events(tool, mon.events.INSTRUCTION)
    try:
        return n.m()
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)


FORMS = [
    ("n.m()", lambda: n.m(), False),
    ("n.m(1, k=2)", lambda: n.m(1, k=2), False),
    ("(n.m)()", lambda: (n.m)(), False),
    ("n.m (bare)", lambda: n.m, True),
    ("f = n.m; f()", lambda: (lambda f: f())(n.m), True),
    ("getattr(n, 'm')()", lambda: getattr(n, "m")(), True),
    ("n.m(*a)", lambda: n.m(*(1,)), True),
    ("n.m(**kw)", lambda: n.m(**{"k": 1}), True),
    ("super().m() inside Sub.m", lambda: s.m(), False),
    ("eval('n.m()')", lambda: eval("n.m()", {"n": n}), False),
    ("generator: yield n.m()", lambda: next(gen()), False),
    ("EXTENDED_ARG: n.m()", lambda: ns["big"](n), False),
    ("EXTENDED_ARG: n.m (bare)", lambda: ns["big2"](n), True),
    ("under settrace: n.m()", traced, False),
    ("under sys.monitoring: n.m()", monitored, False),
    ("[x.m() for x in (n,)]", lambda: [x.m() for x in (n,)], False),
    ("functools.partial(n.m)()", lambda: functools.partial(n.m)(), True),
    ("n.m.__call__()", lambda: n.m.__call__(), True),
]
ok_all = True
for label, form, want in FORMS:
    n._fn_set.checks = s._fn_set.checks = 0
    form()
    got = (n._fn_set.checks + s._fn_set.checks) > 0
    ok = got == want
    ok_all &= ok
    print("%-34s checked=%-5s want=%-5s %s" % (label, got, want, "ok" if ok else "WRONG"))


class K:
    def f(self):
        return "f"

    def c(cls):
        return cls

    def st():
        return "s"

    f = PM(f, "f")
    c = NS["_PlugClassMethod"](c, "c")
    st = NS["_PlugStaticMethod"](st, "st")


k = stand_in(K)
for name in ("f", "c", "st"):
    k._fn_set.checks = 0
    getattr(K, name)  # class access: no check
    exec(f"k.{name}()", {"k": k})
    call_checks = k._fn_set.checks
    exec(f"k.{name}", {"k": k})
    bare_checks = k._fn_set.checks - call_checks
    ok = call_checks == 0 and bare_checks == 1
    ok_all &= ok
    print("%-34s call checks=%d bare checks=%d %s" % ("kind " + name, call_checks, bare_checks, "ok" if ok else "WRONG"))
print("all exact:", ok_all)
sys.exit(0 if ok_all else 1)
