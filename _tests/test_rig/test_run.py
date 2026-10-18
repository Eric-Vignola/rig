"""``rig.run``: a file as if in a fresh Python (``rig/_internal/run.py``).

* `TestPorted`: the 29 checks of the regen project's ``_tests/test_plain.py``, on rig.run's API
  (one ``inputs`` mapping, ``shared=``, the file as ``__main__``, lazy claim, the narrower restore).
* `TestProbes`: every probe of the evaluation (runs/features/regen_eval: review P1-P24, the
  synthesis view probe, the critic's C1-C3) as a regression test.
* `TestWeakRegistries`: ``_ALL_MEMOIZED`` / ``_ALL_NODEOP_CACHES`` hold their members weakly.

The Maya side (undo, rig shared, cgmath, options, callbacks) is in ``test_run_maya.py``.
"""

import gc
import getpass
import importlib
import importlib.machinery
import importlib.util
import io
import linecache
import logging
import os
import py_compile
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from unittest import mock

import rig
from rig._internal import memoize as memoize_module
from rig._internal import run as run_module
from rig._internal.node_ops import NodeOp

WIN = os.name == "nt"


def _write(root, rel, text):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(textwrap.dedent(text).lstrip("\n"))
    return path


def _edit(root, rel, old, new):
    path = os.path.join(root, *rel.split("/"))
    with open(path) as fh:
        text = fh.read()
    assert text.count(old) == 1, (rel, old)
    with open(path, "w", newline="\n") as fh:
        fh.write(text.replace(old, new))


def _quiet(*args, **kwargs):
    """rig.run(...) with its stderr captured: (run, stderr text)."""
    err, saved = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        return rig.run(*args, **kwargs), err.getvalue()
    finally:
        sys.stderr = saved


class _Tree(unittest.TestCase):
    """A temp folder of packages on sys.path; everything it adds to the session goes in tearDownClass."""

    TOPS = ()

    @classmethod
    def make(cls, files, paths=("lib",)):
        cls.WORK = tempfile.mkdtemp(prefix="rigrun_")
        for rel, text in files.items():
            _write(cls.WORK, rel, text)
        cls._path, cls._argv = list(sys.path), list(sys.argv)
        sys.path[:0] = [os.path.join(cls.WORK, p) for p in paths]

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls._path
        for name in [n for n in sys.modules if n.partition(".")[0] in cls.TOPS]:
            del sys.modules[name]
        shutil.rmtree(cls.WORK, ignore_errors=True)
        super().tearDownClass()

    @classmethod
    def leaks(cls, keys):
        return {n for n in set(sys.modules) ^ keys if n.partition(".")[0] in cls.TOPS}


# ---------------------------------------------------------------------------------------------
# The 29 checks of regen's test_plain.py

_PORTED = {
    "lib/rrpkg/__init__.py": """
        import logging
        VERSION = "v1"
        STATE = []
        STATE.append(len(STATE) + 1)
        logging.getLogger("rrpkg").addHandler(logging.NullHandler())
    """,
    "lib/rrpkg/core.py": """
        class Base:
            tag = "v1"

        def lazy():
            from rrpkg import deep
            return deep.VALUE
    """,
    "lib/rrpkg/deep.py": 'VALUE = "v1"\n',
    "lib/rrns/sub/mod.py": 'VALUE = "v1"\n',
    "lib/rrnew.py": 'VALUE = "v1"\n',
    "lib/rrmixed/__init__.py": 'VALUE = "v1"\n',
    "site-packages/rrinst/__init__.py": 'VALUE = "v1"\n',
    "scripts/script.py": """
        import logging
        import sys
        import rrpkg, rrpkg.core, rrnew, rrmixed, rrinst
        from rrpkg.core import Base
        from rrns.sub import mod

        class Thing(Base):
            pass

        seen = {
            "pkg": rrpkg.VERSION, "base": Thing.tag, "ns": mod.VALUE, "newlib": rrnew.VALUE,
            "mixed": rrmixed.VALUE, "instpkg": rrinst.VALUE, "state": list(rrpkg.STATE),
            "x": globals().get("X"), "path0": sys.path[0], "argv": list(sys.argv), "name": __name__,
            "probe": probe() if "probe" in globals() else None,
            "main_is_me": sys.modules["__main__"].__dict__ is globals(),
        }
        sys.displayhook = lambda value: None                # process state the run changes
        sys.excepthook = lambda *exc: None
        sys.stdout = sys.stdout
        logging.getLogger("rrrunonly").addHandler(logging.NullHandler())

        def late_deep():
            return rrpkg.core.lazy()

        def late_direct():
            import rrpkg.deep
            return rrpkg.deep.VALUE

        def late_dynamic():
            import importlib
            return importlib.import_module("rrpkg.deep").VALUE

        def pickle_base():
            import pickle
            return len(pickle.dumps(rrpkg.core.Base()))

        def main(n):
            return n * 2

        if globals().get("BOOM"):
            raise RuntimeError("boom")
    """,
}


def _probe():                                    # session code, called by the script
    import rrpkg.core as core
    return core.Base.tag


class TestPorted(_Tree):
    TOPS = {"rrpkg", "rrnew", "rrns", "rrmixed", "rrinst"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.make(_PORTED, paths=("lib", "site-packages"))
        W = cls.WORK
        cls.SCRIPT = os.path.join(W, "scripts", "script.py")
        cls.FWD = cls.SCRIPT.replace("\\", "/") if WIN else cls.SCRIPT
        import rrpkg, rrpkg.core, rrns.sub.mod, rrmixed, rrinst  # noqa: E401,F401 -- the session has these
        fast = types.ModuleType("rrmixed._fast")              # stand-in for a compiled member
        fast.__spec__ = importlib.machinery.ModuleSpec(
            "rrmixed._fast", importlib.machinery.ExtensionFileLoader("rrmixed._fast", os.path.join(W, "x.pyd")))
        sys.modules["rrmixed._fast"] = fast
        cls.HOST = {n: sys.modules[n] for n in ("rrpkg", "rrpkg.core", "rrns", "rrns.sub", "rrns.sub.mod",
                                                "rrmixed", "rrinst")}
        cls.KEYS = set(sys.modules)
        cls.PATH, cls.ARGV = list(sys.path), list(sys.argv)
        cls.HOOKS = (sys.displayhook, sys.excepthook, sys.stdout, sys.stderr)
        cls.MAIN = sys.modules["__main__"]
        o = cls.o = {}

        mine = {"X": 5, "probe": _probe}
        run1, _ = _quiet(cls.SCRIPT, mine, argv=["a"])
        o["run1"], o["mine"] = run1, mine
        o["main_after"] = sys.modules["__main__"]
        o["late1"] = (run1["late_deep"](), run1["late_direct"]())
        o["hooks_after"] = (sys.displayhook, sys.excepthook, sys.stdout, sys.stderr)
        o["runonly"] = len(logging.getLogger("rrrunonly").handlers)
        logging.getLogger("rrrunonly").handlers.clear()

        # same second, same size: a __pycache__ timestamp check would miss every one of these
        for rel in ("lib/rrpkg/__init__.py", "lib/rrpkg/deep.py", "lib/rrns/sub/mod.py", "lib/rrnew.py",
                    "lib/rrmixed/__init__.py", "site-packages/rrinst/__init__.py"):
            _edit(W, rel, '"v1"', '"v2"')
        _edit(W, "lib/rrpkg/core.py", 'tag = "v1"', 'tag = "v2"')

        run2, o["err2"] = _quiet(cls.SCRIPT, {"probe": _probe})
        logging.getLogger("rrrunonly").handlers.clear()
        o["run2"] = run2
        o["leaks2"] = cls.leaks(cls.KEYS)
        o["late2"] = (run2["late_deep"](), run2["late_direct"](), run2["late_dynamic"]())
        o["late1b"] = (run1["late_direct"](), run1["late_deep"]())
        o["leaks_late"] = cls.leaks(cls.KEYS)

    def test_01_run1_sees_v1(self):
        s1 = self.o["run1"]["seen"]
        self.assertTrue(all(s1[k] == "v1" for k in ("pkg", "base", "ns", "newlib", "mixed", "instpkg")), s1)

    def test_02_inputs_copied_your_dict_untouched(self):
        self.assertEqual(self.o["run1"]["seen"]["x"], 5)
        self.assertEqual(set(self.o["mine"]), {"X", "probe"})

    def test_03_name_path0_argv_like_python(self):
        s1 = self.o["run1"]["seen"]
        self.assertEqual((s1["name"], s1["path0"], s1["argv"]),
                         ("__main__", os.path.dirname(self.SCRIPT), [self.FWD, "a"]))

    def test_04_the_file_is_main_during_the_run_only(self):
        self.assertTrue(self.o["run1"]["seen"]["main_is_me"])
        self.assertIs(self.o["main_after"], self.MAIN)

    def test_05_late_import_after_the_run_run1_world(self):
        self.assertEqual(self.o["late1"], ("v1", "v1"))

    def test_06_run2_sees_every_edit(self):
        s2 = self.o["run2"]["seen"]
        self.assertEqual((s2["pkg"], s2["base"], s2["ns"], s2["newlib"]), ("v2",) * 4, s2)

    def test_07_module_state_starts_fresh(self):
        self.assertEqual(self.o["run2"]["seen"]["state"], [1])

    def test_08_session_modules_untouched_no_reload(self):
        import rrpkg
        self.assertEqual((rrpkg.VERSION, rrpkg.core.Base.tag), ("v1", "v1"))
        self.assertTrue(all(sys.modules[n] is m for n, m in self.HOST.items()))

    def test_09_nothing_leaked(self):
        self.assertEqual(self.o["leaks2"], set())

    def test_10_session_code_during_the_run_one_view(self):
        # lazy claim + one view: the file imported rrpkg, so session code sees the run's copy
        self.assertEqual(self.o["run2"]["seen"]["probe"], "v2")

    def test_11_late_import_run2_import_and_import_module(self):
        self.assertEqual(self.o["late2"], ("v2", "v2", "v2"))

    def test_12_run1_still_in_its_own_world(self):
        self.assertEqual(self.o["late1b"], ("v1", "v1"))

    def test_13_late_imports_leaked_nothing(self):
        self.assertEqual(self.o["leaks_late"], set())

    def test_14_shared_packages_old_code_and_the_run_says_so(self):
        s2, err2 = self.o["run2"]["seen"], self.o["err2"]
        self.assertEqual((s2["mixed"], s2["instpkg"]), ("v1", "v1"))
        self.assertIn("rrmixed (not plain .py", err2)
        self.assertIn("rrinst (installed)", err2)

    def test_15_streams_and_hooks_given_back_a_package_setup_kept(self):
        self.assertEqual(self.o["hooks_after"], self.HOOKS)
        self.assertEqual(self.o["runonly"], 1)                     # logging is the session's: not restored

    def test_16_sys_path_and_argv_given_back(self):
        self.assertEqual((sys.path, sys.argv), (self.PATH, self.ARGV))

    def test_17_call_returns_the_result(self):
        self.assertEqual(self.o["run2"].call("main", 21), 42)

    def test_18_pickle_of_a_run_object_inside_call(self):
        run2 = self.o["run2"]
        self.assertGreater(run2.call("pickle_base"), 0)
        with self.assertRaises(Exception):
            run2["pickle_base"]()

    def test_19_inputs_skip_modules_and_refuse_dunders(self):
        run, _ = _quiet(self.SCRIPT, {"os": os, "X": 3})
        self.assertNotIn("os", run)
        self.assertEqual(run["seen"]["x"], 3)
        with self.assertRaisesRegex(ValueError, "__author__"):
            rig.run(self.SCRIPT, {"__author__": "me"})
        with self.assertRaises(TypeError):
            rig.run(self.SCRIPT, [("X", 1)])
        with self.assertRaises(TypeError):
            rig.run(self.SCRIPT, argv="--fast")

    def test_20_shared_names_share_the_session_copy_and_say_it_changed(self):
        run4, err4 = _quiet(self.SCRIPT, shared=("rrpkg",))
        self.assertEqual(run4["seen"]["pkg"], "v1")
        self.assertIn("rrpkg (shared)", err4)
        self.assertNotIn("rrpkg", run4.owned)

    def test_21_run1_function_inside_run5_keeps_run1_modules(self):
        run5, _ = _quiet(self.SCRIPT, {"probe": self.o["run1"]["late_direct"]})
        self.assertEqual(run5["seen"]["probe"], "v1")

    def test_22_an_error_raises_restores_and_carries_the_run(self):
        with self.assertRaises(RuntimeError) as caught:
            _quiet(self.SCRIPT, {"BOOM": True})
        self.assertIsInstance(caught.exception.rig_run, run_module.Run)
        self.assertIn("late_deep", caught.exception.rig_run)          # what the file made before the error
        self.assertEqual(self.leaks(self.KEYS), set())
        self.assertTrue(all(sys.modules[n] is m for n, m in self.HOST.items()))


class TestPortedPaths(unittest.TestCase):
    """Checks 23-29: ~ and variables expanded until nothing changes, .. walked, forward slashes."""

    NAMES = ("HOME", "USER", "RR_DIR", "RR_UP", "RR_TWO", "RR_X", "RR_TILDE", "RR_LOOP1", "RR_LOOP2", "RR_GROW")

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="rigrunp_")
        self.script = _write(self.work, "scripts/script.py", "import sys\nPATH0 = sys.path[0]\nARGV0 = sys.argv[0]\n")
        self.fwd = self.script.replace("\\", "/") if WIN else self.script
        self.saved = {name: os.environ.get(name) for name in self.NAMES}
        os.environ.update({
            "HOME": os.path.join(self.work, "Documents"),       # what Maya sets HOME to on Windows
            "RR_DIR": self.work, "RR_UP": "scripts/sub/..", "RR_TWO": "$RR_DIR/$RR_UP", "RR_X": "x.py",
            "RR_TILDE": "~/$RR_X", "RR_LOOP1": "$RR_LOOP2", "RR_LOOP2": "$RR_LOOP1", "RR_GROW": "$RR_GROW/x",
        })
        self.userdir = os.environ["USERPROFILE"] if WIN else os.environ["HOME"]

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self.work, ignore_errors=True)

    def in_user(self, name):
        path = os.path.join(self.userdir, name)
        return path.replace("\\", "/") if WIN else path

    def test_23_tilde_is_the_user_folder(self):
        self.assertEqual(run_module._resolve("~/x.py"), self.in_user("x.py"))
        if WIN:
            self.assertEqual(run_module._resolve(r"~\x.py"), self.in_user("x.py"))

    def test_24_home_is_tilde_user_as_set_else_login(self):
        homes = ["$HOME/x.py", "${HOME}/x.py"] + (["%HOME%/x.py", r"$home\x.py"] if WIN else [])
        self.assertEqual([run_module._resolve(p) for p in homes], [self.in_user("x.py")] * len(homes))
        os.environ["USER"] = "rruser"
        self.assertEqual(os.path.basename(run_module._resolve("$USER")), "rruser")
        del os.environ["USER"]
        self.assertEqual(os.path.basename(run_module._resolve("$USER")), getpass.getuser())

    def test_25_var_forms_through_rig_run(self):
        refs = ["$RR_DIR/scripts/script.py", "${RR_DIR}/scripts/script.py"] + (
            ["%RR_DIR%/scripts/script.py"] if WIN else [])
        self.assertEqual([rig.run(ref).path for ref in refs], [self.fwd] * len(refs))

    def test_26_expanded_until_nothing_changes_then_dotdot(self):
        self.assertEqual(rig.run("$RR_TWO/script.py").path, self.fwd)
        self.assertEqual(run_module._resolve("$RR_TILDE"), self.in_user("x.py"))

    def test_27_forward_slashes_sys_path0_as_the_system_writes_it(self):
        result = rig.run("$RR_TWO/script.py")
        self.assertEqual((result.path, result["__file__"], result["ARGV0"]), (self.fwd,) * 3)
        self.assertEqual(result["PATH0"], os.path.dirname(self.script))

    def test_28_a_missing_file_names_the_unset_variable(self):
        path = list(sys.path)
        with self.assertRaisesRegex(FileNotFoundError, r"\$RR_UNSET is not set"):
            rig.run("$RR_UNSET/script.py")
        self.assertEqual(sys.path, path)

    def test_29_variables_that_refer_to_each_other_raise(self):
        for ref in ("$RR_LOOP1", "$RR_GROW"):
            with self.assertRaises(ValueError):
                run_module._resolve(ref)


# ---------------------------------------------------------------------------------------------
# The evaluation's probes

_PROBES = {
    "lib/rqpkg/__init__.py": 'VERSION = "v1"\n',
    "lib/rqpkg/deep.py": 'VALUE = "v1"\nclass Thing:\n    pass\n',
    "lib/rqpkg/rel.py": 'from . import deep\nfrom .deep import VALUE\n',
    "lib/rqcirc/__init__.py": 'from . import a\n',
    "lib/rqcirc/a.py": 'from . import b\nNAME = "a"\n',
    "lib/rqcirc/b.py": 'from . import a\ndef get():\n    return a.NAME\n',
    "lib/rqres/__init__.py": '',
    "lib/rqres/data.txt": 'hello-v1',
    "lib/rqhelper.py": 'import rqpkg.deep\nimport rqbrandnew\nVALUE = rqpkg.deep.VALUE\n',
    "lib/rqbrandnew/__init__.py": 'X = 1\n',
    "lib/rqsessnew/__init__.py": 'X = 1\n',
    "lib/rqswapper.py": """
        import sys
        class _M:
            VALUE = "v1"
        sys.modules[__name__] = _M()
    """,
    "lib/rqvirt/__init__.py": """
        import sys, types
        VALUE = "v1"
        sys.modules["rqvirt.generated"] = types.ModuleType("rqvirt.generated")
    """,
    "lib/rqwalk/__init__.py": '',
    "lib/rqwalk/plug_a.py": 'VALUE = "a"\n',
    "lib/rqwalk/inner/__init__.py": '',
    "lib/rqwalk/inner/plug_b.py": 'VALUE = "b"\n',
    "lib/rqwithpyc/__init__.py": 'from . import fast\nVALUE = "v1"\n',
    "lib/rqwithpyc/_fast_src.py": 'SPEED = 1\n',
    "site-packages/rqinst/__init__.py": """
        import logging, sys
        class Finder:
            @staticmethod
            def find_spec(name, path=None, target=None):
                return None
        FINDER = Finder()
        sys.meta_path.append(FINDER)
        log = logging.getLogger("rqinst")
        log.setLevel(logging.WARNING)
        HANDLER = logging.NullHandler()
        log.addHandler(HANDLER)
    """,
    "lib/rqstudio/__init__.py": "import sys\nsys._rq_loads = getattr(sys, '_rq_loads', 0) + 1\n",
    "lib/rqstudio/helpers.py": "VALUE = 'session-v1'\n",
    "lib/rqstudio/cb.py": """
        def on_node_added():          # a session callback Maya would fire mid-build
            from rqstudio import helpers
            return helpers.VALUE

        def peek():                   # shared code reading sys.modules by name (cgmath's _field_hints)
            import sys
            return getattr(sys.modules.get('rqstudio.helpers'), 'VALUE', None)
    """,
    "lib/rqtd/__init__.py": "VERSION = 'v1'\n",
    "lib/rqtd/deep.py": "VALUE = 'v1'\n",
    "lib/rqcb/__init__.py": "class Thing: pass\n",
    "lib/rqchain/__init__.py": "VERSION = 'v1'\n",
    "lib/rqreload.py": "V = 'v1'\n",
    "lib/rqlines.py": "def boom():\n    raise ValueError('v1')\n",
}


def _session_worker(out):
    try:
        import rqpkg.deep as d                    # session code: no run tag in this module
        out.append(d.VALUE)
    except BaseException as exc:                  # noqa: BLE001
        out.append(f"{type(exc).__name__}: {exc}")


class TestProbes(_Tree):
    TOPS = {"rqpkg", "rqcirc", "rqres", "rqhelper", "rqbrandnew", "rqsessnew", "rqswapper", "rqvirt",
            "rqwalk", "rqwithpyc", "rqinst", "rqstudio", "rqtd", "rqcb", "rqchain", "rqreload", "rqlines"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.make(_PROBES, paths=("lib", "site-packages"))
        src = os.path.join(cls.WORK, "lib", "rqwithpyc", "_fast_src.py")
        py_compile.compile(src, cfile=os.path.join(cls.WORK, "lib", "rqwithpyc", "fast.pyc"))
        os.remove(src)
        import rqpkg.deep, rqstudio.cb, rqstudio.helpers, rqtd.deep  # noqa: E401,F401 -- the session's
        cls.session = {n: sys.modules[n] for n in ("rqpkg", "rqpkg.deep", "rqstudio.helpers", "rqtd", "rqtd.deep")}
        time.sleep(0.02)
        _edit(cls.WORK, "lib/rqpkg/deep.py", '"v1"', '"v2"')        # disk now differs from the session's copy
        _write(cls.WORK, "lib/rqtd/deep.py", "VALUE = 'v2'\n")
        _write(cls.WORK, "lib/rqstudio/helpers.py", "VALUE = 'disk-v2'\n")

    def script(self, name, body):
        return _write(self.WORK, f"scripts/{name}.py", body)

    def run_(self, name, body, inputs=None, **kwargs):
        return _quiet(self.script(name, body), inputs, **kwargs)[0]

    # P1 / P2: a session thread importing an owned module during a run is not blocked
    def test_p01_p02_session_thread_during_a_run(self):
        out = []
        result = self.run_("thread_join", """
            import threading, rqpkg.deep
            t = threading.Thread(target=worker, args=(out,), daemon=True)
            t.start()
            t.join(2.0)
            alive = t.is_alive()
        """, {"worker": _session_worker, "out": out})
        self.assertFalse(result["alive"])
        self.assertEqual(out, ["v2"])                          # one view: the run's copy, at once

    # P3-P7: library code importing by name on the run's behalf
    def test_p03_mock_patch(self):
        result = self.run_("mockpatch", """
            from unittest import mock
            import rqpkg.deep
            with mock.patch("rqpkg.deep.VALUE", "PATCHED"):
                inside = rqpkg.deep.VALUE
        """)
        self.assertEqual(result["inside"], "PATCHED")

    def test_p04_pkgutil_resolve_name_and_walk_packages(self):
        before = set(sys.modules)
        result = self.run_("pkgutilres", """
            import pkgutil, rqpkg.deep, rqwalk
            resolved = pkgutil.resolve_name("rqpkg.deep:VALUE")
            walked = sorted(m.name for m in pkgutil.walk_packages(rqwalk.__path__, "rqwalk."))
        """)
        self.assertEqual(result["resolved"], "v2")
        self.assertEqual(result["walked"], ["rqwalk.inner", "rqwalk.inner.plug_b", "rqwalk.plug_a"])
        self.assertEqual([n for n in set(sys.modules) - before if n.startswith("rqwalk")], [])

    def test_p05_helper_loaded_by_path(self):
        result = self.run_("bypath", """
            import importlib.util, rqpkg.deep
            spec = importlib.util.spec_from_file_location("rqhelper", HELPER)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            helper_value = mod.VALUE
        """, {"HELPER": os.path.join(self.WORK, "lib", "rqhelper.py")})
        self.assertEqual(result["helper_value"], "v2")
        self.assertNotIn("rqbrandnew", sys.modules)

    def test_p06_exec_and_runpy_inside_a_run(self):
        sub = _write(self.WORK, "scripts/sub_script.py", 'import rqpkg.deep\nVALUE = rqpkg.deep.VALUE\n')
        result = self.run_("execsub", """
            import runpy, rqpkg.deep
            ns = {}
            exec("import rqpkg.deep\\nVALUE = rqpkg.deep.VALUE", ns)
            rp = runpy.run_path(SUB)
        """, {"SUB": sub})
        self.assertEqual((result["ns"]["VALUE"], result["rp"]["VALUE"]), ("v2", "v2"))

    def test_p07_dunder_import_with_empty_globals(self):
        result = self.run_("dunderimp", 'm = __import__("rqpkg.deep", {}, {}, ["VALUE"])\ngot = m.VALUE\n')
        self.assertEqual(result["got"], "v2")

    # P8-P10: what a run makes goes with the run (key-diff harvest)
    def test_p08_module_that_replaces_itself(self):
        script = self.script("swap", 'import rqswapper\nVALUE = rqswapper.VALUE\n')
        first, _ = _quiet(script)
        self.assertNotIn("rqswapper", sys.modules)
        _edit(self.WORK, "lib/rqswapper.py", '"v1"', '"v2"')
        second, _ = _quiet(script)
        self.assertEqual((first["VALUE"], second["VALUE"]), ("v1", "v2"))

    def test_p09_synthetic_submodule(self):
        script = self.script("s_virt", 'import rqvirt\nVALUE = rqvirt.VALUE\n')
        _quiet(script)
        self.assertEqual([n for n in sys.modules if n.startswith("rqvirt")], [])
        _edit(self.WORK, "lib/rqvirt/__init__.py", '"v1"', '"v2"')
        self.assertEqual(_quiet(script)[0]["VALUE"], "v2")

    def test_p10_sourceless_member(self):
        script = self.script("pyc", 'import rqwithpyc\nVALUE = rqwithpyc.VALUE\n')
        _quiet(script)
        self.assertEqual([n for n in sys.modules if n.startswith("rqwithpyc")], [])
        self.assertIn("rqwithpyc", _quiet(script)[0].owned)

    # P11: the file is __main__ while it runs
    def test_p11_dataclass_type_hints_pickle(self):
        result = self.run_("dc", """
            from __future__ import annotations
            import dataclasses, pickle, typing
            from typing import ClassVar

            @dataclasses.dataclass
            class Cfg:
                count: ClassVar[int] = 3
                name: str = "x"

            fields = [f.name for f in dataclasses.fields(Cfg)]
            hints = sorted(typing.get_type_hints(Cfg))
            pick = len(pickle.dumps(Cfg()))
        """)
        self.assertEqual(result["fields"], ["name"])
        self.assertEqual(result["hints"], ["count", "name"])
        self.assertGreater(result["pick"], 0)

    # P13: a shared package first imported in a run keeps its own setup
    def test_p13_shared_package_setup_kept(self):
        self.run_("instlib_use", "import rqinst\n")
        import rqinst
        try:
            self.assertIn(rqinst.FINDER, sys.meta_path)
            self.assertEqual(logging.getLogger("rqinst").level, logging.WARNING)
            self.assertIn(rqinst.HANDLER, logging.getLogger("rqinst").handlers)
        finally:
            sys.meta_path.remove(rqinst.FINDER)
            logging.getLogger("rqinst").handlers.clear()

    # P15, P16: resources, getsource, relative and circular imports, tracebacks
    def test_p15_resources_getsource_relative_circular(self):
        result = self.run_("res", """
            import importlib.resources, inspect, rqres, rqpkg.rel, rqcirc, rqcirc.b
            data = importlib.resources.files("rqres").joinpath("data.txt").read_text()
            first = inspect.getsource(rqpkg.rel).splitlines()[0]
            rel = rqpkg.rel.VALUE
            circ = rqcirc.b.get()
        """)
        self.assertEqual((result["data"], result["first"], result["rel"], result["circ"]),
                         ("hello-v1", "from . import deep", "v2", "a"))

    def test_p16_traceback_of_run_code(self):
        _write(self.WORK, "lib/rqtb.py", 'def boom():\n    raise ValueError("v1 line 2")\n')
        self.addCleanup(sys.modules.pop, "rqtb", None)
        first, _ = _quiet(self.script("tb", "import rqtb\n"))
        try:
            first["rqtb"].boom()
        except ValueError as exc:
            frame = __import__("traceback").extract_tb(exc.__traceback__)[-1]
        self.assertEqual((os.path.basename(frame.filename), frame.lineno, frame.line),
                         ("rqtb.py", 2, 'raise ValueError("v1 line 2")'))

    # P17-P21: nesting, interrupts, exits, folders, session imports of new packages
    def test_p17_run_inside_a_run(self):
        inner = self.script("inner", "import rqpkg.deep\nVALUE = rqpkg.deep.VALUE\n")
        result = self.run_("outer", """
            import rig, rqpkg.deep
            mine = rqpkg.deep
            r = rig.run(INNER)
            after = rqpkg.deep is mine
        """, {"INNER": inner})
        self.assertEqual((result["r"]["VALUE"], result["after"]), ("v2", True))

    def test_p18_keyboard_interrupt(self):
        path = list(sys.path)
        with self.assertRaises(KeyboardInterrupt):
            _quiet(self.script("kbi", "import rqpkg.deep\nraise KeyboardInterrupt\n"))
        self.assertEqual((run_module._ACTIVE, sys.path), ([], path))
        self.assertIs(sys.modules["rqpkg.deep"], self.session["rqpkg.deep"])

    def test_p19_sys_exit(self):
        with self.assertRaises(RuntimeError) as caught:
            _quiet(self.script("exit1", "import sys\nA = 1\nsys.exit(3)\n"))
        self.assertEqual((caught.exception.code, caught.exception.rig_run["A"]), (3, 1))
        self.assertIsInstance(caught.exception.__cause__, SystemExit)
        result, _ = _quiet(self.script("exit0", "import sys\nA = 1\nsys.exit()\nB = 2\n"))
        self.assertEqual((result["A"], "B" in result), (1, False))

    def test_p20_a_folder(self):
        with self.assertRaisesRegex(FileNotFoundError, "Not a file"):
            rig.run(os.path.join(self.WORK, "scripts"))

    def test_p21_session_callback_imports_a_new_package_mid_run(self):
        holder = {}

        def callback():
            import rqsessnew
            holder["mod"] = rqsessnew
        result = self.run_("cb", "cb()\n", {"cb": callback})
        self.assertIn("rqsessnew", result.owned)
        self.assertNotIn("rqsessnew", sys.modules)

    def test_p22_apostrophe_and_variable(self):
        os.environ["RQ_D"] = "scripts"
        self.addCleanup(os.environ.pop, "RQ_D")
        self.assertTrue(run_module._resolve("C:/O'Brien/$RQ_D/x.py").endswith("/O'Brien/scripts/x.py"))

    # P24: late imports of code a run made: the fast path swaps nothing
    def test_p24_late_import_fast_path(self):
        result = self.run_("late", """
            import rqpkg.deep
            def late_owned():
                import rqpkg.deep
                from rqpkg import deep
                return deep.VALUE
            def late_stdlib():
                import os
                return os.sep
        """)
        with mock.patch.object(run_module, "_enter", side_effect=AssertionError("view swapped")):
            self.assertEqual(result["late_owned"](), "v2")
            self.assertEqual(result["late_stdlib"](), os.sep)

    # the synthesis' view probe: lazy claim + one view
    def test_view_lazy_claim(self):
        import rqstudio.cb
        box = []

        def worker():
            started = time.perf_counter()
            import rqtd.deep
            box.append((rqtd.deep.VALUE, time.perf_counter() - started))
        loads = sys._rq_loads
        result = self.run_("view", """
            import pkgutil, threading, rqtd.deep
            R = {"script": rqtd.deep.VALUE, "resolve": pkgutil.resolve_name("rqtd.deep:VALUE"),
                 "peek": PEEK(), "fire": sorted({FIRE() for _ in range(50)})}
            th = threading.Thread(target=WORKER, daemon=True)
            th.start()
            th.join(2.0)
            R["thread done"] = not th.is_alive()
        """, {"FIRE": rqstudio.cb.on_node_added, "PEEK": rqstudio.cb.peek, "WORKER": worker})
        self.assertEqual(result["R"], {"script": "v2", "resolve": "v2", "peek": "session-v1",
                                       "fire": ["session-v1"], "thread done": True})
        self.assertEqual(sys._rq_loads, loads)                   # the studio tool never re-executed
        self.assertEqual(box[0][0], "v2")
        self.assertIs(sys.modules["rqstudio.helpers"], self.session["rqstudio.helpers"])
        self.assertEqual(sys.modules["rqtd.deep"].VALUE, "v1")

    # the critic's C1: a thread a run started keeps importing after the run
    def test_c1_run_thread_after_the_run_never_swaps(self):
        for i in range(60):
            _write(self.WORK, f"lib/rqtd/lazy{i}.py", "import time\ntime.sleep(0.002)\n")
        importlib.invalidate_caches()
        result = self.run_("c1", """
            import threading, time, importlib, rqtd
            STOP = threading.Event()
            def worker():
                for i in range(60):
                    if STOP.is_set():
                        break
                    importlib.import_module(f"rqtd.lazy{i}")
            T = threading.Thread(target=worker, daemon=True)
            T.start()
        """)
        anomalies = []
        started = time.perf_counter()
        while result["T"].is_alive() and time.perf_counter() - started < 3.0:
            try:
                import rqtd as t
                if t is not self.session["rqtd"]:
                    anomalies.append("run copy")
            except Exception as exc:                            # noqa: BLE001
                anomalies.append(repr(exc))
        result["STOP"].set()
        result["T"].join(5)
        self.assertEqual(anomalies, [])
        for name in [n for n in sys.modules if n.startswith("rqtd.lazy")]:
            del sys.modules[name]

    # the critic's C2: a callback a run made imports from that run (isinstance holds)
    def test_c2_callback_keeps_its_run(self):
        result = self.run_("c2", """
            import rqcb
            OBJ = rqcb.Thing()
            def on_click():
                from rqcb import Thing
                return isinstance(OBJ, Thing)
        """)
        self.assertTrue(result["on_click"]())
        self.assertNotIn("rqcb", sys.modules)

    # the critic's C3: a Run passed on: names go, module objects don't
    def test_c3_chaining_skips_modules(self):
        first = self.run_("a", "import rqchain\nRESULT = {'joints': 3}\n")
        with self.assertRaises(NameError) as caught:
            self.run_("b", "GOT = RESULT\nSEEN = rqchain.VERSION\n", first)
        self.assertEqual(caught.exception.rig_run["GOT"], {"joints": 3})

    # P14: a button the run made reloads one of the run's modules, after the run
    def test_p14_reload_in_a_callback_and_in_call(self):
        result = self.run_("reloadbtn", """
            import importlib, rqreload
            def on_button():
                import rqreload
                return importlib.reload(rqreload).V
        """)
        _edit(self.WORK, "lib/rqreload.py", "'v1'", "'v2'")
        self.assertEqual(result["on_button"](), "v2")
        _edit(self.WORK, "lib/rqreload.py", "'v2'", "'v3'")
        self.assertEqual(result.call("on_button"), "v3")
        self.assertNotIn("rqreload", sys.modules)
        self.assertEqual(result.modules["rqreload"].V, "v3")

    # the Run shows the dict the file runs in: what run.call sets later shows
    def test_call_sets_a_global_the_run_shows(self):
        result = self.run_("liveglobals", """
            OUT = None
            def main(n):
                global OUT
                OUT = n * 2
                return OUT
        """)
        self.assertEqual(result.call("main", 4), 8)
        self.assertEqual(result["OUT"], 8)
        self.assertNotIn(run_module._MARK, dict(result))
        self.assertEqual(len(result), len(dict(result)))
        with self.assertRaises(KeyError):
            result[run_module._MARK]
        nofunc = self.run_("nofunc", "A = 1\n")
        self.assertEqual((nofunc["A"], run_module._MARK in nofunc), (1, False))

    # exit() / quit() close sys.stdin before raising SystemExit: the session gets an open stdin back
    def test_exit_keeps_stdin_open(self):
        read_fd, write_fd = os.pipe()
        stdin = open(read_fd, "r", encoding="utf-8", closefd=False)
        saved = sys.stdin
        sys.stdin = stdin
        try:
            self.run_("quitter", "A = 1\nexit()\nB = 2\n")
            self.assertFalse(sys.stdin.closed)
            self.assertEqual(sys.stdin.fileno(), read_fd)
            os.write(write_fd, b"typed\n")
            self.assertEqual(sys.stdin.readline(), "typed\n")
        finally:
            sys.stdin = saved
            stdin.close()
            os.close(read_fd)
            os.close(write_fd)

    # linecache: an entry a run wrote follows the disk, as Python's own do (a session reload, an edit)
    def test_linecache_follows_the_disk(self):
        self.run_("lines", "import rqlines\n")
        path = os.path.join(self.WORK, "lib", "rqlines.py")
        self.assertEqual(linecache.getline(path, 2).strip(), "raise ValueError('v1')")
        time.sleep(0.02)
        _write(self.WORK, "lib/rqlines.py", "# a line on top\ndef boom():\n    raise ValueError('v2')\n")
        linecache.checkcache(path)
        self.assertEqual(linecache.getline(path, 3).strip(), "raise ValueError('v2')")

    # a thread calling rig.run while a run goes on gets an error at once instead of waiting for the run
    def test_rig_run_from_another_thread_during_a_run(self):
        tiny = self.script("tiny", "X = 1\n")
        result = self.run_("threadrun", """
            import threading, rig
            box = []
            def worker():
                try:
                    rig.run(TINY)
                    box.append("ran")
                except RuntimeError as err:
                    box.append(str(err))
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(3.0)
            alive = t.is_alive()
        """, {"TINY": tiny})
        self.assertFalse(result["alive"])
        self.assertIn("another thread", result["box"][0])

    # a Run made by an older copy of rig (a reload, a purge) still passes its names on
    def test_a_run_from_a_reloaded_rig_chains(self):
        spec = importlib.util.spec_from_file_location("_rig_run_copy", run_module.__file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        old = copy.Run("old.py", set())
        old._globals = {"__name__": "__main__", "__file__": "old.py", "SIDE": "L"}
        self.assertNotIsInstance(old, run_module.Run)
        self.assertEqual(self.run_("chain_old", "GOT = SIDE\n", old)["GOT"], "L")

    def test_undo_chunk_per_run(self):
        names = []
        with mock.patch.object(run_module, "undo_chunk", side_effect=lambda name: names.append(name) or
                               __import__("contextlib").nullcontext()):
            self.run_("chunk", "X = 1\n")
        self.assertEqual(names, ["rig.run chunk.py"])

    def test_options_given_back(self):
        before = rig.get_options()
        self.run_("opts", "import rig\nrig.set_options(create_containers=not rig.get_options()['create_containers'])\n")
        self.assertEqual(rig.get_options(), before)

    def test_rig_and_its_closure_shared(self):
        result = self.run_("usesrig", "import rig, cgmath\nNODE = rig.Node\n")
        self.assertIs(result["NODE"], rig.Node)
        self.assertEqual((result.shared["rig"], result.shared["cgmath"]), ("rig", "rig"))
        self.assertIn("cgmath", run_module._RIG_TOPS)


class TestWeakRegistries(unittest.TestCase):
    """The memo registries hold their members weakly: a memoized function or a NodeOp that
    nothing uses any more (a run's module that ended) goes, with its cache."""

    def test_memoized_function_goes(self):
        @rig.memoize
        def gone(i):
            return i
        self.assertIn(gone, memoize_module._ALL_MEMOIZED)
        count = len(memoize_module._ALL_MEMOIZED)
        del gone
        gc.collect()
        self.assertEqual(len(memoize_module._ALL_MEMOIZED), count - 1)

    def test_nodeop_goes(self):
        op = NodeOp("rr_probe")
        self.assertIn(op, memoize_module._ALL_NODEOP_CACHES)
        count = len(memoize_module._ALL_NODEOP_CACHES)
        del op
        gc.collect()
        self.assertEqual(len(memoize_module._ALL_NODEOP_CACHES), count - 1)

    def test_seeded_random_cache_stays_with_its_function(self):
        from rig import random as rig_random
        holders = [h for h in memoize_module._ALL_NODEOP_CACHES if type(h).__name__ == "_SeedCacheHolder"]
        self.assertTrue(holders)
        self.assertTrue(any(getattr(f, "_cache_holder", None) in holders for f in vars(rig_random).values()))
