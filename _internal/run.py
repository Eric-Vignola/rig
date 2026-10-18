"""
``rig.run``: run a Python file as if in a fresh Python, without leaving Maya.

    import rig
    result = rig.run("D:/chars/bob/arm.py", {"side": "L"})   # the file runs as __main__
    result["ctrls"]                                          # its globals (read-only)
    result.call("main", lastBuildStep=None)                  # one of its functions, inside the run

The file runs through :func:`runpy.run_path` as ``__main__``, as ``python arm.py``
runs it; rig never transforms its code. Every module it imports from your own
code (plain ``.py`` outside an install) is executed again from the file's
current bytes, so an edit is picked up with no reload, and the session's own
copies are never touched. Shared with the session, never re-executed: the
standard library, an install (site-packages, the interpreter, Maya, Program
Files), a package with a compiled member, ``rig`` and the packages rig itself
uses (cgmath), and the top-level names given in ``shared=``.

A run hides a session package only once the file's own code imports it ("lazy
claim"): code that is not the file's (a session callback, a thread) sees what
``sys.modules`` shows. Code a run made keeps importing (and ``importlib.reload``
-ing) that run's modules when it runs later (a callback). Only Maya's thread (or
the thread inside ``rig.run``) swaps modules in and out of ``sys.modules``; a
``rig.run`` from another thread while a run goes on raises RuntimeError.

One run is one undo step, ``rig.run <file name>``; a run inside a run joins the
outer step. Afterwards the session gets back ``sys.argv``, ``sys.path``, the std
streams, ``sys.excepthook`` / ``displayhook`` and rig's options.
"""

from __future__ import annotations

import builtins
import collections.abc
import contextlib
import errno
import getpass
import hashlib
import importlib
import importlib.machinery
import importlib.util
import linecache
import os
import re
import runpy
import sys
import sysconfig
import threading
import types

from rig._internal.container import get_options, set_options
from rig._internal.undo import undo_chunk

_MARK      = "__rig_run__"          # globals key: the Run whose code this is
_PYTHON    = {"__main__", "__mp_main__", "builtins"}
_STDLIB    = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(sys.builtin_module_names)
_SITE_DIRS = {"site-packages", "dist-packages"}
_SOURCE    = importlib.machinery.SourceFileLoader
_LOCK      = threading.RLock()
_ACTIVE    = []                     # (run, hidden modules, saved sys.path) per view shown, innermost last
_OWNER     = [None]                 # the thread showing them
_MAIN      = threading.main_thread().ident
_RIG_TOPS  = set()                  # rig and the dev packages its modules use: always shared
_ROOTS     = ()                     # install folders (see _install_roots)
_CODE      = {}                     # path -> (digest, code object)
_STAMPS    = {}                     # path -> (mtime_ns, size) when first seen
_SEEN      = set()                  # session module names already stamped
_PASS, _SWAP = object(), object()
_original_import        = builtins.__import__
_original_import_module = importlib.import_module
_original_reload        = importlib.reload


class RunExitError(RuntimeError):
    """The file called ``sys.exit()`` with a failure code (``.code``); the host keeps running."""


# ---------------------------------------------------------------------------
# Public API

def run(path, inputs=None, *, argv=(), shared=()):
    """Run the Python file at ``path`` as ``__main__``, as if in a fresh Python, and return its :class:`Run`.

    ``inputs`` (a mapping, or the :class:`Run` of an earlier round) is copied into the file's
    starting globals: module objects are skipped, and a dunder name raises ValueError. ``argv``
    becomes ``sys.argv[1:]``; ``shared`` names top-level packages to share with the session
    instead of re-executing them. The run is one undo step, ``rig.run <file name>``.

    ``path`` may use ``~`` and environment variables (``$VAR``, ``${VAR}``, ``%VAR%`` on Windows),
    expanded until nothing changes, and ``..``; it comes out with forward slashes. A path that is
    no file raises FileNotFoundError, naming any variable in it that isn't set. An error in the
    file propagates with ``err.rig_run``, the run so far; ``sys.exit(n)`` with a failure code
    raises :class:`RunExitError` carrying ``.code``.
    """
    path = _resolve(path)
    if not os.path.isfile(path):
        why = "Not a file" if os.path.exists(path) else _missing(path)
        raise FileNotFoundError(errno.ENOENT, why, path)
    names = _inputs(inputs)
    argv, shared = _strings("argv", argv), set(_strings("shared", shared))
    with _LOCK:                                              # never held while the file's code runs
        _check_thread()
        _install()
        _stamp()
    result          = Run(path, shared)
    result.sys_path = [os.path.normpath(os.path.dirname(path))] + sys.path
    options = get_options()
    stdin   = _stdin_state()
    saved   = (sys.argv, sys.stdin, sys.stdout, sys.stderr, sys.displayhook, sys.excepthook)
    entered = False
    importlib.invalidate_caches()
    try:
        with undo_chunk(f"rig.run {os.path.basename(path)}"):
            with _LOCK:
                _check_thread()                              # re-checked with the view entered atomically
                _enter(result)
                entered = True
            try:
                sys.argv = [path, *argv]
                result._bind(runpy.run_path(path, {**names, _MARK: result}, "__main__"))
            finally:
                with _LOCK:
                    _leave()
    except SystemExit as err:
        result._globals = _globals_of(err, result)
        if err.code not in (None, 0):
            exit_error = RunExitError(f"rig.run: {os.path.basename(path)} called sys.exit({err.code!r})")
            exit_error.code, exit_error.rig_run = err.code, result
            raise exit_error from err
    except BaseException as err:
        if entered:
            result._globals = _globals_of(err, result)
            with contextlib.suppress(Exception):
                err.rig_run = result                       # post-mortem: sys.last_value.rig_run
        raise
    finally:
        if entered:
            sys.argv, sys.stdin, sys.stdout, sys.stderr, sys.displayhook, sys.excepthook = saved
            _reopen_stdin(stdin)
            if get_options() != options:
                set_options(**options)
    return result


class Run(collections.abc.Mapping):
    """One run of a file: its globals (``run["name"]``, read-only; ``dict(run)`` for a copy), the
    very dict the file ran in, so what its functions set later shows, and the modules it executed
    (``run.modules``)."""

    _rig_run_result = True           # a Run, whichever copy of this module made it (a reload, a purge)

    def __init__(self, path, shared):
        self.path     = path
        self.owned    = set()        # top-level packages this run executes itself
        self.shared   = {}           # top-level name -> why the session's copy is used
        self.modules  = {}           # this run's modules, while no view shows them
        self.sys_path = None
        self._extra   = shared
        self._globals = {}           # the file's globals (with _MARK, hidden here)

    def __getitem__(self, key):
        if key == _MARK:
            raise KeyError(key)
        return self._globals[key]

    def __iter__(self):
        return (key for key in list(self._globals) if key != _MARK)

    def __len__(self):
        return len(self._globals) - (_MARK in self._globals)

    def __repr__(self):
        return f"<rig.run {self.path!r}: {len(self.modules)} modules of its own>"

    def _bind(self, scope):
        """runpy returns a copy of the file's globals: keep the dict its functions write to."""
        self._globals = _live_scope(scope, self) or scope

    def _adopt(self, func):
        scope = getattr(func, "__globals__", None) if isinstance(func, types.FunctionType) else None
        if scope is not None and scope is not self._globals and _is_file_scope(scope, self):
            self._globals = scope

    @contextlib.contextmanager
    def active(self):
        """Show this run's modules in ``sys.modules`` (and its ``sys.path``) for the duration."""
        with _LOCK:                                          # never held while the caller's code runs
            entered = _may_swap() and not (_ACTIVE and _ACTIVE[-1][0] is self)
            if entered:
                _enter(self)
        try:
            yield self
        finally:
            if entered:
                with _LOCK:
                    _leave()

    def call(self, func, *args, **kwargs):
        """Call ``func`` (a name in the file's globals, or any callable) with this run's modules shown."""
        if isinstance(func, str):
            func = self[func]
        self._adopt(func)
        with self.active():
            return func(*args, **kwargs)


def _is_file_scope(scope, run):
    return isinstance(scope, dict) and scope.get(_MARK) is run and scope.get("__name__") == "__main__"


def _live_scope(scope, run):
    """The dict the file ran in, from a function (or a class's method) it defined, or None."""
    for value in list(scope.values()):
        members = list(vars(value).values()) if isinstance(value, type) else (value,)
        for member in members:
            if isinstance(member, (staticmethod, classmethod, types.MethodType)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if isinstance(member, types.FunctionType) and _is_file_scope(member.__globals__, run):
                return member.__globals__
    return None


def _check_thread():
    if _ACTIVE and _OWNER[0] != threading.get_ident():
        raise RuntimeError("rig.run: a run is going on in another thread; start runs from the thread that runs "
                           "them (Maya's), one after the other")


def _stdin_state():
    """sys.stdin as the run found it, to put back open: exit() / quit() close it before SystemExit."""
    stdin = sys.stdin
    try:
        if stdin is None or stdin.closed:
            return None
        return stdin, stdin.fileno(), stdin.encoding, stdin.errors, sys.__stdin__ is stdin
    except Exception:                                        # no fd (a GUI stream): nothing to put back
        return None


def _reopen_stdin(state):
    if state is None:
        return
    stdin, fd, encoding, errors, dunder = state
    if sys.stdin is not stdin or not getattr(stdin, "closed", False):
        return
    try:
        fresh = open(fd, "r", encoding=encoding, errors=errors, closefd=False)
    except (OSError, ValueError, LookupError):
        return
    sys.stdin = fresh
    if dunder and sys.__stdin__ is stdin:
        sys.__stdin__ = fresh


def _inputs(inputs):
    if inputs is None:
        return {}
    if not isinstance(inputs, collections.abc.Mapping):
        raise TypeError(f"rig.run: inputs is a mapping of names (a dict, a Run), not {type(inputs).__name__}")
    names = {}
    is_run = getattr(type(inputs), "_rig_run_result", False) is True
    for key, value in inputs.items():
        if not isinstance(key, str):
            raise TypeError(f"rig.run: an input name is a str, not {key!r}")
        if key.startswith("__") and key.endswith("__"):
            if is_run:
                continue                                     # the earlier round's own __file__ ...
            raise ValueError(f"rig.run: the input {key!r} is a name Python sets for the file; use another name")
        if not isinstance(value, types.ModuleType):          # a module rides along stale: the file imports it
            names[key] = value
    return names


def _strings(what, values):
    values = tuple(values) if not isinstance(values, (str, bytes)) else None
    if values is None or not all(isinstance(value, str) for value in values):
        raise TypeError(f"rig.run: {what} is a list of str, as {what}=[\"name\"]")
    return values


def _globals_of(err, run):
    """The file's globals at the error, from its frame in the traceback (none: it never started)."""
    tb = err.__traceback__
    while tb is not None:
        scope = tb.tb_frame.f_globals
        if _is_file_scope(scope, run):
            return scope
        tb = tb.tb_next
    return {}


# ---------------------------------------------------------------------------
# The path: ~ and variables expanded until nothing changes, .. walked, forward slashes

_WINDOWS    = os.name == "nt"
_SEPARATORS = "/\\" if _WINDOWS else "/"
_PASSES     = 32
_NAME_END   = r"(?![\w-])" if _WINDOWS else r"(?!\w)"
_FLAGS      = re.ASCII | (re.IGNORECASE if _WINDOWS else 0)
_ANY_REF    = re.compile(r"\$([\w-]+)|\$\{([^}]*)\}|%([^%]+)%" if _WINDOWS else r"\$(\w+)|\$\{([^}]*)\}", _FLAGS)


def _reference(name):
    forms = [rf"\${name}{_NAME_END}", rf"\$\{{{name}\}}"] + ([rf"%{name}%"] if _WINDOWS else [])
    return re.compile("|".join(forms), _FLAGS)


_HOME_REF, _USER_REF = _reference("HOME"), _reference("USER")


def _resolve(path):
    """The absolute path of ``path``: ``~`` and ``$HOME`` are the user folder (USERPROFILE on Windows,
    never the Documents folder Maya sets HOME to), ``$USER`` is the login name when USER isn't set,
    every other variable as defined (an unset one stays as written)."""
    path = os.fsdecode(os.fspath(path))
    seen = [path]
    while True:
        expanded = _expand(path)
        if expanded == path:
            break
        if expanded in seen or len(seen) > _PASSES:
            raise ValueError(f"rig.run: the variables in {seen[0]!r} refer to each other, so it never stops expanding")
        seen.append(expanded)
        path = expanded
    path = os.path.abspath(path)
    if _WINDOWS and not path.startswith("\\\\?\\"):          # extended-length paths need their backslashes
        path = path.replace("\\", "/")
    return path


def _expand(path):
    path = _HOME_REF.sub(lambda match: _home(), path)
    user = None if os.environ.get("USER") else _login()
    if user:
        path = _USER_REF.sub(lambda match: user, path)
    # not os.path.expandvars: on Windows it reads a ' as a quote and expands nothing after it
    path = _ANY_REF.sub(lambda match: os.environ.get(next(filter(None, match.groups()), ""), match.group(0)), path)
    if path[:1] == "~" and (len(path) == 1 or path[1] in _SEPARATORS):
        return _home() + path[1:]
    return os.path.expanduser(path)


def _home():
    if _WINDOWS and os.environ.get("USERPROFILE"):
        return os.environ["USERPROFILE"]
    return os.path.expanduser("~")


def _login():
    try:
        return getpass.getuser()
    except Exception:                                          # KeyError, ImportError, OSError (3.13)
        return None


def _missing(path):
    unset = [match.group(0) for match in _ANY_REF.finditer(path)
             if next(filter(None, match.groups()), "") not in os.environ]
    if not unset:
        return "No such file"
    return f"No such file; {', '.join(unset)} {'is' if len(unset) == 1 else 'are'} not set"


# ---------------------------------------------------------------------------
# Which packages a run executes itself

def _install_roots():
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    paths = sysconfig.get_paths()
    roots.update(filter(None, (paths.get("stdlib"), paths.get("platstdlib"))))
    roots.update(filter(None, map(os.environ.get, ("MAYA_LOCATION", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"))))
    return tuple(os.path.join(os.path.normcase(os.path.abspath(root)), "") for root in roots)


def _spec_of(module):
    try:
        return object.__getattribute__(module, "__dict__").get("__spec__")    # never wakes a lazy module
    except (AttributeError, TypeError):
        return None


def _kind(spec):
    """'source' for a plain .py file, 'namespace' for a folder without __init__.py, else None."""
    if spec is None:
        return None
    if spec.origin in (None, "namespace") and spec.submodule_search_locations is not None:
        return "namespace"
    if type(spec.loader) in (_SOURCE, _RunLoader) and spec.origin and os.path.isabs(spec.origin):
        return "source"
    return None


def _installed(spec, kind):
    paths = list(spec.submodule_search_locations) if kind == "namespace" else [spec.origin]
    for path in paths:
        norm = os.path.normcase(os.path.abspath(path))
        if norm.startswith(_ROOTS) or not _SITE_DIRS.isdisjoint(norm.split(os.sep)):
            return True
    return False


def _rule(run, top):
    """Why ``top`` is shared whatever its files are, or None."""
    if top in _PYTHON or top in _STDLIB:
        return "stdlib"
    if top in _RIG_TOPS:
        return "rig"
    if top in run._extra:
        return "shared"
    return None


def _why_shared(members):
    """None when every loaded module of a package is a plain .py outside an install, else why not."""
    for name, module in members:
        if module is None:
            continue
        spec = _spec_of(module)
        kind = _kind(spec)
        if kind is None:
            return f"not plain .py: {name}"
        if _installed(spec, kind):
            return "installed"
    return None


def _members(top):
    prefix = top + "."
    return [(name, module) for name, module in list(sys.modules.items()) if name == top or name.startswith(prefix)]


def _claim(run, top):
    """The file imports ``top``: hide the session's copy for the rest of the run, or share it."""
    if not top or top in run.owned or top in run.shared:
        return
    why = _rule(run, top)
    if why is None:
        members = _members(top)
        if not members:
            return                                           # never loaded: _Finder decides when it finds it
        why = _why_shared(members)
        if why is None:
            hidden = _ACTIVE[-1][1]
            for name, module in members:
                hidden[name] = sys.modules.pop(name)
            run.owned.add(top)
            return
    run.shared[top] = why
    _report_stale(top, why)


_IMPORTS = re.compile(rb"^[ \t]*(?:from[ \t]+(\w+)[\w.]*[ \t]+import\b|import[ \t]+([\w.]+(?:[ \t]+as[ \t]+\w+)?"
                      rb"(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?)*)[ \t]*(?:#.*)?\r?$)", re.M)


def _files_of(top):
    """The .py files of a dev package (plain .py, outside an install), found without importing it."""
    module = sys.modules.get(top)
    try:
        spec = _spec_of(module) if module is not None else importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return []
    kind = _kind(spec)
    if kind is None or _installed(spec, kind):
        return []
    if spec.submodule_search_locations is None:
        return [spec.origin]
    files = []
    for folder in spec.submodule_search_locations:
        for root, dirs, names in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in ("_tests", "examples", "__pycache__")]
            files.extend(os.path.join(root, name) for name in names if name.endswith(".py"))
    return files


def _rig_closure():
    """rig and every dev package its code imports, at the top or inside a function (cgmath), and
    theirs: a run must never execute them again (the session's classes would stop matching)."""
    tops, todo = set(), ["rig"]
    while todo:
        top = todo.pop()
        if top in tops:
            continue
        tops.add(top)
        for path in _files_of(top):
            with contextlib.suppress(OSError), open(path, "rb") as fh:
                for match in _IMPORTS.finditer(fh.read()):
                    for part in (match.group(1) or match.group(2)).split(b","):
                        name = part.strip().partition(b".")[0].partition(b" ")[0].decode("ascii", "replace")
                        if name not in _STDLIB and name not in tops:
                            todo.append(name)
    return tops


def _source_of(module):
    spec = _spec_of(module)
    if _kind(spec) != "source" or os.path.normcase(os.path.abspath(spec.origin)).startswith(_ROOTS):
        return None
    try:
        st = os.stat(spec.origin)
    except OSError:
        return None
    return spec.origin, (st.st_mtime_ns, st.st_size)


def _stamp():
    """First sight of each session module: the version the session runs."""
    if _ACTIVE:
        return
    for name, module in list(sys.modules.items()):
        if name not in _SEEN:
            _SEEN.add(name)
            found = _source_of(module)
            if found:
                _STAMPS.setdefault(*found)


def _report_stale(top, why):
    if why == "stdlib":
        return
    stale = []
    for name, module in _members(top):
        found = _source_of(module)
        if found and _STAMPS.get(found[0], found[1]) != found[1]:
            stale.append(f"{name} ({why})")
    if stale:
        more = f" and {len(stale) - 8} more" if len(stale) > 8 else ""
        sys.stderr.write("rig.run: changed on disk, but shared with this session, so this run uses the old code"
                         f" (restart to pick it up): {', '.join(stale[:8])}{more}\n")


# ---------------------------------------------------------------------------
# Loading a run's modules

def _compile(path):
    """Code for the file's current bytes: compiled once per version, never through __pycache__."""
    with open(path, "rb") as fh:
        data = fh.read()
        try:
            stamp = os.fstat(fh.fileno()).st_mtime
        except OSError:
            stamp = None
    digest = hashlib.blake2b(data, digest_size=16).digest()
    known  = _CODE.get(path)
    if known is not None and known[0] == digest:
        return known[1]
    code        = compile(data, path, "exec", dont_inherit=True)
    _CODE[path] = (digest, code)
    if stamp is not None:                                    # the file's stamp: checkcache follows the disk
        linecache.cache[path] = (len(data), stamp, importlib.util.decode_source(data).splitlines(True), path)
    return code


class _RunLoader(_SOURCE):
    """Compiles from the file's current bytes and tags the module with its run."""

    def __init__(self, fullname, path, run):
        super().__init__(fullname, path)
        self.run = run

    def get_code(self, fullname):
        return _compile(self.get_filename(fullname))

    def exec_module(self, module):
        module.__dict__[_MARK] = self.run
        super().exec_module(module)


class _Finder:
    """First on sys.meta_path: hands each module the shown run executes to a _RunLoader."""

    _rig_run_finder = True

    @staticmethod
    def find_spec(name, path=None, target=None):
        run = _shown()
        if run is None:
            return None
        top = name.partition(".")[0]
        new = top not in run.owned
        if new:
            why = None if top in run.shared else _rule(run, top)
            if why:
                run.shared[top] = why
            if name != top or top in run.shared:
                return None                                  # inside a shared package, or never the run's
        spec = None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if finder is not _Finder and find is not None:
                spec = find(name, path, target)
                if spec is not None:
                    break
        if spec is None:
            return None
        kind = _kind(spec)
        if new:                                              # the first import of this package anywhere
            if kind is None or _installed(spec, kind):
                run.shared[top] = "installed" if kind else f"not plain .py: {name}"
                return spec
            run.owned.add(top)
        if kind == "source" and type(spec.loader) is _SOURCE:
            spec.loader = _RunLoader(spec.name, spec.origin, run)
        return spec


# ---------------------------------------------------------------------------
# Views of sys.modules

def _shown():
    """The run whose view sys.modules shows, or None (read from any thread: the owner may pop it)."""
    try:
        return _ACTIVE[-1][0]
    except IndexError:
        return None


def _may_swap():
    """Only one thread swaps: the one showing a view, else Maya's (the main) thread."""
    me = threading.get_ident()
    return _OWNER[0] == me if _ACTIVE else me == _MAIN


def _harvest(run):
    """Every name under ``run``'s packages that sys.modules holds now, whatever the object."""
    if not run.owned:
        return {}
    return {name: module for name, module in list(sys.modules.items()) if name.partition(".")[0] in run.owned}


def _enter(run):
    if _ACTIVE:
        outer = _ACTIVE[-1][0]
        outer.modules = _harvest(outer)
    else:
        _OWNER[0] = threading.get_ident()
    hidden = _harvest(run)
    for name in hidden:
        del sys.modules[name]
    sys.modules.update(run.modules)
    _ACTIVE.append((run, hidden, sys.path[:]))
    if run.sys_path is not None:
        sys.path[:] = run.sys_path
    if sys.meta_path[:1] != [_Finder]:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_Finder)
        sys.meta_path.insert(0, _Finder)


def _leave():
    run, hidden, path = _ACTIVE.pop()
    run.modules = _harvest(run)
    for name in run.modules:
        sys.modules.pop(name, None)
    sys.modules.update(hidden)
    run.sys_path = sys.path[:]
    sys.path[:]  = path
    if not _ACTIVE:
        _OWNER[0] = None


# ---------------------------------------------------------------------------
# Import hooks: code a run made imports that run's modules

def _install():
    global _original_import, _original_import_module, _original_reload, _ROOTS
    if builtins.__import__ is _import:
        return
    previous, previous_module = builtins.__import__, importlib.import_module   # a reloaded copy's hooks
    _original_import        = getattr(previous, "_rig_run_original", previous)
    _original_import_module = getattr(previous_module, "_rig_run_original", previous_module)
    _original_reload        = getattr(importlib.reload, "_rig_run_original", importlib.reload)
    sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, "_rig_run_finder", False)]
    _import._rig_run_original, _import_module._rig_run_original = _original_import, _original_import_module
    _reload._rig_run_original = _original_reload
    builtins.__import__, importlib.import_module, importlib.reload = _import, _import_module, _reload
    _ROOTS = _install_roots()
    _RIG_TOPS.update(_rig_closure())


def _route(run, absolute, fromlist, whole, fast=True):
    """_PASS (import as usual), _SWAP (import with ``run``'s view shown), or what the import returns:
    the module (``whole``, or a ``from`` import) or its top-level package."""
    top = absolute.partition(".")[0]
    if _shown() is run:                                      # the file's code, its view shown
        if _OWNER[0] == threading.get_ident():
            _claim(run, top)
        return _PASS
    if top not in run.owned:
        return _PASS                                         # not the run's: what sys.modules shows
    module = run.modules.get(absolute) if fast else None
    if module is not None:                                   # loaded already: no view swap
        names = list(fromlist)
        if "*" in names:
            names = [n for n in names if n != "*"] + list(getattr(module, "__all__", ()))
        if all(isinstance(n, str) and hasattr(module, n) for n in names):
            return module if (fromlist or whole) else run.modules.get(top, _SWAP)
    return _SWAP if _may_swap() else _PASS


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    scope = globals or sys._getframe(1).f_globals
    run   = scope.get(_MARK) if isinstance(scope, dict) else None
    if run is None:
        return _original_import(name, globals, locals, fromlist, level)
    try:
        absolute = importlib._bootstrap._resolve_name(name, importlib._bootstrap._calc___package__(scope), level) \
            if level else name
    except Exception:                                        # Python raises its own error below
        return _original_import(name, globals, locals, fromlist, level)
    route = _route(run, absolute, fromlist or (), False, fast=bool(fromlist) or not level)
    if route is _PASS:
        return _original_import(name, globals, locals, fromlist, level)
    if route is not _SWAP:
        return route
    with run.active():
        return _original_import(name, globals, locals, fromlist, level)


def _import_module(name, package=None):
    run = sys._getframe(1).f_globals.get(_MARK)
    if run is None:
        return _original_import_module(name, package)
    try:
        absolute = importlib.util.resolve_name(name, package)
    except (ImportError, ValueError, AttributeError, TypeError):
        return _original_import_module(name, package)
    route = _route(run, absolute, (), True)
    if route is _PASS:
        return _original_import_module(name, package)
    if route is not _SWAP:
        return route
    with run.active():
        return _original_import_module(name, package)


def _reload(module):
    """importlib.reload of one of a run's modules (a button the run made): reloaded with its run shown."""
    try:
        name = object.__getattribute__(module, "__dict__").get("__name__")
        candidates = (object.__getattribute__(module, "__dict__").get(_MARK), sys._getframe(1).f_globals.get(_MARK))
    except (AttributeError, TypeError):
        return _original_reload(module)
    for run in candidates:
        modules = getattr(run, "modules", None) if getattr(type(run), "_rig_run_result", False) else None
        if modules is not None and modules.get(name) is module and _shown() is not run:
            with run.active():
                return _original_reload(module)
    return _original_reload(module)
