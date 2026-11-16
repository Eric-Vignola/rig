"""
``rig.run``: run a Python file as if in a fresh Python, without leaving Maya; or
several files, one after the other, in one namespace. ``rig.exec``: the same for a
string of code, as ``python -c`` runs it.

    import rig
    result = rig.run("D:/chars/bob/arm.py", {"side": "L"})   # the file runs as __main__
    result["ctrls"]                                          # its globals (read-only)
    result.call("main", lastBuildStep=None)                  # one of its functions, inside the run
    rig.run(["D:/chars/lib.py", "D:/chars/bob/arm.py"])      # in turn, in one namespace
    rig.exec("import arm; arm.build()")                      # a string, as python -c runs it

The file runs as ``__main__``, as ``python arm.py`` runs it: rig executes its code
itself, as :func:`runpy.run_path` does, and never transforms it. Several files run
one after the other in one ``__main__`` namespace, as one long file would, each
with its own ``__file__`` and ``sys.argv[0]`` while it runs. The code of a
``rig.exec`` runs as ``__main__`` with no ``__file__``, ``sys.argv[0] == "-c"`` and
the session's ``sys.path`` as it is, dedented first as Python 3.14's ``-c`` does;
everything else is as for a file. Every module a run imports from your own code
(plain ``.py`` outside an install) is executed again from the file's current
bytes, so an edit is picked up with no reload, and the session's own copies are
never touched. Shared with the session, never
re-executed: the standard library, an install (site-packages, the interpreter,
Maya, Program Files), a package with a compiled member, ``rig`` and the packages
rig itself uses (cgmath), and the top-level names given in ``shared=``. A plain
package that code of a ``shared=`` package imports for the first time during a
run is the session's too, so the session's copy never holds a module the run
takes away.

A run hides a session package only once the file's own code imports it ("lazy
claim"): code that is not the file's (a session callback, a thread) sees what
``sys.modules`` shows. Code a run made keeps importing (and ``importlib.reload``
-ing) that run's modules when it runs later (a callback). Only Maya's thread (or
the thread inside ``rig.run``) swaps modules in and out of ``sys.modules``; a
``rig.run`` from another thread while a run goes on raises RuntimeError.

One run is one undo step, ``rig.run a.py`` (``rig.run a.py, b.py`` for several
files, ``rig.exec`` and the code's first line for a string); a run inside a run
joins the outer step. Afterwards the session gets back
``sys.argv``, ``sys.path``, the std streams, ``sys.excepthook`` / ``displayhook``
and rig's options.
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
import io
import linecache
import os
import pkgutil
import re
import sys
import sysconfig
import textwrap
import threading
import types
import weakref

from rig._internal.container import get_options, set_options
from rig._internal.undo import undo_chunk

_MARK      = "__rig_run__"          # globals key: the Run whose code this is
_PYTHON    = {"__main__", "__mp_main__", "builtins"}
_STDLIB    = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(sys.builtin_module_names)
_SITE_DIRS = {"site-packages", "dist-packages"}
_SOURCE    = importlib.machinery.SourceFileLoader
_LOCK      = threading.RLock()
_ACTIVE    = []                     # (run, hidden modules, saved sys.path, saved __main__) per view shown, innermost last
_OWNER     = [None]                 # the thread showing them
_MAIN      = threading.main_thread().ident
_RIG_TOPS  = set()                  # rig and the dev packages its modules use: always shared
_ROOTS     = ()                     # install folders (see _install_roots)
_CODE      = {}                     # path -> (digest, code object)
_STAMPS    = {}                     # path -> (mtime_ns, size) when first seen
_SEEN      = set()                  # session module names already stamped
_IMPORTER  = threading.local()      # .shared: code of a shared= package is importing, on this thread
_REACHED   = [False]                # an import went through rig's hook (the trial of _wrapped_around_ours)
_SOURCES   = {}                     # <rig.exec digest> -> live Runs of that code (linecache keeps its lines)
_PASS, _SWAP = object(), object()
_original_import        = builtins.__import__
_original_import_module = importlib.import_module
_original_reload        = importlib.reload


class RunExitError(RuntimeError):
    """The file (or the code of a ``rig.exec``) called ``sys.exit()`` with a failure code (``.code``); the
    host keeps running."""


# ---------------------------------------------------------------------------
# Public API

def run(path, inputs=None, *, argv=(), shared=()):
    """Run the Python file at ``path`` as ``__main__``, as if in a fresh Python, and return its :class:`Run`.

    ``path`` (``str | os.PathLike | list``) is one path (a str, a :class:`pathlib.Path`), or a
    list of paths that run one after the other in ONE namespace, as one long file would: a later
    file uses what an earlier one defined (the same name: the later one wins), and
    :meth:`Run.call` reaches every function they define. Each file sees its own ``__file__`` and
    ``sys.argv[0]`` while it runs; every file's folder is first on ``sys.path`` (once each, in
    order) for the whole run and for :meth:`Run.call`. ``rig.run("a.py")`` is
    ``rig.run(["a.py"])``. A string of code runs the same way with :func:`exec`.

    ``inputs`` (a mapping, or the :class:`Run` of an earlier round) is copied into the starting
    globals, once: module objects are skipped, and a dunder name raises ValueError. ``argv``
    becomes every file's ``sys.argv[1:]``; ``shared`` names top-level packages to share with the
    session instead of re-executing them, with every plain package their code imports for the
    first time during the run. The run is one undo step, ``rig.run a.py``
    (``rig.run a.py, b.py`` for several).

    A path may use ``~`` and environment variables (``$VAR``, ``${VAR}``, ``%VAR%`` on Windows),
    expanded until nothing changes, and ``..``; it comes out with forward slashes. Every path is
    checked before any file runs: one that is no file raises FileNotFoundError, naming any
    variable in it that isn't set; an empty list raises ValueError; an item that is no path
    raises TypeError. Then the files run in order. An error in one (a SyntaxError
    too) propagates with ``err.rig_run``, the run so far, and the files after it never run;
    ``sys.exit()`` ends the run in the file that calls it: normally with 0 or None, else as
    :class:`RunExitError` carrying ``.code`` and naming that file.
    """
    paths = _paths(path)
    names = _inputs(inputs)
    argv, shared = _strings("argv", argv), set(_strings("shared", shared))
    folders = list(dict.fromkeys(os.path.normpath(os.path.dirname(p)) for p in paths))
    return _start(Run(paths, shared), names, argv, folders, f"rig.run {', '.join(map(os.path.basename, paths))}")


def exec(source, inputs=None, *, argv=(), shared=()):
    """Run the code in ``source`` as ``python -c`` runs it, as if in a fresh Python, and return its :class:`Run`.

    ``rig.exec("import arm; arm.build()")`` is :func:`run` for a string: the code runs as
    ``__main__`` with no ``__file__``, ``sys.argv[0]`` is ``"-c"``, and every module it imports from
    your own code is executed again from the file's current bytes, on every call. The session's
    ``sys.path`` stays as it is: nothing goes first (``python -c`` puts the working folder there,
    ``python -P`` doesn't). The code is dedented first (:func:`textwrap.dedent`), as Python 3.14's
    ``-c`` dedents it, so a triple-quoted block indented inside a function runs as written.

    ``inputs``, ``argv`` and ``shared`` are :func:`run`'s, and so is what comes back: a :class:`Run`
    (``result["name"]``, ``result.call``), with ``result.source`` the code that ran (dedented),
    ``result.path`` None and ``result.paths`` empty. A traceback names the code
    ``<rig.exec 1a2b3c4d>`` (its digest) and shows its lines, as long as a Run of that code lives
    (the error's ``rig_run``, a callback the code made). The run is one undo step, named
    ``rig.exec`` and the code's first line (40 characters at most). An error (a SyntaxError too)
    propagates with ``err.rig_run``, the run so far; ``sys.exit()`` ends the code: normally with 0
    or None, else as :class:`RunExitError` carrying ``.code``.

    ``source`` is a str, else TypeError. ``rig.exec`` is not in ``rig.__all__``: ``from rig import *``
    leaves Python's built-in ``exec`` alone.
    """
    if not isinstance(source, str):
        raise TypeError(f"rig.exec: the code is a str, not {type(source).__name__}")
    names = _inputs(inputs, "rig.exec", "code")
    argv, shared = _strings("argv", argv, "rig.exec"), set(_strings("shared", shared, "rig.exec"))
    code = textwrap.dedent(source)
    return _start(Run((), shared, code), names, argv, [], f"rig.exec {_first_line(code)}".rstrip(), "rig.exec")


def _start(result, names, argv, folders, label, who="rig.run"):
    """The run of :func:`run` and :func:`exec`: ``folders`` go first on ``sys.path``, ``label`` names
    the undo step; afterwards the session gets back what the module docstring lists."""
    with _LOCK:                                              # never held while the run's code runs
        _check_thread(who)
        _install()
        _stamp()
    result.sys_path = folders + sys.path
    options         = get_options()
    stdin           = _stdin_state()
    saved           = (sys.argv, sys.stdin, sys.stdout, sys.stderr, sys.displayhook, sys.excepthook)
    entered         = False
    importlib.invalidate_caches()
    try:
        with undo_chunk(label):
            with _LOCK:
                _check_thread(who)                           # re-checked with the view entered atomically
                _enter(result)
                entered = True
            try:
                _run_files(result, names, argv)
            finally:
                with _LOCK:
                    _leave()
    except BaseException as err:
        if entered:
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
    """One run of a file, of several in one namespace, or of the code of a :func:`exec`: its globals
    (``run["name"]``, read-only; ``dict(run)`` for a copy), the very dict the code ran in, so what its
    functions set later shows; the modules it executed (``run.modules``); and what ran: its files,
    ``run.path`` (the first) and ``run.paths`` (every one given, in order: one for a single path), or
    ``run.source``, the code of a ``rig.exec`` (dedented), whose ``path`` is None and ``paths``
    empty (``source`` is None for files)."""

    _rig_run_result = True           # a Run, whichever copy of this module made it (a reload, a purge)

    def __init__(self, paths, shared, source=None):
        self.path     = paths[0] if paths else None
        self.paths    = paths
        self.source   = source
        self.owned    = set()                         # top-level packages this run executes itself
        self.shared   = {}                            # top-level name -> why the session's copy is used
        self.modules  = {}                            # this run's modules, while no view shows them
        self.sys_path = None
        self._extra   = shared
        self._main    = types.ModuleType("__main__")  # the files' module: sys.modules["__main__"] while shown
        self._globals = self._main.__dict__           # the files' namespace (with _MARK, hidden here)

    def __getitem__(self, key):
        if key == _MARK:
            raise KeyError(key)
        return self._globals[key]

    def __iter__(self):
        return (key for key in list(self._globals) if key != _MARK)

    def __len__(self):
        return len(self._globals) - (_MARK in self._globals)

    def __repr__(self):
        if self.source is not None:
            return f"<rig.exec {_first_line(self.source)!r}: {len(self.modules)} modules of its own>"
        more = len(self.paths) - 1
        also = f" and {more} more file{'s' if more > 1 else ''}" if more else ""
        return f"<rig.run {self.path!r}{also}: {len(self.modules)} modules of its own>"

    @contextlib.contextmanager
    def active(self):
        """Show this run's modules and its ``__main__`` in ``sys.modules`` (and its ``sys.path``) for the duration."""
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
        """Call ``func`` (a name in the files' globals, or any callable) with this run's modules and its
        ``__main__`` shown: ``pickle`` and ``typing.get_type_hints`` find a class the files defined."""
        if isinstance(func, str):
            func = self[func]
        with self.active():
            return func(*args, **kwargs)


def _check_thread(who="rig.run"):
    if _ACTIVE and _OWNER[0] != threading.get_ident():
        raise RuntimeError(f"{who}: a run is going on in another thread; start runs from the thread that runs "
                           "them (Maya's), one after the other")


def _first_line(code, width=40):
    """The first line of ``code`` that holds anything, at most ``width`` characters: what names a rig.exec."""
    line = next((line.strip() for line in code.splitlines() if line.strip()), "")
    return line if len(line) <= width else line[:width - 3] + "..."


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


def _inputs(inputs, who="rig.run", what="file"):
    if inputs is None:
        return {}
    if not isinstance(inputs, collections.abc.Mapping):
        raise TypeError(f"{who}: inputs is a mapping of names (a dict, a Run), not {type(inputs).__name__}")
    names = {}
    is_run = getattr(type(inputs), "_rig_run_result", False) is True
    for key, value in inputs.items():
        if not isinstance(key, str):
            raise TypeError(f"{who}: an input name is a str, not {key!r}")
        if key.startswith("__") and key.endswith("__"):
            if is_run:
                continue                                     # the earlier round's own __file__ ...
            raise ValueError(f"{who}: the input {key!r} is a name Python sets for the {what}; use another name")
        if not isinstance(value, types.ModuleType):          # a module rides along stale: the file imports it
            names[key] = value
    return names


def _strings(what, values, who="rig.run"):
    values = tuple(values) if not isinstance(values, (str, bytes)) else None
    if values is None or not all(isinstance(value, str) for value in values):
        raise TypeError(f"{who}: {what} is a list of str, as {what}=[\"name\"]")
    return values


def _run_files(result, names, argv):
    """What :func:`runpy.run_path` does for a file, for each file in turn, in ONE namespace, or what
    ``python -c`` does for the code of a :func:`exec`: the ``__dict__`` of the run's ``__main__`` module,
    which :func:`_enter` shows as ``sys.modules["__main__"]`` (for the run, and again in :meth:`Run.call`)
    and :func:`_leave` puts back. ``sys.exit()`` ends the run in the file that calls it: normally with 0
    or None, else as :class:`RunExitError` naming that file (or the code)."""
    scope = result._globals
    scope.update(names)
    scope[_MARK] = result
    for path in result.paths or (None,):                     # no path: the code of a rig.exec
        if path is None:
            sys.argv, who = ["-c", *argv], "rig.exec: the code"
            scope.update(__name__="__main__", __doc__=None, __loader__=None, __package__="", __spec__=None)
            code = _code_of_source(result.source, result)
        else:
            sys.argv, who = [path, *argv], f"rig.run: {os.path.basename(path)}"
            scope.update(__name__="__main__", __file__=path, __cached__=None, __doc__=None, __loader__=None,
                         __package__="", __spec__=None)
            code = _code_of(path)
        try:
            builtins.exec(code, scope)                       # Python's: this module's exec is rig.exec
        except SystemExit as err:
            if err.code in (None, 0):
                return
            exit_error      = RunExitError(f"{who} called sys.exit({err.code!r})")
            exit_error.code = err.code
            raise exit_error from err


def _code_of(path):
    """The file's code, read as runpy reads it: a compiled file as it is, else its source, compiled
    without this module's ``__future__`` flags (``annotations`` would make the file's annotations
    strings)."""
    with io.open_code(path) as fh:
        code = pkgutil.read_code(fh)
        if code is None:
            fh.seek(0)
            code = compile(fh.read(), path, "exec", dont_inherit=True)
    return code


def _source_name(source):
    """The name a rig.exec's code is compiled under, ``<rig.exec 1a2b3c4d>``: the code's digest."""
    return f"<rig.exec {hashlib.blake2b(source.encode('utf-8', 'surrogatepass'), digest_size=4).hexdigest()}>"


def _code_of_source(source, run):
    """The code of a rig.exec, compiled as ``python -c`` compiles it, without this module's
    ``__future__`` flags, under a name its tracebacks show the lines of (:func:`_source_name`).
    linecache keeps the lines (an entry with no mtime, which ``checkcache`` leaves as it is) as long
    as a Run of that code lives: ``run`` is counted in here, and out when it goes."""
    name = _source_name(source)

    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    _SOURCES[name]        = _SOURCES.get(name, 0) + 1
    finalizer             = weakref.finalize(run, _drop_source, name)
    finalizer.atexit      = False                            # nothing to do at interpreter exit
    return compile(source, name, "exec", dont_inherit=True)


def _drop_source(name):
    """One Run of the code ``name`` is gone: its lines go once no Run of that code is left."""
    left = _SOURCES.get(name, 0) - 1
    if left > 0:
        _SOURCES[name] = left
        return
    _SOURCES.pop(name, None)
    linecache.cache.pop(name, None)


# ---------------------------------------------------------------------------
# The paths: one or several, each with ~ and variables expanded until nothing changes, .. walked,
# forward slashes

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


def _paths(path):
    """The files to run, resolved, in order, each checked to be a file before any runs: one path (a
    str, bytes or os.PathLike), or a list of them."""
    if isinstance(path, (str, bytes)) or not isinstance(path, collections.abc.Iterable):
        path = [path]  # one path; a pathlib.Path is no iterable
    items = tuple(path)
    if not items:
        raise ValueError("rig.run: no file to run: the list of paths is empty")
    paths = []
    for item in items:
        if not isinstance(item, (str, bytes, os.PathLike)):
            raise TypeError(f"rig.run: a path is a str, bytes or os.PathLike, not {item!r}")
        path = _resolve(item)
        if not os.path.isfile(path):
            why = "Not a file" if os.path.exists(path) else _missing(path)
            raise FileNotFoundError(errno.ENOENT, why, path)
        paths.append(path)
    return tuple(paths)


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
    _report_stale(run, top, why)


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


def _report_stale(run, top, why):
    if why == "stdlib":
        return
    stale = []
    for name, module in _members(top):
        found = _source_of(module)
        if found and _STAMPS.get(found[0], found[1]) != found[1]:
            stale.append(f"{name} ({why})")
    if stale:
        more = f" and {len(stale) - 8} more" if len(stale) > 8 else ""
        who  = "rig.run" if getattr(run, "source", None) is None else "rig.exec"
        sys.stderr.write(f"{who}: changed on disk, but shared with this session, so this run uses the old code"
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
            if getattr(_IMPORTER, "shared", False) or _shared_importer():
                return None                                  # a shared= package's code imports it: the session's
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
    main = sys.modules.get("__main__")                       # the session's (or the outer run's): back at _leave
    if getattr(run, "_main", None) is not None:              # a Run of another copy of this module may have none
        sys.modules["__main__"] = run._main
    _ACTIVE.append((run, hidden, sys.path[:], main))
    if run.sys_path is not None:
        sys.path[:] = run.sys_path
    if sys.meta_path[:1] != [_Finder]:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_Finder)
        sys.meta_path.insert(0, _Finder)


def _leave():
    run, hidden, path, main = _ACTIVE.pop()
    run.modules = _harvest(run)
    for name in run.modules:
        sys.modules.pop(name, None)
    sys.modules.update(hidden)
    if main is None:
        sys.modules.pop("__main__", None)
    else:
        sys.modules["__main__"] = main
    run.sys_path = sys.path[:]
    sys.path[:]  = path
    if not _ACTIVE:
        _OWNER[0] = None


# ---------------------------------------------------------------------------
# Import hooks: code a run made imports that run's modules

def _wrapped_around_ours(previous):
    """Whether ``previous``, ``builtins.__import__`` as found, is a foreign wrapper around rig's hook (a
    debugger's, a reloader's): rig's hook was installed before, ``previous`` is no copy of it, and a
    trial import reaches it. Installing on top of such a wrapper would make every import recurse."""
    if not hasattr(_import, "_rig_run_original") or hasattr(previous, "_rig_run_original"):
        return False
    _REACHED[0] = False
    try:
        previous("sys")
    except Exception:
        return False
    return _REACHED[0]


def _install():
    global _original_import, _original_import_module, _original_reload, _ROOTS
    previous = builtins.__import__
    if previous is _import or _wrapped_around_ours(previous):
        return                                               # ours, or a foreign wrapper around ours: left as is
    previous_module         = importlib.import_module         # a reloaded copy's hook, or a replacement
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


def _from_shared(scope):
    """Whether ``scope`` is the globals of a module of a package the shown run names in ``shared=``."""
    run  = _shown()
    name = scope.get("__name__") if run is not None and isinstance(scope, dict) else None
    return isinstance(name, str) and name.partition(".")[0] in run._extra


def _shared_importer():
    """Whether the import `_Finder` finds comes from code of a ``shared=`` package, read off the
    stack: the innermost frame outside the import machinery (importlib, this module). Maya 2027's
    ``builtins.__import__`` swaps itself for a ``__lazy_import__`` while it imports, so an import
    made by a module being imported skips rig's hook (`_import`), and ``_IMPORTER.shared`` still
    says what the import around it was."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__")
        if name != __name__ and not (isinstance(name, str) and (
                name == "importlib" or name.startswith(("importlib.", "_frozen_importlib")))):
            return _from_shared(frame.f_globals)
        frame = frame.f_back
    return False


def _imported(shared, do, *args):
    """``do(*args)``, an import, with this thread marked as importing for a ``shared=`` package's code, or not."""
    before = getattr(_IMPORTER, "shared", False)
    _IMPORTER.shared = shared
    try:
        return do(*args)
    finally:
        _IMPORTER.shared = before


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    _REACHED[0] = True
    scope       = globals or sys._getframe(1).f_globals
    run         = scope.get(_MARK) if isinstance(scope, dict) else None
    if run is None:
        if _ACTIVE and _from_shared(scope):
            return _imported(True, _original_import, name, globals, locals, fromlist, level)
        return _original_import(name, globals, locals, fromlist, level)
    try:
        absolute = importlib._bootstrap._resolve_name(name, importlib._bootstrap._calc___package__(scope), level) \
            if level else name
    except Exception:                                        # Python raises its own error below
        return _original_import(name, globals, locals, fromlist, level)
    route = _route(run, absolute, fromlist or (), False, fast=bool(fromlist) or not level)
    if route is _PASS:
        return _imported(False, _original_import, name, globals, locals, fromlist, level)
    if route is not _SWAP:
        return route
    with run.active():
        return _imported(False, _original_import, name, globals, locals, fromlist, level)


def _import_module(name, package=None):
    _REACHED[0] = True
    scope       = sys._getframe(1).f_globals
    run         = scope.get(_MARK)
    if run is None:
        if _ACTIVE and _from_shared(scope):
            return _imported(True, _original_import_module, name, package)
        return _original_import_module(name, package)
    try:
        absolute = importlib.util.resolve_name(name, package)
    except (ImportError, ValueError, AttributeError, TypeError):
        return _original_import_module(name, package)
    route = _route(run, absolute, (), True)
    if route is _PASS:
        return _imported(False, _original_import_module, name, package)
    if route is not _SWAP:
        return route
    with run.active():
        return _imported(False, _original_import_module, name, package)


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
