"""Tests for ``_defaults.toml`` settings file loader.

Verifies the TOML loader in :func:`_internal.container._load_defaults_from_file`:
  * missing file -> in-code defaults remain
  * valid file -> values override the matching ContainerOptions attr
  * malformed file -> in-code defaults remain (fail open)
  * unknown key -> silently ignored
  * unknown section -> silently ignored
"""

import contextlib
import os
import tempfile
import textwrap
import unittest
from unittest import mock

from rig._internal import container as container_mod
from rig._internal.container import _load_defaults_from_file, ContainerOptions


def _toml_parser_available():
    """Mirror the loader's own import ladder.

    ``_load_defaults_from_file`` returns immediately when neither ``tomllib``
    (3.11+) nor ``tomli`` can be imported. Maya 2022 is Python 3.7 and ships
    no ``tomli``, so ``_defaults.toml`` is inert there and none of these
    behaviours can be exercised.
    """
    for module in ("tomllib", "tomli"):
        try:
            __import__(module)
            return True
        except ImportError:
            continue
    return False


@unittest.skipUnless(
    _toml_parser_available(),
    "no tomllib/tomli available -- _load_defaults_from_file() is a no-op",
)
class TestDefaultsLoader(unittest.TestCase):
    """Exercise ``_load_defaults_from_file`` against a fake config path."""

    def setUp(self):
        # Snapshot in-code defaults so we can restore after each test.
        self._snapshot = {
            "create_containers":  ContainerOptions.create_containers,
            "publish_attributes": ContainerOptions.publish_attributes,
            "flatten_containers": ContainerOptions.flatten_containers,
            "use_shorthand":      ContainerOptions.use_shorthand,
            "skip_selection":     ContainerOptions.skip_selection,
            "cleanup_on_exit":    ContainerOptions.cleanup_on_exit,
            "maya_version":       ContainerOptions.maya_version,
        }

    def tearDown(self):
        # Restore to pre-test values.
        for k, v in self._snapshot.items():
            setattr(ContainerOptions, k, v)

    def _run_loader_with_temp(self, contents):
        """Write ``contents`` to a temp file and trick the loader into
        reading it instead of the real ``_defaults.toml``."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(contents)
            temp_path = f.name
        try:
            # ExitStack rather than a parenthesized multi-manager ``with``:
            # mayapy 2022 is Python 3.7, which parses ``with (a, b):`` as a
            # tuple and fails with ``AttributeError: __enter__``.
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch("os.path.join", side_effect=lambda *_: temp_path)
                )
                stack.enter_context(mock.patch("os.path.isfile", return_value=True))
                _load_defaults_from_file()
        finally:
            os.unlink(temp_path)

    # ------------------------------------------------------------------- #

    def test_missing_file_keeps_in_code_defaults(self):
        # When the file doesn't exist, ContainerOptions stays untouched.
        with mock.patch("os.path.isfile", return_value=False):
            _load_defaults_from_file()
        for k, v in self._snapshot.items():
            self.assertEqual(getattr(ContainerOptions, k), v)

    def test_valid_file_overrides_options(self):
        contents = textwrap.dedent(
            """
            [options]
            flatten_containers = false
            publish_attributes = false
            """
        )
        self._run_loader_with_temp(contents)
        self.assertFalse(ContainerOptions.flatten_containers)
        self.assertFalse(ContainerOptions.publish_attributes)
        # Untouched options keep their pre-load values.
        self.assertEqual(
            ContainerOptions.create_containers,
            self._snapshot["create_containers"],
        )

    def test_malformed_file_keeps_in_code_defaults(self):
        # Garbage TOML -> loader logs warning + falls back silently.
        self._run_loader_with_temp("this is not = valid [toml")
        for k, v in self._snapshot.items():
            self.assertEqual(getattr(ContainerOptions, k), v)

    def test_unknown_key_silently_ignored(self):
        contents = textwrap.dedent(
            """
            [options]
            this_key_does_not_exist = true
            """
        )
        self._run_loader_with_temp(contents)
        # No error, no setattr -- defaults unchanged.
        for k, v in self._snapshot.items():
            self.assertEqual(getattr(ContainerOptions, k), v)
        # Sanity: the unknown key does NOT get added.
        self.assertFalse(hasattr(ContainerOptions, "this_key_does_not_exist"))

    def test_unknown_section_silently_ignored(self):
        contents = textwrap.dedent(
            """
            [other_section]
            x = 1

            [options]
            flatten_containers = false
            """
        )
        self._run_loader_with_temp(contents)
        # The [options] block still applies; [other_section] ignored.
        self.assertFalse(ContainerOptions.flatten_containers)


class TestShippedDefaultsFile(unittest.TestCase):
    """Validate the REAL shipped ``_defaults.toml`` (not a temp fixture).

    The loader silently ignores any ``[options]`` key that is not a
    ContainerOptions attribute, so a typo'd key is a no-op that no other
    test would catch. These tests parse the actual file and assert its
    active keys are valid AND that the options deemed unsafe to expose as a
    committed team-wide default stay commented out (see the rationale in
    ``_defaults.toml`` itself).
    """

    # Options intentionally NOT exposed as active file defaults. Each is a
    # debug toggle, an experimental capability, or an environment pin whose
    # silent committed non-default would be dangerous -- they must stay
    # commented in the shipped file.
    _UNSAFE_TO_EXPOSE = frozenset(
        {
            "use_shorthand",  # debug-only dispatch toggle
            "maya_version",  # environment pin -- must track live runtime
            "native_multi_publish",  # experimental -- degrades Node Editor
            "constant_folding",  # debug/demo aid -- use force_nodes()
        }
    )

    def _load_options_table(self):
        """Parse the real ``_defaults.toml`` and return its ``[options]`` dict.

        Locates the file exactly as ``_load_defaults_from_file`` does, and
        mirrors the loader's TOML-lib resolution so the test skips cleanly
        on a Python without ``tomllib`` / ``tomli`` (e.g. older Maya).
        """
        try:
            import tomllib  # Python 3.11+ (Maya 2026+)
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                self.skipTest("no TOML library available")
        rig_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(container_mod.__file__))
        )
        path = os.path.join(rig_dir, "_defaults.toml")
        self.assertTrue(os.path.isfile(path), f"missing shipped file: {path}")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("options", {})

    def test_file_parses_with_options_table(self):
        options = self._load_options_table()
        self.assertIsInstance(options, dict)
        self.assertTrue(options, "shipped [options] table is empty")

    def test_active_keys_are_valid_container_options(self):
        # Each active key must map to a real ContainerOptions attribute,
        # else the loader silently ignores it (a typo'd key = silent no-op).
        options = self._load_options_table()
        for key in options:
            self.assertTrue(
                hasattr(ContainerOptions, key),
                f"_defaults.toml [options] key {key!r} is not a "
                f"ContainerOptions attribute (typo? silently ignored).",
            )

    def test_unsafe_options_stay_commented(self):
        # The debug / experimental / pin options must NOT ship active.
        options = self._load_options_table()
        for key in self._UNSAFE_TO_EXPOSE:
            self.assertNotIn(
                key,
                options,
                f"{key!r} must stay commented out in _defaults.toml "
                f"(unsafe as a committed team-wide default).",
            )

    def test_absorb_unit_conversions_is_exposed(self):
        # The newly-exposed safe structural knob should be active.
        options = self._load_options_table()
        self.assertIn("absorb_unit_conversions", options)

    def test_loading_real_file_does_not_change_unsafe_options(self):
        # End-to-end: loading the real file must leave every unsafe option at
        # its in-code default (since they are commented out, not overridden).
        snapshot = {k: getattr(ContainerOptions, k) for k in self._UNSAFE_TO_EXPOSE}
        try:
            _load_defaults_from_file()
            for key, before in snapshot.items():
                self.assertEqual(
                    getattr(ContainerOptions, key),
                    before,
                    f"loading _defaults.toml changed {key!r} -- it must "
                    f"stay commented out so the in-code default wins.",
                )
        finally:
            for key, before in snapshot.items():
                setattr(ContainerOptions, key, before)


if __name__ == "__main__":
    unittest.main()