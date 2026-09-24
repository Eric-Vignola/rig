"""
``rig.maya`` -- the Maya object-model layer the ``rig`` DSL is built on.

The typed node wrappers and the :class:`~rig.maya.nodetypes.Attribute` plug
wrapper live in :mod:`rig.maya.nodetypes`; :mod:`rig.maya.plugins` sits
alongside it as a peer.

Deliberately re-exports nothing -- submodules are always imported explicitly::

    from rig.maya.nodetypes import Attribute, Transform
    from rig.maya.plugins import load_plugin

Keeping this module inert avoids import cycles: :mod:`rig.maya.nodetypes`
imports :mod:`rig.maya.plugins` at module scope.

Note that ``import maya.cmds`` inside this package still resolves to Autodesk's
top-level ``maya``, not to this one: Python 3 imports are absolute. That holds
only while the *parent* of ``rig/`` is on ``sys.path``. Putting ``rig/`` itself
on ``sys.path`` would shadow Autodesk's ``maya`` package with this one.
"""
