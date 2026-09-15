"""
``rig.maya`` -- the Maya object-model layer the ``rig`` DSL is built on.

The typed node wrappers live in :mod:`rig.maya.nodetypes`; the supporting
modules (:mod:`~rig.maya.attribute`, :mod:`~rig.maya.node_name`,
:mod:`~rig.maya.constants`, :mod:`~rig.maya.plugins`,
:mod:`~rig.maya.pycmds`) sit alongside it as peers.

Deliberately re-exports nothing -- submodules are always imported explicitly::

    from rig.maya import pycmds
    from rig.maya.attribute import Attribute
    from rig.maya.nodetypes import Transform

Keeping this module inert avoids import cycles: :mod:`rig.maya.nodetypes`
imports several of its peers at module scope.

Note that ``import maya.cmds`` inside this package still resolves to Autodesk's
top-level ``maya``, not to this one: Python 3 imports are absolute. That holds
only while the *parent* of ``rig/`` is on ``sys.path``. Putting ``rig/`` itself
on ``sys.path`` would shadow Autodesk's ``maya`` package with this one.
"""
