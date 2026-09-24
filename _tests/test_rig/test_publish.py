"""Tests for v4.G container-publishing API: ``container.publish_input``,
``container.publish_output``, the ``>>`` shortcut, and the
``flatten_containers`` global option."""

from maya import cmds
from rig import container, get_options, Node, PlugList, set_options
from rig._internal.maya_version import get_maya_version
from rig.spec import Float
from rig._tests._base import MayaTestCase


def _bound_plug(ctn, name):
    """Return the real inner/host plug bound to the published ``name`` on
    container ``ctn``.

    Native publishing makes ``ctn.<name>`` a ``publishName`` / ``bindAttr``
    ALIAS, so topology / value / range queries must target the real bound plug
    (an inner-node attr for member publishes, or ``<ctn>_host.<name>`` for
    created knobs / multis). Returns ``None`` if ``name`` is not published.
    """
    from rig._internal.container import _resolve_published

    resolved = _resolve_published(ctn, name)
    return str(resolved) if resolved is not None else None


def _bind_pairs(ctn):
    """Return ``{published_name: inner_plug}`` from the container's native
    ``bindAttr`` table (flat ``[innerPlug, pubName, ...]``)."""
    bind = cmds.container(ctn, query=True, bindAttr=True) or []
    return {bind[i + 1]: bind[i] for i in range(0, len(bind) - 1, 2)}


class TestPublishOption(MayaTestCase):
    """v4.G: ``flatten_containers`` global option."""

    TEST_START_NEW_SCENE = True

    def test_default_flatten_is_true(self):
        # New session -- flatten_containers defaults to True.
        self.assertTrue(get_options()["flatten_containers"])

    def test_set_options_toggles(self):
        try:
            set_options(flatten_containers=False)
            self.assertFalse(get_options()["flatten_containers"])
            set_options(flatten_containers=True)
            self.assertTrue(get_options()["flatten_containers"])
        finally:
            set_options(flatten_containers=True)


class TestPublishInput(MayaTestCase):
    """v4.G: ``container.publish_input`` -- expose a knob on the active container."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_publish_input_creates_attr_and_drives_source(self):
        with container("outer"):
            n = Node.create("transform", name="cube1")
            n << Float("weight", dv=0.5) << 0.7
            pub = container.publish_input(n.weight, "weight")

            self.assertIsNotNone(pub)
            self.assertTrue(cmds.attributeQuery("weight", node="outer", exists=True))
            # Native publish: ``outer.weight`` is a bindAttr ALIAS for the real
            # inner plug (no separate attr + connectAttr). The bind table maps
            # the published name to ``cube1.weight``.
            self.assertEqual(_bind_pairs("outer").get("weight"), "cube1.weight")

    def test_publish_input_copies_initial_value_by_default(self):
        with container("outer"):
            n = Node.create("transform", name="cube2")
            n << Float("blend", dv=0.0) << 0.42
            container.publish_input(n.blend, "blend")
            self.assertAlmostEqual(cmds.getAttr("outer.blend"), 0.42)

    def test_publish_input_value_false_skips_copy(self):
        with container("outer"):
            n = Node.create("transform", name="cube3")
            n << Float("blend", dv=0.0) << 0.99
            container.publish_input(n.blend, "blend", value=False)
            # Native publish of an INNER member plug binds it directly --
            # ``outer.blend`` IS ``cube3.blend`` (an alias), so it shares the
            # live value (0.99). There is no separate attr to seed, so the
            # ``value`` flag is a no-op here; it only gates source-wiring for
            # EXTERNAL sources (see test_value_false_skips_wiring_external_plug).
            self.assertAlmostEqual(cmds.getAttr("outer.blend"), 0.99)

    def test_publish_input_collision_raises(self):
        with container("outer"):
            n = Node.create("transform", name="cube4")
            n << Float("a")
            n << Float("b")
            container.publish_input(n.a, "weight")
            with self.assertRaises(AttributeError):
                container.publish_input(n.b, "weight")

    def test_publish_input_required_name_raises(self):
        with container("outer"):
            n = Node.create("transform", name="cube5")
            n << Float("weight")
            with self.assertRaises(ValueError):
                container.publish_input(n.weight, "")

    def test_publish_input_fires_at_top_level_under_flatten(self):
        # Under the leaf-frame-realness rule (v4.P): top-level container is
        # always real (flatten only flattens NESTED scopes), so publish fires
        # at top level even with flatten_containers=True. This is what makes
        # ``set_options(create_containers=True)`` alone give a published
        # interface -- no need to also set flatten=False.
        set_options(flatten_containers=True)
        with container("flat"):
            n = Node.create("transform", name="cube6")
            n << Float("weight")
            pub = container.publish_input(n.weight, "weight")
            # Publish FIRED (not passthrough): the name is on the container's
            # native publish table. (Native publish of a member returns the
            # real inner plug, so ``pub == n.weight`` -- that is expected now.)
            self.assertIn(
                "weight", cmds.container("flat", q=True, publishName=True) or []
            )
            self.assertTrue(cmds.attributeQuery("weight", node="flat", exists=True))

    def test_publish_input_passthrough_when_nested_under_flatten(self):
        # Nested under flatten=True: the inner ``with container():`` does NOT
        # create a real Maya sub-container (leaf frame's container_node is
        # None), so publish_input passes through. This preserves module-
        # factory isolation -- a factory called inside a user's outer container
        # doesn't pollute the outer with its internals.
        set_options(flatten_containers=True)
        with container("outer"):
            with container("inner"):
                n = Node.create("transform", name="cube6")
                n << Float("weight")
                pub = container.publish_input(n.weight, "weight")
                self.assertEqual(str(pub), str(n.weight))  # passthrough
                self.assertFalse(
                    cmds.attributeQuery("weight", node="outer", exists=True),
                    "nested factory must not pollute outer container",
                )

    def test_publish_input_passthrough_when_create_containers_false(self):
        set_options(create_containers=False)
        try:
            with container("xx"):
                n = Node.create("transform", name="cube7")
                n << Float("weight")
                pub = container.publish_input(n.weight, "weight")
                self.assertEqual(str(pub), str(n.weight))  # passthrough
        finally:
            set_options(create_containers=True)


class TestPublishOutput(MayaTestCase):
    """v4.G: ``container.publish_output`` -- expose a result on the active container."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_publish_output_creates_attr_and_source_drives_it(self):
        with container("outer"):
            m = Node.create("multiplyDivide", name="mul1")
            m.input1X << 5
            m.input2X << 2
            pub = container.publish_output(m.outputX, "result")

            self.assertIsNotNone(pub)
            self.assertTrue(cmds.attributeQuery("result", node="outer", exists=True))
            # Native publish: ``outer.result`` is a bindAttr ALIAS for the real
            # source plug ``mul1.outputX``.
            self.assertEqual(_bind_pairs("outer").get("result"), "mul1.outputX")
            self.assertAlmostEqual(cmds.getAttr("outer.result"), 10.0)

    def test_publish_same_plug_as_input_and_output_passthrough(self):
        """Identity / passthrough: the SAME plug published under two names
        must yield two DISTINCT bound published names.

        Maya's ``bindAttr`` binds a given plug only ONCE -- a naive second
        bind is silently skipped, leaving a dangling (unbound) published
        name. The publish path must instead route the second publish through
        a distinct host carrier attr. This mirrors the ``in_linear`` ease,
        whose output IS its input.
        """
        with container("ident") as c:
            n = Node.create("transform", name="cube1")
            n << Float("val", dv=0.0) << 0.25
            in_plug = container.publish_input(n.val, "input")
            container.publish_output(in_plug, "output")

        ctn = "ident"
        # Both published names exist on the container.
        self.assertTrue(cmds.attributeQuery("input", node=ctn, exists=True))
        self.assertTrue(cmds.attributeQuery("output", node=ctn, exists=True))
        # Both resolve to REAL bound plugs (not dangling) via __getattr__.
        self.assertIsNotNone(c.input)
        self.assertIsNotNone(c.output)
        # "output" must bind a DISTINCT carrier (a plug binds only once).
        bind  = cmds.container(ctn, query=True, bindAttr=True) or []
        pairs = {bind[i + 1]: bind[i] for i in range(0, len(bind) - 1, 2)}
        self.assertIn("input", pairs)
        self.assertIn("output", pairs)
        self.assertNotEqual(pairs["input"], pairs["output"])
        # Identity holds: container.output tracks container.input.
        cmds.setAttr(f"{ctn}.input", 0.8)
        self.assertAlmostEqual(cmds.getAttr(f"{ctn}.output"), 0.8)


class TestRshiftContainerShortcut(MayaTestCase):
    """v4.G: ``plug >> container`` shortcut auto-routes via attribute writability."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_writable_plug_publishes_as_input(self):
        with container("outer"):
            n = Node.create("transform", name="cube1")
            n << Float("blend", dv=0.0)
            pub = n.blend >> container

            self.assertIsNotNone(pub)
            # Native publish: ``outer.blend`` is a bindAttr ALIAS for the real
            # member plug ``cube1.blend`` (writable input bound directly).
            self.assertEqual(_bind_pairs("outer").get("blend"), "cube1.blend")

    def test_readonly_plug_publishes_as_output(self):
        with container("outer"):
            m = Node.create("multiplyDivide", name="mul1")
            m.input1X << 5
            m.input2X << 2
            pub = m.outputX >> container

            self.assertIsNotNone(pub)
            # Native publish: ``outer.outputX`` is a bindAttr ALIAS for the real
            # source plug ``mul1.outputX`` (read-only output bound directly).
            self.assertEqual(_bind_pairs("outer").get("outputX"), "mul1.outputX")

    def test_rshift_fires_at_top_level_under_flatten(self):
        # Mirror of test_publish_input_fires_at_top_level_under_flatten
        # but for the ``>>`` shortcut. Top-level under flatten=True still
        # has a real container, so publishing fires.
        set_options(flatten_containers=True)
        with container("flat"):
            n = Node.create("transform", name="cube1")
            n << Float("blend")
            pub = n.blend >> container
            # Publish FIRED (not passthrough): name is on the native publish
            # table. (>> of a member returns the real inner plug now.)
            self.assertIn(
                "blend", cmds.container("flat", q=True, publishName=True) or []
            )
            self.assertTrue(cmds.attributeQuery("blend", node="flat", exists=True))

    def test_rshift_passthrough_when_nested_under_flatten(self):
        # Nested under flatten=True: inner scope is flattened, so ``>>`` is
        # passthrough. Preserves module-factory isolation.
        set_options(flatten_containers=True)
        with container("outer"):
            with container("inner"):
                n = Node.create("transform", name="cube1")
                n << Float("blend")
                pub = n.blend >> container
                self.assertEqual(str(pub), str(n.blend))  # passthrough
                self.assertFalse(
                    cmds.attributeQuery("blend", node="outer", exists=True),
                    "nested factory must not pollute outer container",
                )

    def test_rshift_to_container_node_directly(self):
        # When passed an explicit container Node (not the singleton),
        # >> still works and routes via writability.
        with container("outer") as ctn:
            n = Node.create("transform", name="cube1")
            n << Float("blend", dv=0.5)
            pub = n.blend >> ctn

            self.assertIsNotNone(pub)
            # Native publish: ``outer.blend`` is a bindAttr ALIAS for the real
            # member plug ``cube1.blend`` (routed via writability to input).
            self.assertEqual(_bind_pairs("outer").get("blend"), "cube1.blend")


class TestPublishInputExternalSource(MayaTestCase):
    """v4.G+ extension: ``container.publish_input`` accepts external Plugs,
    scalars, sequences, and ``None`` as the source -- creates the container
    attr fresh (with optional addAttr kwargs) and wires the source IN.

    Enables the module-factory pattern:

        def lerp(a, b, w):
            with container("lerp"):
                a = container.publish_input(a, "input1")
                b = container.publish_input(b, "input2")
                w = container.publish_input(w, "weight", min=0, max=1, dv=0.5)
                return container.publish_output((b - a) * w + a, "output")
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_external_plug_source_wires_inward(self):
        # External Plug source -> container.<name> is created and the
        # external plug DRIVES it (opposite direction from internal-plug
        # form, which had the published attr drive an internal plug).
        external = Node.create("transform", name="src")
        external.tx << 7.5

        with container("outer"):
            pub = container.publish_input(external.tx, "input1")

            self.assertIsNotNone(pub)
            self.assertTrue(cmds.attributeQuery("input1", node="outer", exists=True))
            # outer.input1 is fed BY src.tx (external -> container).
            sources = (
                cmds.listConnections(
                    "outer.input1", source=True, destination=False, plugs=True
                )
                or []
            )
            self.assertIn("src.translateX", sources)

    def test_external_plug_clones_spec_when_no_kwargs(self):
        # Without addAttr kwargs and with an external Plug, the new attr
        # is created via _clone_attribute -- it gets a numeric type
        # matching the source. (Note: _clone_attribute may collapse
        # doubleLinear/doubleAngle to double; pin the broad assertion
        # only.)
        external = Node.create("transform", name="src")

        with container("outer"):
            container.publish_input(external.tx, "input1")
            attr_type = cmds.getAttr("outer.input1", type=True)
            self.assertIn(attr_type, ("double", "doubleLinear", "float"))

    def test_scalar_source_seeds_value(self):
        # Scalar source -> attr created (auto-typed double), value set.
        with container("outer"):
            pub = container.publish_input(0.42, "weight")

            self.assertIsNotNone(pub)
            self.assertTrue(cmds.attributeQuery("weight", node="outer", exists=True))
            self.assertAlmostEqual(cmds.getAttr("outer.weight"), 0.42)
            # No incoming connection -- it's just a SET.
            self.assertIsNone(
                cmds.listConnections("outer.weight", source=True, destination=False)
            )

    def test_addattr_kwargs_are_forwarded(self):
        # min / max / dv / keyable kwargs reach cmds.addAttr.
        with container("outer"):
            container.publish_input(0.5, "weight", min=0, max=1, dv=0.25)
            # A created knob (scalar source) lives on the host node; min/max are
            # NOT visible through the published alias, so query the real plug.
            host_node = _bound_plug("outer", "weight").split(".")[0]
            self.assertEqual(
                cmds.attributeQuery("weight", node=host_node, min=True), [0.0]
            )
            self.assertEqual(
                cmds.attributeQuery("weight", node=host_node, max=True), [1.0]
            )
            # Default for the new branch is keyable=True (channel-box visible).
            self.assertTrue(cmds.getAttr("outer.weight", keyable=True))

    def test_int_source_creates_long_attr(self):
        with container("outer"):
            container.publish_input(7, "count")
            self.assertEqual(cmds.getAttr("outer.count", type=True), "long")
            self.assertEqual(cmds.getAttr("outer.count"), 7)

    def test_bool_source_creates_bool_attr(self):
        with container("outer"):
            container.publish_input(True, "switch")
            self.assertEqual(cmds.getAttr("outer.switch", type=True), "bool")
            self.assertTrue(cmds.getAttr("outer.switch"))

    def test_vec3_list_source_creates_double3_attr(self):
        # A flat literal vec3 must publish as a double3 compound (not a
        # scalar). Regression: _infer_attr_type fell through to
        # {"at": "double"} for a flat [x, y, z], so publish_input crashed
        # injecting a 3-sequence into a scalar attr -- which broke
        # slerp([0,0,1],[1,0,0]) and lerp(...) on literal vectors.
        with container("outer"):
            container.publish_input([0, 0, 1], "vec")
        real = _bound_plug("outer", "vec")
        host = real.split(".")[0]
        self.assertEqual(
            cmds.attributeQuery("vec", node=host, attributeType=True), "double3"
        )
        self.assertFalse(cmds.attributeQuery("vec", node=host, multi=True))
        actual = cmds.getAttr(real)[0]
        for a, e in zip(actual, (0.0, 0.0, 1.0)):
            self.assertAlmostEqual(a, e)

    def test_vec4_list_source_creates_double4_attr(self):
        # A flat literal vec4 (e.g. a raw quaternion) must publish as a
        # double4 compound.
        with container("outer"):
            container.publish_input([0.0, 0.0, 1.0, 0.0], "quat")
        real = _bound_plug("outer", "quat")
        host = real.split(".")[0]
        self.assertEqual(
            cmds.attributeQuery("quat", node=host, attributeType=True), "double4"
        )
        self.assertFalse(cmds.attributeQuery("quat", node=host, multi=True))

    def test_none_source_with_explicit_attr_type(self):
        # None source -> attr created; nothing wired or set.
        with container("outer"):
            pub = container.publish_input(None, "blend", at="float", dv=0.3)
            self.assertIsNotNone(pub)
            self.assertEqual(cmds.getAttr("outer.blend", type=True), "float")
            self.assertAlmostEqual(cmds.getAttr("outer.blend"), 0.3)
            self.assertIsNone(
                cmds.listConnections("outer.blend", source=True, destination=False)
            )

    def test_value_false_skips_wiring_external_plug(self):
        external = Node.create("transform", name="src")
        external.tx << 9.0
        with container("outer"):
            container.publish_input(external.tx, "input1", value=False)
            # No incoming connection -- value=False suppressed the wire.
            self.assertIsNone(
                cmds.listConnections("outer.input1", source=True, destination=False)
            )

    def test_external_source_collision_raises(self):
        with container("outer"):
            container.publish_input(1.0, "weight")
            with self.assertRaises(AttributeError):
                container.publish_input(2.0, "weight")

    def test_external_source_fires_at_top_level_under_flatten(self):
        # Under the leaf-frame-realness rule (v4.P): top-level container is
        # real even with flatten=True, so external-source publish_input fires.
        set_options(flatten_containers=True)
        with container("flat"):
            external = Node.create("transform", name="src")
            pub      = container.publish_input(external.tx, "input1")
            # Publish FIRED (not passthrough): name is on the native publish
            # table. (``src`` is created inside the container, so it is a member
            # bound directly -- ``pub == external.tx`` is expected now.)
            self.assertIn(
                "input1", cmds.container("flat", q=True, publishName=True) or []
            )
            self.assertTrue(cmds.attributeQuery("input1", node="flat", exists=True))

    def test_external_source_passthrough_when_nested_under_flatten(self):
        # Nested under flatten=True: leaf is flattened (no real container)
        # so publish_input passes through. Module-factory isolation preserved.
        set_options(flatten_containers=True)
        with container("outer"):
            with container("inner"):
                external = Node.create("transform", name="src_nested")
                pub      = container.publish_input(external.tx, "input1")
                self.assertEqual(str(pub), str(external.tx))  # passthrough
                self.assertFalse(
                    cmds.attributeQuery("input1", node="outer", exists=True),
                    "nested factory must not pollute outer container",
                )

    def test_external_source_passthrough_when_create_containers_false(self):
        # v4.G+ (unified passthrough policy): scalar source passes
        # through unchanged when create_containers=False.
        set_options(create_containers=False)
        try:
            with container("xx"):
                pub = container.publish_input(1.0, "weight")
                self.assertEqual(pub, 1.0)  # passthrough
        finally:
            set_options(create_containers=True)

    def test_module_factory_pattern_end_to_end(self):
        # The motivating use case: a lerp() function builds a container
        # whose four published attrs are the entire public interface.
        # External callers don't touch internal nodes.
        a    = Node.create("transform", name="src_a")
        b    = Node.create("transform", name="src_b")
        ctrl = Node.create("transform", name="lerp_ctrl")
        a.tx << 10.0
        b.tx << 30.0
        ctrl << Float("blend", min=0, max=1, dv=0.25)

        def lerp(input1, input2, weight):
            with container("lerp"):
                input1 = container.publish_input(input1, "input1")
                input2 = container.publish_input(input2, "input2")
                weight = container.publish_input(weight, "weight", min=0, max=1, dv=0.5)
                result = (input2 - input1) * weight + input1
                return container.publish_output(result, "output")

        target = Node.create("transform", name="lerp_target")
        target.tx << lerp(a.tx, b.tx, ctrl.blend)

        # Find the actual container name (Maya may suffix it; in a fresh
        # scene the first one is just "lerp", but tests run sequentially
        # so be defensive).
        containers = [n for n in cmds.ls(type="container") if n.startswith("lerp")]
        self.assertTrue(containers, "no lerp container created")
        ctn_name = containers[0]

        # Verify the public interface exists.
        for attr in ("input1", "input2", "weight", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=ctn_name, exists=True),
                f"{ctn_name}.{attr} should have been published",
            )

        # Initial: (30 - 10) * 0.25 + 10 = 15.
        self.assertAlmostEqual(cmds.getAttr("lerp_target.tx"), 15.0)

        # Slide weight at the public interface; inner network re-evaluates.
        cmds.setAttr("lerp_ctrl.blend", 0.75)
        self.assertAlmostEqual(cmds.getAttr("lerp_target.tx"), 25.0)


class TestPublishAttributesOption(MayaTestCase):
    """v4.G+ extension: ``publish_attributes`` option toggles whether
    ``publish_input`` / ``publish_output`` create container attrs.

    When False, both methods passthrough -- return the source unchanged --
    so module-factory functions can be flipped between published and raw
    modes without code changes. Useful for debugging (peek at the raw
    network without the container interface boundary) or for testing
    the math in isolation.

    Gating priority (top wins):
      1. ``create_containers=False``  -> ``None``  (no Maya container at all)
      2. ``publish_attributes=False`` -> source    (passthrough)
      3. ``flatten_containers=True``  -> ``None``  (current flat default)
      4. else                          -> publish normally
    """

    TEST_START_NEW_SCENE = True

    def test_default_publish_attributes_is_true(self):
        # Defaults preserved -- existing behavior unchanged.
        self.assertTrue(get_options()["publish_attributes"])

    def test_set_options_toggles(self):
        try:
            set_options(publish_attributes=False)
            self.assertFalse(get_options()["publish_attributes"])
            set_options(publish_attributes=True)
            self.assertTrue(get_options()["publish_attributes"])
        finally:
            set_options(publish_attributes=True)

    def test_publish_input_passthrough_with_external_plug(self):
        try:
            set_options(flatten_containers=False, publish_attributes=False)
            external = Node.create("transform", name="src")
            with container("outer"):
                pub = container.publish_input(external.tx, "input1")
                # Passthrough: returns the external plug, no container attr.
                self.assertEqual(str(pub), str(external.tx))
                self.assertFalse(
                    cmds.attributeQuery("input1", node="outer", exists=True)
                )
        finally:
            set_options(flatten_containers=True, publish_attributes=True)

    def test_publish_input_passthrough_with_scalar(self):
        try:
            set_options(flatten_containers=False, publish_attributes=False)
            with container("outer"):
                pub = container.publish_input(0.42, "weight")
                self.assertEqual(pub, 0.42)
                self.assertFalse(
                    cmds.attributeQuery("weight", node="outer", exists=True)
                )
        finally:
            set_options(flatten_containers=True, publish_attributes=True)

    def test_publish_output_passthrough(self):
        try:
            set_options(flatten_containers=False, publish_attributes=False)
            with container("outer"):
                m = Node.create("multiplyDivide", name="mul1")
                m.input1X << 5
                m.input2X << 2
                pub = container.publish_output(m.outputX, "result")
                # Passthrough: returns the source plug, no container attr.
                self.assertEqual(str(pub), str(m.outputX))
                self.assertFalse(
                    cmds.attributeQuery("result", node="outer", exists=True)
                )
        finally:
            set_options(flatten_containers=True, publish_attributes=True)

    def test_passthrough_overrides_flatten(self):
        # publish_attributes=False is higher priority than
        # flatten_containers=True, so we get passthrough not None.
        try:
            set_options(flatten_containers=True, publish_attributes=False)
            external = Node.create("transform", name="src")
            with container("flat"):
                pub = container.publish_input(external.tx, "input1")
                self.assertEqual(str(pub), str(external.tx))  # passthrough
        finally:
            set_options(flatten_containers=True, publish_attributes=True)

    def test_create_containers_false_also_passthroughs(self):
        # v4.G+ (unified passthrough policy): all three "no-publish"
        # gates -- create_containers=False, publish_attributes=False,
        # flatten_containers=True -- now passthrough the source. So
        # combining create_containers=False with publish_attributes=False
        # produces the same result as either alone: source returned.
        try:
            set_options(create_containers=False, publish_attributes=False)
            external = Node.create("transform", name="src")
            with container("flat"):
                pub = container.publish_input(external.tx, "input1")
                self.assertEqual(str(pub), str(external.tx))  # passthrough
        finally:
            set_options(create_containers=True, publish_attributes=True)

    def test_rshift_shortcut_passthrough(self):
        # plug >> container also honors publish_attributes=False.
        try:
            set_options(flatten_containers=False, publish_attributes=False)
            external = Node.create("transform", name="src")
            with container("outer"):
                pub = external.tx >> container
                self.assertEqual(str(pub), str(external.tx))
                self.assertFalse(
                    cmds.attributeQuery("translateX", node="outer", exists=True)
                )
        finally:
            set_options(flatten_containers=True, publish_attributes=True)

    def test_lerp_module_works_in_passthrough_mode(self):
        # Same lerp() function -- in passthrough mode, no published attrs
        # are created, math nodes connect DIRECTLY to external sources.
        try:
            set_options(flatten_containers=False, publish_attributes=False)

            a    = Node.create("transform", name="src_a")
            b    = Node.create("transform", name="src_b")
            ctrl = Node.create("transform", name="lerp_ctrl")
            a.tx << 10.0
            b.tx << 30.0
            ctrl << Float("blend", min=0, max=1, dv=0.25)

            def lerp(input1, input2, weight):
                with container("lerp"):
                    input1 = container.publish_input(input1, "input1")
                    input2 = container.publish_input(input2, "input2")
                    weight = container.publish_input(
                        weight, "weight", min=0, max=1, dv=0.5
                    )
                    result = (input2 - input1) * weight + input1
                    return container.publish_output(result, "output")

            target = Node.create("transform", name="lerp_target")
            target.tx << lerp(a.tx, b.tx, ctrl.blend)

            # Container exists but has NO published attrs (passthrough
            # mode skipped them entirely).
            containers = [n for n in cmds.ls(type="container") if n.startswith("lerp")]
            self.assertTrue(containers)
            ctn_name = containers[0]
            for attr in ("input1", "input2", "weight", "output"):
                self.assertFalse(
                    cmds.attributeQuery(attr, node=ctn_name, exists=True),
                    f"{ctn_name}.{attr} should NOT exist in passthrough mode",
                )

            # But the math network is still wired: target.tx gets the
            # correct value, fed by sources directly through the network.
            self.assertAlmostEqual(cmds.getAttr("lerp_target.tx"), 15.0)
            cmds.setAttr("lerp_ctrl.blend", 0.75)
            self.assertAlmostEqual(cmds.getAttr("lerp_target.tx"), 25.0)
        finally:
            set_options(flatten_containers=True, publish_attributes=True)


class TestPublishMultiCompound(MayaTestCase):
    """v4.G+ extension: compound-multi (e.g. ``at='double3', multi=True``)
    publishing creates parent + X/Y/Z children correctly, accepts
    sequence-of-vec3 / sequence-of-Plug sources for both inputs and
    outputs, auto-infers shape from 2D arrays, and routes range kwargs
    (min/max/dv) onto the children."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_publish_input_compound_multi_explicit_kwargs(self):
        with container("outer"):
            pub = container.publish_input(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "vectors",
                at    = "double3",
                multi = True,
            )
            self.assertIsNotNone(pub)
            # Multi attrs are NOT published on the container -- they live on the
            # per-container host node and resolve through the registry. Resolve
            # the real plug and assert against the host.
            real = _bound_plug("outer", "vectors")
            host = real.split(".")[0]
            # Parent + X/Y/Z children created on the host.
            for child in ("vectors", "vectorsX", "vectorsY", "vectorsZ"):
                self.assertTrue(
                    cmds.attributeQuery(child, node=host, exists=True),
                    f"{host}.{child} should exist",
                )
            # 3 indices populated with values.
            for i, expected in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
                actual = cmds.getAttr(f"{real}[{i}]")[0]
                for a, e in zip(actual, expected):
                    self.assertAlmostEqual(a, e)

    def test_publish_input_compound_multi_auto_infer_from_2d_list(self):
        # No kwargs -- type inference detects 2D list-of-3 -> double3 multi.
        with container("outer"):
            container.publish_input([[1, 2, 3], [4, 5, 6]], "vectors")
            # Multi attrs live on the host node (registry-resolved), not the
            # container. Maya reports multi-of-compound parent's getAttr type as
            # "TdataCompound"; check via attributeQuery (which gives the
            # element type) and verify children exist on the host.
            real = _bound_plug("outer", "vectors")
            host = real.split(".")[0]
            self.assertEqual(
                cmds.attributeQuery("vectors", node=host, attributeType=True),
                "double3",
            )
            indices = cmds.getAttr(real, multiIndices=True) or []
            self.assertEqual(sorted(indices), [0, 1])
            for axis in "XYZ":
                self.assertTrue(
                    cmds.attributeQuery(f"vectors{axis}", node=host, exists=True),
                    f"{host}.vectors{axis} should exist",
                )

    def test_publish_input_compound_multi_pluglist_source(self):
        # PlugList-of-vec3-plugs source -> per-index live connection.
        sources = [Node.create("transform", name=f"src{i}") for i in range(3)]
        for i, s in enumerate(sources):
            s.t << [(i + 1), (i + 1) * 2, (i + 1) * 3]

        with container("outer"):
            container.publish_input(
                PlugList([s.t for s in sources]),
                "input_vectors",
                at    = "double3",
                multi = True,
            )
            # Multi attrs live on the host (registry-resolved). Verify per-
            # element connection there (Maya stores compound-to-compound
            # connections at the parent vector level).
            real = _bound_plug("outer", "input_vectors")
            for i in range(3):
                srcs = (
                    cmds.listConnections(
                        f"{real}[{i}]",
                        source      = True,
                        destination = False,
                        plugs       = True,
                    )
                    or []
                )
                # Maya may report the connection as the compound parent
                # or as one of the children -- accept either form.
                self.assertTrue(
                    any(s.startswith(f"src{i}.translate") for s in srcs),
                    f"input_vectors[{i}] should source from src{i}.translate, got {srcs}",
                )

    def test_publish_input_compound_multi_with_range_kwargs(self):
        # min / max / dv on a compound-multi -> applied to children.
        with container("outer"):
            container.publish_input(
                [[0.5, 0.5, 0.5]],
                "weights",
                at    = "double3",
                multi = True,
                min   = 0,
                max   = 1,
                dv    = 0.5,
            )
            # Children (on the host) should have the min/max applied.
            host = _bound_plug("outer", "weights").split(".")[0]
            for axis in "XYZ":
                self.assertEqual(
                    cmds.attributeQuery(f"weights{axis}", node=host, min=True),
                    [0.0],
                )
                self.assertEqual(
                    cmds.attributeQuery(f"weights{axis}", node=host, max=True),
                    [1.0],
                )

    def test_publish_output_multi_from_pluglist_scalar(self):
        # publish_output of a PlugList of single-Plug results -> scalar multi.
        with container("outer"):
            mults = [Node.create("multiplyDivide", name=f"mul{i}") for i in range(3)]
            for i, m in enumerate(mults):
                m.input1X << float(i + 1)
                m.input2X << 2.0  # so outputX = 2 * (i+1)
            container.publish_output(
                PlugList([m.outputX for m in mults]),
                "results",
                at    = "double",
                multi = True,
            )
            # Multi output lives on the host (registry-resolved).
            real = _bound_plug("outer", "results")
            for i, expected in enumerate([2.0, 4.0, 6.0]):
                self.assertAlmostEqual(cmds.getAttr(f"{real}[{i}]"), expected)

    def test_publish_output_compound_multi_from_pluglist(self):
        # publish_output of a PlugList of vec3 Plugs -> compound multi.
        with container("outer"):
            adds = [Node.create("plusMinusAverage", name=f"add{i}") for i in range(2)]
            for i, a in enumerate(adds):
                a.input3D[0] << [i + 1, i + 2, i + 3]
            container.publish_output(
                PlugList([a.output3D for a in adds]),
                "vec_out",
                at    = "double3",
                multi = True,
            )
            # Multi output lives on the host (registry-resolved).
            real = _bound_plug("outer", "vec_out")
            host = real.split(".")[0]
            # X/Y/Z children created on the host.
            for axis in "XYZ":
                self.assertTrue(
                    cmds.attributeQuery(f"vec_out{axis}", node=host, exists=True)
                )
            # 2 indices live-driven.
            for i in range(2):
                srcs = (
                    cmds.listConnections(
                        f"{real}[{i}]",
                        source      = True,
                        destination = False,
                        plugs       = True,
                    )
                    or []
                )
                self.assertTrue(
                    any(s.startswith(f"add{i}.output3D") for s in srcs),
                    f"vec_out[{i}] should source from add{i}.output3D, got {srcs}",
                )

    def test_vector_average_via_publish_end_to_end(self):
        # The motivating use case: a vector_average() factory that uses
        # publish_input / publish_output for compound-multi attrs --
        # NO manual ``ctn << Vector(...)`` workaround needed.
        from rig import functions as rf

        a = Node.create("transform", name="src_a")
        b = Node.create("transform", name="src_b")
        a.t << [1, 2, 3]
        b.t << [10, 20, 30]

        with container("vec_avg"):
            in1 = container.publish_input(
                PlugList([a.t]),
                "input1",
                at    = "double3",
                multi = True,
            )
            in2 = container.publish_input(
                PlugList([b.t]),
                "input2",
                at    = "double3",
                multi = True,
            )
            results = [rf.avg([in1[0], in2[0]])]
            container.publish_output(
                PlugList(results),
                "output",
                at    = "double3",
                multi = True,
            )

        # All three are multi attrs -- published via the registry on the host,
        # so resolve them through the container rather than ``attributeQuery``.
        for attr in ("input1", "input2", "output"):
            self.assertIsNotNone(
                _bound_plug("vec_avg", attr),
                f"vec_avg.{attr} should be published",
            )

        # output[0] = avg([1,2,3], [10,20,30]) = (5.5, 11, 16.5)
        actual = cmds.getAttr(f"{_bound_plug('vec_avg', 'output')}[0]")[0]
        for a_val, e_val in zip(actual, [5.5, 11.0, 16.5]):
            self.assertAlmostEqual(a_val, e_val)


class TestPublishMultiParentResolution(MayaTestCase):
    """Fix: publishing a multi-PARENT plug (e.g. ``worldMatrix``, which is a
    per-instance multi) as a SINGULAR input/output auto-resolves it to its
    ``[0]`` element.

    Without this, ``publish_input`` mirrors the multi shape (via
    ``_clone_attribute`` / ``_infer_attr_type``), producing a *multi*
    published attr. That attr then cannot connect onward to a single-value
    consumer -- Maya raises "Incompatible multi-attribute parent levels" --
    which the old swallow-on-failure ``<<`` hid, silently leaving the
    consumer unconnected (computing on an identity default).
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_worldmatrix_publishes_as_single_not_multi(self):
        a = Node.create("transform", name="a")
        with container("outer"):
            container.publish_input(a.wm, "matrix0")
        self.assertTrue(cmds.attributeQuery("matrix0", node="outer", exists=True))
        # A published alias ALWAYS reports multi=True (Maya quirk); the real
        # resolved matrix [0] lives on the host and is a single, not a multi.
        host = _bound_plug("outer", "matrix0").split(".")[0]
        self.assertFalse(
            cmds.attributeQuery("matrix0", node=host, multi=True),
            "multi-parent worldMatrix must resolve to [0]; published attr "
            "should be a single matrix, not a multi",
        )

    def test_worldmatrix_published_input_connects_onward(self):
        # The whole point: the published single matrix must be connectable
        # into a single-matrix consumer without raising.
        a = Node.create("transform", name="a")
        with container("outer"):
            pub = container.publish_input(a.wm, "matrix0")
            dec = Node.create("decomposeMatrix", name="dec")
            dec.inputMatrix << pub  # must NOT raise
        srcs = (
            cmds.listConnections(
                "dec.inputMatrix", source=True, destination=False, plugs=True
            )
            or []
        )
        # ``pub`` is the real host plug (the published name is an alias), so the
        # consumer's source is the host plug, not ``outer.matrix0``.
        self.assertIn(_bound_plug("outer", "matrix0"), srcs)

    def test_worldmatrix_input_is_driven_by_an_element(self):
        # Source side: outer.matrix0 is fed by the resolved worldMatrix
        # element (not the bare multi parent).
        a = Node.create("transform", name="a")
        with container("outer"):
            container.publish_input(a.wm, "matrix0")
        srcs = (
            cmds.listConnections(
                "outer.matrix0", source=True, destination=False, plugs=True
            )
            or []
        )
        self.assertTrue(
            srcs and any("worldMatrix" in s for s in srcs),
            f"outer.matrix0 should be driven by a worldMatrix element, got {srcs}",
        )

    def test_explicit_multi_kwarg_is_respected(self):
        # When the caller EXPLICITLY asks for a multi attr, do not resolve --
        # the sequence / compound-multi forms rely on this.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 2, 3]
        b.t << [4, 5, 6]
        with container("outer"):
            container.publish_input([a.t, b.t], "vectors", at="double3", multi=True)
        # multi attrs live on the host (registry-resolved); query it directly.
        host = _bound_plug("outer", "vectors").split(".")[0]
        self.assertTrue(
            cmds.attributeQuery("vectors", node=host, multi=True),
            "explicit multi=True must still create a multi attr",
        )

    def test_rshift_shortcut_resolves_multi_parent(self):
        # #5: the ``>>`` publish shortcut routes through ``_publish_to_container``
        # directly (NOT through publish_input/publish_output), so it must ALSO
        # resolve a bare multi-parent (worldMatrix) to its ``[0]`` element.
        # Before the fix, ``a.wm >> container`` cloned a *multi* and raised
        # "Incompatible multi-attribute parent levels" -- diverging from the
        # ``container.publish_input(a.wm, ...)`` spelling.
        a = Node.create("transform", name="a")
        with container("outer"):
            a.wm >> container  # must NOT raise
        self.assertTrue(
            cmds.attributeQuery("worldMatrix", node="outer", exists=True),
            "the >> shortcut should publish a 'worldMatrix' attr",
        )
        # Alias reports multi=True (quirk); the resolved [0] on the host is a
        # single matrix.
        host = _bound_plug("outer", "worldMatrix").split(".")[0]
        self.assertFalse(
            cmds.attributeQuery("worldMatrix", node=host, multi=True),
            "multi-parent published via >> must be a single matrix, not a multi",
        )

    def test_scalar_multi_source_is_not_truncated(self):
        # #6: ONLY matrix multi-parents (worldMatrix family, data_type
        # 'matrix') should resolve to [0]. A scalar/vector multi such as
        # plusMinusAverage.input1D (data_type 'compound') must be published
        # WHOLE -- not silently truncated to element [0].
        from rig._internal.container import _resolve_multi_parent_source

        pma          = Node.create("plusMinusAverage", name="pma")
        scalar_multi = pma.input1D
        self.assertTrue(scalar_multi.is_multi)
        result = _resolve_multi_parent_source(scalar_multi, {})
        # NB: compare via str -- Plug.__eq__ is overloaded to build a
        # comparison node, so it cannot be used as a boolean assertion.
        self.assertEqual(str(result), str(scalar_multi))  # NOT truncated to [0]
        self.assertNotIn("[0]", str(result))
        # A genuine matrix multi-parent must STILL resolve to its [0] element.
        t = Node.create("transform", name="t")
        self.assertEqual(str(_resolve_multi_parent_source(t.wm, {})), str(t.wm[0]))

    def test_resolution_swallows_introspection_error_and_returns_source(self):
        # Coverage: the defensive ``except (...): pass`` + fall-through
        # ``return source`` in ``_resolve_multi_parent_source``. A healthy
        # Attribute never raises on ``is_multi``; patch it to raise so the
        # swallow path runs and the original source is returned unchanged.
        from unittest import mock

        from rig.maya.nodetypes import Attribute
        from rig._internal.container import _resolve_multi_parent_source

        a = Node.create("transform", name="a")
        with mock.patch.object(
            type(a.wm),
            "is_multi",
            new_callable = mock.PropertyMock,
            side_effect  = RuntimeError("boom"),
        ):
            result = _resolve_multi_parent_source(a.wm, {})
        self.assertIsInstance(result, Attribute)
        self.assertEqual(str(result), str(a.wm))
        self.assertNotIn("[0]", str(result))

    def test_explicit_multi_kwarg_skips_resolution_for_attribute_source(self):
        # Coverage: the ``add_attr_kwargs.get("multi")`` early-return in
        # ``_resolve_multi_parent_source``. With an Attribute source AND an
        # explicit ``multi=True``, even a genuine matrix multi-parent
        # (``worldMatrix``) must be returned WHOLE -- the compound-multi /
        # sequence publishing forms depend on the unresolved multi shape.
        from rig._internal.container import _resolve_multi_parent_source

        a      = Node.create("transform", name="a")
        result = _resolve_multi_parent_source(a.wm, {"multi": True})
        self.assertEqual(str(result), str(a.wm))  # NOT resolved to [0]
        self.assertNotIn("[0]", str(result))


class TestNodeOpFanOutPublishing(MayaTestCase):
    """v4.P: NodeOp framework's SCOPE_SCALAR + compound fan-out wraps in a
    container AND publishes input1/input2/.../output on it, so external
    callers get a stable named interface -- not the raw _constant aggregator.

    Regression for: ``(node1.t % node2.t).input1`` raising AttributeError
    because the framework created the container but never published.

    These tests use the leaf-frame-realness rule's defaults: ``flatten=True``
    (the standard out-of-the-box config). Top-level NodeOp containers are
    real, so publishing fires automatically when ``create_containers=True``.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        # These tests force ``maya_version=2026``, which only downgrades
        # dispatch on a newer Maya -- it cannot add native node types an older
        # Maya does not ship.
        if get_maya_version() < 2024:
            self.skipTest("native 2024+ math node types require Maya 2024+")
        super().setUp()
        # Verify these tests run under defaults: flatten=True (the user's
        # exact scenario) -- leaf-frame-realness rule still publishes at
        # top level.
        set_options(flatten_containers=True)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True, maya_version=None)

    def test_compound_modulo_publishes_input1_input2_output(self):
        """``node1.t % node2.t`` (compound) -> outer ``modulo1`` container has
        published ``input1``, ``input2``, and ``output`` attrs."""
        set_options(maya_version=2026)
        node1  = Node.create("transform", name="src1_for_mod")
        node2  = Node.create("transform", name="src2_for_mod")
        result = node1.t % node2.t

        # Container exists.
        self.assertTrue(
            cmds.objExists("modulo1"),
            "modulo1 container should exist for compound fan-out",
        )

        # Published attrs exist on the container.
        for attr in ("input1", "input2", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node="modulo1", exists=True),
                f"modulo1.{attr} should be published",
            )

        # Result is the published output: the real plug bound to ``output``
        # (native publish returns the inner plug; ``modulo1.output`` aliases it).
        self.assertEqual(_bound_plug("modulo1", "output"), str(result))

    def test_compound_logical_and_publishes_interface(self):
        """``a.t & b.t`` (compound) -> outer ``logical_and1`` container has
        published ``input1``/``input2``/``output``."""
        set_options(maya_version=2026)
        a      = Node.create("transform", name="src1_for_and")
        b      = Node.create("transform", name="src2_for_and")
        result = a.t & b.t

        self.assertTrue(cmds.objExists("logical_and1"))
        for attr in ("input1", "input2", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node="logical_and1", exists=True),
                f"logical_and1.{attr} should be published",
            )
        self.assertEqual(_bound_plug("logical_and1", "output"), str(result))

    def test_compound_invert_publishes_single_input(self):
        """``~a.t`` (arity-1) -> outer container has published ``input``
        (singular, no number suffix) and ``output``."""
        set_options(maya_version=2026)
        a      = Node.create("transform", name="src_for_invert")
        result = ~a.t

        self.assertTrue(cmds.objExists("logical_not1"))
        for attr in ("input", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node="logical_not1", exists=True),
                f"logical_not1.{attr} should be published",
            )
        self.assertEqual(_bound_plug("logical_not1", "output"), str(result))

    def test_published_input_drives_internal_modulo_nodes(self):
        """The published ``input1``/``input2`` actually wire into the per-
        channel native nodes (not just static attrs)."""
        set_options(maya_version=2026)
        node1 = Node.create("transform", name="drive_src")
        node2 = Node.create("transform", name="drive_mod")
        _     = node1.t % node2.t

        # Each per-channel modulo node should have its `input` plug fed by
        # one channel of the published `modulo1.input1` attr.
        modulo_nodes = sorted(
            n
            for n in (cmds.container("modulo1", q=True, nodeList=True) or [])
            if cmds.nodeType(n) == "modulo"
        )
        self.assertEqual(
            len(modulo_nodes),
            3,
            f"expected 3 native modulo nodes (one per channel), got {modulo_nodes}",
        )
        # First channel (X) should be driven by modulo1.input1[0] or similar.
        src_for_x = (
            cmds.listConnections(
                f"{modulo_nodes[0]}.input",
                source      = True,
                destination = False,
                plugs       = True,
            )
            or []
        )
        # ``input1`` is a created knob on the host; its per-channel children
        # (``input1X`` ...) drive the internal modulo nodes.
        real_in1 = _bound_plug("modulo1", "input1")
        self.assertTrue(
            any(real_in1 in s for s in src_for_x),
            f"expected {real_in1} to drive {modulo_nodes[0]}.input; got {src_for_x}",
        )

    def test_user_repro_node_dot_input1_works(self):
        """Verbatim user repro: ``(node1.t % node2.t).input1`` should not
        raise AttributeError; should return the published Plug."""
        set_options(maya_version=2026)
        node1 = Node.create("transform", name="user_src1")
        node2 = Node.create("transform", name="user_src2")
        node  = node1.t % node2.t
        # Pre-fix: AttributeError("'output_plug1.value' has no child or
        # sibling attribute 'input1'.")
        input1_plug = node.input1
        self.assertIsNotNone(input1_plug)
        # ``node.input1`` resolves through the container to the real bound plug
        # (the created knob on the host); ``modulo1.input1`` is its alias.
        self.assertEqual(_bound_plug("modulo1", "input1"), str(input1_plug))


class TestNativePublishCapabilities(MayaTestCase):
    """New native-publish hybrid capabilities: removeContainer-safety,
    ``Container.__getattr__`` resolution, host-node lifecycle, the
    ``native_multi_publish`` flag, and assorted ``publish_output`` forms."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True, native_multi_publish=False)

    # -- twin requirement: removeContainer reveals internals without breaking -- #
    def test_remove_container_preserves_external_data_flow(self):
        consumer = Node.create("transform", name="consumer")  # external
        with container("box"):
            m = Node.create("multiplyDivide", name="inner_mul")
            m.input2X << 5.0
            knob = container.publish_input(0.0, "weight")  # created host knob
            m.input1X << knob  # knob drives internal math
            out = container.publish_output(m.outputX, "result")
        consumer.tx << out  # external consumer reads the published output
        host_weight = _bound_plug("box", "weight")  # box_host.weight
        cmds.setAttr("box.weight", 2.0)
        self.assertAlmostEqual(cmds.getAttr("consumer.tx"), 10.0)
        # Dissolve the container -- internals are REVEALED, not destroyed.
        cmds.container("box", edit=True, removeContainer=True)
        self.assertFalse(cmds.objExists("box"))
        self.assertTrue(cmds.objExists("inner_mul"))
        # Data still flows through the revealed network: weight 2 -> 3 lifts the
        # consumer 10 -> 15. The host survives because it is transitively
        # reachable to the external consumer.
        self.assertTrue(cmds.objExists(host_weight))
        cmds.setAttr(host_weight, 3.0)
        self.assertAlmostEqual(cmds.getAttr("consumer.tx"), 15.0)

    # -- Container.__getattr__ resolution -- #
    def test_container_getattr_resolves_published_genuine_and_raises(self):
        with container("gg") as c:
            n = Node.create("transform", name="gn")
            n << Float("blend", dv=0.0) << 0.4
            container.publish_input(n.blend, "blend")  # native single alias
            c << Float("knob", dv=0.7)  # genuine on-container attr
        # Published single resolves to the REAL inner plug (not the alias).
        self.assertEqual(str(c.blend), "gn.blend")
        # Genuine on-container attr resolves via the inherited Node.__getattr__.
        self.assertAlmostEqual(cmds.getAttr(str(c.knob)), 0.7)
        # Unknown name raises.
        with self.assertRaises(AttributeError):
            _ = c.does_not_exist
        # ``_``-prefixed name short-circuits to AttributeError (base guard).
        with self.assertRaises(AttributeError):
            _ = c._not_a_real_private

    # -- host-node lifecycle -- #
    def test_host_node_created_once_reused_and_tagged(self):
        from rig._internal import container as _C

        with container("hh"):
            container.publish_input(1.0, "knobA")  # creates the host
            # Force a cache miss so the second publish exercises the member-scan
            # fallback that re-discovers the existing host.
            _C._HOST_CACHE.clear()
            container.publish_input(2.0, "knobB")  # reuses host via scan
        members = cmds.container("hh", q=True, nodeList=True) or []
        hosts = [
            m
            for m in members
            if cmds.attributeQuery("__rl_host__", node=m, exists=True)
        ]
        self.assertEqual(len(hosts), 1, f"exactly one host expected, got {hosts}")
        host = hosts[0]
        self.assertTrue(cmds.getAttr(f"{host}.__rl_host__"))
        # Host must NOT carry the rig GC-ownership tag (or cleanup could sweep it).
        self.assertFalse(cmds.attributeQuery("__rig__", node=host, exists=True))
        # Both created knobs live on the one host.
        self.assertTrue(cmds.attributeQuery("knobA", node=host, exists=True))
        self.assertTrue(cmds.attributeQuery("knobB", node=host, exists=True))

    # -- native_multi_publish capability flag -- #
    def test_native_multi_publish_flag_publishes_input_multi_on_container(self):
        set_options(native_multi_publish=True)
        with container("nm_in"):
            container.publish_input(
                [[1, 2, 3], [4, 5, 6]], "vectors", at="double3", multi=True
            )
        self.assertIn(
            "vectors", cmds.container("nm_in", q=True, publishName=True) or []
        )
        self.assertTrue(cmds.attributeQuery("vectors", node="nm_in", exists=True))

    def test_native_multi_publish_flag_publishes_output_multi(self):
        set_options(native_multi_publish=True)
        with container("nm_out"):
            mults = [Node.create("multiplyDivide", name=f"nm{i}") for i in range(2)]
            for i, mm in enumerate(mults):
                mm.input1X << float(i + 1)
                mm.input2X << 2.0
            container.publish_output(
                PlugList([mm.outputX for mm in mults]),
                "results",
                at    = "double",
                multi = True,
            )
        self.assertIn(
            "results", cmds.container("nm_out", q=True, publishName=True) or []
        )

    def test_native_multi_publish_default_keeps_multi_off_container(self):
        with container("nm_def"):
            container.publish_input([[1, 2, 3]], "vectors", at="double3", multi=True)
        # Default OFF: not on the container, but resolvable via the registry.
        self.assertFalse(cmds.attributeQuery("vectors", node="nm_def", exists=True))
        self.assertIsNotNone(_bound_plug("nm_def", "vectors"))

    def test_publish_internal_member_multi_registers(self):
        # Publishing an INNER member plug that is itself a multi (default flag
        # off) registers it rather than natively publishing -- it stays on the
        # real inner node and resolves through the registry.
        with container("imm"):
            pma = Node.create("plusMinusAverage", name="imm_pma")  # input3D is multi
            container.publish_input(pma.input3D, "arr")
        self.assertFalse(cmds.attributeQuery("arr", node="imm", exists=True))
        real = _bound_plug("imm", "arr")
        self.assertIsNotNone(real)
        self.assertTrue(real.startswith("imm_pma."))  # the REAL inner plug

    # -- >> shortcut, external writable plug routes to a host input -- #
    def test_rshift_external_writable_plug_routes_to_host_input(self):
        ext = Node.create("transform", name="ext_src")  # external (pre-container)
        with container("ee"):
            ext.tx >> container  # external + writable -> host input knob
        real = _bound_plug("ee", "translateX")
        self.assertIsNotNone(real)
        self.assertIn("_host.", real)

    # -- identity / passthrough: same member plug published as two inputs -- #
    def test_publish_same_member_plug_as_two_inputs(self):
        with container("ii"):
            n = Node.create("transform", name="inode")
            n << Float("val", dv=0.0) << 0.3
            container.publish_input(n.val, "knobA")  # binds inode.val directly
            container.publish_input(n.val, "knobB")  # already bound -> host carrier
        pairs = _bind_pairs("ii")
        self.assertEqual(pairs.get("knobA"), "inode.val")
        self.assertIsNotNone(_bound_plug("ii", "knobB"))
        self.assertNotEqual(pairs.get("knobB"), pairs.get("knobA"))  # distinct carrier

    # -- publish_output: inferred compound from a scalar list (no kwargs) -- #
    def test_publish_output_infers_compound_from_scalar_list(self):
        with container("ic"):
            m = Node.create("multiplyDivide", name="mm")
            m.input1X << 2.0
            m.input1Y << 3.0
            m.input1Z << 4.0
            m.input2X << 1.0
            m.input2Y << 1.0
            m.input2Z << 1.0
            # 3 scalar Plugs, NO kwargs -> inferred double3 compound (not multi).
            container.publish_output([m.outputX, m.outputY, m.outputZ], "vec")
        real = _bound_plug("ic", "vec")
        host = real.split(".")[0]
        # Inferred as a 3-element compound (double3 / float3 by element type) --
        # crucially NOT a multi (the v4.R scalar-list -> compound heuristic).
        self.assertIn(
            cmds.attributeQuery("vec", node=host, attributeType=True),
            ("double3", "float3"),
        )
        self.assertFalse(cmds.attributeQuery("vec", node=host, multi=True))
        self.assertAlmostEqual(cmds.getAttr(real)[0][0], 2.0)

    # -- publish_output: external single Plug with explicit kwargs -- #
    def test_publish_output_external_plug_with_kwargs(self):
        ext = Node.create("multiplyDivide", name="ext_mul")  # external
        ext.input1X << 3.0
        ext.input2X << 4.0  # outputX = 12
        with container("eo"):
            container.publish_output(ext.outputX, "result", at="double")
        real = _bound_plug("eo", "result")
        self.assertAlmostEqual(cmds.getAttr(real), 12.0)

    # -- publish_output: external multi Plug registers (not natively published) -- #
    def test_publish_output_external_multi_plug_registers(self):
        ext = Node.create("plusMinusAverage", name="ext_pma")  # input3D is multi
        with container("eom"):
            container.publish_output(ext.input3D, "arr")
        self.assertFalse(cmds.attributeQuery("arr", node="eom", exists=True))
        self.assertIsNotNone(_bound_plug("eom", "arr"))

    # -- publish_output: a scalar constant value -- #
    def test_publish_output_scalar_value(self):
        with container("so"):
            container.publish_output(7.5, "constant")
        real = _bound_plug("so", "constant")
        self.assertAlmostEqual(cmds.getAttr(real), 7.5)


class TestNativePublishHelpers(MayaTestCase):
    """Defensive-input coverage for the native-publish helper functions --
    they must degrade gracefully (return a falsey default) on invalid nodes /
    plugs rather than raise."""

    TEST_START_NEW_SCENE = True

    def test_is_host_false_on_invalid_node(self):
        from rig._internal.container import _is_host

        self.assertFalse(_is_host("does_not_exist"))

    def test_plug_is_multi_false_on_non_plug(self):
        from rig._internal.container import _plug_is_multi

        self.assertFalse(_plug_is_multi("not_a_plug"))

    def test_plug_is_bound_false_on_invalid_container(self):
        from rig._internal.container import _plug_is_bound

        self.assertFalse(_plug_is_bound("does_not_exist", "some.plug"))

    def test_is_container_member_false_on_bad_plug_and_bad_container(self):
        from rig._internal.container import _is_container_member

        n = Node.create("transform", name="member_probe")
        # Bad plug object (no ``.node``) -> owner lookup fails -> False.
        self.assertFalse(_is_container_member("does_not_exist", object()))
        # Valid plug but non-existent container -> nodeList query fails -> False.
        self.assertFalse(_is_container_member("does_not_exist", n.tx))

    def test_published_name_exists_false_on_invalid_container(self):
        from rig._internal.container import _published_name_exists

        self.assertFalse(_published_name_exists("does_not_exist", "x"))

    def test_register_multi_noop_on_invalid_nodes(self):
        from rig._internal.container import _register_multi

        # Must degrade to a no-op (not raise) on unresolvable nodes.
        _register_multi("does_not_exist", "x", "also_missing")

    def test_resolve_published_none_on_invalid_container(self):
        from rig._internal.container import _resolve_published

        self.assertIsNone(_resolve_published("does_not_exist", "x"))

    def test_destroy_published_name_false_on_invalid_container(self):
        from rig._internal.container import _destroy_published_name

        self.assertFalse(_destroy_published_name("does_not_exist", "x"))


class TestInferAttrTypeVectorLiterals(MayaTestCase):
    """``_infer_attr_type`` must recognise a FLAT all-numeric vec3 / vec4
    literal as a ``double3`` / ``double4`` compound. Without this it fell
    through to scalar ``{"at": "double"}``, breaking ``publish_input`` (and
    thus ``slerp`` / ``lerp``) on raw ``[x, y, z]`` inputs. Mirrors the
    PlugList-of-3/4 and 2D-array compound heuristics for flat literals."""

    TEST_START_NEW_SCENE = True

    def _infer(self, source):
        from rig._internal.container import _infer_attr_type

        return _infer_attr_type(source)

    def test_flat_vec3_infers_double3(self):
        self.assertEqual(self._infer([0, 0, 1]), {"at": "double3"})

    def test_flat_vec3_tuple_infers_double3(self):
        self.assertEqual(self._infer((1.0, 2.0, 3.0)), {"at": "double3"})

    def test_flat_vec4_infers_double4(self):
        self.assertEqual(self._infer([0, 0, 1, 0]), {"at": "double4"})

    def test_len5_numeric_list_stays_scalar(self):
        # Lengths other than 3/4/16 keep the scalar fallback (unchanged).
        self.assertEqual(self._infer([1, 2, 3, 4, 5]), {"at": "double"})

    def test_len3_non_numeric_stays_scalar(self):
        # A 3-element sequence that is not all-numeric must NOT become a
        # compound -- covers the ``all(...)`` False branch.
        self.assertEqual(self._infer([1, 2, "x"]), {"at": "double"})

    def test_len3_with_bool_stays_scalar(self):
        # bool is excluded (it is an int subclass but not a vector channel),
        # so a triple containing a bool stays scalar.
        self.assertEqual(self._infer([1.0, 2.0, True]), {"at": "double"})

    def test_len16_still_matrix(self):
        # The pre-existing flat-16 -> matrix case is untouched.
        self.assertEqual(self._infer([0.0] * 16), {"dt": "matrix"})