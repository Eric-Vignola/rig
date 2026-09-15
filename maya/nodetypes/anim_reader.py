"""
AnimReader node type for working with .anm animation clips.
Provides a clean API for creating, connecting, and managing AnimReader nodes
with RT Rigs, following the rig.maya.nodetypes pattern.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import maya.api.OpenMaya as om
from maya import cmds
from rig.maya.nodetypes.joint import Joint

LOG = logging.getLogger(__name__)


class AnimReaderError(Exception):
    """Exception raised for AnimReader operation errors"""

    pass


class AnimReaderNode:
    """
    AnimReader node wrapper for working with .anm animation clips.
    Provides high-level API for creating and managing AnimReader nodes with RT Rigs.
    """

    NODE_TYPE                 = "animReader"
    _global_joint_names_cache = None  # Class-level cache for joint names

    def __init__(self, node_name: str, cached_joint_names: Optional[List[str]] = None):
        """
        Initialize AnimReader node wrapper.

        Args:
            node_name: Name of the AnimReader node in Maya
            cached_joint_names: Optional cached joint names to avoid repeated getRTHierarchy calls
        """
        self.name                = node_name
        self._cached_joint_names = cached_joint_names
        if not cmds.objExists(node_name):
            raise AnimReaderError(f"AnimReader node '{node_name}' does not exist")

    @classmethod
    def _get_joint_names_cached(cls) -> List[str]:
        """Get joint names with class-level caching to avoid repeated getRTHierarchy calls."""
        if cls._global_joint_names_cache is None:
            try:
                cls._global_joint_names_cache = cmds.getRTHierarchy() or []
                LOG.info(
                    f"Cached {len(cls._global_joint_names_cache)} joint names from RT hierarchy"
                )
            except Exception as e:
                LOG.warning(f"Error getting joint names from getRTHierarchy: {e}")
                cls._global_joint_names_cache = []
        return cls._global_joint_names_cache

    @classmethod
    def clear_joint_names_cache(cls):
        """Clear the cached joint names (useful when RT hierarchy changes)."""
        cls._global_joint_names_cache = None

    @classmethod
    def create(cls, name: str = None) -> AnimReaderNode:
        """
        Create a new AnimReader node.

        Args:
            name: Optional name for the node. If None, Maya will auto-generate.

        Returns:
            New AnimReader node instance
        """
        try:
            # Check if animReader plugin is loaded
            if not cmds.pluginInfo("AnimReader", query=True, loaded=True):
                LOG.info("AnimReader plugin not loaded, attempting to load...")
                try:
                    cmds.loadPlugin("AnimReader")
                    LOG.info("Successfully loaded AnimReader plugin")
                except Exception as plugin_e:
                    raise AnimReaderError(
                        f"Failed to load AnimReader plugin: {plugin_e}"
                    )

            # Verify animReader node type is available
            available_types = cmds.allNodeTypes()
            if "animReader" not in available_types:
                raise AnimReaderError(
                    "animReader node type not available even after loading plugin"
                )

            # Create the animReader node
            LOG.info("Creating animReader node...")
            node = cmds.createNode("animReader")
            LOG.info(f"Created node: {node}")

            # Verify the node was actually created and exists
            if not cmds.objExists(node):
                raise AnimReaderError(f"Node creation failed - {node} does not exist")

            # Rename it if a specific name was requested
            if name:
                LOG.info(f"Renaming node to: {name}")
                node = cmds.rename(node, name)
                LOG.info(f"Renamed to: {node}")

                # Verify renamed node exists
                if not cmds.objExists(node):
                    raise AnimReaderError(
                        f"Node lost after rename - {node} does not exist"
                    )

            LOG.info(f"Successfully created AnimReader node: {node}")
            return cls(node)

        except Exception as e:
            LOG.error(f"Error creating AnimReader node: {e}")
            raise AnimReaderError(f"Failed to create AnimReader node: {e}")

    @classmethod
    def create_for_rt_root_joint(
        cls,
        rt_root_joint:  str,
        anm_file_paths: List[str],
        logger:         Optional[logging.Logger] = None,
    ) -> Tuple[AnimReaderNode, int]:
        """
        Create and configure an AnimReader node for an RT root joint.

        Args:
            rt_root_joint: RT root joint name (e.g., "RT_00:skelanim_RTRig_Root" or "RT_00:RTRig_Root")
            anm_file_paths: List of .anm file paths to load
            logger: Optional logger for output messages

        Returns:
            Tuple of (AnimReader node instance, connected_joint_count)

        Raises:
            AnimReaderError: If creation or connection fails
        """
        if not logger:
            logger = LOG

        if not anm_file_paths:
            raise AnimReaderError("No .anm file paths provided")

        # Extract RT rig name from the root joint
        if ":" in rt_root_joint:
            rt_rig_name = rt_root_joint.split(":")[0]
        else:
            # For non-namespaced joints, use a default namespace or the joint name itself
            rt_rig_name = "RT_00"  # Default namespace used by getRTHierarchy

        # Verify the root joint exists
        if not cmds.objExists(rt_root_joint):
            raise AnimReaderError(f"RT root joint does not exist: {rt_root_joint}")

        # Check if AnimReader already exists and delete it
        expected_name = cls.get_expected_name_for_rt_rig(rt_rig_name)
        if cmds.objExists(expected_name):
            logger.info(
                f"AnimReader '{expected_name}' already exists for RT rig '{rt_rig_name}', deleting and rebuilding..."
            )
            cmds.delete(expected_name)
            logger.info(f"Deleted existing AnimReader: {expected_name}")

        try:
            logger.info(f"Step 1: Creating AnimReader node with name: {expected_name}")
            # Get joint names once during creation and cache them at class level
            joint_names = cls._get_joint_names_cached()
            logger.info(f"Retrieved {len(joint_names)} joint names from RT hierarchy")

            # Create the AnimReader node with expected name and cached joint names
            anim_reader = cls.create(expected_name)
            anim_reader._cached_joint_names = (
                joint_names  # Cache the joint names on instance too
            )

            logger.info(f"Step 2: Setting animation file paths: {anm_file_paths}")
            anim_reader.set_animation_file_paths(anm_file_paths)
            logger.info("Step 2 Complete: Animation file paths set")

            logger.info("Step 3: Refreshing node to load clips")
            anim_reader.refresh()
            logger.info("Step 3 Complete: Node refreshed")

            logger.info(f"Step 4: Connecting to skeletal joints for {rt_root_joint}")
            connected_joints = anim_reader.connect_to_rt_rig_skeleton(
                None, rt_root_joint, logger
            )
            logger.info(f"Step 4 Complete: Connected {connected_joints} joints")

            logger.info("Step 5: Connecting time node")
            anim_reader.connect_time_node()
            logger.info("Step 5 Complete: Time node connected")

            logger.info("Step 6: Setting Maya timeline to match animation duration")
            anim_reader.set_timeline_to_animation_duration(logger)
            logger.info("Step 6 Complete: Timeline updated")

            logger.info(
                f"Animation setup complete for {rt_rig_name}! "
                f"Connected {connected_joints} joints."
            )

            return anim_reader, connected_joints

        except Exception as e:
            logger.error(f"Error in create_for_rt_root_joint: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # DO NOT delete the node - leave it for debugging and manual cleanup
            if cmds.objExists(expected_name):
                logger.warning(
                    f"Node {expected_name} still exists after failure - leaving for manual inspection"
                )
            else:
                logger.info(f"Node {expected_name} does not exist after failure")

            raise AnimReaderError(f"Failed to create AnimReader for {rt_rig_name}: {e}")

    @classmethod
    def get_node_from_skeleton(cls, root_joint: str) -> Optional[AnimReaderNode]:
        """
        Get existing AnimReader node for an RT rig by checking what's actually connected to the RT root joint.

        Args:
            rt_rig_name: Name of the RT rig (e.g., "RT_00" or "RTRig_Root" for non-namespaced)

        Returns:
            AnimReader node instance if exists, None otherwise
        """

        connections = cmds.listConnections(
            f"{root_joint}.tx", source=True, destination=False
        )
        if connections:
            for connection in connections:
                # Check if this connection comes from an animReader node
                if cmds.nodeType(connection) == "animReader":
                    return cls(connection)

        return None

    @classmethod
    def get_expected_name_for_rt_rig(cls, rt_rig_name: str) -> str:
        """
        Get the expected AnimReader node name for an RT rig.

        Args:
            rt_rig_name: Name of the RT rig

        Returns:
            Expected AnimReader node name
        """
        return f"{rt_rig_name}_animReader"

    def set_animation_file_paths(self, file_paths: List[str]) -> None:
        """
        Set animation file paths on the AnimReader node.

        Args:
            file_paths: List of .anm file paths
        """
        # Convert paths to forward slashes for Maya
        normalized_paths = [path.replace("\\", "/") for path in file_paths]

        # Set up animation file paths using Maya API
        selection_list = om.MSelectionList()
        selection_list.add(self.name)
        node_obj    = selection_list.getDependNode(0)
        dep_node_fn = om.MFnDependencyNode(node_obj)

        anim_file_paths_attr = dep_node_fn.attribute("animFilePaths")
        anim_file_paths_plug = om.MPlug(node_obj, anim_file_paths_attr)

        # Set the animation file paths
        for i, file_path in enumerate(normalized_paths):
            element_plug = anim_file_paths_plug.elementByLogicalIndex(i)
            element_plug.setString(file_path)

    def refresh(self) -> None:
        """Force evaluation to load the clips."""
        cmds.dgdirty(self.name)
        cmds.refresh()

    def connect_to_rt_rig_skeleton(
        self, prefix: str, rt_root_joint: str, logger: Optional[logging.Logger] = None
    ) -> int:
        """
        Connect AnimReader outputs to RT rig skeletal joints.
        Gets actual joints from the RT root joint hierarchy using listRelatives and matches them
        with getRTHierarchy joint names for correct indexing.

        Args:
            rt_root_joint: RT root joint name (e.g., "avatar_artcastle_RTRig_Root")
            logger: Optional logger for messages

        Returns:
            Number of joints successfully connected

        Raises:
            AnimReaderError: If no joints are found or connection fails
        """
        if not logger:
            logger = LOG

        # Verify the root joint exists
        if not cmds.objExists(rt_root_joint):
            raise AnimReaderError(f"RT root joint does not exist: {rt_root_joint}")

        # Force the node to evaluate to ensure clips are loaded
        cmds.dgdirty(self.name)
        cmds.refresh()

        # Get all actual joints in the RT rig hierarchy
        actual_joints = (
            cmds.listRelatives(rt_root_joint, allDescendents=True, type="joint") or []
        )
        actual_joints.insert(0, rt_root_joint)  # Include root joint
        logger.info(f"Found {len(actual_joints)} actual joints in RT rig hierarchy")

        # Get RT hierarchy joint names from getRTHierarchy for correct indexing
        rt_hierarchy_joint_names = cmds.getRTHierarchy()
        logger.info(
            f"Got {len(rt_hierarchy_joint_names)} joint names from getRTHierarchy"
        )

        # Create a mapping from short joint names to actual full joint names
        joint_name_mapping = {}
        for actual_joint in actual_joints:
            # Extract the short name (after namespace and prefixes)
            short_name = actual_joint.split(":")[-1]  # Remove namespace
            # Remove common prefixes like "skelanim_" or "avatar_artcastle_"
            if "skelanim_" in short_name:
                short_name = short_name.split("skelanim_")[-1]
            elif "_" in short_name:
                # Handle cases like "avatar_artcastle_RTRig_Root" -> "RTRig_Root"
                parts = short_name.split("_")
                # Look for the RTRig part and take from there
                for i, part in enumerate(parts):
                    if part == "RTRig" and i + 1 < len(parts):
                        short_name = "_".join(parts[i:])
                        break

            joint_name_mapping[short_name] = actual_joint
            logger.debug(f"Mapped joint: {short_name} -> {actual_joint}")

        connected_count = 0
        failed_joints   = []

        # Connect joints using the correct mapping
        for i, hier_joint_name in enumerate(rt_hierarchy_joint_names):
            # Find the actual joint that matches this hierarchy joint name
            actual_joint = joint_name_mapping.get(hier_joint_name)

            if not actual_joint:
                # Try alternative matching patterns
                for mapped_name, mapped_joint in joint_name_mapping.items():
                    if hier_joint_name in mapped_name or mapped_name.endswith(
                        hier_joint_name
                    ):
                        actual_joint = mapped_joint
                        break

            if not actual_joint:
                failed_joints.append(f"{hier_joint_name} (no matching joint found)")
                logger.warning(
                    f"Could not find actual joint for hierarchy joint: {hier_joint_name}"
                )
                continue

            if not cmds.objExists(actual_joint):
                failed_joints.append(
                    f"{hier_joint_name} -> {actual_joint} (joint does not exist)"
                )
                logger.warning(f"Actual joint does not exist: {actual_joint}")
                continue

            try:
                # Reset joint attributes to zero first
                self._reset_joint_attributes(actual_joint)

                # Connect all transform channels using the correct AnimReader index (i)
                self._connect_joint_transforms(i, actual_joint)
                connected_count += 1
                logger.debug(
                    f"Connected joint {i}: {hier_joint_name} -> {actual_joint}"
                )

            except Exception as e:
                failed_joints.append(f"{hier_joint_name} -> {actual_joint} ({e})")
                logger.warning(f"Failed to connect joint {actual_joint}: {e}")

        if failed_joints:
            logger.warning(
                f"Failed to connect {len(failed_joints)} joints: {failed_joints}"
            )

        logger.info(
            f"Successfully connected {connected_count} out of {len(rt_hierarchy_joint_names)} joints"
        )
        return connected_count

    def connect_time_node(self) -> None:
        """Connect Maya's time node to drive the AnimReader."""
        time_nodes = cmds.ls(type="time")
        if time_nodes:
            time_node = time_nodes[0]
            cmds.connectAttr(f"{time_node}.outTime", f"{self.name}.time")

    def set_timeline_to_animation_duration(
        self, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Set Maya's timeline end frame to match the total duration of loaded animation clips.

        Args:
            logger: Optional logger for output messages
        """
        if not logger:
            logger = LOG

        try:
            # Get debug info to find total duration
            debug_info    = self.get_debug_info()
            end_frame_str = debug_info.get("end_frame", "0")

            try:
                end_frame = int(float(end_frame_str))
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse end frame '{end_frame_str}' as a number"
                )
                return

            if end_frame <= 0:
                logger.warning(f"Invalid end frame value: {end_frame}")
                return

            # Get current timeline settings
            current_start = cmds.playbackOptions(query=True, minTime=True)
            current_end   = cmds.playbackOptions(query=True, maxTime=True)

            # Set the timeline end frame to match animation duration
            cmds.playbackOptions(maxTime=end_frame)

            logger.info(
                f"Updated Maya timeline: Start={current_start}, End={current_end} -> End={end_frame}"
            )
            logger.info(
                f"Timeline now matches animation duration of {end_frame} frames"
            )

        except Exception as e:
            logger.warning(f"Failed to set timeline duration: {e}")

    def disconnect_from_rt_rig(
        self, rt_root_joint: str, logger: Optional[logging.Logger] = None
    ) -> int:
        """
        Disconnect AnimReader and properly match skeleton to base skeleton.
        This creates a clean duplicate skeleton hierarchy that replaces the animated one.

        Args:
            rt_root_joint: RT root joint name (e.g., "RT_00:skelanim_RTRig_Root" or "RT_00:RTRig_Root")
            logger: Optional logger for output messages

        Returns:
            Number of joints successfully matched

        Raises:
            AnimReaderError: If disconnection fails
        """
        if not logger:
            logger = LOG

        # Verify the root joint exists
        if not cmds.objExists(rt_root_joint):
            raise AnimReaderError(f"RT root joint does not exist: {rt_root_joint}")

        logger.info(f"Starting disconnect process for RT root joint: {rt_root_joint}")

        try:
            # Step 1: Create base skeleton using getRTHierarchy
            logger.info("Step 1: Creating base skeleton using getRTHierarchy")
            base_joint_names = cmds.getRTHierarchy(generate=True, prefix="base_")
            if not base_joint_names:
                raise AnimReaderError("getRTHierarchy returned no joint names")

            base_root_joint = base_joint_names[0]  # First joint is the root
            logger.info(f"Created base skeleton with root: {base_root_joint}")

            # Step 2: Get all joints from both hierarchies
            logger.info("Step 2: Getting joint hierarchies for matching")
            rt_joints = (
                cmds.listRelatives(rt_root_joint, allDescendents=True, type="joint")
                or []
            )
            rt_joints.insert(0, rt_root_joint)  # Include root joint

            base_joints = (
                cmds.listRelatives(base_root_joint, allDescendents=True, type="joint")
                or []
            )
            base_joints.insert(0, base_root_joint)  # Include root joint

            logger.info(
                f"Found {len(rt_joints)} RT joints and {len(base_joints)} base joints"
            )

            # Step 3: Disconnect AnimReader connections FIRST
            logger.info("Step 3: Disconnecting AnimReader connections")
            self._disconnect_all_anim_reader_connections(rt_root_joint, logger)

            # Step 4: Delete the AnimReader node
            logger.info("Step 4: Deleting AnimReader node")
            cmds.delete(self.name)
            logger.info(f"Deleted AnimReader node: {self.name}")

            # Step 5: Match joint transforms using Joint class's match_matrix method
            logger.info("Step 5: Matching joint transforms to base pose")
            matched_count = 0

            # Ensure we have the same number of joints
            min_joints = min(len(rt_joints), len(base_joints))

            for i in range(min_joints):
                try:
                    rt_joint_obj   = Joint(rt_joints[i])
                    base_joint_obj = Joint(base_joints[i])

                    # Copy the transform from base skeleton to RT joint (reset to base pose)
                    rt_joint_obj.match_matrix(base_joint_obj, world_space=False)
                    matched_count += 1
                    logger.debug(
                        f"Reset joint {i} to base pose: {rt_joints[i]} <- {base_joints[i]}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to reset joint {rt_joints[i]} to base pose from {base_joints[i]}: {e}"
                    )

            logger.info(f"Successfully reset {matched_count} joints to base pose")

            # Step 6: Clean up base skeleton (it was only used for matching)
            logger.info("Step 6: Cleaning up temporary base skeleton")
            cmds.delete(base_joint_names)
            logger.info("Cleaned up temporary base skeleton")

            logger.info(
                f"Successfully disconnected AnimReader and matched {matched_count} joints"
            )
            return matched_count

        except Exception as e:
            logger.error(f"Error in disconnect_from_rt_rig: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise AnimReaderError(f"Failed to disconnect from RT rig: {e}")

    def get_animation_info(self) -> dict:
        """
        Get information about loaded animations and connections.

        Returns:
            Dictionary with animation information
        """
        info = {
            "node_name":    self.name,
            "loaded_clips": [],
            "total_joints": 0,
            "joint_names":  [],
        }

        try:
            # Use cached joint names if available, otherwise get them from class cache
            if self._cached_joint_names is None:
                self._cached_joint_names = self._get_joint_names_cached()

            info["joint_names"]  = self._cached_joint_names
            info["total_joints"] = len(self._cached_joint_names)

            # Get loaded clips info from debug attributes instead of removed animFilePaths
            try:
                total_clips          = cmds.getAttr(f"{self.name}.debugTotalClips") or "0"
                info["loaded_clips"] = [f"clip_{i}" for i in range(int(total_clips))]
            except Exception as clips_e:
                LOG.warning(
                    f"Error getting clips info from debug attributes: {clips_e}"
                )
                info["loaded_clips"] = []

        except Exception as e:
            LOG.warning(f"Error getting animation info for {self.name}: {e}")

        return info

    def get_debug_info(self) -> dict:
        """
        Get debug information from the AnimReader node's debug attributes.

        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "start_frame":   "0",
            "end_frame":     "0",
            "total_clips":   "0",
            "current_clip":  "0",
            "current_phase": "idle",
        }

        try:
            # Get debug attributes from the node
            debug_info["start_frame"] = (
                cmds.getAttr(f"{self.name}.debugStartFrame") or "0"
            )
            debug_info["end_frame"] = cmds.getAttr(f"{self.name}.debugEndFrame") or "0"
            debug_info["total_clips"] = (
                cmds.getAttr(f"{self.name}.debugTotalClips") or "0"
            )
            debug_info["current_clip"] = (
                cmds.getAttr(f"{self.name}.debugCurrentClip") or "0"
            )
            debug_info["current_phase"] = (
                cmds.getAttr(f"{self.name}.debugCurrentPhase") or "idle"
            )

        except Exception as e:
            LOG.warning(f"Error getting debug info for {self.name}: {e}")

        return debug_info

    def get_rt_rig_connection_info(self, rt_rig_name: str) -> dict:
        """
        Get detailed connection information for a specific RT rig.

        Args:
            rt_rig_name: Name of the RT rig

        Returns:
            Dictionary with connection information
        """
        info = {
            "rt_rig_name":      rt_rig_name,
            "connected_joints": 0,
            "failed_joints":    [],
            "missing_joints":   [],
        }

        try:
            # Use cached joint names if available, otherwise get them from class cache
            if self._cached_joint_names is None:
                self._cached_joint_names = self._get_joint_names_cached()

            joint_names = self._cached_joint_names

            for joint_name in joint_names:
                target_joint = f"{rt_rig_name}:skelanim_{joint_name}"

                if not cmds.objExists(target_joint):
                    info["missing_joints"].append(joint_name)
                    continue

                # Check if any transform attribute is connected
                transform_attrs = ["tx", "ty", "tz", "rx", "ry", "rz"]
                is_connected    = False

                for attr in transform_attrs:
                    connections = cmds.listConnections(
                        f"{target_joint}.{attr}", source=True, destination=False
                    )
                    if connections and self.name in connections:
                        is_connected = True
                        break

                if is_connected:
                    info["connected_joints"] += 1
                else:
                    info["failed_joints"].append(joint_name)

        except Exception as e:
            LOG.warning(f"Error getting connection info for {rt_rig_name}: {e}")

        return info

    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================

    def _find_rt_root_joint(self, rt_rig_name: str) -> Optional[str]:
        """
        Find existing RT root joint in the scene.

        Args:
            rt_rig_name: Name of the RT rig

        Returns:
            RT root joint name if found, None otherwise
        """
        # Check for traditional skeletal drive support
        traditional_skeletal_root = f"{rt_rig_name}:skelanim_RTRig_Root"
        if cmds.objExists(traditional_skeletal_root):
            return traditional_skeletal_root

        # Check for new namespaced skeleton (created by getRTHierarchy)
        namespaced_root = f"{rt_rig_name}:RTRig_Root"
        if cmds.objExists(namespaced_root):
            return namespaced_root

        return None

    def _get_joint_names_from_rt_root(self, rt_root_joint: str) -> List[str]:
        """
        Extract joint names from existing RT root joint hierarchy in the scene.

        Args:
            rt_root_joint: The RT root joint found in the scene

        Returns:
            List of joint names from the hierarchy
        """
        # Get all joints in the hierarchy
        all_joints = (
            cmds.listRelatives(rt_root_joint, allDescendents=True, type="joint") or []
        )
        all_joints.insert(0, rt_root_joint)  # Include root joint
        return all_joints

    def _get_joint_names_from_hierarchy(self) -> List[str]:
        """
        Get joint names from getRTHierarchy command as fallback.

        Args:
            rt_rig_name: Name of the RT rig
            logger: Logger for messages

        Returns:
            Tuple of (joint_names_list, target_prefix)
        """
        joint_names = cmds.getRTHierarchy()
        return joint_names

    def _reset_joint_attributes(self, joint: str) -> None:
        """Reset joint transform and orient attributes to zero."""
        reset_attrs = [
            "tx",
            "ty",
            "tz",
            "rx",
            "ry",
            "rz",
            "jointOrientX",
            "jointOrientY",
            "jointOrientZ",
        ]

        for attr in reset_attrs:
            try:
                cmds.setAttr(f"{joint}.{attr}", 0)
            except Exception:
                pass  # Attribute might be locked or non-existent

    def _connect_joint_transforms(self, joint_index: int, target_joint: str) -> None:
        """Connect all transform channels from AnimReader to target joint."""
        # Translation channels
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].translateX",
            f"{target_joint}.translateX",
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].translateY",
            f"{target_joint}.translateY",
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].translateZ",
            f"{target_joint}.translateZ",
        )

        # Rotation channels
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].rotateX",
            f"{target_joint}.rotateX",
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].rotateY",
            f"{target_joint}.rotateY",
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].rotateZ",
            f"{target_joint}.rotateZ",
        )

        # Scale channels
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].scaleX", f"{target_joint}.scaleX"
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].scaleY", f"{target_joint}.scaleY"
        )
        cmds.connectAttr(
            f"{self.name}.outTransforms[{joint_index}].scaleZ", f"{target_joint}.scaleZ"
        )

    def _reset_skeletal_to_base_pose(
        self,
        rt_rig_name:   str,
        base_skeleton: str,
        skeletal_root: str,
        logger:        logging.Logger,
    ) -> int:
        """
        Reset all skeletal joints to match their corresponding base joints.

        Returns:
            Number of joints successfully reset
        """
        # Get all joints from both hierarchies
        skeletal_joints = cmds.listRelatives(skeletal_root, ad=True, type="joint") or []
        skeletal_joints.insert(0, skeletal_root)  # Include root

        base_joints = cmds.listRelatives(base_skeleton, ad=True, type="joint") or []
        base_joints.insert(0, base_skeleton)  # Include root

        # Create mapping from joint names to base joints
        base_joint_map = {}
        for base_joint in base_joints:
            joint_name = base_joint.split(":")[-1]  # Remove namespace
            if joint_name.startswith("base_"):
                joint_name = joint_name[5:]  # Remove "base_" prefix
            base_joint_map[joint_name] = base_joint

        # Reset each skeletal joint
        reset_count   = 0
        failed_joints = []

        for skeletal_joint in skeletal_joints:
            try:
                # Extract joint name
                joint_name = skeletal_joint.split(":")[-1]  # Remove namespace
                if joint_name.startswith("skelanim_"):
                    joint_name = joint_name[9:]  # Remove "skelanim_" prefix

                # Find corresponding base joint
                base_joint = base_joint_map.get(joint_name)
                if not base_joint:
                    failed_joints.append(f"{skeletal_joint} (no base joint)")
                    continue

                # Disconnect and reset the joint
                self._disconnect_and_reset_joint(skeletal_joint, base_joint)
                reset_count += 1

            except Exception as e:
                failed_joints.append(f"{skeletal_joint} ({e})")
                logger.warning(f"Failed to reset joint {skeletal_joint}: {e}")

        if failed_joints:
            logger.warning(
                f"Failed to reset {len(failed_joints)} joints: {failed_joints}"
            )

        return reset_count

    def _disconnect_and_reset_joint(self, skeletal_joint: str, base_joint: str) -> None:
        """Disconnect incoming connections and reset joint to base pose."""
        transform_attrs    = ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"]
        joint_orient_attrs = ["jointOrientX", "jointOrientY", "jointOrientZ"]

        # Disconnect incoming connections
        for attr in transform_attrs:
            connections = cmds.listConnections(
                f"{skeletal_joint}.{attr}", source=True, destination=False, plugs=True
            )
            if connections:
                for connection in connections:
                    try:
                        cmds.disconnectAttr(connection, f"{skeletal_joint}.{attr}")
                    except Exception:
                        pass  # Connection might already be broken

        # Copy values from base joint
        for attr in transform_attrs + joint_orient_attrs:
            try:
                base_value = cmds.getAttr(f"{base_joint}.{attr}")
                cmds.setAttr(f"{skeletal_joint}.{attr}", base_value)
            except Exception:
                pass  # Attribute might be locked or non-existent

    def _disconnect_all_anim_reader_connections(
        self, rt_root_joint: str, logger: Optional[logging.Logger] = None
    ) -> int:
        """
        Disconnect all AnimReader connections from the RT rig skeleton hierarchy.

        Args:
            rt_root_joint: RT root joint name
            logger: Optional logger for messages

        Returns:
            Number of joints disconnected
        """
        if not logger:
            logger = LOG

        # Get all joints in the RT rig hierarchy
        all_joints = (
            cmds.listRelatives(rt_root_joint, allDescendents=True, type="joint") or []
        )
        all_joints.insert(0, rt_root_joint)  # Include root joint

        disconnected_count = 0
        transform_attrs    = ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"]

        for joint in all_joints:
            joint_disconnected = False

            for attr in transform_attrs:
                connections = cmds.listConnections(
                    f"{joint}.{attr}", source=True, destination=False, plugs=True
                )
                if connections:
                    for connection in connections:
                        # Check if this connection comes from our AnimReader node
                        if self.name in connection:
                            try:
                                cmds.disconnectAttr(connection, f"{joint}.{attr}")
                                joint_disconnected = True
                                logger.debug(
                                    f"Disconnected {connection} from {joint}.{attr}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to disconnect {connection} from {joint}.{attr}: {e}"
                                )

            if joint_disconnected:
                disconnected_count += 1

        logger.info(f"Disconnected AnimReader from {disconnected_count} joints")
        return disconnected_count


# =============================================================================
# CONVENIENCE FUNCTIONS - Simple wrappers for common operations
# =============================================================================


def create_anim_reader_for_rt_root_joint(
    rt_root_joint:  str,
    anm_file_paths: List[str],
    logger:         Optional[logging.Logger] = None,
) -> Tuple[str, int]:
    """
    Convenience function to create an AnimReader for an RT root joint.

    Args:
        rt_root_joint: RT root joint name (e.g., "RT_00:skelanim_RTRig_Root" or "RT_00:RTRig_Root")
        anm_file_paths: List of .anm file paths
        logger: Optional logger

    Returns:
        Tuple of (anim_reader_node_name, connected_joint_count)
    """
    anim_reader, connected_joints = AnimReaderNode.create_for_rt_root_joint(
        rt_root_joint, anm_file_paths, logger
    )
    return anim_reader.name, connected_joints


def disconnect_anim_reader_from_rt_rig(
    rt_rig_name: str, logger: Optional[logging.Logger] = None
) -> int:
    """
    Convenience function to disconnect AnimReader from an RT rig.

    Args:
        rt_rig_name: Name of the RT rig
        logger: Optional logger

    Returns:
        Number of joints reset
    """
    anim_reader = AnimReaderNode.get_node_from_skeleton(rt_rig_name)
    if not anim_reader:
        raise AnimReaderError(f"No AnimReader found for RT rig '{rt_rig_name}'")

    # Find the RT root joint for this rig
    rt_root_joint = anim_reader._find_rt_root_joint(rt_rig_name)
    if not rt_root_joint:
        raise AnimReaderError(f"No RT root joint found for RT rig '{rt_rig_name}'")

    return anim_reader.disconnect_from_rt_rig(rt_root_joint, logger)


def has_anim_reader(rt_rig_name: str) -> bool:
    """
    Convenience function to check if RT rig has an AnimReader.

    Args:
        rt_rig_name: Name of the RT rig

    Returns:
        True if AnimReader exists, False otherwise
    """
    anim_reader = AnimReaderNode.get_node_from_skeleton(rt_rig_name)
    return anim_reader is not None


def get_anim_reader_info(rt_rig_name: str) -> dict:
    """
    Convenience function to get AnimReader information.

    Args:
        rt_rig_name: Name of the RT rig

    Returns:
        Dictionary with AnimReader information
    """
    anim_reader = AnimReaderNode.get_node_from_skeleton(rt_rig_name)
    if not anim_reader:
        return {"rt_rig_name": rt_rig_name, "has_anim_reader": False}

    info            = anim_reader.get_animation_info()
    connection_info = anim_reader.get_rt_rig_connection_info(rt_rig_name)

    return {
        "rt_rig_name":      rt_rig_name,
        "has_anim_reader":  True,
        "anim_reader_node": anim_reader.name,
        "loaded_clips":     info["loaded_clips"],
        "total_joints":     info["total_joints"],
        "connected_joints": connection_info["connected_joints"],
        "failed_joints":    connection_info["failed_joints"],
        "missing_joints":   connection_info["missing_joints"],
    }