"""Stub visualization utilities.

The full ``SceneVisualizer`` is only needed for interactive notebook
visualization (``make_scene``).  This placeholder prevents import errors
when the module is referenced but not exercised in the baseline's pipeline.
"""

from __future__ import annotations


class SceneVisualizer:
    """Minimal stub — raises if any method is actually called."""

    @staticmethod
    def object_pointcloud(**kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "SceneVisualizer.object_pointcloud requires the full visualization "
            "module. It is not needed for the baseline pipeline."
        )
