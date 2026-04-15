from __future__ import annotations


class DonutRenderError(RuntimeError):
    """Base exception for the draft DonutRenderPy API."""


class RuntimeNotInitializedError(DonutRenderError):
    """Raised when create_scene() is called before init()."""


class InvalidStateError(DonutRenderError):
    """Raised when the API is used in an invalid lifecycle state."""


class SceneNotInitializedError(DonutRenderError):
    """Raised when Scene.update_*() or render_frame() is called before Scene.init()."""


class SceneDestroyedError(DonutRenderError):
    """Raised when a destroyed Scene is used again."""


class UnsupportedFeatureError(DonutRenderError):
    """Raised when the draft API accepts an object but the current backend cannot execute it yet."""
