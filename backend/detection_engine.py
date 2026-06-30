"""
Compatibility shim for DetectionEngine.
This file restores the historical import path `backend.detection_engine.DetectionEngine`
by aliasing the newer `CROSREngine` implementation.
"""
from .crosr_engine import CROSREngine


class DetectionEngine(CROSREngine):
    """Alias for backward compatibility with older imports.

    Inherits all behavior from `CROSREngine` so existing code that imports
    `DetectionEngine` continues to work without modification.
    """
    pass
