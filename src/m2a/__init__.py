"""
Manhwa to Anime (m2a) pipeline package.
"""

__version__ = "0.1.0"

from m2a.state import RunState, PanelJSON, CharacterBible
from m2a.config import PipelineConfig

__all__ = [
    "RunState",
    "PipelineConfig",
    "PanelJSON",
    "CharacterBible",
]
