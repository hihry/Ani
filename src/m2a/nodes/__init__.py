"""Node definitions for the Manhwa-to-Anime pipeline."""

from m2a.nodes.ingest import ingest
from m2a.nodes.segment import segment
from m2a.nodes.understand import understand
from m2a.nodes.clean_plates import clean_plates
from m2a.nodes.route import route
from m2a.nodes.generate import generate
from m2a.nodes.qc import qc
from m2a.nodes.tts import tts_node
from m2a.nodes.assemble import assemble

__all__ = [
    "ingest",
    "segment",
    "understand",
    "clean_plates",
    "route",
    "generate",
    "qc",
    "tts_node",
    "assemble",
]
