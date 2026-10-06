"""Human evaluation rubric for generated videos."""
from dataclasses import dataclass
import logging
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class RubricScore:
    style_fidelity: int = 1
    motion_appeal: int = 1
    story_clarity: int = 1
    audio_fit: int = 1

    def __post_init__(self):
        for val in [self.style_fidelity, self.motion_appeal, self.story_clarity, self.audio_fit]:
            if not (1 <= val <= 5):
                raise ValueError("Scores must be between 1 and 5")

@dataclass
class RubricResult:
    reviewer_name: str
    clip_id: str
    scores: RubricScore
    comments: str = ""

    def aggregate_score(self) -> float:
        """Returns the average score across all criteria."""
        vals = [
            self.scores.style_fidelity,
            self.scores.motion_appeal,
            self.scores.story_clarity,
            self.scores.audio_fit
        ]
        return sum(vals) / len(vals)
        
def print_rubric_table(results: List[RubricResult]) -> str:
    """Generates a markdown table for rubric results."""
    # TODO: Implement actual markdown table formatting
    table = "| Reviewer | Clip ID | Style | Motion | Story | Audio | Avg |\n"
    table += "|----------|---------|-------|--------|-------|-------|-----|\n"
    for r in results:
        table += f"| {r.reviewer_name} | {r.clip_id} | {r.scores.style_fidelity} | {r.scores.motion_appeal} | {r.scores.story_clarity} | {r.scores.audio_fit} | {r.aggregate_score():.2f} |\n"
    return table
