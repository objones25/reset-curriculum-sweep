"""Per-IC learning-potential scores. Importing the package registers every score."""

from resetsweep.scores import (
    controls,
    critic,
    slope,
    steps_to_reward,
    success_band,
    visit_count,
)
from resetsweep.scores.base import SCORES, BaseScore, ICTable, Score, TableScore

__all__ = [
    "SCORES",
    "BaseScore",
    "ICTable",
    "Score",
    "TableScore",
    "controls",
    "critic",
    "slope",
    "steps_to_reward",
    "success_band",
    "visit_count",
]
