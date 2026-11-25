"""Per-IC learning-potential scores. Importing the package registers every score."""

from resetsweep.scores import controls, slope, success_band
from resetsweep.scores.base import SCORES, BaseScore, ICTable, Score, TableScore

__all__ = [
    "SCORES",
    "BaseScore",
    "ICTable",
    "Score",
    "TableScore",
    "controls",
    "slope",
    "success_band",
]
