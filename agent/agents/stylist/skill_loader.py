"""Loader for the repository Fashion Styling Skill."""

import logging
from functools import lru_cache
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "fashion_styling"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _read_skill() -> str:
    logger.info("function_start function=_read_skill")
    return (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")


def load_fashion_skill() -> str:
    """Load the complete fashion styling skill."""
    logger.info("function_start function=load_fashion_skill")
    return _read_skill()
