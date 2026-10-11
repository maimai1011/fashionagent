"""Explicit progressive loader for the repository Fashion Styling Skill."""

from functools import lru_cache
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "fashion_styling"


@lru_cache(maxsize=16)
def _read(relative_path: str) -> str:
    path = (SKILL_ROOT / relative_path).resolve()
    if SKILL_ROOT.resolve() not in path.parents:
        raise ValueError("skill reference escapes skill directory")
    return path.read_text(encoding="utf-8")


def load_fashion_skill(
    preferences: dict[str, object] | None = None,
    feedback: str = "",
) -> str:
    """Load core instructions plus references relevant to this request."""
    context = f"{preferences or {}} {feedback}".lower()
    references = {"color_matching.md", "outfit_composition.md", "styling_constraints.md"}
    occasion_terms = (
        "work", "office", "wedding", "party", "date", "formal", "casual", "occasion",
    )
    if any(word in context for word in occasion_terms):
        references.add("occasion_guidelines.md")
    sections = [_read("SKILL.md")]
    sections.extend(_read(f"references/{name}") for name in sorted(references))
    return "\n\n".join(sections)
