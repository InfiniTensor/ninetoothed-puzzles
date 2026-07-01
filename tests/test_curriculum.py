from ntpuzzles.curriculum import MODULES_BY_SLUG, QUESTS, QUESTS_BY_SLUG, text
from ntpuzzles.game import (
    CODE_ANSWER_REPLACEMENTS,
    FILL_ANSWER_REPLACEMENTS,
    _inferred_required_fragments,
    reference_solution_source,
)


def test_curriculum_has_consistent_metadata():
    slugs = [quest.slug for quest in QUESTS]
    assert len(slugs) == len(set(slugs))
    assert QUESTS_BY_SLUG.keys() == set(slugs)

    for quest in QUESTS:
        assert quest.module in MODULES_BY_SLUG
        assert quest.kind in {"answer", "code"}
        for field in (quest.title, quest.subtitle, quest.question, quest.hint, quest.success):
            assert text(field, "zh")
            assert text(field, "en")
        assert quest.concepts


def test_every_level_has_reference_completion():
    for quest in QUESTS:
        if quest.kind == "answer":
            assert quest.slug in FILL_ANSWER_REPLACEMENTS
            source = reference_solution_source(quest)
            assert "___" not in source
            assert "def solve" in source
        else:
            assert quest.slug in CODE_ANSWER_REPLACEMENTS
            assert quest.starter_name
            assert quest.validator


def test_fill_references_use_required_ninetoothed_fragments():
    for quest in QUESTS:
        if quest.kind != "answer":
            continue
        source = reference_solution_source(quest)
        source_without_comments = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for fragment in _inferred_required_fragments(quest):
            assert fragment in source_without_comments, f"{quest.slug} should use {fragment}"
