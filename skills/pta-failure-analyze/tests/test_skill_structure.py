from pathlib import Path


def test_skill_markers_and_boundaries_present():
    skill_md = Path(__file__).resolve().parents[1] / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")

    assert "## When to use" in text
    assert "## When not to use" in text
    assert "## Stage 0: Gather Evidence" in text
    assert "## Stage 1: Capture Canonical Facts" in text
    assert "## Stage 2: Check Existing Knowledge" in text
    assert "## Stage 3: Diagnose the Failure" in text
    assert "## Stage 4: Validate and Close" in text
    assert "manual skill" in text.lower()
    assert "must not" in text.lower()
    assert "known_failure" in text
    assert "operator" in text
    assert "failure-showcase.md" in text


def test_skill_non_use_boundaries_present():
    skill_md = Path(__file__).resolve().parents[1] / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8").lower()

    assert "accuracy drift" in text
    assert "performance" in text
    assert "setup" in text
    assert "higher-level `failure-agent`".lower() in text
