from pathlib import Path

import yaml


def test_skill_yaml_manual_contract():
    skill_root = Path(__file__).resolve().parents[1]
    manifest = skill_root / "skill.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0.0"
    assert data["name"] == "pta-failure-analyze"
    assert data["entry"]["type"] == "manual"
    assert data["entry"]["path"] == "SKILL.md"
    assert data["outputs"]["report_schema"] == "skills/_shared/contract/report.schema.json"
    assert data["outputs"]["out_dir_layout"] == "runs/<run_id>/out/"
    assert data["permissions"]["network"] == "none"
    assert data["permissions"]["filesystem"] == "workspace-write"


def test_manifest_dependencies_are_honest_and_minimal():
    skill_root = Path(__file__).resolve().parents[1]
    manifest = skill_root / "skill.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    assert sorted(data["dependencies"]["tools"]) == ["bash", "rg"]
    assert data["dependencies"]["python"] == []
