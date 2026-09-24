from __future__ import annotations

import json
from pathlib import Path

import importlib.metadata as metadata
import perturbo

from perturbo import cli


class _Distribution:
    def __init__(self, *, version: str, package: Path, editable: Path | None = None):
        self.version = version
        self._package = package
        self._editable = editable

    def locate_file(self, name: str) -> Path:
        assert name == "perturbo"
        return self._package

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        if self._editable is None:
            return None
        return json.dumps({"url": self._editable.as_uri(), "dir_info": {"editable": True}})


def _source_tree(root: Path, version: str | None) -> Path:
    package = root / "src" / "perturbo"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("")
    if version is not None:
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "perturbo"\nversion = "{version}"\n'
        )
    return init


def test_source_provenance_recognizes_the_current_editable(tmp_path, monkeypatch):
    project = tmp_path / "checkout"
    init = _source_tree(project, "2.0.0rc11")
    dist = _Distribution(
        version="2.0.0rc11",
        package=tmp_path / "site-packages" / "perturbo",
        editable=project,
    )
    monkeypatch.setattr(perturbo, "__file__", str(init))
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)

    version, source, note = cli._source_provenance()
    assert version == "2.0.0rc11"
    assert source == str(init.parent.resolve())
    assert note is None


def test_source_provenance_detects_another_editable_shadowed_by_pythonpath(tmp_path, monkeypatch):
    active = tmp_path / "active"
    installed_project = tmp_path / "old-editable"
    init = _source_tree(active, "2.0.0rc11")
    _source_tree(installed_project, "2.0.0rc5")
    dist = _Distribution(
        version="2.0.0rc5",
        package=tmp_path / "site-packages" / "perturbo",
        editable=installed_project,
    )
    monkeypatch.setattr(perturbo, "__file__", str(init))
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)

    version, _source, note = cli._source_provenance()
    assert version == "2.0.0rc11"
    assert note is not None
    assert "2.0.0rc5" in note
    assert str(installed_project.resolve()) in note
    assert "shadowed and NOT running" in note


def test_source_provenance_uses_normal_wheel_metadata(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "perturbo"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("")
    dist = _Distribution(version="2.0.0", package=package)
    monkeypatch.setattr(perturbo, "__file__", str(init))
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)

    version, source, note = cli._source_provenance()
    assert version == "2.0.0"
    assert source == str(package.resolve())
    assert note is None


def test_source_copy_without_pyproject_does_not_borrow_installed_version(tmp_path, monkeypatch):
    copied_init = _source_tree(tmp_path / "copied", None)
    installed = tmp_path / "site-packages" / "perturbo"
    installed.mkdir(parents=True)
    dist = _Distribution(version="2.0.0rc5", package=installed)
    monkeypatch.setattr(perturbo, "__file__", str(copied_init))
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)

    version, _source, note = cli._source_provenance()
    assert version == "unknown (source tree without package metadata)"
    assert note is not None
    assert "2.0.0rc5" in note
