"""The results archive's download order: the release first, then its mirrors.

This exercises ``scripts/results_archive.py``, the repository's script rather
than a package, which is why it lives here. ``file://`` URLs stand in for the
release and the Zenodo mirror, so nothing touches the network.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("results_archive", REPO / "scripts/results_archive.py")
results_archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results_archive)


def _manifest(tmp_path, url, mirrors):
    good = tmp_path / "good.tar.gz"
    good.write_bytes(b"the archive the manifest pins")
    return {"archive": "results.tar.gz", "sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
            "url": url, "mirrors": mirrors}, good


def test_a_failed_release_falls_back_to_the_mirror(tmp_path):
    missing = (tmp_path / "not-there.tar.gz").as_uri()
    manifest, good = _manifest(tmp_path, missing, [])
    manifest["mirrors"] = [good.as_uri()]

    archive = results_archive.download(manifest, tmp_path / "out")

    assert archive.read_bytes() == good.read_bytes()


def test_a_location_serving_other_bytes_is_skipped(tmp_path):
    wrong = tmp_path / "wrong.tar.gz"
    wrong.write_bytes(b"something else")
    manifest, good = _manifest(tmp_path, wrong.as_uri(), [])
    manifest["mirrors"] = [good.as_uri()]

    assert results_archive.download(manifest, tmp_path / "out").read_bytes() == good.read_bytes()


def test_no_good_location_is_an_error_not_a_bad_install(tmp_path):
    wrong = tmp_path / "wrong.tar.gz"
    wrong.write_bytes(b"something else")
    manifest, _ = _manifest(tmp_path, (tmp_path / "gone").as_uri(), [wrong.as_uri()])

    with pytest.raises(SystemExit, match="no location served"):
        results_archive.download(manifest, tmp_path / "out")


def test_the_committed_manifest_names_a_release_and_a_mirror():
    import json

    manifest = json.loads((REPO / "data/results_manifest.json").read_text())
    assert manifest["url"].startswith("https://github.com/leejjoon/tellurix/releases/download/")
    assert any("zenodo.org" in url for url in manifest["mirrors"])
