import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from tellurix import download as download_module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_aer_archive(path: Path, unsafe_name: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        name = unsafe_name or "aer_v_3.9/line_files_By_Molecule/01_H2O/01_H2O"
        payload = b"H2O lines\n"
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def test_download_mt_ckd_verifies_and_reuses_file(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.nc"
    payload = b"synthetic MT_CKD netCDF"
    source.write_bytes(payload)
    monkeypatch.setattr(download_module, "MT_CKD_URL", source.as_uri())
    monkeypatch.setattr(download_module, "MT_CKD_SHA256", sha256(payload))
    output = tmp_path / "installed"

    first = download_module.download_mt_ckd(output)
    second = download_module.download_mt_ckd(output)

    assert first == output / "mt_ckd/absco-ref_wv-mt-ckd.nc"
    assert first.read_bytes() == payload
    assert second == first
    assert "Using verified" in capsys.readouterr().out


def test_download_aer_lines_extracts_expected_layout(tmp_path, monkeypatch):
    archive = tmp_path / "source.tgz"
    archive_data = make_aer_archive(archive)
    license_file = tmp_path / "LICENSE.md"
    license_data = b"AER test license\n"
    license_file.write_bytes(license_data)
    monkeypatch.setattr(download_module, "AER_LINES_URL", archive.as_uri())
    monkeypatch.setattr(download_module, "AER_LINES_SHA256", sha256(archive_data))
    monkeypatch.setattr(download_module, "AER_LICENSE_URL", license_file.as_uri())
    monkeypatch.setattr(download_module, "AER_LICENSE_SHA256", sha256(license_data))
    output = tmp_path / "installed"

    installed = download_module.download_aer_lines(output)

    assert (installed / "line_files_By_Molecule/01_H2O/01_H2O").read_bytes() == b"H2O lines\n"
    assert (installed / "LICENSE.md").read_bytes() == license_data


def test_aer_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tgz"
    make_aer_archive(archive, "aer_v_3.9/../../outside")

    with pytest.raises(RuntimeError, match="unsafe path"):
        download_module._extract_aer_archive(archive, tmp_path / "aer_v_3.9")

    assert not (tmp_path / "outside").exists()


def test_download_hitran_h2o_builds_the_table_once_per_line(tmp_path, monkeypatch):
    import numpy as np
    from tellurix import PressureShiftTable

    def record(nu, delta):
        text = f" 11{nu:12.6f}{1e-24:10.3E}{1e-5:10.3E}.0700.3500{100.0:10.4f}0.70{delta:8.5f}"
        return (text + " " * 160)[:160].encode()

    served = {(0.0, 100.0): [record(50.0, -0.01), record(100.0, -0.02)],
              (100.0, 150.0): [record(100.0, -0.02), record(120.0, -0.03)]}
    calls = []

    def fetch(ids, lower, upper):
        calls.append((lower, upper))
        return b"\n".join(served[(lower, upper)]) + b"\n"

    monkeypatch.setattr(download_module, "_fetch_hitran_par", fetch)
    first = download_module.download_hitran_h2o(tmp_path, upper_cm1=150.0, step_cm1=100.0)
    download_module.download_hitran_h2o(tmp_path, upper_cm1=150.0, step_cm1=100.0)

    assert first == tmp_path / "hitran/h2o_pressure_shift.npz"
    assert calls == [(0.0, 100.0), (100.0, 150.0)]  # the raw ranges are kept and reused
    table = PressureShiftTable.load(first)
    assert table.wavenumber_cm1.tolist() == [50.0, 100.0, 120.0]  # the shared boundary line once
    np.testing.assert_allclose(table.delta_air, [-0.01, -0.02, -0.03])
    assert table.provenance["edition"] == download_module.HITRAN_EDITION
    assert set(table.provenance["par_sha256"]) == {"H2O_00000_00100.par", "H2O_00100_00150.par"}
