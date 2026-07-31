"""Updater: version comparison, integrity refusal, and silent-failure behaviour.

The install path executes a downloaded file, so the tests that matter most here
are the ones proving it refuses to when the digest cannot be verified.
"""

import hashlib
import json

import pytest

from svrspec import __version__, update
from svrspec.update import Release, UpdateError, check, download, is_newer, parse_version


def _release(assets: dict[str, str], tag: str = "v9.9.9") -> Release:
    return Release(version=tag.lstrip("v"), tag=tag, notes="", page_url="https://example/r",
                   assets=assets)


# -- version comparison ---------------------------------------------------


def test_parse_version_handles_the_shapes_github_tags_take():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("v0.1.0-rc1") == (0, 1, 0)
    assert parse_version("garbage") == (0,)


def test_is_newer_orders_correctly():
    assert is_newer("v0.2.0", "0.1.0")
    assert is_newer("v0.1.1", "0.1.0")
    assert is_newer("v1.0.0", "0.9.9")
    assert not is_newer("v0.1.0", "0.1.0")
    assert not is_newer("v0.0.9", "0.1.0")


def test_the_shipped_version_is_not_newer_than_itself():
    assert not is_newer(__version__)


# -- the check is allowed to fail, never to raise --------------------------


def test_check_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv(update.DISABLE_ENV, "1")

    def explode(*a, **k):
        raise AssertionError("the check must not touch the network when disabled")

    monkeypatch.setattr(update, "_get", explode)
    assert check() is None


def test_check_returns_none_when_offline(monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)

    def offline(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(update, "_get", offline)
    assert check() is None


def test_check_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    monkeypatch.setattr(update, "_get", lambda *a, **k: b"not json")
    assert check() is None


def test_check_ignores_an_older_or_equal_release(monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    payload = json.dumps({"tag_name": f"v{__version__}", "assets": []}).encode()
    monkeypatch.setattr(update, "_get", lambda *a, **k: payload)
    assert check() is None


def test_check_reports_a_newer_release(monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    payload = json.dumps({
        "tag_name": "v99.0.0",
        "body": "새 릴리스",
        "html_url": "https://github.com/rokaproj/svrspec/releases/tag/v99.0.0",
        "assets": [
            {"name": "svrspec-99.0.0-setup.exe", "browser_download_url": "https://x/setup.exe"},
            {"name": "svrspec-99.0.0-win64.msi", "browser_download_url": "https://x/app.msi"},
            {"name": "SHA256SUMS", "browser_download_url": "https://x/sums"},
        ],
    }).encode()
    monkeypatch.setattr(update, "_get", lambda *a, **k: payload)

    r = check()
    assert r is not None
    assert r.tag == "v99.0.0"
    # setup.exe wins over the MSI when both are published.
    assert r.installer_name == "svrspec-99.0.0-setup.exe"


def test_installer_falls_back_to_the_msi():
    r = _release({"svrspec-9-win64.msi": "https://x/app.msi", "SHA256SUMS": "https://x/s"})
    assert r.installer_name == "svrspec-9-win64.msi"


# -- integrity: the part that must not be lenient -------------------------


def test_download_refuses_without_a_sums_file(monkeypatch, tmp_path):
    r = _release({"svrspec-9-setup.exe": "https://x/setup.exe"})
    monkeypatch.setattr(update, "_get", lambda *a, **k: b"payload")
    with pytest.raises(UpdateError, match="SHA256SUMS"):
        download(r, tmp_path)


def test_download_refuses_on_a_digest_mismatch(monkeypatch, tmp_path):
    r = _release({"svrspec-9-setup.exe": "https://x/setup.exe", "SHA256SUMS": "https://x/s"})

    def fake_get(url, timeout, accept="application/json"):
        if url.endswith("/s"):
            return b"0" * 64 + b"  svrspec-9-setup.exe\n"
        return b"tampered installer bytes"

    monkeypatch.setattr(update, "_get", fake_get)
    with pytest.raises(UpdateError, match="SHA-256"):
        download(r, tmp_path)
    assert not list(tmp_path.iterdir())  # nothing written when it fails


def test_download_refuses_when_the_asset_is_absent_from_sums(monkeypatch, tmp_path):
    r = _release({"svrspec-9-setup.exe": "https://x/setup.exe", "SHA256SUMS": "https://x/s"})

    def fake_get(url, timeout, accept="application/json"):
        if url.endswith("/s"):
            return b"abc  some-other-file.exe\n"
        return b"bytes"

    monkeypatch.setattr(update, "_get", fake_get)
    with pytest.raises(UpdateError, match="항목이 없다"):
        download(r, tmp_path)


def test_download_writes_the_file_when_the_digest_matches(monkeypatch, tmp_path):
    blob = b"a genuine installer"
    digest = hashlib.sha256(blob).hexdigest()
    r = _release({"svrspec-9-setup.exe": "https://x/setup.exe", "SHA256SUMS": "https://x/s"})

    def fake_get(url, timeout, accept="application/json"):
        if url.endswith("/s"):
            # sha256sum's own format uses a '*' marker for binary mode.
            return f"{digest} *svrspec-9-setup.exe\n".encode()
        return blob

    monkeypatch.setattr(update, "_get", fake_get)
    out = download(r, tmp_path)
    assert out.read_bytes() == blob
    assert out.name == "svrspec-9-setup.exe"


def test_download_refuses_a_release_with_no_installer(monkeypatch, tmp_path):
    r = _release({"SHA256SUMS": "https://x/s", "notes.txt": "https://x/n"})
    with pytest.raises(UpdateError, match="설치 파일이 없다"):
        download(r, tmp_path)


def test_launch_installer_refuses_off_windows(tmp_path):
    import os

    if os.name == "nt":
        pytest.skip("this guard only applies off Windows")
    with pytest.raises(UpdateError, match="Windows"):
        update.launch_installer(tmp_path / "setup.exe")


# -- the GUI payload ------------------------------------------------------


def test_update_payload_shape_when_disabled(monkeypatch):
    from svrspec.gui import update_payload

    monkeypatch.setenv(update.DISABLE_ENV, "1")
    d = update_payload()
    assert d == {"available": False, "current": __version__}


def test_update_payload_shape_when_available(monkeypatch):
    from svrspec.gui import update_payload

    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    monkeypatch.setattr(update, "check", lambda *a, **k: _release(
        {"svrspec-9-setup.exe": "https://x/e", "SHA256SUMS": "https://x/s"}))
    d = update_payload()
    assert d["available"] and d["tag"] == "v9.9.9"
    assert d["installer"] == "svrspec-9-setup.exe"


def test_the_packaging_keeps_pywebview_unzipped():
    """pywebview reads its own JS off the filesystem, so it cannot be zipped.

    `webview.util.load_js_files` globs `webview/js/**/*.js` and concatenates
    what it finds. Inside cx_Freeze's library.zip that glob matches nothing:
    `api.js` never runs, `window.pywebview` is never created, `pywebviewready`
    never fires. The import still succeeds and the window still opens -- it
    just renders a page whose dropdowns never fill and whose bridge times out.
    A release shipped exactly that.

    The same rule covers `svrspec` itself, whose catalogue loader resolves JSON
    with `Path(__file__).parent`.
    """
    import ast
    from pathlib import Path

    setup_py = Path(__file__).resolve().parent.parent / "installer" / "setup_cxfreeze.py"
    tree = ast.parse(setup_py.read_text(encoding="utf-8"))

    # Only this one key is read: the dict also holds `str(ROOT / ...)` calls,
    # which literal_eval cannot evaluate and which this test does not need.
    unzipped: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "build_exe_options" for t in node.targets
        )):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == "zip_exclude_packages":
                unzipped = set(ast.literal_eval(value))
    assert unzipped, "zip_exclude_packages not found in build_exe_options"
    assert "webview" in unzipped, (
        "pywebview must stay on disk or its JS bridge never loads"
    )
    assert "svrspec" in unzipped, "the catalogue is read from disk"


def test_the_release_smoke_test_checks_the_bridge_files():
    """Opening a window proves nothing; the bridge can be dead behind it.

    The smoke test used to assert only that the process was still alive after
    twelve seconds, which a build with no JS bridge passes.
    """
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")

    assert "webview\\js" in workflow or "webview/js" in workflow
    assert "api.js" in workflow and "finish.js" in workflow


def test_the_installer_runs_silently_so_an_update_is_not_a_reinstall(monkeypatch, tmp_path):
    """Pressing the update button must not open a setup wizard.

    The user already consented by pressing it. Making them agree to a licence
    page and re-pick a destination they chose once is friction, not consent.
    Silent plus Restart Manager is what turns "run an installer" into "the
    window closes and comes back updated".
    """
    from svrspec import update

    installer = tmp_path / "svrspec-9.9.9-setup.exe"
    installer.write_bytes(b"MZ")

    seen: dict = {}

    class _Popen:
        def __init__(self, args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr(update.os, "name", "nt")
    monkeypatch.setattr("subprocess.Popen", _Popen)
    update.launch_installer(installer)

    assert seen["args"][0] == str(installer)
    switches = seen["args"][1:]
    for required in ("/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"):
        assert required in switches, switches
    # An app update may never reboot the machine.
    assert "/NORESTART" in switches
    # Detached: the installer is about to close this very process.
    assert seen["kwargs"].get("close_fds") is True


def test_the_wizard_is_still_reachable_when_silent_is_refused(monkeypatch, tmp_path):
    """Somebody debugging a failed silent install needs to see the wizard."""
    from svrspec import update

    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"MZ")
    opened: list = []

    monkeypatch.setattr(update.os, "name", "nt")
    monkeypatch.setattr(update.os, "startfile", opened.append, raising=False)
    update.launch_installer(installer, silent=False)
    assert opened == [str(installer)]


def test_the_installer_declares_it_can_be_closed_and_restarted():
    """The switches only work if Setup was built to honour Restart Manager."""
    from pathlib import Path

    iss = (
        Path(__file__).resolve().parent.parent / "installer" / "svrspec.iss"
    ).read_text(encoding="utf-8")

    assert "CloseApplications=yes" in iss
    assert "RestartApplications=yes" in iss
    assert "AlwaysRestart=no" in iss


def test_a_silent_launch_failure_is_reported_not_swallowed(monkeypatch, tmp_path):
    from svrspec import update

    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"MZ")

    def _boom(*args, **kwargs):
        raise OSError("access denied")

    monkeypatch.setattr(update.os, "name", "nt")
    monkeypatch.setattr("subprocess.Popen", _boom)
    with pytest.raises(update.UpdateError, match="실행할 수 없다"):
        update.launch_installer(installer)


def test_the_build_copies_the_webview_js_and_refuses_to_ship_without_it():
    """Unzipping the package is necessary but not sufficient.

    cx_Freeze copies modules, not the data files sitting beside them, so
    `webview/js/*.js` has to be named in `include_files` the same way the
    catalogue is. And because a build missing them fails *silently* -- the
    window opens and simply never fills in -- the build must refuse rather
    than produce that installer.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "installer" / "setup_cxfreeze.py"
    ).read_text(encoding="utf-8")

    assert "_webview_js_dir()" in source
    assert '"lib/webview/js"' in source
    # The guard, not just the copy.
    assert "raise SystemExit" in source
    assert "api.js" in source and "finish.js" in source
