"""Update from GitHub Releases.

Publishing a release is the whole deployment: CI builds the installers, attaches
them plus a SHA256SUMS file, and installed copies find the new version from the
same place.

Two rules shape this module.

**Never execute an unverified download.** The update path runs an installer, so
the downloaded file is checked against the SHA256SUMS asset published in the same
release before anything is launched. No sums file, or a mismatch, means no
install -- the caller is pointed at the release page to do it by hand instead.

**Never make the tool hang or phone home unasked.** The check is a single request
with a short timeout that fails silently, and it is skipped entirely when
`SVRSPEC_NO_UPDATE_CHECK` is set -- an air-gapped server must not stall on a
socket that will never connect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__

REPO = "rokaproj/svrspec"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
SUMS_ASSET = "SHA256SUMS"

#: Set to any non-empty value to disable the check completely.
DISABLE_ENV = "SVRSPEC_NO_UPDATE_CHECK"

DEFAULT_TIMEOUT = 4.0
#: Refuse anything larger than this, so a wrong asset cannot fill the disk.
MAX_ASSET_BYTES = 200 * 1024 * 1024

USER_AGENT = f"svrspec/{__version__}"


class UpdateError(Exception):
    """Raised only on the install path, where silence would be wrong."""


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    notes: str
    page_url: str
    #: asset name -> download URL
    assets: dict[str, str]

    @property
    def installer_name(self) -> str | None:
        """Prefer the setup.exe; fall back to the MSI."""
        for suffix in ("-setup.exe", ".msi"):
            for name in self.assets:
                if name.endswith(suffix):
                    return name
        return None


def parse_version(text: str) -> tuple[int, ...]:
    """`v1.2.3` / `1.2.3` -> (1, 2, 3). Unparseable parts become 0."""
    cleaned = text.strip().lstrip("vV")
    parts = re.split(r"[.\-+]", cleaned)
    out: list[int] = []
    for part in parts:
        digits = re.match(r"\d+", part)
        if not digits:
            break
        out.append(int(digits.group()))
    return tuple(out) or (0,)


def is_newer(candidate: str, current: str = __version__) -> bool:
    return parse_version(candidate) > parse_version(current)


def _get(url: str, timeout: float, accept: str = "application/json") -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": accept}
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        try:
            length = int(response.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length > MAX_ASSET_BYTES:
            raise UpdateError(f"내려받을 파일이 너무 크다: {length:,} 바이트")
        blob = response.read(MAX_ASSET_BYTES + 1)
        if len(blob) > MAX_ASSET_BYTES:
            raise UpdateError(f"내려받은 파일이 너무 크다: {MAX_ASSET_BYTES:,} 바이트를 넘었다")
        return blob


def check(timeout: float = DEFAULT_TIMEOUT) -> Release | None:
    """The newest release if it is newer than this build, else None.

    Returns None on any network or parsing problem: a failed check must never
    become an error the user has to dismiss.
    """
    if os.environ.get(DISABLE_ENV):
        return None
    try:
        payload = json.loads(_get(API_LATEST, timeout))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, UpdateError):
        return None

    tag = str(payload.get("tag_name") or "")
    if not tag or not is_newer(tag):
        return None

    assets = {
        str(a.get("name")): str(a.get("browser_download_url"))
        for a in payload.get("assets") or []
        if a.get("name") and a.get("browser_download_url")
    }
    return Release(
        version=tag.lstrip("vV"),
        tag=tag,
        notes=str(payload.get("body") or "").strip(),
        page_url=str(payload.get("html_url") or RELEASES_PAGE),
        assets=assets,
    )


def _expected_sha256(release: Release, asset_name: str, timeout: float) -> str:
    """The published digest for one asset, from the release's SHA256SUMS."""
    if SUMS_ASSET not in release.assets:
        raise UpdateError(
            f"릴리스에 {SUMS_ASSET} 가 없어 무결성을 확인할 수 없다. "
            f"자동 설치를 중단한다 — {release.page_url} 에서 직접 받아라."
        )
    try:
        text = _get(release.assets[SUMS_ASSET], timeout, accept="text/plain").decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
        raise UpdateError(f"{SUMS_ASSET} 를 받지 못했다: {exc}") from exc

    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == asset_name:
            return parts[0].lower()
    raise UpdateError(f"{SUMS_ASSET} 에 {asset_name} 항목이 없다. 자동 설치를 중단한다.")


def download(
    release: Release,
    directory: Path | None = None,
    timeout: float = 120.0,
    asset_name: str | None = None,
) -> Path:
    """Fetch the installer and verify it against the published digest.

    Raises rather than returning a file that failed verification -- this file is
    about to be executed.
    """
    name = asset_name or release.installer_name
    if not name:
        raise UpdateError(
            f"릴리스에 Windows 설치 파일이 없다 — {release.page_url} 를 확인해 주세요."
        )

    if Path(name).is_absolute() or Path(name).name != name:
        raise UpdateError(f"설치 파일 이름은 경로가 아닌 파일 이름이어야 한다: {name!r}")
    if name not in release.assets:
        raise UpdateError(f"릴리스에 {name} 의 내려받기 주소가 없다")

    expected = _expected_sha256(release, name, timeout=min(timeout, 30.0))

    target_dir = directory or Path(os.environ.get("TEMP") or "/tmp") / "svrspec-update"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name

    try:
        blob = _get(release.assets[name], timeout, accept="application/octet-stream")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"설치 파일을 받지 못했다: {exc}") from exc

    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise UpdateError(
            "내려받은 파일의 SHA-256이 릴리스에 게시된 값과 다르다. 실행하지 않는다.\n"
            f"  기대 {expected}\n  실제 {actual}"
        )

    target.write_bytes(blob)
    return target


#: Inno Setup switches that turn "run an installer" into "apply an update".
#:
#:   /SILENT                 progress only, no wizard to click through
#:   /SUPPRESSMSGBOXES       no prompts behind a window that is about to close
#:   /NORESTART              never reboot the machine over an app update
#:   /CLOSEAPPLICATIONS      let it replace files this process is holding open
#:   /RESTARTAPPLICATIONS    and reopen the app afterwards
#:
#: The last two are what make it feel like an update rather than a reinstall:
#: the window closes, a progress bar runs, the window comes back on the new
#: version. Without them Windows cannot overwrite the running .exe and the
#: install fails or defers.
SILENT_SWITCHES = (
    "/SILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/CLOSEAPPLICATIONS",
    "/RESTARTAPPLICATIONS",
)


def launch_installer(path: Path, silent: bool = True) -> None:
    """Hand the verified installer to Windows and let this process exit.

    Silent by default. The user already consented by pressing the update
    button, and making them agree to a licence page and a destination folder
    they chose once already is not consent, it is friction.

    `silent=False` falls back to the ordinary wizard -- worth keeping for the
    case where a silent install fails and somebody needs to see why.
    """
    if os.name != "nt":
        raise UpdateError(f"이 설치 파일은 Windows 전용이다: {path}")
    if not silent:
        os.startfile(str(path))  # noqa: S606 - a verified installer, by design
        return

    # ShellExecute cannot pass arguments, so the silent path needs a real spawn.
    # Detached, because this process is about to be closed by the installer
    # itself and must not be waited on.
    import subprocess  # noqa: PLC0415 - only needed on this branch

    creation = 0
    for flag in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        creation |= getattr(subprocess, flag, 0)
    try:
        subprocess.Popen(  # noqa: S603 - a hash-verified installer, by design
            [str(path), *SILENT_SWITCHES],
            close_fds=True,
            creationflags=creation,
        )
    except OSError as exc:
        raise UpdateError(f"설치 프로그램을 실행할 수 없다: {exc}") from exc


def describe(release: Release | None) -> str:
    if release is None:
        return f"최신 버전이다 (v{__version__})"
    return f"업데이트 있음: v{__version__} → {release.tag}"
