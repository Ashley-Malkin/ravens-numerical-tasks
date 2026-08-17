#!/usr/bin/env python3
"""Put TinyDialogues GPT-2 finals onto Modal Volume ``ravens-td-base``.

Two modes:

1. **From local folders** (recommended when Drive blocks unauthenticated
   ``gdown``) — upload already-downloaded finals from your machine::

    modal run babylm_finetune/scripts/upload_td_base_volume.py \\
      --from-local /path/to/TD_parent_or_checkpoints

2. **From Google Drive on Modal** — requires the folder to be shared as
   *Anyone with the link* (viewer)::

    modal run babylm_finetune/scripts/upload_td_base_volume.py
    modal run babylm_finetune/scripts/upload_td_base_volume.py --budgets 10M,50M

``--from-local`` may point at either the Drive parent folder (containing
``GPT2-small_TD_*`` children) or a directory that *is* one of those folders.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import modal

_SCRIPT_DIR = Path(__file__).resolve().parent
_BABYLM_FT_DIR = _SCRIPT_DIR.parent
REPO_ROOT = _BABYLM_FT_DIR.parent
_SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ravens_numerical.models.registry import (  # noqa: E402
    TD_BASE_VOLUME_MOUNT,
    TD_BASE_VOLUME_NAME,
    TD_BUDGETS,
    TD_DRIVE_PARENT_ID,
    TD_FOLDER_BY_BUDGET,
    assert_td_checkpoint_dir,
    td_folder_name,
)
from ravens_numerical.paths import CONTAINER_RAVENS_ROOT  # noqa: E402

STAGING_ROOT = Path("/tmp/td_drive_staging")

_DRIVE_SHARE_HINT = (
    "Google Drive blocked unauthenticated download (gdown). "
    "Either share the folder as 'Anyone with the link' (Viewer), or download "
    "the five final GPT2-small_TD_* folders in a browser and upload with:\n"
    "  modal run babylm_finetune/scripts/upload_td_base_volume.py "
    "--from-local /path/to/folder\n"
    f"Drive folder: https://drive.google.com/drive/folders/{TD_DRIVE_PARENT_ID}"
)

app = modal.App("ravens-td-base-upload")

td_base_volume = modal.Volume.from_name(TD_BASE_VOLUME_NAME, create_if_missing=True)

upload_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gdown>=5.2", "requests>=2.31")
    .env({"PYTHONPATH": f"{CONTAINER_RAVENS_ROOT}:{CONTAINER_RAVENS_ROOT}/src"})
    .add_local_dir(
        str(REPO_ROOT / "src" / "ravens_numerical"),
        remote_path=f"{CONTAINER_RAVENS_ROOT}/src/ravens_numerical",
        copy=True,
    )
)


def _parse_budgets(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return TD_BUDGETS
    out: list[str] = []
    for part in raw.split(","):
        b = part.strip()
        if not b:
            continue
        if b.lower().endswith("m") and not b.endswith("M"):
            b = b[:-1] + "M"
        if b not in TD_FOLDER_BY_BUDGET:
            raise ValueError(f"Unknown budget {b!r}; expected one of {TD_BUDGETS}")
        out.append(b)
    if not out:
        raise ValueError("No budgets requested")
    return tuple(out)


def _find_folder(staging: Path, folder_name: str) -> Path | None:
    direct = staging / folder_name
    if direct.is_dir():
        return direct
    matches = [p for p in staging.rglob(folder_name) if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _resolve_local_checkpoint(root: Path, folder_name: str) -> Path | None:
    """Locate a final TD folder under ``root`` (parent or the folder itself)."""
    if root.name == folder_name and root.is_dir():
        return root
    found = _find_folder(root, folder_name)
    if found is not None:
        return found
    # Flat layout: root contains config.json for a single budget.
    if (root / "config.json").is_file() and folder_name in str(root):
        return root
    return None


def _list_drive_child_folders(parent_id: str) -> dict[str, str]:
    """Map folder name → Drive file id via the public Drive API (best effort)."""
    import requests

    name_to_id: dict[str, str] = {}
    page_token: str | None = None
    while True:
        params: dict[str, str] = {
            "q": f"'{parent_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": "100",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=60,
        )
        if resp.status_code != 200:
            print(
                f"Drive list HTTP {resp.status_code}: {resp.text[:300]} "
                "(will fall back to full-folder gdown)",
                flush=True,
            )
            return {}
        payload = resp.json()
        for item in payload.get("files", []):
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                name_to_id[item["name"]] = item["id"]
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return name_to_id


def _gdown_folder(url: str, output: Path) -> None:
    """Call gdown; re-raise Drive/auth failures as plain RuntimeError."""
    import gdown

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"gdown → {output}", flush=True)
    print(f"  {url}", flush=True)
    try:
        gdown.download_folder(
            url=url,
            output=str(output),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:  # noqa: BLE001 — map any gdown failure for Modal
        raise RuntimeError(f"{_DRIVE_SHARE_HINT}\n\nUnderlying error: {exc}") from None


def _upload_local_to_volume(local_root: Path, budgets: list[str]) -> dict:
    """Client-side Volume upload from already-downloaded checkpoint dirs."""
    if not local_root.is_dir():
        raise FileNotFoundError(f"--from-local path not found: {local_root}")

    summary: dict[str, str] = {}
    print(
        f"Uploading from {local_root} → Volume {TD_BASE_VOLUME_NAME!r}",
        flush=True,
    )
    with td_base_volume.batch_upload(force=True) as batch:
        for budget in budgets:
            folder = td_folder_name(budget)
            src = _resolve_local_checkpoint(local_root, folder)
            if src is None:
                raise FileNotFoundError(
                    f"Missing {folder!r} under {local_root}. "
                    "Download that final checkpoint folder (not int-ckpt) first."
                )
            assert_td_checkpoint_dir(src)
            remote = f"/{folder}"
            print(f"  {src} → {remote}", flush=True)
            batch.put_directory(str(src), remote)
            summary[budget] = f"{TD_BASE_VOLUME_MOUNT}/{folder}"

    manifest = {
        "volume": TD_BASE_VOLUME_NAME,
        "mount": TD_BASE_VOLUME_MOUNT,
        "source": "local",
        "local_root": str(local_root.resolve()),
        "budgets": list(budgets),
        "paths": summary,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.function(
    image=upload_image,
    volumes={TD_BASE_VOLUME_MOUNT: td_base_volume},
    timeout=6 * 60 * 60,
    # Modal requires ephemeral_disk in [524288, 3145728] MiB (512 GiB–3 TiB).
    ephemeral_disk=524288,
)
def fetch_td_to_volume(budgets: list[str]) -> dict:
    """Download requested TD finals from Drive onto ``/td-base``."""
    try:
        return _fetch_td_to_volume_impl(budgets)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — keep Modal client deserializable
        raise RuntimeError(f"{_DRIVE_SHARE_HINT}\n\nUnderlying error: {exc}") from None


def _fetch_td_to_volume_impl(budgets: list[str]) -> dict:
    td_base_volume.reload()
    dest_root = Path(TD_BASE_VOLUME_MOUNT)
    dest_root.mkdir(parents=True, exist_ok=True)

    wanted = {b: td_folder_name(b) for b in budgets}
    wanted_names = set(wanted.values())
    summary: dict[str, str] = {}

    name_to_id = _list_drive_child_folders(TD_DRIVE_PARENT_ID)
    per_folder = bool(name_to_id) and wanted_names.issubset(name_to_id.keys())

    if per_folder:
        print(
            f"Downloading {len(wanted)} final folder(s) by Drive id "
            "(skipping int-ckpt / data / other)",
            flush=True,
        )
        for budget, folder in wanted.items():
            dest = dest_root / folder
            if dest.exists():
                shutil.rmtree(dest)
            staging = STAGING_ROOT / folder
            if staging.exists():
                shutil.rmtree(staging)
            _gdown_folder(
                f"https://drive.google.com/drive/folders/{name_to_id[folder]}",
                staging,
            )
            src = _find_folder(staging, folder) or staging
            if not (src / "config.json").is_file() and (staging / "config.json").is_file():
                src = staging
            shutil.copytree(src, dest)
            assert_td_checkpoint_dir(dest)
            summary[budget] = str(dest)
            shutil.rmtree(staging, ignore_errors=True)
    else:
        print(
            "Per-folder Drive ids unavailable; downloading parent folder to staging "
            "then copying finals only.",
            flush=True,
        )
        if STAGING_ROOT.exists():
            shutil.rmtree(STAGING_ROOT)
        _gdown_folder(
            f"https://drive.google.com/drive/folders/{TD_DRIVE_PARENT_ID}",
            STAGING_ROOT,
        )
        missing: list[str] = []
        for budget, folder in wanted.items():
            src = _find_folder(STAGING_ROOT, folder)
            if src is None:
                missing.append(folder)
                continue
            dest = dest_root / folder
            if dest.exists():
                shutil.rmtree(dest)
            print(f"Copying {folder} → {dest}", flush=True)
            shutil.copytree(src, dest)
            assert_td_checkpoint_dir(dest)
            summary[budget] = str(dest)
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)
        if missing:
            raise RuntimeError(
                "Missing final TD folders after Drive download:\n  - "
                + "\n  - ".join(missing)
            )

    td_base_volume.commit()
    manifest = {
        "volume": TD_BASE_VOLUME_NAME,
        "mount": TD_BASE_VOLUME_MOUNT,
        "drive_parent_id": TD_DRIVE_PARENT_ID,
        "budgets": list(budgets),
        "paths": summary,
        "per_folder_download": per_folder,
        "source": "gdrive",
    }
    manifest_path = dest_root / "upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    td_base_volume.commit()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.local_entrypoint()
def main(
    budgets: Optional[str] = None,
    from_local: Optional[str] = None,
) -> None:
    """Upload TD finals onto Volume ``ravens-td-base``.

    ``budgets`` — comma-separated list (default: all five).
    ``from_local`` — path to downloaded checkpoints (skips Drive/gdown).
    """
    parsed = list(_parse_budgets(budgets))
    if from_local:
        result = _upload_local_to_volume(Path(from_local).expanduser(), parsed)
    else:
        print(
            f"Fetching TinyDialogues budgets {parsed} from Drive → "
            f"Volume {TD_BASE_VOLUME_NAME!r}",
            flush=True,
        )
        result = fetch_td_to_volume.remote(parsed)

    print("\n=== TD base upload complete ===", flush=True)
    for budget, path in sorted(result.get("paths", {}).items()):
        print(f"  {budget}: {path}", flush=True)
    print(
        f"Models ready at {TD_BASE_VOLUME_MOUNT}/<folder> for SFT/eval.",
        flush=True,
    )
