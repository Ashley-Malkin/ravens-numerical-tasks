#!/usr/bin/env python3
"""Put TinyStories GPT-2-small 10M finals onto Modal Volume ``ravens-ts-base``.

Uploads local folder under ``tiny-stories-model/`` (default)::

    modal run babylm_finetune/scripts/upload_tinystories_base_volume.py
    modal run babylm_finetune/scripts/upload_tinystories_base_volume.py \\
      --from-local /path/to/tiny-stories-model
"""

from __future__ import annotations

import json
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
    TS_BASE_VOLUME_MOUNT,
    TS_BASE_VOLUME_NAME,
    TS_FOLDER_NAME,
    TS_LOCAL_ROOT,
    assert_hf_checkpoint_dir,
    ts_folder_name,
)

app = modal.App("ravens-ts-base-upload")

ts_base_volume = modal.Volume.from_name(TS_BASE_VOLUME_NAME, create_if_missing=True)


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
    if root.name == folder_name and root.is_dir():
        return root
    found = _find_folder(root, folder_name)
    if found is not None:
        return found
    if (root / "config.json").is_file() and folder_name in str(root):
        return root
    return None


def _upload_local_to_volume(local_root: Path) -> dict:
    if not local_root.is_dir():
        raise FileNotFoundError(f"--from-local path not found: {local_root}")

    folder = ts_folder_name()
    print(
        f"Uploading from {local_root} → Volume {TS_BASE_VOLUME_NAME!r}",
        flush=True,
    )
    src = _resolve_local_checkpoint(local_root, folder)
    if src is None:
        raise FileNotFoundError(f"Missing {folder!r} under {local_root}.")
    assert_hf_checkpoint_dir(src, label="TinyStories checkpoint")
    remote = f"/{folder}"
    with ts_base_volume.batch_upload(force=True) as batch:
        print(f"  {src} → {remote}", flush=True)
        batch.put_directory(str(src), remote)

    manifest = {
        "volume": TS_BASE_VOLUME_NAME,
        "mount": TS_BASE_VOLUME_MOUNT,
        "source": "local",
        "local_root": str(local_root.resolve()),
        "folder": folder,
        "paths": {folder: f"{TS_BASE_VOLUME_MOUNT}/{folder}"},
    }
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.local_entrypoint()
def main(from_local: Optional[str] = None) -> None:
    """Upload TinyStories final onto Volume ``ravens-ts-base``.

    ``from_local`` — checkpoint parent (default: repo ``tiny-stories-model/``).
    """
    local_root = Path(from_local).expanduser() if from_local else TS_LOCAL_ROOT
    result = _upload_local_to_volume(local_root)

    print("\n=== TinyStories base upload complete ===", flush=True)
    for name, path in sorted(result.get("paths", {}).items()):
        print(f"  {name}: {path}", flush=True)
    print(
        f"Model ready at {TS_BASE_VOLUME_MOUNT}/{TS_FOLDER_NAME} for SFT/eval.",
        flush=True,
    )
