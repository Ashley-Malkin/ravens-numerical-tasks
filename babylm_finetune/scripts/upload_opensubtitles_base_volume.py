#!/usr/bin/env python3
"""Put OpenSubtitles GPT-2-small 20M finals onto Modal Volume ``ravens-os-base``.

Uploads local folders under ``open-subtitles-models/`` (default)::

    modal run babylm_finetune/scripts/upload_opensubtitles_base_volume.py
    modal run babylm_finetune/scripts/upload_opensubtitles_base_volume.py --seeds 42
    modal run babylm_finetune/scripts/upload_opensubtitles_base_volume.py \\
      --from-local /path/to/open-subtitles-models
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
    OS_BASE_VOLUME_MOUNT,
    OS_BASE_VOLUME_NAME,
    OS_FOLDER_BY_SEED,
    OS_LOCAL_ROOT,
    OS_SEEDS,
    assert_hf_checkpoint_dir,
    os_folder_name,
)

app = modal.App("ravens-os-base-upload")

os_base_volume = modal.Volume.from_name(OS_BASE_VOLUME_NAME, create_if_missing=True)


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return OS_SEEDS
    out: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        seed = int(s)
        if seed not in OS_FOLDER_BY_SEED:
            raise ValueError(f"Unknown OpenSubtitles seed {seed!r}; expected one of {OS_SEEDS}")
        out.append(seed)
    if not out:
        raise ValueError("No seeds requested")
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
    if root.name == folder_name and root.is_dir():
        return root
    found = _find_folder(root, folder_name)
    if found is not None:
        return found
    if (root / "config.json").is_file() and folder_name in str(root):
        return root
    return None


def _upload_local_to_volume(local_root: Path, seeds: list[int]) -> dict:
    if not local_root.is_dir():
        raise FileNotFoundError(f"--from-local path not found: {local_root}")

    summary: dict[str, str] = {}
    print(
        f"Uploading from {local_root} → Volume {OS_BASE_VOLUME_NAME!r}",
        flush=True,
    )
    with os_base_volume.batch_upload(force=True) as batch:
        for seed in seeds:
            folder = os_folder_name(seed)
            src = _resolve_local_checkpoint(local_root, folder)
            if src is None:
                raise FileNotFoundError(
                    f"Missing {folder!r} under {local_root}."
                )
            assert_hf_checkpoint_dir(src, label="OpenSubtitles checkpoint")
            remote = f"/{folder}"
            print(f"  {src} → {remote}", flush=True)
            batch.put_directory(str(src), remote)
            summary[str(seed)] = f"{OS_BASE_VOLUME_MOUNT}/{folder}"

    manifest = {
        "volume": OS_BASE_VOLUME_NAME,
        "mount": OS_BASE_VOLUME_MOUNT,
        "source": "local",
        "local_root": str(local_root.resolve()),
        "seeds": list(seeds),
        "paths": summary,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.local_entrypoint()
def main(
    seeds: Optional[str] = None,
    from_local: Optional[str] = None,
) -> None:
    """Upload OpenSubtitles finals onto Volume ``ravens-os-base``.

    ``seeds`` — comma-separated list (default: 0,42,123).
    ``from_local`` — checkpoint parent (default: repo ``open-subtitles-models/``).
    """
    parsed = list(_parse_seeds(seeds))
    local_root = Path(from_local).expanduser() if from_local else OS_LOCAL_ROOT
    result = _upload_local_to_volume(local_root, parsed)

    print("\n=== OpenSubtitles base upload complete ===", flush=True)
    for seed, path in sorted(result.get("paths", {}).items(), key=lambda x: int(x[0])):
        print(f"  seed {seed}: {path}", flush=True)
    print(
        f"Models ready at {OS_BASE_VOLUME_MOUNT}/<folder> for SFT/eval.",
        flush=True,
    )
