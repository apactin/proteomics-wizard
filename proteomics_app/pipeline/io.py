from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import re


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    logs_dir: Path


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "run"


def make_run_dir(
    base_dir: Optional[Union[str, Path]] = None,
    *,
    prefix: str = "run",
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> RunContext:
    """
    If base_dir is None, defaults to ./runs (next to where Streamlit was launched).
    """
    if base_dir is None:
        base_dir = Path.cwd() / "runs"
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    name_parts = [prefix]
    if run_name:
        name_parts.append(_slugify(run_name))

    run_dir = base_dir / f"{timestamp}_{'_'.join(name_parts)}"

    uploads_dir = run_dir / "uploads"
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"

    for p in (uploads_dir, outputs_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_dir=run_dir,
        uploads_dir=uploads_dir,
        outputs_dir=outputs_dir,
        logs_dir=logs_dir,
    )


def save_uploaded_file_as(
    uploaded_file,
    dest_path: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Save a Streamlit UploadedFile to dest_path.
    Returns dest_path.

    uploaded_file: streamlit.runtime.uploaded_file_manager.UploadedFile
      (but we keep it untyped to avoid importing streamlit here).
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and (not overwrite):
        raise FileExistsError(f"Destination already exists: {dest_path}")

    # Streamlit UploadedFile supports getbuffer()
    data = uploaded_file.getbuffer()
    with open(dest_path, "wb") as f:
        f.write(data)

    return dest_path
