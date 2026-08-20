from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services.image_signature import (
    create_signed_record,
    embed_signature_in_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PREDICTOR_SCRIPT = PROJECT_ROOT.parent / "inject_wm_with_predictor.py"
ROBIN_SCRIPT = PROJECT_ROOT.parent / "inject_wm_baseline.py"

WM_PATH = (
    PROJECT_ROOT.parent
    / "ckpts"
    / "optimized_wm5-30_embedding-step-50.pt"
)

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# We are not building accounts/login for the dissertation prototype.
# This identifies images created through this web application.
DEFAULT_CREATOR_ID = "WEBAPP-USER"


@dataclass
class GenerationResult:
    prompt: str
    method: str
    seed: Optional[str]
    runtime_sec: float
    predictor_inference_sec: Optional[float]
    output_image: Optional[str]
    image_id: Optional[str]
    signature_added: bool
    log_tail: str
    ok: bool


def _run_command(cmd: list[str], cwd: Path):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    return proc.stdout, proc.returncode


def _find_latest_image(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None

    candidates = []

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(folder.rglob(ext))

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def _create_signed_output(
    source_image: Path,
    prompt: str,
    method: str,
) -> tuple[Path, str]:
    """
    Create the final PNG used by the web application.

    The image pixels are unchanged. A signed Image Signature
    record is embedded in the PNG metadata.
    """

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = int(time.time())

    output_name = (
        f"generated_{timestamp}_signed.png"
    )

    destination = OUTPUTS_DIR / output_name

    # --------------------------------------------------
    # Create cryptographically signed image record
    # --------------------------------------------------

    signed_record = create_signed_record(
        image_path=source_image,
        creator_id=DEFAULT_CREATOR_ID,
        method=method,
        prompt=prompt,
    )

    # --------------------------------------------------
    # Embed the record into the final PNG
    # --------------------------------------------------

    embed_signature_in_png(
        source_image=source_image,
        destination_image=destination,
        signed_record=signed_record,
    )

    image_id = signed_record["record"]["image_id"]

    return destination, image_id


def run_generation(
    prompt: str,
    method: str,
    seed: Optional[str] = None,
) -> GenerationResult:
    """
    Run ROBIN or the predictor-backed ROBIN pipeline.

    After generation, the final web-app output is converted
    to PNG and receives a cryptographically signed Image
    Signature.

    ROBIN watermarking and Image Signature remain separate.
    """

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save prompt for the existing generation scripts
    # --------------------------------------------------

    prompt_file = (
        RUNTIME_DIR
        / f"prompt_{int(time.time())}.txt"
    )

    prompt_file.write_text(
        prompt.strip() + "\n",
        encoding="utf-8",
    )

    method = (
        method or "predictor"
    ).lower().strip()

    # --------------------------------------------------
    # Select generation method
    # --------------------------------------------------

    if method == "robin":
        script_path = ROBIN_SCRIPT
    else:
        method = "predictor"
        script_path = PREDICTOR_SCRIPT

    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing generation script: {script_path}"
        )

    if not WM_PATH.exists():
        raise FileNotFoundError(
            f"Missing watermark checkpoint: {WM_PATH}"
        )

    # --------------------------------------------------
    # Build command
    # --------------------------------------------------

    cmd = [
        sys.executable,
        str(script_path),

        "--dataset",
        "custom",

        "--prompt_file",
        str(prompt_file),

        "--start",
        "0",

        "--end",
        "1",

        "--model_id",
        MODEL_ID,

        "--wm_path",
        str(WM_PATH),
    ]

    if seed:
        cmd.extend(
            [
                "--gen_seed",
                str(seed),
            ]
        )

    # --------------------------------------------------
    # Run generation
    # --------------------------------------------------

    start = time.time()

    log, return_code = _run_command(
        cmd,
        cwd=PROJECT_ROOT.parent,
    )

    runtime_sec = time.time() - start

    # --------------------------------------------------
    # Extract predictor inference time
    # --------------------------------------------------

    predictor_inference_sec = None

    match = re.search(
        r"PREDICTOR_INFERENCE_TIME=([0-9.]+)",
        log,
    )

    if match:
        predictor_inference_sec = float(
            match.group(1)
        )

    # --------------------------------------------------
    # Find generated ROBIN image
    # --------------------------------------------------

    latest_img = _find_latest_image(
        PROJECT_ROOT.parent
        / "generated_imgs"
    )

    output_image = None
    image_id = None
    signature_added = False

    # --------------------------------------------------
    # Create signed final PNG
    # --------------------------------------------------

    if (
        return_code == 0
        and latest_img is not None
    ):
        try:
            signed_output, image_id = (
                _create_signed_output(
                    source_image=latest_img,
                    prompt=prompt,
                    method=method,
                )
            )

            output_image = (
                signed_output.name
            )

            signature_added = True

        except Exception as exc:
            # Generation succeeded, but signing failed.
            # We deliberately report this rather than
            # pretending the image was successfully signed.
            log += (
                "\nIMAGE_SIGNATURE_ERROR="
                + str(exc)
            )

    # --------------------------------------------------
    # Overall status
    # --------------------------------------------------

    ok = (
        return_code == 0
        and output_image is not None
        and signature_added
    )

    # --------------------------------------------------
    # Return result to Flask
    # --------------------------------------------------

    return GenerationResult(
        prompt=prompt,
        method=method,
        seed=seed,
        runtime_sec=runtime_sec,
        predictor_inference_sec=predictor_inference_sec,
        output_image=output_image,
        image_id=image_id,
        signature_added=signature_added,
        log_tail=log[-4000:],
        ok=ok,
    )
