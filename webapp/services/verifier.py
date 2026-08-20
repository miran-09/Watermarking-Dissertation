from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler

# ---------------------------------------------------------
# Make the ROBIN project importable from the webapp
# ---------------------------------------------------------

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
ROBIN_ROOT = WEBAPP_ROOT.parent

if str(ROBIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBIN_ROOT))

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from optim_utils import get_watermarking_mask, transform_img


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "runwayml/stable-diffusion-v1-5"

WATERMARK_CHECKPOINT = (
    ROBIN_ROOT
    / "ckpts"
    / "optimized_wm5-30_embedding-step-50.pt"
)

WATERMARK_STEP = 35
TEST_INFERENCE_STEPS = 50
DETECTION_THRESHOLD = 37.5


# ---------------------------------------------------------
# Result object
# ---------------------------------------------------------

@dataclass
class VerificationResult:
    filename: str
    prompt: str
    detection_score: float
    verification_time_sec: float

    # We deliberately leave this optional until the
    # threshold has been calibrated on real data.
    detected: Optional[bool]

    threshold: Optional[float]
    message: str


# ---------------------------------------------------------
# ROBIN configuration
# ---------------------------------------------------------

def _build_robin_args() -> SimpleNamespace:
    """
    Configuration matching the watermark settings used
    by the dissertation evaluation scripts.
    """

    return SimpleNamespace(
        # watermark configuration
        w_seed=999999,
        w_channel=0,
        w_pattern="rand",
        w_mask_shape="circle",
        w_up_radius=30,
        w_low_radius=5,
        w_measurement="l1_complex",
        w_injection="complex",
        w_pattern_const=0.0,

        # required internally by some ROBIN helpers
        w_radius=10,

        # diffusion verification settings
        num_inference_steps=TEST_INFERENCE_STEPS,
        test_num_inference_steps=TEST_INFERENCE_STEPS,

        # image settings
        image_length=512,
    )


# ---------------------------------------------------------
# Load the ROBIN pipeline
# ---------------------------------------------------------

_PIPELINE = None
_DEVICE = None
_WATERMARK = None
_MASK = None
_ARGS = None


def _load_robin_components():
    """
    Load the diffusion pipeline, optimized watermark,
    and watermarking mask once and reuse them for
    subsequent verification requests.
    """

    global _PIPELINE
    global _DEVICE
    global _WATERMARK
    global _MASK
    global _ARGS

    if (
        _PIPELINE is not None
        and _WATERMARK is not None
        and _MASK is not None
        and _ARGS is not None
    ):
        return _PIPELINE, _DEVICE, _WATERMARK, _MASK, _ARGS

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

    print(f"[Verifier] Loading ROBIN pipeline on {_DEVICE}...")

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
    )

    _PIPELINE = InversableStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=dtype,
        revision="main",
    )

    _PIPELINE = _PIPELINE.to(_DEVICE)

    if not WATERMARK_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Watermark checkpoint not found:\n{WATERMARK_CHECKPOINT}"
        )

    checkpoint = torch.load(
        WATERMARK_CHECKPOINT,
        map_location="cpu",
    )

    if "opt_wm" not in checkpoint:
        raise KeyError(
            "The watermark checkpoint does not contain 'opt_wm'."
        )

    _WATERMARK = checkpoint["opt_wm"].to(_DEVICE)

    _ARGS = _build_robin_args()

    # Create a latent with the same size used by ROBIN
    # and derive the same watermark mask.
    init_latents = _PIPELINE.get_random_latents(
        height=512,
        width=512,
    )

    _MASK = get_watermarking_mask(
        init_latents,
        _ARGS,
        _DEVICE,
    )

    print("[Verifier] ROBIN verifier ready.")

    return _PIPELINE, _DEVICE, _WATERMARK, _MASK, _ARGS


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

def verify_uploaded_image(
    image_path: Path,
    prompt: str = "",
) -> VerificationResult:
    """
    Verify an uploaded image using the same reverse-diffusion
    procedure used by the ROBIN evaluation code.

    IMPORTANT:
    The returned detection score is a ROBIN L1 watermark
    distance. LOWER means a closer match to the target
    watermark.

    A binary threshold is intentionally not hard-coded yet.
    We will calibrate it from known watermarked images
    and camera/real images.
    """

    start_time = time.perf_counter()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    pipe, device, opt_wm, watermarking_mask, args = (
        _load_robin_components()
    )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    # -----------------------------------------------------
    # Load and preprocess image
    # -----------------------------------------------------

    pil_image = Image.open(image_path).convert("RGB")

    transformed = transform_img(pil_image)

    image_tensor = (
        transformed
        .unsqueeze(0)
        .to(dtype=dtype, device=device)
    )

    # -----------------------------------------------------
    # Convert image to Stable Diffusion latent space
    # -----------------------------------------------------

    with torch.inference_mode():

        image_latents = pipe.get_image_latents(
            image_tensor,
            sample=False,
        )

        # -------------------------------------------------
        # Same text embedding used by the research detector
        # -------------------------------------------------

        tester_prompt = ""

        text_embeddings = pipe.get_text_embedding(
            tester_prompt
        )

        text_embeddings = text_embeddings.to(
            device=device,
            dtype=dtype,
        )

        # -------------------------------------------------
        # Reverse diffusion
        #
        # This mirrors the research evaluation:
        #
        # image
        #   â†“
        # image latent
        #   â†“
        # forward_diffusion(...)
        #   â†“
        # latents_b
        #   â†“
        # latents_b[35]
        # -------------------------------------------------

        latents_b = []

        (
            reversed_latents,
            latents_b,
            noise_b,
        ) = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=args.test_num_inference_steps,
            latents_b=latents_b,
        )

        # -------------------------------------------------
        # Retrieve the same watermark step used in
        # your completed evaluation.
        # -------------------------------------------------

        if len(latents_b) <= WATERMARK_STEP:
            raise RuntimeError(
                f"Not enough reverse-diffusion states were "
                f"produced. Expected index {WATERMARK_STEP}, "
                f"but only {len(latents_b)} states are available."
            )

        recovered_latent = latents_b[WATERMARK_STEP]

        # -------------------------------------------------
        # Convert recovered latent to frequency domain
        # -------------------------------------------------

        recovered_fft = torch.fft.fftshift(
            torch.fft.fft2(recovered_latent),
            dim=(-1, -2),
        )

        # -------------------------------------------------
        # ROBIN l1_complex watermark distance
        #
        # Same basic calculation as eval_watermark():
        #
        # | recovered FFT - target watermark | mean
        #
        # LOWER = closer to watermark
        # -------------------------------------------------

        score = torch.abs(
            recovered_fft[watermarking_mask]
            - opt_wm[watermarking_mask]
        ).mean().item()

    verification_time = time.perf_counter() - start_time

    # -----------------------------------------------------
    # Watermark decision
    #
    # ROBIN uses an L1 distance:
    # LOWER score = closer match to the expected watermark.
    #
    # This threshold was provisionally calibrated using
    # known watermarked images and real camera photographs.
    # -----------------------------------------------------

    detected = score <= DETECTION_THRESHOLD

    if detected:
        message = "ROBIN watermark detected. "
    else:
        message = "No ROBIN watermark detected. "

    return VerificationResult(
        filename=image_path.name,
        prompt=prompt,
        detection_score=float(score),
        verification_time_sec=float(verification_time),
        detected=detected,
        threshold=DETECTION_THRESHOLD,
        message=message,
    )
