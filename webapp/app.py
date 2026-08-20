from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from services.generator import run_generation
from services.verifier import verify_uploaded_image
from services.image_signature import verify_image_signature

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"

APP_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = APP_ROOT / "uploads"
OUTPUTS_DIR = APP_ROOT / "outputs"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        method = request.form.get("method", "predictor")
        seed = request.form.get("seed", "").strip() or None

        if not prompt:
            return render_template("generate.html", error="Please enter a prompt.")

        try:
            result = run_generation(prompt=prompt, method=method, seed=seed)
            return render_template(
                "result.html",
                prompt=result.prompt,
                method=result.method,
                seed=result.seed,
                runtime_sec=f"{result.runtime_sec:.3f}",
                predictor_inference_sec=(
                    f"{result.predictor_inference_sec:.6f}"
                    if result.predictor_inference_sec is not None
                    else None
                ),
                output_image=result.output_image,
                image_id=result.image_id,
                signature_added=result.signature_added,
                ok=result.ok,
                log_tail=result.log_tail,
            )
        except Exception as exc:
            print(f"Generate error: {exc}")
            return render_template("generate.html", error=str(exc), prompt=prompt)

    return render_template("generate.html")


@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("verify.html", error="Please choose an image.", prompt=prompt)

        if not allowed_file(file.filename):
            return render_template(
                "verify.html",
                error="Please upload a PNG, JPG, JPEG, or WEBP image.",
                prompt=prompt,
            )

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = UPLOADS_DIR / filename
        file.save(upload_path)

        signature_result = verify_image_signature(upload_path)

        try:
            result = verify_uploaded_image(upload_path, prompt=prompt)
            return render_template(
                "verify.html",
                prompt=prompt,
                preview_url=url_for("uploaded_file", filename=filename),

                # ROBIN watermark result
                detected=result.detected,
                detection_score=f"{result.detection_score:.4f}",
                verification_time=f"{result.verification_time_sec:.3f}",
                verification_message=result.message,

                # Image Signature result
                signature_found=signature_result["found"],
                signature_valid=signature_result["signature_valid"],
                signature_integrity_valid=signature_result["integrity_valid"],
                signature_record=signature_result["record"],
                signature_message=signature_result["message"],
            )
        except Exception as exc:
            print(f"Verify error: {exc}")
            return render_template(
                "verify.html",
                error=str(exc),
                prompt=prompt,
                uploaded_image=filename,
            )

    return render_template("verify.html")

@app.route("/verify-generated", methods=["POST"])
def verify_generated():
    filename = request.form.get("filename", "").strip()
    prompt = request.form.get("prompt", "").strip()

    if not filename:
        return render_template(
            "generate.html",
            error="No generated image was found to verify."
        )

    image_path = OUTPUTS_DIR / filename

    if not image_path.exists():
        return render_template(
            "verify.html",
            error="Generated image file not found.",
            prompt=prompt,
        )

    try:
        # -------------------------------------------------
        # 1. ROBIN watermark check
        # -------------------------------------------------
        result = verify_uploaded_image(
            image_path,
            prompt=prompt
        )

        # -------------------------------------------------
        # 2. IMAGE SIGNATURE check
        # -------------------------------------------------
        signature_result = verify_image_signature(
            image_path
        )

        # -------------------------------------------------
        # Show both results on Verify page
        # -------------------------------------------------
        return render_template(
            "verify.html",
            prompt=prompt,

            # Image preview
            preview_url=url_for(
                "outputs",
                filename=filename
            ),

            # -----------------------------
            # ROBIN watermark result
            # -----------------------------
            detected=result.detected,
            detection_score=f"{result.detection_score:.4f}",
            verification_time=f"{result.verification_time_sec:.3f}",
            verification_message=result.message,

            # -----------------------------
            # Image Signature result
            # -----------------------------
            signature_found=signature_result["found"],
            signature_valid=signature_result["signature_valid"],
            signature_integrity_valid=signature_result["integrity_valid"],
            signature_record=signature_result["record"],
            signature_message=signature_result["message"],
        )

    except Exception as exc:
        print(f"Verify-generated error: {exc}")

        return render_template(
            "verify.html",
            error=str(exc),
            prompt=prompt,
            preview_url=url_for(
                "outputs",
                filename=filename
            ),
        )

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/download/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
