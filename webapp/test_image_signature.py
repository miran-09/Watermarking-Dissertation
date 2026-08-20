from pathlib import Path

from services.image_signature import (
    create_signed_record,
    embed_signature_in_png,
    verify_image_signature,
)


WEBAPP_ROOT = Path(__file__).resolve().parent

SOURCE_IMAGE = (
    WEBAPP_ROOT
    / "uploads"
    / "b0511d510eeb4056b42ffe1dbdf17c79_Gemini_Generated_Image_52wf4g52wf4g52wf.png"
)

OUTPUT_IMAGE = (
    WEBAPP_ROOT
    / "outputs"
    / "signature_test.png"
)


signed_record = create_signed_record(
    image_path=SOURCE_IMAGE,
    creator_id="USER-001",
    method="predictor",
    prompt="A cinematic photograph of a city at night.",
)

embed_signature_in_png(
    source_image=SOURCE_IMAGE,
    destination_image=OUTPUT_IMAGE,
    signed_record=signed_record,
)

print("Created signed image:")
print(OUTPUT_IMAGE)

print()

result = verify_image_signature(OUTPUT_IMAGE)

print("IMAGE SIGNATURE TEST")
print("--------------------")
print("Found:           ", result["found"])
print("Signature valid: ", result["signature_valid"])
print("Integrity valid: ", result["integrity_valid"])
print("Message:         ", result["message"])

if result["record"]:
    print()
    print("Image ID:        ", result["record"]["image_id"])
    print("Creator:         ", result["record"]["creator_id"])
    print("Method:          ", result["record"]["method"])
