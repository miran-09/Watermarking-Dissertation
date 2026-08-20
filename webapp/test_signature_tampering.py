from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin

from services.image_signature import (
    SIGNATURE_METADATA_FIELD,
    extract_signature,
    verify_image_signature,
)


WEBAPP_ROOT = Path(__file__).resolve().parent

source = WEBAPP_ROOT / "outputs" / "signature_test.png"
modified = WEBAPP_ROOT / "outputs" / "signature_test_modified.png"


# Read the original signed image
original = Image.open(source)

# Extract the existing signature metadata
signature_data = original.info.get(SIGNATURE_METADATA_FIELD)

if not signature_data:
    raise RuntimeError("No image signature metadata found in the source image.")

# Make a copy of the image pixels
image = original.convert("RGB")

# Make a small visible modification
draw = ImageDraw.Draw(image)
draw.rectangle(
    (10, 10, 20, 20),
    fill=(255, 0, 0),
)

# Re-attach the ORIGINAL signature metadata
metadata = PngImagePlugin.PngInfo()
metadata.add_text(
    SIGNATURE_METADATA_FIELD,
    signature_data,
)

# Save the modified image while preserving the signature record
image.save(
    modified,
    format="PNG",
    pnginfo=metadata,
)

print("Modified image created:")
print(modified)

# Verify the modified image
result = verify_image_signature(modified)

print()
print("TAMPER TEST")
print("-----------")
print("Found:             ", result["found"])
print("Signature valid:   ", result["signature_valid"])
print("Integrity valid:   ", result["integrity_valid"])
print("Message:            ", result["message"])
