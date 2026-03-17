from google import genai
from google.genai import types
import pytesseract
from PIL import Image


def extract_text(image_path, api_key):
    # -------- TRY GEMINI --------
    try:
        client = genai.Client(api_key=api_key)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                "Extract all handwritten text exactly as written."
            ]
        )

        return getattr(response, "text", "")

    # -------- FALLBACK TO TESSERACT --------
    except Exception as e:
        print("Gemini failed, using Tesseract OCR:", e)

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        return text