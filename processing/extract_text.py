import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


device = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def get_trocr_model():
    global _processor, _model

    if _processor is None or _model is None:
        _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        _model.to(device)
        _model.eval()

    return _processor, _model


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    return img, thresh


def split_lines(image_path):
    img, thresh = preprocess_image(image_path)

    horizontal_sum = np.sum(thresh, axis=1)

    lines = []
    start = None
    min_line_height = 18

    for i, val in enumerate(horizontal_sum):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            if end - start >= min_line_height:
                line_img = img[start:end, :]
                if line_img.shape[0] > 0 and line_img.shape[1] > 0:
                    lines.append((start, line_img))
            start = None

    if start is not None:
        end = len(horizontal_sum)
        if end - start >= min_line_height:
            line_img = img[start:end, :]
            if line_img.shape[0] > 0 and line_img.shape[1] > 0:
                lines.append((start, line_img))

    lines.sort(key=lambda x: x[0])
    return [line_img for _, line_img in lines]


def trocr_extract(image_path):
    try:
        processor, model = get_trocr_model()
        line_images = split_lines(image_path)

        if not line_images:
            return ""

        extracted_lines = []

        for line in line_images:
            pil_img = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((384, 384))

            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if text:
                extracted_lines.append(text)

        return " ".join(extracted_lines)

    except Exception as e:
        print("TrOCR failed:", e)
        return ""


def tesseract_extract(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print("Tesseract failed:", e)
        return ""


def gemini_extract(image_path, api_key):
    try:
        raise Exception("Gemini disabled")
    except Exception as e:
        print("Gemini failed:", e)
        return ""


def extract_text(image_path, api_key):
    gemini_text = gemini_extract(image_path, api_key)
    if gemini_text:
        print("Using Gemini OCR")
        return gemini_text

    trocr_text = trocr_extract(image_path)
    if trocr_text:
        print("Using TrOCR OCR")
        return trocr_text

    print("Using Tesseract OCR")
    return tesseract_extract(image_path)