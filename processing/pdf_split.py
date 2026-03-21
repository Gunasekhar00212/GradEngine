from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder="data/pages"):
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_path,dpi=300)
    paths = []

    for i, img in enumerate(images):
        path = f"{output_folder}/page_{i}.png"
        img.save(path, "PNG")
        paths.append(path)

    return paths