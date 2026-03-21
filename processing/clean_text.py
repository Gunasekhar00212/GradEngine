import re

def clean_text(text):
    text = text.lower()

    # keep letters, digits, spaces, and question marks for OCR repair
    text = re.sub(r'[^a-z0-9\s?]', ' ', text)

    # remove long numeric garbage like 0000005000
    text = re.sub(r'\d{2,}', ' ', text)

    # remove isolated single digits like 1 2 3
    text = re.sub(r'\b\d\b', ' ', text)

    # domain-aware science token fixes
    text = re.sub(r'co\s*\?', 'co2', text)
    text = re.sub(r'c0\s*2', 'co2', text)
    text = re.sub(r'c02', 'co2', text)
    text = re.sub(r'o\s*2', 'o2', text)
    text = re.sub(r'h\s*2\s*o', 'h2o', text)
    text = re.sub(r'h20', 'h2o', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text