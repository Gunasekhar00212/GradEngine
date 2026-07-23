import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from processing.extract_text import extract_text

images = {
    "q22a": "data/student1/q22a.png",
    "q22b_p1": "data/student1/q22b_p1.png",
    "q22b_p2": "data/student1/q22b_p2.png",
}

for name, path in images.items():
    text = extract_text(path, api_key=None)
    print(name, "→", text) 