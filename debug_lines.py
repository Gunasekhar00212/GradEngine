from processing.extract_text import split_lines
import cv2

lines = split_lines("data/student1/q22a.png")
for i, line in enumerate(lines):
    cv2.imwrite(f"data/student1/debug_line_{i}.png", line)
print(f"{len(lines)} lines found")