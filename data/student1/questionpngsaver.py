import fitz  # pymupdf

doc = fitz.open("Biology.pdf")

def save_page_crop(page_num, out_name):
    page = doc[page_num - 1]  # 0-indexed
    pix = page.get_pixmap(dpi=300)
    pix.save(out_name)

save_page_crop(7, "q22a.png")
save_page_crop(8, "q22b_p1.png")
save_page_crop(9, "q22b_p2.png")

print("Done")