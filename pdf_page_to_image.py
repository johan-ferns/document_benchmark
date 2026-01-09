import pymupdf

_pdf_path = "data/pdf/input/example_1.pdf"
dpi = 300
doc = pymupdf.open(_pdf_path)

# # For all pages
# page_images = dict()
# for page_no in range(doc.page_count):
#     page = doc.load_page(page_no)
#     pixmap = page.get_pixmap(dpi=dpi)
#     page_images[page_no + 1] = pixmap.pil_image()


# For a single page
page_no = 3
page = doc.load_page(page_no)
pixmap = page.get_pixmap(dpi=dpi)
pil_image = pixmap.pil_image()

pil_image.save("data/pdf/output/example_1_page_3.jpg")
