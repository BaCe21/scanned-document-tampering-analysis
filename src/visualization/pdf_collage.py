import os
import random
from pdf2image import convert_from_path
from PIL import Image

# Parametry
pdf_folder = 'pdfs'
num_pdfs = 6  # ile PDFów do kolażu
cols = 3      # ile kolumn w kolażu
thumb_size = (300, 400)  # rozmiar miniatury każdej strony

# Pobierz pliki PDF
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
selected_pdfs = random.sample(pdf_files, min(num_pdfs, len(pdf_files)))

# Wczytaj pierwsze strony PDF-ów jako obrazy
thumbnails = []
for pdf_file in selected_pdfs:
    path = os.path.join(pdf_folder, pdf_file)
    try:
        pages = convert_from_path(path, first_page=1, last_page=1)
        if pages:
            thumb = pages[0].resize(thumb_size)
            thumbnails.append(thumb)
    except Exception as e:
        print(f"Problem z {pdf_file}: {e}")

# Wylicz układ kolażu
rows = (len(thumbnails) + cols - 1) // cols
collage_width = cols * thumb_size[0]
collage_height = rows * thumb_size[1]

# Utwórz pusty obraz kolażu
collage = Image.new('RGB', (collage_width, collage_height), color='white')

# Wklej miniatury w siatkę
for index, thumb in enumerate(thumbnails):
    x = (index % cols) * thumb_size[0]
    y = (index // cols) * thumb_size[1]
    collage.paste(thumb, (x, y))

# Zapisz kolaż
collage.save('collage_pdf.png')
print("Zapisano kolaż jako 'collage.png'")
