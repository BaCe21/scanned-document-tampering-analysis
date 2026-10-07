import os
import glob
import pytesseract
from PIL import Image
import pandas as pd
import re

folder = 'fake_imgs'
max_imgs = 50   # ile maksymalnie zdjęć przetworzyć (możesz zwiększyć)
output_csv = 'ocr_results.csv'

# Funkcja do sortowania po numerze pliku (np. fake_12.png => 12)
def extract_num(path):
    fname = os.path.basename(path)
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(os.path.join(folder, '*.png')), key=extract_num)[:max_imgs]

results = []

print("Plik\tRozpoznany znak")
for f in files:
    img = Image.open(f).convert('RGB')
    recognized = pytesseract.image_to_string(img, config='--psm 10').strip()
    results.append({'file': os.path.basename(f), 'recognized': recognized})
    print(f"{os.path.basename(f)}\t{recognized}")

# Zapisz do CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\nWyniki zapisane do {output_csv}")
