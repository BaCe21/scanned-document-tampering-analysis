import os
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt

folder = 'fake_imgs'
collage_path = 'collage_fakes.png'
n_images = 100   # ile obrazków w kolażu
rows = 10        # np. 4 wiersze
cols = 10        # np. 7 kolumn

# Wybierz 28 losowych plików .png
all_files = sorted(glob.glob(os.path.join(folder, '*.png')))
if len(all_files) < n_images:
    print(f"W folderze {folder} jest tylko {len(all_files)} obrazków!")
    n_images = len(all_files)
collage_files = random.sample(all_files, n_images)

# Załaduj obrazy
imgs = [Image.open(f).convert('RGB') for f in collage_files]
img_size = imgs[0].size  # zakładamy wszystkie są tego samego rozmiaru

# Stwórz pusty obraz do kolażu
collage_img = Image.new('RGB', (cols * img_size[0], rows * img_size[1]), color=(255, 255, 255))

# Wklejaj litery
for idx, img in enumerate(imgs):
    r = idx // cols
    c = idx % cols
    collage_img.paste(img, (c * img_size[0], r * img_size[1]))

# Zapisz i/lub pokaż
collage_img.save(collage_path)
collage_img.show()    # możesz wykomentować jeśli nie chcesz od razu otwierać okienka

print(f"Kolaż zapisany jako {collage_path}")
