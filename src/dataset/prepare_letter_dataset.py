import os
import random
import csv
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw, ImageFont

def sanitize_filename(filename):
    # Convert to ASCII, ignoring non-ASCII characters.
    filename = filename.encode("ascii", "ignore").decode()
    # Remove any character that is not a letter, digit, space, hyphen, or underscore.
    return re.sub(r'[^A-Za-z0-9 _\-]+', '', filename)

# Step 1: Load PDF and convert to an image (assumes one page for simplicity)
def load_pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path, dpi=800)
    return images[0]

# Step 2: Use OCR to extract word boxes from the image.
def get_word_boxes(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxes = []
    for i, word in enumerate(data['text']):
        if word.strip() != "":
            box = {
                'word': word,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            }
            boxes.append(box)
    return boxes

# Helper: Generate a fake word with the same length as the original.
def generate_fake_word(length):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for _ in range(length))

# Step 3: Replace some words on the image.
def replace_words(image, word_boxes, replacement_prob=0.5):
    draw = ImageDraw.Draw(image)
    replaced_boxes = []
    # Adjust the font path as needed.
    font_path = "arial.ttf"
    for box in word_boxes:
        if random.random() < replacement_prob:
            original_word = box['word']
            fake_word = generate_fake_word(len(original_word))
            left, top, width, height = box['left'], box['top'], box['width'], box['height']
            # Erase the original word area.
            draw.rectangle([left, top, left + width, top + height], fill="white")
            try:
                font = ImageFont.truetype(font_path, size=height)
            except Exception as e:
                print("Could not load truetype font. Using default. Error:", e)
                font = ImageFont.load_default()
            # Write the fake word.
            draw.text((left, top), fake_word, font=font, fill="black")
            box['fake_word'] = fake_word
            replaced_boxes.append(box)
    return replaced_boxes

# Step 4: Save the modified document image in a separate folder.
def save_modified_document(image, base_name):
    modified_folder = "modified_documents"
    os.makedirs(modified_folder, exist_ok=True)
    output_path = os.path.join(modified_folder, f"{base_name}_modified.png")
    image.save(output_path)
    print(f"Modified document saved as {output_path}")

# Step 5: Extract letter boxes using pytesseract.image_to_data with confidence filtering.
def get_letter_boxes(image, conf_threshold=96):
    # Use psm 6 so that Tesseract handles the full-page text better.
    data = pytesseract.image_to_data(
        image, lang='eng', config="--psm 6 --oem 1", output_type=pytesseract.Output.DICT
    )
    letter_boxes = []
    n_boxes = len(data['level'])
    print(f"Total boxes detected: {n_boxes}")
    for i in range(n_boxes):
        text = data['text'][i].strip()
        try:
            conf = float(data['conf'][i])
        except ValueError:
            conf = -1
        if text != "" and conf >= conf_threshold:
            left = int(data['left'][i])
            top = int(data['top'][i])
            width = int(data['width'][i])
            height = int(data['height'][i])
            right = left + width
            bottom = top + height

            # For boxes with multiple characters, split them.
            if len(text) > 1:
                n_letters = len(text)
                sub_width = width / n_letters
                for j in range(n_letters):
                    # Add a margin to help avoid including parts of adjacent letters.
                    margin = 2
                    sub_left = left + int(j * sub_width) + margin
                    sub_right = left + int((j + 1) * sub_width) - margin
                    # Ensure valid sub-box dimensions.
                    if sub_left >= sub_right:
                        sub_left = left + int(j * sub_width)
                        sub_right = left + int((j + 1) * sub_width)
                    # Check aspect ratio: if sub-box is too wide relative to height, skip it.
                    if (sub_right - sub_left) > 1.2 * height:
                        continue
                    letter_boxes.append({
                        'char': text[j],
                        'conf': conf,
                        'left': sub_left,
                        'top': top,
                        'right': sub_right,
                        'bottom': bottom
                    })
            else:
                # For single-character boxes, also filter out boxes that are too wide.
                if width > 1.2 * height:
                    continue
                letter_boxes.append({
                    'char': text,
                    'conf': conf,
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom
                })
    print(f"Detected {len(letter_boxes)} letter boxes after filtering.")
    return letter_boxes

# Helper: Check if the letter box overlaps a replaced (fake) word box.
def is_letter_fake(letter_box, replaced_boxes):
    center_x = (letter_box['left'] + letter_box['right']) / 2
    center_y = (letter_box['top'] + letter_box['bottom']) / 2
    for box in replaced_boxes:
        if (center_x >= box['left'] and center_x <= box['left'] + box['width'] and
            center_y >= box['top'] and center_y <= box['top'] + box['height']):
            return True
    return False

# Step 6: Crop each letter, save as an image, and return CSV rows.
def save_letter_images_and_get_csv_rows(image, letter_boxes, replaced_boxes, letters_folder, base_name):
    rows = []
    width, height = image.size
    for i, letter_box in enumerate(letter_boxes):
        # Clamp the crop coordinates within the image boundaries.
        left = max(letter_box['left'], 0)
        top = max(letter_box['top'], 0)
        right = min(letter_box['right'], width)
        bottom = min(letter_box['bottom'], height)
        # Skip invalid boxes.
        if left >= right or top >= bottom:
            continue
        letter_img = image.crop((left, top, right, bottom))
        filename = f"{base_name}_letter_{i}.png"
        letter_img.save(os.path.join(letters_folder, filename))
        label = 1 if is_letter_fake(letter_box, replaced_boxes) else 0
        recognized_letter = letter_box['char']
        rows.append([filename, label, recognized_letter])
    return rows

# Process a single PDF file and return its CSV rows.
def process_pdf(pdf_path, letters_folder):
    base_name = sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])
    print(f"Processing PDF: {pdf_path}")

    image = load_pdf_to_image(pdf_path)
    word_boxes = get_word_boxes(image)
    replaced_boxes = replace_words(image, word_boxes, replacement_prob=0.5)
    save_modified_document(image, base_name)
    # Use the updated get_letter_boxes with adjusted config.
    letter_boxes = get_letter_boxes(image, conf_threshold=96)
    rows = save_letter_images_and_get_csv_rows(image, letter_boxes, replaced_boxes, letters_folder, base_name)
    return rows

# Main workflow: Process all PDFs in a folder and produce a unified CSV.
def main(pdf_folder):
    if not os.path.isdir(pdf_folder):
        print(f"Folder '{pdf_folder}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    unified_rows = []  # Collect CSV rows from all PDFs.
    letters_folder = "letters"
    os.makedirs(letters_folder, exist_ok=True)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        rows = process_pdf(pdf_path, letters_folder)
        unified_rows.extend(rows)

    # Write the unified CSV file.
    csv_filename = "letters_labels.csv"
    with open(csv_filename, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "letter"])
        writer.writerows(unified_rows)
    print(f"Unified CSV saved as '{csv_filename}'")

if __name__ == "__main__":
    pdf_folder = "pdfs"  # Replace with the path to your PDF folder.
    main(pdf_folder)
