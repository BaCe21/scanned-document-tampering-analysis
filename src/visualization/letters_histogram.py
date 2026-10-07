import csv
import matplotlib.pyplot as plt
from collections import Counter

csv_path = "letters_labels_cleaned.csv"  # dopasuj jeśli inna nazwa
output_path = "histogram.png"            # ścieżka do zapisu histogramu

def draw_histogram(csv_file, save_path=None):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        letters = [row['letter'] for row in reader if row['letter'].isalpha()]

    counter = Counter(letters)

    # Posortuj alfabetycznie A-Z + a-z
    sorted_keys = sorted(counter.keys(), key=lambda c: (c.islower(), c))
    values = [counter[k] for k in sorted_keys]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_keys, values)
    plt.title("Histogram liter w zbiorze (rozróżnienie wielkości)")
    plt.xlabel("Litera")
    plt.ylabel("Liczba wystąpień")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Histogram zapisany jako: {save_path}")

    plt.show()

if __name__ == "__main__":
    draw_histogram(csv_path, save_path=output_path)
