"""
Grayscale Histogram Processing Pipeline (Interactive Selection).

Converts RGB images to grayscale, generates histograms, and computes
intensity statistics for a user-selected image.

Usage:
    python grayscale_histogram_02.py

Workflow:
    1. List available images
    2. User selects image by number or name
    3. Convert to grayscale, compute histogram and statistics
    4. Display original, grayscale, and histogram side-by-side
    5. Save results to results/grayscale/ and results/histograms/
"""

from pathlib import Path
import csv
from typing import Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "training_data"
GRAY_DIR = BASE_DIR / "results" / "grayscale"
HIST_DIR = BASE_DIR / "results" / "histograms"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HISTOGRAM_BINS = 256
HISTOGRAM_RANGE = (0, 255)

GRAY_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)


def get_available_images():
    """List all available images from training_data."""
    images = sorted(
        p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VALID_EXT
    )
    return images


def display_image_list(images):
    """Display list of available images for user selection."""
    print("\n" + "="*50)
    print("DAFTAR GAMBAR TERSEDIA:")
    print("="*50)
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}] {img_path.name}")
    print("="*50)


def get_user_selection(images):
    """Get user input for image selection."""
    while True:
        try:
            user_input = input("\nMasukkan nomor gambar (atau nama file): ").strip()
            
            # Try as index
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(images):
                    return images[idx]
                else:
                    print(f"Nomor harus antara 1-{len(images)}. Coba lagi.")
                    continue
            
            # Try as filename
            for img in images:
                if user_input.lower() in img.name.lower():
                    return img
            
            print("File tidak ditemukan. Coba lagi.")
        except Exception as e:
            print(f"Error: {e}")


def compute_statistics(gray_image: np.ndarray) -> Dict[str, float]:
    """Compute intensity statistics from grayscale image."""
    return {
        "MeanIntensity": float(np.mean(gray_image)),
        "StandardDeviation": float(np.std(gray_image)),
        "MinimumIntensity": int(np.min(gray_image)),
        "MaximumIntensity": int(np.max(gray_image)),
        "MedianIntensity": float(np.median(gray_image)),
    }


def process_single_image(image_path):
    """Process single image: convert to grayscale, compute stats, histogram."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Gagal membaca: {image_path.name}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    histogram = cv2.calcHist([gray_image], [0], None, [HISTOGRAM_BINS], HISTOGRAM_RANGE)
    
    # Compute statistics
    stats = compute_statistics(gray_image)
    
    return {
        "original": image_rgb,
        "grayscale": gray_image,
        "histogram": histogram,
        "stats": stats
    }


def display_results(image_path, results):
    """Display original, grayscale, and histogram side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grayscale & Histogram: {image_path.name}", fontsize=14, fontweight='bold')
    
    # Original image
    axes[0].imshow(results["original"])
    axes[0].set_title("Original (RGB)", fontweight='bold')
    axes[0].axis('off')
    
    # Grayscale image
    axes[1].imshow(results["grayscale"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale", fontweight='bold')
    axes[1].axis('off')
    
    # Histogram
    axes[2].plot(results["histogram"], color='black')
    axes[2].set_xlim([0, 255])
    axes[2].set_xlabel("Pixel Intensity")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Histogram", fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics text
    stats = results["stats"]
    stats_text = f"Mean: {stats['MeanIntensity']:.2f}\n"
    stats_text += f"Std: {stats['StandardDeviation']:.2f}\n"
    stats_text += f"Min: {stats['MinimumIntensity']}\n"
    stats_text += f"Max: {stats['MaximumIntensity']}\n"
    stats_text += f"Median: {stats['MedianIntensity']:.2f}"
    
    axes[2].text(0.98, 0.97, stats_text, 
                transform=axes[2].transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def save_results(image_path, results):
    """Save grayscale image and histogram plot to disk."""
    # Save grayscale image
    gray_output_path = GRAY_DIR / f"{image_path.stem}_grayscale{image_path.suffix}"
    cv2.imwrite(str(gray_output_path), results["grayscale"])
    
    # Save histogram plot
    hist_output_path = HIST_DIR / f"{image_path.stem}_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results["histogram"], color='black')
    ax.set_xlim([0, 255])
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram: {image_path.name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_output_path, dpi=150)
    plt.close(fig)
    
    # Save statistics to CSV
    csv_path = HIST_DIR / "grayscale_statistics.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ImageName'] + list(results["stats"].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({'ImageName': image_path.name, **results["stats"]})
    
    print(f"\n[OK] Hasil disimpan:")
    print(f"  - Grayscale: {gray_output_path}")
    print(f"  - Histogram: {hist_output_path}")
    print(f"  - Statistics: {csv_path}")


def main():
    """Main workflow."""
    images = get_available_images()
    
    if not images:
        print(f"Tidak ada citra di {INPUT_DIR}")
        return
    
    display_image_list(images)
    selected_image = get_user_selection(images)
    
    if not selected_image:
        print("Tidak ada gambar yang dipilih.")
        return
    
    print(f"\nMemproses: {selected_image.name}...")
    results = process_single_image(selected_image)
    
    if results:
        display_results(selected_image, results)
        save_results(selected_image, results)
        print("[OK] Selesai!")


if __name__ == "__main__":
    main()
