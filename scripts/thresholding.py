"""
Thresholding Experiment Script (Interactive Selection).
Compares global, adaptive, Otsu, and multilevel thresholding methods on a user-selected image.

Usage:
    python thresholding.py

Workflow:
    1. List available images
    2. User selects image by number or name
    3. Apply 4 thresholding methods to selected image
    4. Display original + all thresholded versions side-by-side
    5. Save results to results/thresholding/<method>/
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_multiotsu

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "training_data"
OUTPUT_DIR = BASE_DIR / "results" / "thresholding"
REFERENCE_DIR = OUTPUT_DIR / "reference"
METHODS = ["global", "adaptive", "otsu", "multilevel"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
for method in METHODS:
    (OUTPUT_DIR / method).mkdir(parents=True, exist_ok=True)


def get_available_images():
    """List all available images from training_data."""
    images = sorted(
        p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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


def process_single_image(image_path):
    """Process single image with all thresholding methods. Return dict of results."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Gagal membaca: {image_path.name}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    results = {
        "original": image_rgb
    }
    
    # 1. Global thresholding (Otsu's method)
    global_thresh = threshold_otsu(gray_image)
    global_binary = gray_image > global_thresh
    global_binary_uint8 = (global_binary * 255).astype(np.uint8)
    results["global"] = cv2.cvtColor(global_binary_uint8, cv2.COLOR_GRAY2RGB)
    
    # 2. Adaptive thresholding (Gaussian)
    adaptive_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results["adaptive"] = cv2.cvtColor(adaptive_image, cv2.COLOR_GRAY2RGB)
    
    # 3. Otsu thresholding (explicit)
    otsu_thresh = threshold_otsu(gray_image)
    otsu_binary = gray_image > otsu_thresh
    otsu_binary_uint8 = (otsu_binary * 255).astype(np.uint8)
    results["otsu"] = cv2.cvtColor(otsu_binary_uint8, cv2.COLOR_GRAY2RGB)
    
    # 4. Multilevel thresholding
    try:
        multilevel_thresh = threshold_multiotsu(gray_image, classes=3)
        multilevel_binary = gray_image <= multilevel_thresh[1]
        multilevel_binary_uint8 = (multilevel_binary * 255).astype(np.uint8)
        results["multilevel"] = cv2.cvtColor(multilevel_binary_uint8, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Error multilevel thresholding: {e}")
        results["multilevel"] = None
    
    return results


def display_results(image_path, results):
    """Display original + all thresholded versions side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Thresholding Results: {image_path.name}", fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define order: original, global, adaptive, otsu, multilevel
    titles = ["Original", "Global (Otsu)", "Adaptive (Gaussian)", "Otsu"]
    keys = ["original", "global", "adaptive", "otsu"]
    
    for ax, title, key in zip(axes, titles, keys):
        if key in results and results[key] is not None:
            ax.imshow(results[key])
            ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_results(image_path, image_bgr, results):
    """Save thresholded images to disk."""
    # Save reference
    reference_path = REFERENCE_DIR / image_path.name
    cv2.imwrite(str(reference_path), image_bgr)
    
    # Save each thresholded image
    for method in ["global", "adaptive", "otsu", "multilevel"]:
        if method in results and results[method] is not None:
            # Convert RGB back to BGR for saving
            img_bgr = cv2.cvtColor(results[method], cv2.COLOR_RGB2BGR)
            output_path = OUTPUT_DIR / method / f"{image_path.stem}_thresholded_{method}{image_path.suffix}"
            cv2.imwrite(str(output_path), img_bgr)
    
    print(f"\n[OK] Hasil disimpan ke: {OUTPUT_DIR}")


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
    image_bgr = cv2.imread(str(selected_image))
    results = process_single_image(selected_image)
    
    if results:
        display_results(selected_image, results)
        save_results(selected_image, image_bgr, results)
        print("[OK] Selesai!")


if __name__ == "__main__":
    main()
