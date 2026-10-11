"""
Filtering and Convolution Experiment Script (Interactive Selection).
Applies mean, median, Gaussian, Laplacian, and Sobel filters to a user-selected image.

Usage:
    python filtering_convolution_04.py

Workflow:
    1. List available images
    2. User selects image by number or name
    3. Apply 5 filters to selected image
    4. Display original + all filtered versions side-by-side
    5. Save results to results/filtering/<method>/
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "training_data"
OUTPUT_DIR = BASE_DIR / "results" / "filtering"
REFERENCE_DIR = OUTPUT_DIR / "reference"
METHODS = ["mean", "median", "gaussian", "laplacian", "sobel"]

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
    """Process single image with all filters. Return dict of results."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Gagal membaca: {image_path.name}")
        return None
    
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    results = {
        "original": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "gray": gray_image
    }
    
    # 1. Mean filter
    mean_kernel = np.ones((3, 3), np.float32) / 9
    mean_filtered = cv2.filter2D(image_bgr, -1, mean_kernel)
    results["mean"] = cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB)
    
    # 2. Median filter
    median_filtered = cv2.medianBlur(image_bgr, 3)
    results["median"] = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB)
    
    # 3. Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(image_bgr, (3, 3), 1)
    results["gaussian"] = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB)
    
    # 4. Laplacian filter
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    laplacian_uint8 = np.clip(laplacian_abs, 0, 255).astype(np.uint8)
    results["laplacian"] = cv2.cvtColor(cv2.cvtColor(laplacian_uint8, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    
    # 5. Sobel filter
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_uint8 = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    results["sobel"] = cv2.cvtColor(cv2.cvtColor(sobel_uint8, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    
    return results


def display_results(image_path, results):
    """Display original + all filtered versions side-by-side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Filtering Results: {image_path.name}", fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define order: original, mean, median, gaussian, laplacian, sobel
    titles = ["Original", "Mean Filter", "Median Filter", "Gaussian Filter", "Laplacian", "Sobel"]
    keys = ["original", "mean", "median", "gaussian", "laplacian", "sobel"]
    
    for ax, title, key in zip(axes, titles, keys):
        if key in results:
            img = results[key]
            if key == "gray":
                ax.imshow(results["gray"], cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_results(image_path, image_bgr, results):
    """Save filtered images to disk."""
    # Save reference
    reference_path = REFERENCE_DIR / image_path.name
    cv2.imwrite(str(reference_path), image_bgr)
    
    # Save each filtered image
    for method in ["mean", "median", "gaussian", "laplacian", "sobel"]:
        if method in results:
            # Convert RGB back to BGR for saving
            img_bgr = cv2.cvtColor(results[method], cv2.COLOR_RGB2BGR)
            output_path = OUTPUT_DIR / method / f"{image_path.stem}_filtered_{method}{image_path.suffix}"
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
