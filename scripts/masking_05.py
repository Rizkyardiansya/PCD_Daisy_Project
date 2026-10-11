"""
Masking Experiment Script (Interactive Selection).
Demonstrates kernel masks (convolution) and binary masks (region selection) on a user-selected image.

Usage:
    python masking_05.py

Workflow:
    1. List available images
    2. User selects image by number or name
    3. Apply kernel masks and binary masks to selected image
    4. Display original + all masked versions
    5. Save results to results/masking/<type>/<mask_name>/
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "training_data"
OUTPUT_DIR = BASE_DIR / "results" / "masking"
REFERENCE_DIR = OUTPUT_DIR / "reference"
KERNEL_DIR = OUTPUT_DIR / "kernel"
BINARY_DIR = OUTPUT_DIR / "binary"
MASKS_DIR = OUTPUT_DIR / "masks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_DIR.mkdir(parents=True, exist_ok=True)
BINARY_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# Define kernel masks for convolution
KERNEL_MASKS = {
    "smoothing_3x3": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9,
    "smoothing_5x5": np.ones((5, 5), dtype=np.float32) / 25,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "edge_h": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "edge_v": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "edge_diagonal": np.array([[1, 1, -1], [1, 0, -1], [-1, -1, 1]], dtype=np.float32),
}

for name in KERNEL_MASKS:
    (KERNEL_DIR / name).mkdir(parents=True, exist_ok=True)

# Binary mask directories
(BINARY_DIR / "threshold_mask").mkdir(parents=True, exist_ok=True)
(BINARY_DIR / "circular_mask").mkdir(parents=True, exist_ok=True)
(BINARY_DIR / "rectangular_mask").mkdir(parents=True, exist_ok=True)


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
    """Process single image with kernel masks and binary masks. Return dict of results."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Gagal membaca: {image_path.name}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    img_height, img_width = image_bgr.shape[:2]
    center = (img_width // 2, img_height // 2)
    
    results = {
        "original": image_rgb,
        "kernels": {},
        "binary_masks": {}
    }
    
    # === KERNEL MASKS (Convolution) ===
    for name, kernel in KERNEL_MASKS.items():
        if name == "sharpen":
            # Sharpen applied to grayscale then convert back
            filtered = cv2.filter2D(gray_image, cv2.CV_8U, kernel)
            filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        else:
            filtered = cv2.filter2D(image_bgr, -1, kernel)
            filtered_bgr = filtered
        
        results["kernels"][name] = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    
    # === BINARY MASKS (Region Selection) ===
    
    # 1. Threshold-based binary mask
    _, thresh_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    masked_thresh = cv2.bitwise_and(image_bgr, image_bgr, mask=thresh_mask)
    results["binary_masks"]["threshold"] = cv2.cvtColor(masked_thresh, cv2.COLOR_BGR2RGB)
    results["mask_images"] = {"threshold": thresh_mask}
    
    # 2. Circular ROI mask
    radius = min(img_width, img_height) // 4
    circular_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)
    masked_circular = cv2.bitwise_and(image_bgr, image_bgr, mask=circular_mask)
    results["binary_masks"]["circular"] = cv2.cvtColor(masked_circular, cv2.COLOR_BGR2RGB)
    results["mask_images"]["circular"] = circular_mask
    
    # 3. Rectangular ROI mask
    rect_w, rect_h = img_width // 2, img_height // 2
    top_left = (img_width // 4, img_height // 4)
    bottom_right = (img_width // 4 + rect_w, img_height // 4 + rect_h)
    rectangular_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.rectangle(rectangular_mask, top_left, bottom_right, 255, -1)
    masked_rect = cv2.bitwise_and(image_bgr, image_bgr, mask=rectangular_mask)
    results["binary_masks"]["rectangular"] = cv2.cvtColor(masked_rect, cv2.COLOR_BGR2RGB)
    results["mask_images"]["rectangular"] = rectangular_mask
    
    return results


def display_results(image_path, results):
    """Display original + kernel masks (first 6) side-by-side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Kernel Masks: {image_path.name}", fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    titles = ["Original", "Smoothing 3x3", "Smoothing 5x5", "Sharpen", "Edge Horizontal", "Edge Vertical"]
    keys = ["original", "smoothing_3x3", "smoothing_5x5", "sharpen", "edge_h", "edge_v"]
    
    for ax, title, key in zip(axes, titles, keys):
        if key in results["kernels"]:
            ax.imshow(results["kernels"][key])
        elif key == "original":
            ax.imshow(results["original"])
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display binary masks
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Binary Masks: {image_path.name}", fontsize=14, fontweight='bold')
    
    titles = ["Original", "Threshold Mask", "Circular Mask", "Rectangular Mask"]
    keys = ["original", "threshold", "circular", "rectangular"]
    
    for ax, title, key in zip(axes, titles, keys):
        if key == "original":
            ax.imshow(results["original"])
        elif key in results["binary_masks"]:
            ax.imshow(results["binary_masks"][key])
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_results(image_path, image_bgr, results):
    """Save masked images to disk."""
    # Save reference
    reference_path = REFERENCE_DIR / image_path.name
    cv2.imwrite(str(reference_path), image_bgr)
    
    # Save kernel masks
    for name in results["kernels"]:
        img_bgr = cv2.cvtColor(results["kernels"][name], cv2.COLOR_RGB2BGR)
        output_path = KERNEL_DIR / name / f"{image_path.stem}_kernel_{name}{image_path.suffix}"
        cv2.imwrite(str(output_path), img_bgr)
    
    # Save binary masked images
    for name in results["binary_masks"]:
        img_bgr = cv2.cvtColor(results["binary_masks"][name], cv2.COLOR_RGB2BGR)
        output_path = BINARY_DIR / f"{name}_mask" / f"{image_path.stem}_masked_{name}{image_path.suffix}"
        cv2.imwrite(str(output_path), img_bgr)
    
    # Save mask visualizations
    for name, mask in results["mask_images"].items():
        mask_path = MASKS_DIR / f"{name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
    
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
