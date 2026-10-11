"""
dataset_exploration_01.py (Interactive Selection)
Tahap 01 - Dataset Exploration
Membaca dataset citra bunga, menghitung jumlah per kelas,
memeriksa dimensi, ukuran, dan kondisi citra (pencahayaan, kontras, noise, ketajaman).

Workflow:
    1. Analyze all images and save documentation
    2. List available images for user selection
    3. User selects image by number or name
    4. Display selected image with detailed analysis
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "training_data"
TEST_DIR = DATASET_DIR / "testing_data"
OUTPUT_DIR = BASE_DIR / "documentations"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(folder: Path) -> list[Path]:
    """Kumpulkan semua file citra di satu folder (non-rekursif)."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def extract_class_name(filename: str) -> str:
    """Ekstrak nama kelas dari filename (prefix sebelum angka)."""
    import re
    match = re.match(r"^([A-Za-z]+)", filename)
    return match.group(1) if match else "Unknown"


def analyze_image(path: Path) -> dict:
    """Analisis citra: dimensi, ukuran file, pencahayaan, kontras, noise, ketajaman."""
    img_bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Dimensi & metadata
    with Image.open(path) as pil_img:
        width, height = pil_img.size
        mode = pil_img.mode
        channels = len(pil_img.getbands())
    
    # Ukuran file (KB)
    file_size_kb = path.stat().st_size / 1024
    
    # Pencahayaan (rata-rata intensitas)
    brightness = float(np.mean(gray))
    
    # Kontras (standar deviasi intensitas)
    contrast = float(np.std(gray))
    
    # Ketajaman (variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    # Noise estimate (via high-freq variance, metode sederhana)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_est = float(np.std(gray - blur))
    
    return {
        "name": path.name,
        "class": extract_class_name(path.name),
        "width": width,
        "height": height,
        "channels": channels,
        "mode": mode,
        "file_size_kb": round(file_size_kb, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise_est": round(noise_est, 2),
    }


def summarize(folder: Path, label: str) -> pd.DataFrame:
    """Buat ringkasan metadata untuk satu split dataset."""
    rows = [analyze_image(p) for p in collect_images(folder)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["split"] = label
    return df


def save_documentation(df: pd.DataFrame) -> None:
    """Simpan statistik dataset ke MD dan CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # CSV lengkap
    csv_path = OUTPUT_DIR / "dataset_exploration.csv"
    df.to_csv(csv_path, index=False)
    
    # Markdown summary
    md_path = OUTPUT_DIR / "dataset_exploration.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Exploration - PCD Daisy Project\n\n")
        f.write(f"Total citra: {len(df)}\n\n")
        
        f.write("## Statistik Per Split\n\n")
        split_stats = df.groupby("split").size()
        for split, count in split_stats.items():
            f.write(f"- **{split}**: {count} citra\n")
        f.write("\n")
        
        f.write("## Statistik Per Kelas\n\n")
        class_stats = df.groupby("class").size()
        for cls, count in class_stats.items():
            f.write(f"- **{cls}**: {count} citra\n")
        f.write("\n")
        
        f.write("```\n")
        f.write(df[["width", "height"]].describe().round(2).to_string())
        f.write("\n```\n\n")

        f.write("## Ukuran File (KB)\n\n```\n")
        f.write(df[["file_size_kb"]].describe().round(2).to_string())
        f.write("\n```\n\n")

        f.write("## Kondisi Citra\n\n")
        f.write("### Pencahayaan (Brightness)\n\n```\n")
        f.write(df.groupby("class")["brightness"].describe().round(2).to_string())
        f.write("\n```\n\n")
        
        f.write("### Kontras\n\n```\n")
        f.write(df.groupby("class")["contrast"].describe().round(2).to_string())
        f.write("\n```\n\n")
        
        f.write("### Ketajaman\n\n```\n")
        f.write(df.groupby("class")["sharpness"].describe().round(2).to_string())
        f.write("\n```\n\n")
        
        f.write("### Noise\n\n```\n")
        f.write(df.groupby("class")["noise_est"].describe().round(2).to_string())
        f.write("\n```\n\n")
    
    print(f"Dokumentasi tersimpan:\n  - {csv_path}\n  - {md_path}")


def display_image_list(df: pd.DataFrame):
    """Display list of available images for user selection."""
    print("\n" + "="*50)
    print("DAFTAR GAMBAR TERSEDIA:")
    print("="*50)
    for idx, row in df.iterrows():
        print(f"[{idx+1}] {row['name']} ({row['split']}) - {row['width']}x{row['height']}")
    print("="*50)


def get_user_selection(df: pd.DataFrame):
    """Get user input for image selection."""
    while True:
        try:
            user_input = input("\nMasukkan nomor gambar (atau nama file, atau 'skip' untuk lewati): ").strip()
            
            if user_input.lower() == 'skip':
                return None
            
            # Try as index
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(df):
                    return df.iloc[idx]
                else:
                    print(f"Nomor harus antara 1-{len(df)}. Coba lagi.")
                    continue
            
            # Try as filename
            matches = df[df['name'].str.lower().str.contains(user_input.lower())]
            if len(matches) == 1:
                return matches.iloc[0]
            elif len(matches) > 1:
                print(f"Ditemukan {len(matches)} gambar cocok. Coba nama lebih spesifik.")
                continue
            
            print("File tidak ditemukan. Coba lagi.")
        except Exception as e:
            print(f"Error: {e}")


def display_selected_image(row, df):
    """Display selected image with detailed analysis."""
    folder = TRAIN_DIR if row['split'] == 'train' else TEST_DIR
    img_path = folder / row['name']
    
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Create figure with image and analysis
    fig = plt.figure(figsize=(14, 6))
    
    # Original image
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"{row['name']}", fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Analysis text
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    analysis_text = f"""
ANALISIS GAMBAR
{'='*40}

Nama: {row['name']}
Kelas: {row['class']}
Split: {row['split']}

DIMENSI & METADATA
  - Ukuran: {row['width']} x {row['height']} px
  - Channels: {row['channels']} ({row['mode']})
  - File size: {row['file_size_kb']} KB

KONDISI CITRA
  - Brightness: {row['brightness']:.2f}
  - Contrast: {row['contrast']:.2f}
  - Sharpness: {row['sharpness']:.2f}
  - Noise estimate: {row['noise_est']:.2f}

PERBANDINGAN DENGAN DATASET
  - Avg brightness: {df['brightness'].mean():.2f}
  - Avg contrast: {df['contrast'].mean():.2f}
  - Avg sharpness: {df['sharpness'].mean():.2f}
"""
    
    ax2.text(0.1, 0.5, analysis_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main workflow."""
    print("="*50)
    print("TAHAP 1: DATASET EXPLORATION")
    print("="*50)
    
    # Analyze all images
    print("\nMenganalisis dataset...")
    train_df = summarize(TRAIN_DIR, "train")
    test_df = summarize(TEST_DIR, "test")
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    if all_df.empty:
        print("Tidak ada citra ditemukan.")
        return
    
    # Save documentation
    save_documentation(all_df)
    
    print(f"\n[OK] Total {len(all_df)} gambar dianalisis")
    print(f"  - Training: {len(train_df)}")
    print(f"  - Testing: {len(test_df)}")
    
    # Interactive selection
    display_image_list(all_df)
    selected_row = get_user_selection(all_df)
    
    if selected_row is not None:
        print(f"\nMemproses: {selected_row['name']}...")
        display_selected_image(selected_row, all_df)
        print("[OK] Selesai!")
    else:
        print("\nSkip tampilan gambar individual.")


if __name__ == "__main__":
    main()
