"""
Daisy Image Processing Pipeline Orchestrator.

Provides an interactive CLI menu to run any or all of the 7 image processing
steps defined in the project README.

Usage:
    python main.py
"""

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

# Tambahkan root directory ke path agar modul di 'scripts/' bisa di-import
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


def run_step_01() -> None:
    """Jalankan Tahap 1: Eksplorasi Dataset."""
    print("\n--- [TAHAP 1] Eksplorasi Dataset ---")
    script_path = BASE_DIR / "scripts" / "dataset_exploration_01.py"
    if not script_path.is_file():
        print("Error: Script scripts/dataset_exploration_01.py belum tersedia.")
        return
    try:
        import subprocess

        # stdio diwariskan agar input interaktif di script bisa diakses user
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(
                f"Gagal menjalankan Tahap 1 (code {result.returncode})"
            )
    except Exception as err:
        print(f"Gagal menjalankan Tahap 1: {err}")


def run_step_02() -> None:
    """Jalankan Tahap 2: Grayscale & Histogram."""
    print("\n--- [TAHAP 2] Grayscale & Histogram ---")
    script_path = BASE_DIR / "scripts" / "grayscale_histogram_02.py"
    if not script_path.is_file():
        print("Error: Script scripts/grayscale_histogram_02.py belum tersedia.")
        return
    try:
        import subprocess

        # stdio diwariskan agar input interaktif di script bisa diakses user
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(
                f"Gagal menjalankan Tahap 2 (code {result.returncode})"
            )
    except Exception as err:
        print(f"Gagal menjalankan Tahap 2: {err}")


def run_step_03() -> None:
    """Jalankan Tahap 3: Thresholding."""
    print("\n--- [TAHAP 3] Thresholding ---")
    script_path = BASE_DIR / "scripts" / "thresholding.py"
    if not script_path.is_file():
        print("Error: Script scripts/thresholding.py belum tersedia.")
        return
    try:
        import subprocess

        # stdio diwariskan agar input interaktif di script bisa diakses user
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(
                f"Gagal menjalankan Tahap 3 (code {result.returncode})"
            )
    except Exception as err:
        print(f"Gagal menjalankan Tahap 3: {err}")


def run_step_04() -> None:
    """Jalankan Tahap 4: Filtering & Convolution."""
    print("\n--- [TAHAP 4] Filtering & Convolution ---")
    script_path = BASE_DIR / "scripts" / "filtering_convolution_04.py"
    if not script_path.is_file():
        print("Error: Script scripts/filtering_convolution_04.py belum tersedia.")
        return
    try:
        import subprocess

        # stdio diwariskan agar input interaktif di script bisa diakses user
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(
                f"Gagal menjalankan Tahap 4 (code {result.returncode})"
            )
    except Exception as err:
        print(f"Gagal menjalankan Tahap 4: {err}")


def run_step_05() -> None:
    """Jalankan Tahap 5: Masking."""
    print("\n--- [TAHAP 5] Masking ---")
    script_path = BASE_DIR / "scripts" / "masking_05.py"
    if not script_path.is_file():
        print("Error: Script scripts/masking_05.py belum tersedia.")
        return
    try:
        import subprocess

        # stdio diwariskan agar input interaktif di script bisa diakses user
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(
                f"Gagal menjalankan Tahap 5 (code {result.returncode})"
            )
    except Exception as err:
        print(f"Gagal menjalankan Tahap 5: {err}")


def run_step_06() -> None:
    """Jalankan Tahap 6: Placeholder (Belum diimplementasikan)."""
    print("\n--- [TAHAP 6] Placeholder ---")
    print("Script tahap 6 belum dibuat.")


def run_step_07() -> None:
    """Jalankan Tahap 7: Placeholder (Belum diimplementasikan)."""
    print("\n--- [TAHAP 7] Placeholder ---")
    print("Script tahap 7 belum dibuat.")


def run_all_steps() -> None:
    """Jalankan semua tahap secara sekuensial (1-7)."""
    print("\n=== MENJALANKAN SELURUH PIPELINE ===")
    run_step_01()
    run_step_02()
    run_step_03()
    run_step_04()
    run_step_05()
    run_step_06()
    run_step_07()
    print("\n=== PIPELINE SELESAI ===")


MENU_ACTIONS: Dict[str, Callable[[], None]] = {
    "1": run_step_01,
    "2": run_step_02,
    "3": run_step_03,
    "4": run_step_04,
    "5": run_step_05,
    "6": run_step_06,
    "7": run_step_07,
    "a": run_all_steps,
}


def display_menu() -> None:
    """Tampilkan menu interaktif utama ke terminal."""
    print("\n" + "=" * 40)
    print("  DAISY IMAGE PROCESSING PIPELINE")
    print("=" * 40)
    print(" [1] Tahap 01: Eksplorasi Dataset")
    print(" [2] Tahap 02: Grayscale & Histogram")
    print(" [3] Tahap 03: Thresholding")
    print(" [4] Tahap 04: Filtering & Convolution")
    print(" [5] Tahap 05: Masking")
    print(" [6] Tahap 06: (Belum dibuat)")
    print(" [7] Tahap 07: (Belum dibuat)")
    print(" [a] Jalankan Semua Tahap (1-7)")
    print(" [0] Keluar")
    print("-" * 40)


def main() -> None:
    """Fungsi utama loop interaktif CLI."""
    while True:
        display_menu()
        choice = input("Pilih menu (0-7 / a): ").strip().lower()

        if choice == "0":
            print("\nTerima kasih telah menggunakan pipeline pengolahan citra daisy.")
            sys.exit(0)
        elif choice in MENU_ACTIONS:
            action = MENU_ACTIONS[choice]
            action()
            input("\nTekan [Enter] untuk kembali ke menu utama...")
        else:
            print("\nPilihan tidak valid. Masukkan angka 0-7 atau 'a'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh pengguna. Keluar...")
        sys.exit(0)
