# PCD Daisy Project

Pipeline pengolahan citra digital untuk dataset bunga Daisy menggunakan Python. Proyek ini mengimplementasikan 7 tahap pemrosesan citra dari eksplorasi dataset hingga evaluasi hasil.

## Fitur

- **Dataset Exploration** — Analisis karakteristik dataset (jumlah, dimensi, format)
- **Grayscale & Histogram** — Konversi dan analisis distribusi intensitas piksel
- **Thresholding** — Segmentasi citra dengan multiple methods (global, adaptive, otsu)
- **Filtering & Convolution** — Noise reduction dan edge detection (mean, median, Gaussian, Laplacian, Sobel)
- **Masking** — Region-based processing (circular, rectangular, threshold masks)
- **Problem-Specific Solution** — Placeholder untuk solusi spesifik
- **Evaluation** — Placeholder untuk evaluasi hasil

## Tech Stack

| Komponen | Fungsi |
|----------|--------|
| Python 3.12 | Bahasa pemrograman utama |
| OpenCV | Operasi pengolahan citra |
| Matplotlib | Visualisasi citra dan histogram |
| NumPy | Operasi numerik array |
| Pandas | Analisis data tabular |
| scikit-image | Utility pengolahan citra tambahan |
| Pillow | Membaca dan menyimpan gambar |
| Seaborn | Visualisasi data statistik |

## Prasyarat

- Python >= 3.12
- pip package manager
- Git (untuk clone dan version control)

## Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd PCD_Daisy_Project
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
```

### 3. Aktifkan Virtual Environment

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Siapkan Dataset

Tempatkan citra Daisy pada folder:
- `dataset/training_data/` — Data untuk eksperimen dan pengembangan
- `dataset/testing_data/` — Data untuk pengujian

Dataset bunga Daisy tersedia di: [Kaggle Flowers Dataset](https://www.kaggle.com/datasets)

## Konfigurasi Environment

### `.gitignore`

File `.gitignore` menentukan file dan folder yang **tidak** di-track oleh Git. Ini mencegah commit file sensitif, dependency besar, dan output generate otomatis. Sudah dikonfigurasi untuk:

- **Secrets**: `.env`, `.env.*` (kecuali `.env.example`)
- **Dependencies**: `node_modules/`, `vendor/`, `.venv/`, `__pycache__/`
- **Build output**: `dist/`, `build/`, `results/`, `output/`
- **Logs & cache**: `*.log`, `*.tmp`, `.cache/`
- **IDE/OS**: `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`

### `.env.example`

File `.env.example` adalah template untuk variabel environment. Salin ke `.env` dan isi nilainya:

```bash
cp .env.example .env
```

**Windows (Command Prompt):**
```bat
copy .env.example .env
```

> ⚠️ **Penting**: `.env` TIDAK boleh di-commit. Sudah diabaikan oleh `.gitignore`.

### Variabel Environment

<!-- TODO: verifikasi — kode saat ini belum memakai env var. Tabel berikut placeholder untuk pengembangan Tahap 06/07. -->

| Nama Variabel | Deskripsi | Wajib | Default |
|---------------|-----------|-------|---------|
| `MPLBACKEND` | Backend Matplotlib (`Agg` = headless, `TkAgg` = GUI) | Tidak | `Agg` |
| `TRAIN_DATA_DIR` | Path dataset training | Tidak | `dataset/training_data` |
| `TEST_DATA_DIR` | Path dataset testing | Tidak | `dataset/testing_data` |
| `RESULTS_DIR` | Folder output hasil pipeline | Tidak | `results` |

## Penggunaan

### Menjalankan Pipeline Interaktif

```bash
python main.py
```

Menu interaktif akan muncul:

```
========================================
  DAISY IMAGE PROCESSING PIPELINE
========================================
 [1] Tahap 01: Eksplorasi Dataset
 [2] Tahap 02: Grayscale & Histogram
 [3] Tahap 03: Thresholding
 [4] Tahap 04: Filtering & Convolution
 [5] Tahap 05: Masking
 [6] Tahap 06: (Belum dibuat)
 [7] Tahap 07: (Belum dibuat)
 [a] Jalankan Semua Tahap (1-7)
 [0] Keluar
```

### Menjalankan Script Individual

```bash
python scripts/dataset_exploration_01.py
python scripts/grayscale_histogram_02.py
python scripts/thresholding.py
python scripts/filtering_convolution_04.py
python scripts/masking_05.py
```

Setiap script meminta input interaktif untuk memilih gambar dari daftar.

### Testing

<!-- TODO: verifikasi — belum ada test suite. Rencana unit testing ada di "Pengembangan Selanjutnya". -->

Belum ada test. Jalankan pipeline untuk verifikasi manual:

```bash
python main.py
# pilih menu a untuk menjalankan semua tahap
```

## Struktur Project

```
PCD_Daisy_Project/
│
├── dataset/
│   ├── training_data/    # 150 citra Daisy untuk eksperimen
│   └── testing_data/     # Citra untuk pengujian
│
├── scripts/
│   ├── dataset_exploration_01.py
│   ├── grayscale_histogram_02.py
│   ├── thresholding.py
│   ├── filtering_convolution_04.py
│   └── masking_05.py
│
├── results/              # Output generate (di-ignore Git)
│   ├── grayscale/        # Citra grayscale
│   ├── histograms/       # Visualisasi histogram
│   ├── thresholding/     # Hasil segmentasi
│   ├── filtering/        # Hasil filtering per method
│   │   ├── mean/
│   │   ├── median/
│   │   ├── gaussian/
│   │   ├── laplacian/
│   │   ├── sobel/
│   │   └── reference/
│   └── masking/          # Hasil masking
│       ├── binary/
│       ├── kernel/
│       ├── masks/
│       └── reference/
│
├── documentations/       # Hasil eksplorasi dataset
│   ├── dataset_exploration.csv
│   └── dataset_exploration.md
│
├── output/               # Output tambahan (di-ignore Git)
│   ├── thresholding_03/
│   └── thresholding_report.md
│
├── main.py               # CLI orchestrator
├── requirements.txt      # Dependencies
├── .gitignore            # File yang diabaikan Git
├── .env.example          # Template variabel environment
└── README.md
```

> <!-- TODO: verifikasi — `documentations/*.csv` dan `documentations/*.md` ada di `.gitignore`, tapi README menyebut file tersebut sebagai output. Tentukan apakah file dokumentasi hasil generate ingin di-commit atau tidak. -->

## Workflow Pipeline

```
Training Data (150 images)
         |
         v
[01] Dataset Exploration
         | → dataset_exploration.csv, .md
         v
[02] Grayscale & Histogram
         | → grayscale images, histogram plots
         v
[03] Thresholding
         | → segmented images (global, adaptive, otsu)
         v
[04] Filtering & Convolution
         | → filtered images (mean, median, gaussian, laplacian, sobel)
         v
[05] Masking
         | → masked images (circular, rectangular, threshold)
         v
[06] Problem-Specific Solution
         | → (placeholder)
         v
Testing Data
         |
         v
[07] Evaluation
         | → (placeholder)
         v
Analysis & Documentation
```

## Detail Setiap Tahap

### Tahap 01: Dataset Exploration

Menganalisis karakteristik dataset:
- Jumlah total citra
- Distribusi dimensi (tinggi, lebar)
- Format file (JPG, PNG, dll)
- Ukuran file
- Jumlah kanal warna

Output: `documentations/dataset_exploration.csv` dan `dataset_exploration.md`

### Tahap 02: Grayscale & Histogram

Mengonversi citra RGB ke grayscale dan menganalisis:
- Distribusi intensitas piksel
- Statistik (mean, std, min, max, median)
- Visualisasi histogram

Output: `results/grayscale/` dan `results/histograms/`

### Tahap 03: Thresholding

Menerapkan berbagai metode thresholding:
- Global thresholding
- Adaptive thresholding (mean, gaussian)
- Otsu's thresholding

Output: `results/thresholding/`

### Tahap 04: Filtering & Convolution

Menerapkan filter untuk:
- Noise reduction: Mean, Median, Gaussian
- Edge detection: Laplacian, Sobel

Output: `results/filtering/<method>/`

### Tahap 05: Masking

Membuat dan menerapkan mask:
- Circular mask
- Rectangular mask
- Threshold-based mask

Output: `results/masking/`

## Status Project

| Tahap | Status | Progress |
|-------|--------|----------|
| 01. Dataset Exploration | ✅ Selesai | 100% |
| 02. Grayscale & Histogram | ✅ Selesai | 100% |
| 03. Thresholding | ✅ Selesai | 100% |
| 04. Filtering & Convolution | ✅ Selesai | 100% |
| 05. Masking | ✅ Selesai | 100% |
| 06. Problem-Specific Solution | 🔲 Belum | 0% |
| 07. Evaluation | 🔲 Belum | 0% |

**Overall Progress: 71% (5/7 tahap)**

## Kontribusi & Git Workflow

Project menggunakan **GitHub Flow**:

```
main (protected, selalu deployable)
  │
  ├── feature/tahap-06      → PR → merge ke main
  ├── feature/tahap-07      → PR → merge ke main
  └── fix/bug-thresholding  → PR → merge ke main
```

### Aturan

1. **Jangan push langsung ke `main`.** Buat branch fitur dulu.
2. Branch fitur dari `main`, nama format: `feature/<deskripsi>` atau `fix/<deskripsi>`.
3. Commit dengan format Conventional Commits: `<type>(<scope>): <subject>`.
   - Contoh: `feat(masking): add circular mask support`
4. Buka Pull Request (PR) untuk review sebelum merge.
5. Setelah PR di-approve, merge ke `main`.

### Contoh Alur Kerja

```bash
# Update dari main
git checkout main
git pull origin main

# Buat branch fitur
git checkout -b feature/tahap-06

# ... kode ...

# Commit & push
git add .
git commit -m "feat(evaluation): add tahap 06 solution"
git push origin feature/tahap-06

# Buka PR di GitHub, lalu merge setelah review
```

## Pengembangan Selanjutnya

1. Implementasi Tahap 06 — Problem-specific solution
2. Implementasi Tahap 07 — Evaluation metrics dan reporting
3. Batch processing untuk multiple images
4. Config file untuk parameter tuning
5. Unit testing

## Catatan Teknis

- Setiap script berjalan independen melalui subprocess
- Input interaktif menggunakan stdin
- Hasil disimpan dengan naming convention: `{filename}_{method}.{ext}`
- Matplotlib backend: Agg (non-interactive) untuk headless execution

## Lisensi

Proyek untuk keperluan pembelajaran Pengolahan Citra Digital.

---

**Project:** PCD Daisy Project  
**Domain:** Digital Image Processing  
**Language:** Python 3.12  
**IDE:** Visual Studio Code  
**Dataset:** Daisy Flower (150 images)
