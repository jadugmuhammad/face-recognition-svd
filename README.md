# face-recognition-svd

Aplikasi Streamlit untuk membandingkan dua wajah (SAMA / BEDA orang) memakai
**PCA/SVD (Eigenfaces)** klasik — tanpa deep learning. Dirancang agar tetap
robust ketika dua foto diambil pada usia yang berbeda.

## Arsitektur

```
app/      -> lapisan Streamlit (UI saja, tidak ada logika algoritma)
core/     -> pure Python/NumPy: preprocessing -> decomposition -> matching
data/     -> loader untuk dataset (AT&T, LFW, Yale)
scripts/  -> CLI untuk membangun & mengkalibrasi eigenspace sekali di awal
artifacts/-> hasil build (mean face, eigenvectors, statistik kalibrasi)
```

`core/` sengaja tidak tahu apa-apa soal Streamlit, supaya bisa ditest dengan
`pytest` tanpa menjalankan UI.

## Dataset

Semua dataset (Kecuali Yale) diunduh dan diekstrak **secara otomatis** oleh skrip saat Anda membangun *eigenspace*. Anda tidak perlu mengunduhnya secara manual.

| Dataset | Kegunaan |
|---|---|
| **AT&T / ORL** | Membangun eigenspace dasar & Kalibrasi Threshold (Isolasi untuk metrik yang ketat) |
| **LFW (Labeled Faces in the Wild)** | Membangun eigenspace (menyerap variasi wajah alami) |
| **Yale (Extended B)** | Membangun eigenspace (menyerap variasi pencahayaan tajam dan bayangan ekstrem) |

> **Catatan Yale Dataset:** Server asli universitas untuk dataset Extended Yale B sudah *offline*. Untuk menggunakan Yale, Anda harus mengunduhnya secara manual (misalnya melalui Kaggle) dan mengekstrak foldernya ke `data/raw/yale_faces/`. Jika folder ini tidak ada, skrip akan **melewati (skip)** dataset Yale secara otomatis dan hanya menggunakan AT&T + LFW.

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Membangun Model (Wajib Dijalankan Pertama Kali)

Sebelum menjalankan aplikasi, Anda diwajibkan untuk mengunduh dataset dan membangun model *eigenspace*. Proses ini hanya perlu dilakukan satu kali:

```bash
python scripts/build_eigenspace.py
```
*(Catatan: Skrip ini akan mengunduh dataset LFW dan AT&T secara otomatis ke dalam folder `data/raw/` jika belum ada. Pastikan Anda telah meletakkan dataset Yale di `data/raw/yale_faces` secara manual jika ingin memanfaatkannya).*

## 2. Menjalankan Aplikasi

Setelah file model `artifacts/eigenspace.npz` berhasil tercipta, Anda dapat menjalankan UI Streamlit:

```bash
streamlit run app/main.py
```

## Testing

```bash
pytest
```
