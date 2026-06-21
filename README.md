# face-recognition-svd

Aplikasi Streamlit untuk membandingkan dua wajah (SAMA / BEDA orang) memakai
**PCA/SVD (Eigenfaces)** klasik — tanpa deep learning. Dirancang agar tetap
robust ketika dua foto diambil pada usia yang berbeda.

## Arsitektur

```
app/      -> lapisan Streamlit (UI saja, tidak ada logika algoritma)
core/     -> pure Python/NumPy: preprocessing -> decomposition -> matching
data/     -> loader untuk dataset (AT&T, LFW) dengan fitur auto-download
scripts/  -> CLI untuk membangun & mengkalibrasi eigenspace sekali di awal
artifacts/-> hasil build (mean face, eigenvectors, statistik kalibrasi)
```

`core/` sengaja tidak tahu apa-apa soal Streamlit, supaya bisa ditest dengan
`pytest` tanpa menjalankan UI.

## Dataset

Semua dataset diunduh dan diekstrak **secara otomatis** oleh skrip saat Anda membangun *eigenspace*. Anda tidak perlu mengunduhnya secara manual.

| Dataset | Kegunaan |
|---|---|
| **AT&T / ORL** | Membangun eigenspace dasar & Kalibrasi Threshold (Isolasi untuk metrik yang ketat) |
| **LFW (Labeled Faces in the Wild)** | Membangun eigenspace (menyerap variasi senyum, cahaya, dan pose ekstrem) |

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
*(Catatan: Skrip ini akan mengunduh dataset LFW dan AT&T secara otomatis ke dalam folder `data/raw/` jika belum ada).*

## 2. Menjalankan Aplikasi

Setelah file model `artifacts/eigenspace.npz` berhasil tercipta, Anda dapat menjalankan UI Streamlit:

```bash
streamlit run app/main.py
```

## Testing

```bash
pytest
```
