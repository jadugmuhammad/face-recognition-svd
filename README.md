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
| **AT&T / ORL** | Bangun eigenspace dasar & Kalibrasi Threshold (Isolasi untuk metrik yang ketat) |
| **LFW (Labeled Faces in the Wild)** | Bangun eigenspace (menyerap variasi senyum, cahaya, dan pose ekstrem) |

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan

```bash
streamlit run app/main.py
```

Untuk membangun eigenspace + kalibrasi threshold (dataset akan diunduh otomatis ke `data/raw/` jika belum ada):

```bash
python scripts/build_eigenspace.py
```

## Testing

```bash
pytest
```
