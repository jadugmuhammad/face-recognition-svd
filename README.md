# face-recognition-svd

Aplikasi Streamlit untuk membandingkan dua wajah (SAMA / BEDA orang) memakai
**PCA/SVD (Eigenfaces)** klasik — tanpa deep learning. Dirancang agar tetap
robust ketika dua foto diambil pada usia yang berbeda.

## Status

Repo ini telah **selesai diimplementasikan** secara end-to-end. Semua modul di `core/`, `data/loaders/`, dan `app/` telah berfungsi penuh (85/85 tests passed).

- [x] `data/loaders/*` — parsing dataset (AT&T, Yale)
- [x] `core/preprocessing/*` — auto-grayscaling, deteksi wajah, alignment (Haar), normalisasi (CLAHE)
- [x] `core/decomposition/eigenfaces.py` — PCA/SVD dengan **Whitening** (menyeimbangkan bobot varians fitur)
- [x] `core/matching/*` — perbandingan vektor fitur menggunakan murni **Cosine Distance** + threshold berbasis ROC/EER
- [x] `core/pipeline.py` — orkestrasi end-to-end
- [x] `scripts/build_eigenspace.py` — build + kalibrasi artifacts
- [x] `app/*` — UI Streamlit terintegrasi penuh

## Arsitektur

```
app/      -> lapisan Streamlit (UI saja, tidak ada logika algoritma)
core/     -> pure Python/NumPy: preprocessing -> decomposition -> matching
data/     -> loader untuk tiap dataset (AT&T, Yale, FG-NET)
scripts/  -> CLI untuk membangun & mengkalibrasi eigenspace sekali di awal
artifacts/-> hasil build (mean face, eigenvectors, statistik kalibrasi)
```

`core/` sengaja tidak tahu apa-apa soal Streamlit, supaya bisa ditest dengan
`pytest` tanpa menjalankan UI.

## Dataset

Semua dataset diunduh manual (tidak otomatis lewat pip) dan diletakkan di
`data/raw/`:

| Dataset | Kegunaan | Catatan unduh |
|---|---|---|
| [AT&T / ORL](https://cam-orl.co.uk/facedatabase.html) | Bangun eigenspace dasar & Kalibrasi Threshold | Unduh, ekstrak ke `data/raw/att_faces/` |
| [Extended Yale Face Database B](https://paperswithcode.com/dataset/yale-face) | Bangun eigenspace (variasi cahaya) & Kalibrasi Threshold | Ekstrak ke `data/raw/yale_faces/` |

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

Untuk membangun eigenspace + kalibrasi threshold (pastikan semua dataset sudah diunduh ke `data/raw/`):

```bash
python scripts/build_eigenspace.py
```

## Testing

```bash
pytest
```
