# face-recognition-svd

Aplikasi Streamlit untuk membandingkan dua wajah (SAMA / BEDA orang) memakai
**PCA/SVD (Eigenfaces)** klasik — tanpa deep learning. Dirancang agar tetap
robust ketika dua foto diambil pada usia yang berbeda.

## Status

Repo ini baru di tahap **setup struktur**. Modul-modul di `core/` dan
`data/loaders/` masih berupa skeleton (`raise NotImplementedError`) — lihat
checklist di bawah.

- [ ] `data/loaders/*` — parsing dataset
- [ ] `core/preprocessing/*` — deteksi wajah, alignment, normalisasi
- [ ] `core/decomposition/eigenfaces.py` — PCA/SVD
- [ ] `core/matching/*` — multi-metric distance + ensemble + threshold
- [ ] `core/pipeline.py` — orkestrasi end-to-end
- [ ] `scripts/build_eigenspace.py` — build + kalibrasi artifacts
- [x] `app/*` — kerangka UI Streamlit (sudah jalan, tapi logikanya placeholder)

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
| [AT&T / ORL](https://cam-orl.co.uk/facedatabase.html) | Bangun eigenspace dasar | Unduh, ekstrak ke `data/raw/att_faces/` |
| [Extended Yale Face Database B](https://paperswithcode.com/dataset/yale-face) | Bangun eigenspace (variasi cahaya) | Ekstrak ke `data/raw/yale_faces/` |
| [FG-NET](https://yanweifu.github.io/FG_NET_data/) | Validasi cross-age (genuine/impostor pairs) | Perlu request akses/lisensi, ekstrak ke `data/raw/fgnet/` |

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

Untuk membangun eigenspace + kalibrasi threshold (setelah dataset diunduh
dan modul `core`/`data` selesai diimplementasikan):

```bash
python scripts/build_eigenspace.py
```

## Testing

```bash
pytest
```
