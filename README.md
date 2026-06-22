# face-recognition-svd

Aplikasi Streamlit untuk membandingkan dua wajah (SAMA / BEDA orang) memakai
**PCA/SVD (Eigenfaces)** klasik. Dirancang agar tetap
robust ketika dua foto diambil pada usia yang berbeda.

## Fitur Utama

- **Pilihan Metrik Jarak**: Membandingkan wajah menggunakan jarak **Cosine** atau **Euclidean**. Hasil kalibrasi *threshold* akan menyesuaikan secara dinamis.
- **Pengaturan Resolusi PCA**: Anda dapat mengubah jumlah komponen utama (*principal components* / nilai `k`) secara interaktif untuk menganalisis dampaknya.
- **Visualisasi Kalibrasi**: Dilengkapi dengan visualisasi distribusi skor jarak, kurva ROC, dan *Equal Error Rate* (EER) untuk setiap metrik.
- **Rekonstruksi PCA**: Melihat bagaimana PCA merepresentasikan wajah dengan noise yang telah dihilangkan (berdasarkan koefisien yang diproyeksikan ke *eigenspace*).

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

## Instalasi & Run

Karena repositori ini sudah menyertakan file model (*pre-computed eigenspace*) di dalam folder `artifacts/`, Anda tidak perlu membangun model dari awal. Cukup *install dependencies* dan jalankan aplikasi.

1. Buat dan aktifkan *virtual environment*:
   ```bash
   # Windows:
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   
   # Linux / MacOS:
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Instal *dependencies*:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi Streamlit:
   ```bash
   # Windows:
   streamlit.exe run app/main.py

   # Linux / MacOS:
   streamlit run app/main.py
   ```

### (Opsional) Membuat Ulang Model Eigenspace

Jika ingin membuat ulang model *eigenspace* (misalnya untuk mengubah resolusi atau jumlah komponen utama), jalankan skrip berikut (pastikan `.venv` sudah aktif):

```bash
python scripts/build_eigenspace.py
```
*(Catatan: Skrip ini akan mengunduh dataset LFW dan AT&T secara otomatis ke dalam folder `data/raw/` jika belum ada. Pastikan Anda telah meletakkan dataset Yale di `data/raw/yale_faces` secara manual jika ingin memanfaatkannya).*

## Testing

```bash
pytest
```
