import streamlit as st
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Konfigurasi Parameter
IMG_SIZE = (100, 100)
N_COMPONENTS = 50
THRESHOLD = 0.75
BACKGROUND_DATASET = "Datasets/att_faces/Training"

# Memuat model deteksi wajah OpenCV
@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_face_cascade()

# Membangun ruang PCA sekali saja dan menyimpannya di cache memori
@st.cache_data
def train_background_pca(dataset_path):
    X = []
    if not os.path.exists(dataset_path):
        return None, False

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder): continue
            
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")): # [cite: 170]
                image_path = os.path.join(person_folder, filename)
                
                # Baca gambar
                img = cv2.imread(image_path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # [cite: 309]
                
                # Preprocessing [cite: 322-326]
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    face_crop = gray
                else:
                    x, y, w, h = faces[0]
                    face_crop = gray[y:y+h, x:x+w]

                resized = cv2.resize(face_crop, IMG_SIZE)
                normalized = resized / 255.0
                X.append(normalized.flatten())

    if len(X) == 0:
        return None, False

    X = np.array(X)
    pca = PCA(n_components=N_COMPONENTS) # [cite: 178]
    pca.fit(X) 
    
    return pca, True

def process_uploaded_image(uploaded_file):
    """Membaca file dari Streamlit, mendeteksi wajah, dan meratakannya menjadi vektor."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Visualisasi kotak wajah untuk UI
    img_display = img.copy()
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    if len(faces) == 0:
        st.warning("Wajah tidak terdeteksi secara otomatis. Menggunakan seluruh area gambar.")
        face_crop = gray
    else:
        x, y, w, h = faces[0]
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_crop = gray[y:y+h, x:x+w]

    resized = cv2.resize(face_crop, IMG_SIZE)
    normalized = resized / 255.0
    vector = normalized.flatten().reshape(1, -1)
    
    return vector, img_display

# --- ANTARMUKA STREAMLIT ---
st.title("Sistem Verifikasi Wajah (1:1 Matching)")
st.write("Menggunakan ruang PCA untuk membandingkan dua unggahan gambar dan menentukan kemiripannya.")

# Status Dataset Latar Belakang
with st.spinner("Memuat dataset latar belakang dan melatih ruang SVD..."):
    pca_model, is_trained = train_background_pca(BACKGROUND_DATASET)

if not is_trained:
    st.error(f"Gagal memuat dataset dari `{BACKGROUND_DATASET}`. Pastikan direktori tersedia untuk melatih matriks dasar PCA.")
    st.stop()

st.success("Ruang PCA berhasil dilatih. Silakan unggah foto.")

# Layout dua kolom untuk unggahan foto
col1, col2 = st.columns(2)

with col1:
    st.subheader("Foto Subjek 1")
    file_1 = st.file_uploader("Unggah Foto 1", type=["jpg", "jpeg", "png"], key="file1")

with col2:
    st.subheader("Foto Subjek 2")
    file_2 = st.file_uploader("Unggah Foto 2", type=["jpg", "jpeg", "png"], key="file2")

# Tombol Eksekusi
if file_1 is not None and file_2 is not None:
    if st.button("Bandingkan Wajah", use_container_width=True):
        st.markdown("---")
        
        col_res1, col_res2 = st.columns(2)
        
        # Proses gambar 1
        vec_1, img_disp_1 = process_uploaded_image(file_1)
        with col_res1:
            st.image(img_disp_1, caption="Area Terdeteksi 1", use_column_width=True)
            
        # Proses gambar 2
        vec_2, img_disp_2 = process_uploaded_image(file_2)
        with col_res2:
            st.image(img_disp_2, caption="Area Terdeteksi 2", use_column_width=True)
            
        # Proyeksi ke ruang PCA dan hitung similarity [cite: 198-201]
        proj_1 = pca_model.transform(vec_1)
        proj_2 = pca_model.transform(vec_2)
        similarity = cosine_similarity(proj_1, proj_2)[0][0]
        
        # Keputusan
        st.subheader("Hasil Analisis SVD")
        st.metric("Skor Cosine Similarity", f"{similarity:.4f}")
        
        if similarity >= THRESHOLD: # [cite: 202-205]
            st.success("🟢 KEPUTUSAN: WAJAH SAMA (Identitas Terverifikasi)")
        else:
            st.error("🔴 KEPUTUSAN: WAJAH BERBEDA (Di bawah ambang batas)")