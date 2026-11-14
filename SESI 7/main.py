
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. MEMBACA CITRA DOKUMEN HASIL SCAN
# ==================================================

nama_file = "dilation.png"

img = cv2.imread(nama_file)

if img is None:
    raise FileNotFoundError(
        f"Gambar '{nama_file}' tidak ditemukan. "
        "Periksa nama file dan lokasi gambar."
    )

# ==================================================
# 2. KONVERSI CITRA KE GRAYSCALE
# ==================================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# ==================================================
# 3. MENINGKATKAN KONTRAS CITRA
# ==================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

contrast = clahe.apply(gray)

# ==================================================
# 4. THRESHOLDING
# MEMISAHKAN TULISAN DAN LATAR BELAKANG
# ==================================================

binary = cv2.adaptiveThreshold(
    contrast,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21,
    10
)

# ==================================================
# 5. MEMBALIK WARNA CITRA
# TULISAN HITAM MENJADI PUTIH
# ==================================================

binary_inv = cv2.bitwise_not(binary)

# ==================================================
# 6. MEMBUAT STRUCTURING ELEMENT
# ==================================================

kernel = np.ones(
    (2, 2),
    np.uint8
)

# ==================================================
# 7. PROSES DILASI
# MEMPERTEBAL TULISAN
# ==================================================

dilasi = cv2.dilate(
    binary_inv,
    kernel,
    iterations=1
)

# ==================================================
# 8. MENGEMBALIKAN WARNA CITRA
# TULISAN PUTIH MENJADI HITAM
# ==================================================

hasil_dilasi = cv2.bitwise_not(dilasi)

# ==================================================
# 9. MENYIMPAN HASIL
# ==================================================

cv2.imwrite(
    "hasil_perbaikan_dokumen.jpg",
    hasil_dilasi
)

# ==================================================
# 10. MENAMPILKAN PERBANDINGAN HASIL
# ==================================================

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
)
plt.title("1. Dokumen Asli")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("2. Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(contrast, cmap="gray")
plt.title("3. Peningkatan Kontras")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(binary, cmap="gray")
plt.title("4. Thresholding")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(binary_inv, cmap="gray")
plt.title("5. Inversi Citra")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(hasil_dilasi, cmap="gray")
plt.title("6. Hasil Dilasi")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Proses selesai!")
print("Hasil disimpan sebagai hasil_perbaikan_dokumen.jpg")
