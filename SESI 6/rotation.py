import cv2
import matplotlib.pyplot as plt

# Membaca gambar (Pastikan file pemandangan.jpg berada di direktori yang sama)
img = cv2.imread("pemandangan.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tinggi, lebar = img.shape[:2]

# Menentukan titik pusat gambar
pusat = (lebar // 2, tinggi // 2)

# 1. Rotasi 45 derajat
M_45 = cv2.getRotationMatrix2D(pusat, 45, 1.0)
hasil_45 = cv2.warpAffine(img, M_45, (lebar, tinggi))

# 2. Rotasi 90 derajat
M_90 = cv2.getRotationMatrix2D(pusat, 90, 1.0)
hasil_90 = cv2.warpAffine(img, M_90, (lebar, tinggi))

# 3. Rotasi 180 derajat
M_180 = cv2.getRotationMatrix2D(pusat, 180, 1.0)
hasil_180 = cv2.warpAffine(img, M_180, (lebar, tinggi))


# Menampilkan hasil dalam grid 2x2 menggunakan Matplotlib
plt.figure(figsize=(12, 10))

# Citra Asli
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

# Rotasi 45°
plt.subplot(2, 2, 2)
plt.imshow(hasil_45)
plt.title("Rotasi 45 Derajat")
plt.axis("off")

# Rotasi 90°
plt.subplot(2, 2, 3)
plt.imshow(hasil_90)
plt.title("Rotasi 90 Derajat")
plt.axis("off")

# Rotasi 180°
plt.subplot(2, 2, 4)
plt.imshow(hasil_180)
plt.title("Rotasi 180 Derajat")
plt.axis("off")

plt.tight_layout()
plt.show()