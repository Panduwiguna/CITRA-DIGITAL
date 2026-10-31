import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("pemandangan.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Horizontal Flip (flipCode = 1)
# Mencerminkan gambar secara horizontal (kiri ke kanan)
img_h_flip = cv2.flip(img, 1)

# 2. Vertical Flip (flipCode = 0)
# Mencerminkan gambar secara vertikal (atas ke bawah)
img_v_flip = cv2.flip(img, 0)

# Menampilkan hasil menggunakan Matplotlib (Grid 1x3)
plt.figure(figsize=(15, 5))

# Citra Asli
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")

# Horizontal Flip
plt.subplot(1, 3, 2)
plt.imshow(img_h_flip)
plt.title("Horizontal Flip")
plt.axis("off")

# Vertical Flip
plt.subplot(1, 3, 3)
plt.imshow(img_v_flip)
plt.title("Vertical Flip")
plt.axis("off")

plt.tight_layout()
plt.show()