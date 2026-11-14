import cv2
import matplotlib.pyplot as plt
import numpy as np

# Memuat citra asli dalam mode grayscale (keabuan)
img = cv2.imread('Plat2.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Mean Filter (Averaging)
mean_3x3 = cv2.blur(img, (3, 3))
mean_5x5 = cv2.blur(img, (5, 5))

# 2. Gaussian Filter
gaussian_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
gaussian_5x5 = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Median Filter
median_3x3 = cv2.medianBlur(img, 3)
median_5x5 = cv2.medianBlur(img, 5)

# 4. Sharpening Filter (Menggunakan Laplacian Kernel)
kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel_sharpening)

# Visualisasi menggunakan Matplotlib
plt.figure(figsize=(14, 8))
titles = [
    'Citra Asli',
    'Mean 3x3',
    'Mean 5x5',
    'Gaussian 3x3',
    'Gaussian 5x5',
    'Median 3x3',
    'Median 5x5',
    'Sharpening',
]
images = [
    img,
    mean_3x3,
    mean_5x5,
    gaussian_3x3,
    gaussian_5x5,
    median_3x3,
    median_5x5,
    sharpened,
]

for i in range(8):
  plt.subplot(2, 4, i + 1)
  plt.imshow(images[i], cmap='gray')
  plt.title(titles[i], fontsize=10)
  plt.axis('off')

plt.tight_layout()
plt.show()