import cv2
import matplotlib.pyplot as plt
    
img = cv2.imread("C:\\PANDU\\CITRA DIGITAL\\SESI 5\\Plat_noisy.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mean = cv2.mean(rgb)[:3]
gaussian = cv2.GaussianBlur(rgb, (5, 5), 0)
median = cv2.medianBlur(rgb, 5)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for a, im, title in zip(axes, [rgb, gaussian, median], 
["asli", "mean", "gaussian", "median"]):
    a.imshow(im) ; a.set_title(title); a.axis("off")
    plt.tight_layout() : plt.show()