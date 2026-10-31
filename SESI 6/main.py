import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\PANDU\CITRA DIGITAL\SESI 6\Plat_noisy.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')