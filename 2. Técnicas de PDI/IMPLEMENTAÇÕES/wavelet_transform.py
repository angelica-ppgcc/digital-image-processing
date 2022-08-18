from pywt import dwt2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt


img_haar = dwt2(img, "haar")

cA, (cH, cV, cD) = img_haar

plt.figure(2,figsize=(12,12))

plt.subplot(221)
plt.imshow(cA, 'gray')
plt.title('Imagem original')
plt.subplot(222)
plt.imshow(cH, 'gray')
plt.title('Imagem horizontais')
plt.subplot(223)
plt.imshow(cV, 'gray')
plt.title('Imagem verticais')
plt.subplot(224)
plt.imshow(cD, 'gray')
plt.title('Imagem diagonais')
plt.show()