from pywt import dwt2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt



img = cv2.imread("lena_gray_512.jpg", 0)

haar = pywt.Wavelet('haar')

(cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(img, haar, level=3)


plt.subplot(121)
plt.imshow(cA3, 'gray')
plt.title('Imagem original')
plt.subplot(122)
plt.imshow(cD3, 'gray')
plt.show()

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


concatenate = np.concatenate((cH, cV, cD), axis = 1)

cv2.imshow("Result-Haar",concatenate)
cv2.waitKey(0)
cv2.destroyAllWindows()
