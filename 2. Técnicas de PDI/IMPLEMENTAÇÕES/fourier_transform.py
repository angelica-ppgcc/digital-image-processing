import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m


img = cv2.imread('livingroom.jpg',0)
rows, cols = img.shape
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
#plt.figure(2,figsize=(12,12))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title("Imagem original")
#plt.axis('OFF')

plt.subplot(122)
plt.imshow(magnitude_spectrum, 'gray')
plt.title("Espectro em frequencia")
#plt.axis('OFF')

plt.show()

#Calculate the inverse

img_idft = np.fft.ifft2(f)
img_inversa = np.abs(img_idft)

plt.figure(2)
plt.imshow(img_inversa, 'gray')
plt.title("Imagem apos IDFT")
plt.axis('OFF')

plt.show()

def laplacianKernel(h1, h2):
    
    sizeKernel = np.array([h1,h2])

    kernel = -1*np.ones(sizeKernel)

    center = np.uint8(kernel.shape[0]/2)

    kernel[center,center] = (-1*np.sum(kernel)) -1
    
    return kernel

def gaussianKernel(h1, h2):
    

    ## Returns a normalized 2D gauss kernel array for general purporses

    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 2
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) )
    return g / g.sum()

filterKernel = laplacianKernel(rows,cols)

filter_dft = np.fft.fft2(filterKernel)
filter_dft_shift = np.fft.fftshift(filter_dft)
filter_dft_mag = np.abs(filter_dft_shift)

filter_dft_mag = filter_dft_mag

plt.figure(4)
plt.imshow(filter_dft_mag, 'gray')
plt.title("Espectro em frequencia do filtro Gaussiano com spread 2")
plt.show()

filter_img = fshift * filter_dft_mag
filter_img_mag = np.abs(filter_img)

#img_back = np.fft.fftshift(np.fft.ifft2(filter_img))

img_back = np.fft.ifft2(filter_img)

img_back_mag = np.abs(img_back)

plt.figure(5, figsize=(12,12))
plt.subplot(221)
plt.imshow((img), 'gray')
plt.title("Imagem original")

plt.subplot(222)
plt.imshow(20*np.log(magnitude_spectrum), 'gray')
plt.title("Espectro da imagem original")

plt.subplot(223)
plt.imshow(img_back_mag, 'gray')
plt.title("Imagem apos filtragem na frequencia")

plt.subplot(224)
plt.imshow(20*np.log(filter_img_mag), 'gray')
plt.title("Especto da imagem apos filtragem na frequencia")

plt.show()
'''
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])
plt.show()
'''