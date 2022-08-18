import cv2
import numpy as np
from bib_pdi import *

if __name__ == "__main__":
    img = cv2.imread("casa.jpg", 0)
    img = cv2.resize(img, (200, 300))
    
    img_binary_1 = filterMean(img, 3)
    img_binary_2 = filterMean(img, 5)
    img_binary_3 = filterMean(img, 9)
    img_binary_4 = filterMean(img, 15)
    img_binary_5 = filterMean(img, 23)
    img_concatenate1 = np.concatenate((img, img_binary_1), axis = 1)
    img_concatenate2 = np.concatenate((img_concatenate1, img_binary_2), axis = 1)
    
    img_concatenate3 = np.concatenate((img_binary_3, img_binary_4), axis = 1)
    img_concatenate4 = np.concatenate((img_concatenate3, img_binary_5), axis = 1)
    
    img_concatenate = np.concatenate((img_concatenate2, img_concatenate4), axis = 0)
    cv2.imwrite("test_size_kernels_mean.jpg",img_concatenate)
    cv2.imshow("Filtro da media", img_concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()