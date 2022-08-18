import cv2
import numpy as np
from bib_pdi import *

if __name__ == "__main__":
    img = cv2.imread("casa.jpg", 0)
    img = cv2.resize(img, (200, 300))
    
    img_binary1 = filterMedian(img, times = 10)
    img_binary2 = filterMedian(img, times = 20)
    img_concatenate1 = np.concatenate((img, img_binary1), axis = 1)
    img_concatenate = np.concatenate((img_concatenate1, img_binary2), axis = 1)
    
    
    #img_concatenate = np.concatenate((img_concatenate1, img_concatenate2_), axis = 1)
    cv2.imwrite("test_kernels_median_repeat.jpg",img_concatenate)
    cv2.imshow("Aplicacoes do filtro da mediana 3x3", img_concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()