import cv2
import numpy as np
from bib_pdi import *

if __name__ == "__main__":
    img = cv2.imread("casa.jpg", 0)
    img = cv2.resize(img, (200, 300))
    
    img_binary1 = filterSobel(img)
    
    img_concatenate = np.concatenate((img, img_binary1), axis = 1)
    
    #img_concatenate = np.concatenate((img_concatenate1, img_concatenate2_), axis = 1)
    cv2.imwrite("test_sobel.jpg", img_concatenate)
    cv2.imshow("Aplicacao do filtro Sobel", img_concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()