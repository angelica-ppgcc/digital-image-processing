import cv2
import numpy as np
from bib_pdi import *

if __name__ == "__main__":
    img = cv2.imread("gato.jpg", 0)
    img = cv2.resize(img, (400, 300))

    #plotHistogram(img)
    img_equalize = equalizeHistogram(img)

    img_concatenate = np.concatenate((img, img_equalize), axis = 1)

    #plotHistogram(img_equalize)
    
    cv2.imshow("Equalizacao de histograma", img_concatenate)
    cv2.imwrite("gatinhos.jpg", img_concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()