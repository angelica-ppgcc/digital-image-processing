import numpy as np
import cv2
from bib_pdi import *
import collections

def erode(img, kernel, iterations = 1):
    '''
    Function erode binary image
    '''
    i = 0
    while(i != iterations):
        kernelsize = kernel.shape[0]
        width = img.shape[0]
        height = img.shape[1]
        convolution = np.zeros((kernel.shape[0], kernel.shape[1]), np.float32)
        center = kernel.shape[0]/2

        img_copy = np.zeros((width+2*center, height+2*center), np.float32)
        
        img_copy[center:(width+center), center:(height+center)] = img[:,:]

        img_result = img.copy()
        img_result = np.float32(img_result)#Solucao do meu problema 
        
        for x in range(center,width+center):
            for y in range(center,height+center):
                
                convolution = kernel * img_copy[x-kernelsize/2:x+kernelsize/2+1,y-kernelsize/2:y+kernelsize/2+1]
            
                kernel[kernel == 1] = 255
                
                if np.array_equal(convolution,kernel):
                    new_value = 255
    
                else:
                    new_value = 0
                kernel[kernel == 255] = 1
                img_result[x-center, y-center] = int(new_value)
                
        img = img_result.copy()
        i = i + 1
            
    img_result = np.uint8(img_result)
    return img_result

def dilate(img, kernel, iterations = 1):
    '''
    Function dilate binary image
    '''
    i = 0
    while(i != iterations):
        kernelsize = kernel.shape[0]
        width = img.shape[0]
        height = img.shape[1]
        convolution = np.zeros((kernel.shape[0], kernel.shape[1]), np.float32)
        center = kernel.shape[0]/2

        img_copy = np.zeros((width+2*center, height+2*center), np.float32)
        img_copy[center:(width+center), center:(height+center)] = img[:,:]

        img_result = img.copy()
        img_result = np.float32(img_result)#Solucao do meu problema 
        
        for x in range(center,width+center):
            for y in range(center,height+center):
                
                convolution = kernel * img_copy[x-kernelsize/2:x+kernelsize/2+1,y-kernelsize/2:y+kernelsize/2+1]
                
                if 255 in convolution:
                    new_value = 255
    
                else:
                    new_value = 0
                    
                img_result[x-center, y-center] = int(new_value)
        
        img = img_result.copy()
        i = i + 1
        
    img_result = np.uint8(img_result)
    
    return img_result

def grayscale_erode(img, kernel, iterations = 1):
    '''
    Function erode grayscale image
    '''
    
    i = 0
    while(i != iterations):
        kernelsize = kernel.shape[0]
        
        width = img.shape[0]
        height = img.shape[1]
        convolution = np.zeros((kernel.shape[0], kernel.shape[1]), np.float32)
        center = kernel.shape[0]/2

        img_copy = np.zeros((width+2*center, height+2*center), np.float32)
    
        img_copy[center:(width+center), center:(height+center)] = img[:,:]

        img_result = img.copy()
        img_result = np.float32(img_result)#Solucao do meu problema 
        
        for x in range(center,width+center):
            for y in range(center,height+center):
                
                convolution = kernel * img_copy[x-kernelsize/2:x+kernelsize/2+1,y-kernelsize/2:y+kernelsize/2+1]
            
                n = np.count_nonzero(convolution)
                
                if n < np.sum(kernel):
                    new_value = 0
                    
                else:    
                    convolution = convolution[convolution !=0]
                    new_value = np.min(convolution)
                   
                    
                img_result[x-center, y-center] = new_value
        
        img = img_result.copy()
        i = i + 1        
        
    img_result = np.uint8(img_result)
        
    return img_result


def grayscale_dilate(img, kernel, iterations = 1):
    i = 0
    while(i!=iterations):
        
        kernelsize = kernel.shape[0]
    
        width = img.shape[0]
        height = img.shape[1]
        convolution = np.zeros((kernel.shape[0], kernel.shape[1]), np.float32)
        center = kernel.shape[0]/2

        img_copy = np.zeros((width+2*center, height+2*center), np.float32)
    
        img_copy[center:(width+center), center:(height+center)] = img[:,:]

        img_result = img.copy()
        img_result = np.float32(img_result)#Solucao do meu problema 
        
        for x in range(center,width+center):
            for y in range(center,height+center):
                
                convolution = kernel * img_copy[x-kernelsize/2:x+kernelsize/2+1,y-kernelsize/2:y+kernelsize/2+1]
                
                new_value = np.max(convolution)
                    
                img_result[x-center, y-center] = new_value
                
        #img_result = np.uint8(img_result)
        img = img_result.copy()
        i = i + 1
    
    img_result = np.uint8(img_result)
            
    return img_result

def gradient(img, kernel):
    gradient_img = grayscale_dilate(img, kernel) - grayscale_erode(img, kernel)
    return gradient_img

if __name__ == "__main__":
    #img = cv2.imread("lena_gray_512.jpg",0)
    img = cv2.imread("predio.jpg", 0)
    img = cv2.resize(img, (200, 200))
    
    #cv2.imwrite("Resultados/Topico_3/g_img_original.jpg", img)
    #img = threshold(img)
    #img_copy = img.copy()
    #img[img_copy == 0] = 255
    #img[img_copy == 255] = 0
    
    
    # Morphological Gradient
    '''s = 7
    kernel = np.ones((s,s), np.float32)
    
    gradient_img = gradient(img, kernel)
    
    concatenate = np.concatenate((img, gradient_img), axis = 1)
    
    cv2.imshow("Gradient image", concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Resultados/Topico_4/gradient_7.jpg", gradient_img)
    '''
    # Topico 3
    
    #Erosion
    
    '''es1 = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]])
    es2 = np.ones((3,3))
    es3 = np.array([[1, 0, 0],[1, 0, 0],[1, 1, 1]])
    ''''''
    s = 3
    
    es1 = np.zeros((s,s))'''
    '''
    
    '''
    s = 3
    es1 = np.ones((s,s))
    '''
    es1[s/2,:] = 1
    es1[:,s/2] = 1
    
    ''''''
    es1[:,0] = 1
    es1[-1,:] = 1'''
    '''
    s = 7
    
    es2 = np.zeros((s,s))'''
    '''
    
    '''
    s = 7
    es2 = np.ones((s,s))
    '''es2[s/2,:] = 1
    es2[:,s/2] = 1'''
    
    '''
    es2[:,0] = 1
    es2[-1,:] = 1
    '''
    '''
    
    
    es3 = np.zeros((s,s))'''
    '''
    
    '''
    s = 15
    es3 = np.ones((s,s))
    '''es3[s/2,:] = 1
    es3[:,s/2] = 1'''
    
    '''
    es3[:,0] = 1
    es3[-1,:] = 1
    '''
    
    ''''
    eroded_img1 = grayscale_erode(img, es1)
    cv2.imwrite("Resultados/Topico_3/dilate_img1_3.jpg", eroded_img1)
    
    eroded_img2 = grayscale_erode(img, es2)
    cv2.imwrite("Resultados/Topico_3/dilate_img2_3.jpg", eroded_img2)
    
    eroded_img3 = grayscale_erode(img, es3)
    cv2.imwrite("Resultados/Topico_3/dilate_img3_3.jpg", eroded_img3)
    
    concatenate = np.concatenate((img, eroded_img1, eroded_img2, eroded_img3), axis = 1)
    cv2.imshow("Erosao", concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #Dilatacao
    '''
    dilate_img1 = dilate(img, es1)
    dilate_img2 = dilate(img, es2)
    dilate_img3 = dilate(img, es3)
    
    concatenate = np.concatenate((img, dilate_img1, dilate_img2, dilate_img3), axis = 1)
    cv2.imshow("Dilatacao", concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
    '''
    '''
    img_res = grayscale_erode(img, kernel)
    conc = np.concatenate((img, img_res), axis = 1)
    cv2.imshow("resultado", conc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #Grayscale Erode
    
    eroded_img1 = grayscale_erode(img, es1)
    cv2.imwrite("Resultados/Topico_3/g_erode_img1_2.jpg", eroded_img1)
    
    eroded_img2 = grayscale_erode(img, es2)
    cv2.imwrite("Resultados/Topico_3/g_erode_img2_2.jpg", eroded_img2)
    
    eroded_img3 = grayscale_erode(img, es3)
    cv2.imwrite("Resultados/Topico_3/g_erode_img3_2.jpg", eroded_img3)
    
    concatenate = np.concatenate((img, eroded_img1, eroded_img2, eroded_img3), axis = 1)
    cv2.imshow("Erosao", concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()