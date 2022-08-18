import numpy as np
import cv2

def translation(img, T):
    '''
    Function translate image
    '''
    height, width = img.shape
    translated_img = np.zeros((width, height), np.float32)     
    for i in range(width):
        for j in range(height):
            vector = np.array([[i], [j], [1]])
            new = np.dot(T, vector)
            if new[0]>= 0 and new[0]<width and new[1]>= 0 and new[1]< height:
                translated_img[int(new[0]), int(new[1])] = img[i, j]
            
    return np.uint8(translated_img)

def scale(img,  min=0.2, max=0.3):
    '''
    Function resize image 
    '''
    rows,cols = img.shape

    #Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = getRotationMatrix((cols/2,rows/2), 0, scale)
    return translation(img, M)

def getRotationMatrix(center, angle, scale):
    '''
    Function rotate image
    '''
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    
    matrix = np.array([[alpha, beta, (1-alpha)*center[0] - beta*center[1]],[-beta, alpha, beta*center[0]+(1-alpha)*center[1]]])

    return matrix
    
if __name__ == "__main__":
    
    img = cv2.imread("lena.jpg", 0)
    cv2.imwrite("Resultados/Topico_1/grayscale_original.jpg", img)
    h, w = img.shape
    
    #Translation
    tx = 50
    ty = 50
    
    T = np.array([[1, 0,tx],[0, 1, ty]])
    
    translated_img = translation(img, T)
    cv2.imwrite("Resultados/Topico_1/grayscale_translated.jpg", translated_img)
    
    #Rotation
    
    T = getRotationMatrix((w/2, h/2), 90, 1)

    rotated_img = translation(img, T)
    cv2.imwrite("Resultados/Topico_1/grayscale_rotated.jpg", rotated_img)

    #Scale
    
    scaled_img = scale(img)
    cv2.imwrite("Resultados/Topico_1/grayscale_scaled.jpg", scaled_img)

    
    concatenate = np.concatenate((img, translated_img, rotated_img, scaled_img), axis = 1)
    cv2.imshow("img - translation - rotation - scale", concatenate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()