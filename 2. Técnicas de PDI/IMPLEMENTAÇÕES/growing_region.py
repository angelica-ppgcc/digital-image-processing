import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import itertools

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        s_box = x, y
        boxes.append(s_box)

def growing_region(img, seed):
    pixel_difference = 0
    T = 1
    neighbors = [p for p in itertools.product([-1,0,1], repeat=2)]
    neighbors.remove((0,0))
    
    list_points = []
    list_intensities = []
    region_size = 1
    height, width = img.shape
    image_size = height * width
    segmented_img = np.zeros((height, width), np.float32)
    region_mean = img[seed]
    i = 0
    while(pixel_difference < T and region_size < image_size):
        for neighbor in neighbors:
            x = seed[0] + neighbor[0]
            y = seed[1] + neighbor[1]
            
            in_image = (x >= 0) & (x < height) & (y >= 0) & (y < width)
            
            if in_image:
                if segmented_img[x, y] == 0:
                    list_points.append((x,y))
                    list_intensities.append(img[x,y])
                    segmented_img[x,y] = 255
                
        differencies = np.array(list_intensities) - img[seed[0], seed[1]]
        differencies = list(differencies)
        pixel_difference = min(differencies)
        index = differencies.index(pixel_difference)
        seed = list_points[index] 
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1
        
        region_mean = (region_mean*region_size + list_intensities[index])/(region_size+1)
        
        list_intensities.pop(index)
        list_points.pop(index)
        cv2.imwrite("Resultados/Topico_6/iteracao - "+str(i)+".jpg", segmented_img)
        i = i + 1
    return segmented_img

if __name__ == '__main__':
    
    boxes = []
    filename = 'lena.jpg'
    img = cv2.imread(filename, 0)
    cv2.imwrite("Resultados/Topico_6/lena.jpg", img)
    resized = cv2.resize(img,(100,100))
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', on_mouse, 0,)
    cv2.imshow('input', resized)
    
    cv2.waitKey()
    print("Starting region growing based on last click")
    
    seed = boxes[-1]
    res = growing_region(resized, seed)
    cv2.imshow('input', res)

    cv2.waitKey()
    cv2.destroyAllWindows()

        
                              
    
    