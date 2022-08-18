import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def filter2D(img, kernel, filter = 1):
    '''
    --------------------------------------
    Standard Filter 2D
    
        Parameters:
        
        img: Image in 2D
        kernel: mask 2D of filter
        filter: Define the type filter
                1 --> Linear Filter 
                2 --> No Linear Filter
                
    Return: Filtered image
    --------------------------------------
    '''
    width = img.shape[0]
    height = img.shape[1]
    convolution = np.zeros((kernel.shape[0], kernel.shape[1]), np.float32)
    center = kernel.shape[0]/2

    img_copy = np.zeros((width+2*center, height+2*center), np.float32)
    
    img_copy[center:(width+center), center:(height+center)] = img[:,:]

    img_result = img.copy()
    img_result = np.float32(img_result)
    
    for x in range(center,width+center):
        for y in range(center,height+center):
            for i in range(center+1):
                convolution[center-i, center+i] = kernel[center-i, center+i] * img_copy[x-i, y+i]
                convolution[center+i, center-i] = kernel[center+i, center-i] * img_copy[x+i, y-i]
                for j in range(center+1):
                    convolution[center-i, center-j] = kernel[center-i, center-j] * img_copy[x-i, y-j]
                    convolution[center+i, center+j] = kernel[center+i, center+j] * img_copy[x+i, y+j]
              
            if filter == 1:
                new_value = np.sum(convolution)
    
            else:
                numbers = convolution.flatten()
                new_value = np.median(numbers)
                
            img_result[x-center, y-center] = int(new_value)
   
    return img_result

def filterMean(img, kernelsize = 3, times = 1):
    '''
    ----------------------------------------------------------
    Mean filter: Change each pixel by its neighbors mean
    ----------------------------------------------------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
        
    Return: 
        Blurring image 
    ----------------------------------------------------------
        
    '''
    
    kernel = np.ones((kernelsize, kernelsize), np.float32)/(kernelsize ** 2)
    
    for time in range(times):
        img_processed = filter2D(img, kernel, filter = 1)
        img = img_processed[:,:]
        
    img = np.uint8(img)
    return img
        
def filterMedian(img, kernelsize = 3, times = 1):
    '''
    ----------------------------------------------------------
    Median filter: Change each pixel by its neighbors median
    ----------------------------------------------------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    Return:
        Image without noise 
    ----------------------------------------------------------
    '''
    im = img.copy()
    kernel = np.ones((kernelsize, kernelsize), np.float32)
    
    for time in range(times):
        img_processed = filter2D(im, kernel, filter = 2)
        
        im = img_processed[:,:]
        
    im = np.uint8(im)
    return im

def filterLaplacian(img, kernelsize = 3, times = 1, type = "normal"):
    minimum = np.min(img)
    im = img.copy()
    '''
    ----------------------------------------------------------
    Laplacian filter: Change each pixel by its neighbors 
    weighted average
    ----------------------------------------------------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    Return:

    ----------------------------------------------------------      
    '''
    if(type == "normal"):
        kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])    

    else:    
        size = (kernelsize ** 2)-1  

        kernel = np.ones((kernelsize, kernelsize), np.float32)
        
        kernel[kernelsize/2, kernelsize/2] =  -size
        
    
    for time in range(times):
        img_processed = filter2D(im, kernel)
        img_processed[img_processed<0] = 0
        img_processed[img_processed>255] = 255
        im = img_processed[:,:]    
    
    img_processed = np.uint8(img_processed)

    return img_processed 

def filterPrewitt(img, kernelsize = 3, times = 1):
    '''
    ----------------------------------------------------------
    Prewitt Filter: Change each pixel by the 
    ----------------------------------------------------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    Return:
        Processed image
    '''
    im = img.copy()
    k1 = np.array([-1, 0, 1])
    k2 = np.array([1, 1, 1])
    
    kernel1 = np.array([[-1], [0], [1]]) * k2

    kernel2 = np.array([[1], [1], [1]]) * k1
    
    for time in range(times):
         
        img1 = filter2D(im, kernel1)
        img2 = filter2D(im, kernel2)    
        img_processed = np.sqrt(img1**2 + img2**2)
        img_processed[img_processed<0] = 0
        img_processed[img_processed>255] = 255
        im = img_processed[:,:]
        
    
    img_processed = np.uint8(img_processed)


    
    return img_processed

def filterSobel(img, kernelsize = 3, times = 1):
    '''
    --------------- Mean filter ---------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    Return:
        Processed image
    '''
    kernel1 = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    kernel2 = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    
    for time in range(times):
        img1 = filter2D(img, kernel1)
        img2 = filter2D(img, kernel2)
    
        img_processed = np.sqrt(pow(img1,2) + pow(img2,2))
        img_processed[img_processed<0] = 0
        img_processed[img_processed>255] = 255
        img = img_processed[:,:]
    
    img_processed = np.uint8(img_processed)
    
    return img_processed

def gaussian(x, y, sigma):
    factor = (1/np.sqrt(2*math.pi*(pow(sigma,2))))
    return factor*np.exp( -(pow(x,2) + pow(y,2))/(2*(pow(sigma,2))))

def filterGaussian(img, kernelsize = 3, times = 1, sigma = 3):
    '''
    --------------- Mean filter ---------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    
    Return:
        processed image 
    '''
    kernel = np.zeros((kernelsize, kernelsize), np.float32)
    size = kernelsize/2
    for x in range(-size, size+1):
        for y in range(-size, size+1):
            kernel[x+size, y+size] = gaussian(x, y, sigma)
    
    kernel = kernel/np.sum(kernel)
    
    for time in range(times):
        img_processed = filter2D(img, kernel)
        img = img_processed[:,:]
    
    img = np.uint8(img)
    return img

def calcHistogram(img):
    '''
    --------------- Calculate Histogram ---------------
    Parameters:
        img: image in 2D
        kernelsize: size of kernel mask
        times: times of application of filter
    
    Return:
        Dictionary representing the histogram of image
    '''
    img_flatten = img.flatten()
    
    keys = [x for x in range(0,256)]
    values = [0]*len(keys)

    histogram = dict(zip(keys, values))

    for pixel in img_flatten:
        histogram[pixel] = histogram[pixel] + 1 
    
    return histogram

def plotHistogram(img):
    '''
    --------------- Plot Histogram ---------------
    Parameters:
        img: image in 2D
    Output:
        Plot of histogram of image
    '''
    histogram = calcHistogram(img)
    plt.bar(histogram.keys(), histogram.values())
    plt.title('Histogram')
    plt.savefig("histogram.jpg")
    plt.show()
    

def equalizeHistogram(img):
    '''
    --------------- Histogram Equalization ---------------
    Parameters:
        img: image in 2D
    
    Return:
        Equalized image, i.e scattered grayscale image  
    '''
    
    width, height = img.shape

    equalized_img = np.zeros((width, height), np.uint8)

    hist = calcHistogram(img)

    vector_hist = np.array(zip(hist.keys(), hist.values()))
    max_min = vector_hist[vector_hist[:,1] >= 10][:,0]
    
    min_value = np.min(max_min)
    max_value = np.max(max_min)
    
    for x in range(width):
        for y in range(height):
            equalized_img[x,y] = int(255.0 * float(img[x,y] - min_value)/(max_value - min_value)) 

    return equalized_img

def threshold(img, T = 127, type = 'binaria'):
    '''
    --------------- Threshold ---------------
    Parameters:
        img: image in 2D
        T: number between 0 and 1
        type: type of threshold, can be 'binaria', 'saturada' or 'truncada'
    
    Return:
        Thresholded image
    '''
    width, height = img.shape 
    
    img_binary = img.copy()
    img_binary = np.float32(img_binary)
    min_value = np.min(img.flatten())
    max_value = np.max(img.flatten())
   
    thres = {'binaria':[255, 0], 'saturada':[255, 1], 'truncada':[T, 255]}
    
    threshold = thres[type]
    
    for x in range(width):
        for y in range(height):
            if img_binary[x, y] > T:
                img_binary[x, y] = threshold[0] 
            else:
                img_binary[x, y] = threshold[1]*img_binary[x, y] 
                             
    min_value = np.min(img_binary.flatten())
    max_value = np.max(img_binary.flatten())
    
    img_binary = np.uint8(img_binary)
    return img_binary

def multiThresholding(img, thresholds, colors):
    '''
    --------------- MultiThreshoding---------------
    Parameters:
        img: image in 2D
        thresholds: array of thresholds(numbers between 0-255)
        colors: array of grayscale(numbers between 0-255)
    Return:
        Multithresholding image
    '''
    width, height = img.shape 
    
    img_thresholded = img.copy()
    
    numbers_thrs = len(thresholds)
    
    img_thresholded[img_thresholded<thresholds[0]] = colors[0]
    img_thresholded[img_thresholded>thresholds[-1]] = colors[-1]
    
    for t in range(numbers_thrs-1):
        img_thresholded[(img_thresholded > thresholds[t]) & (img_thresholded < thresholds[t+1])] = colors[t+1]
    
    return img_thresholded