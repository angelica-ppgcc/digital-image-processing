import numpy as np
from random import randint
import cv2
import matplotlib.pyplot as plt

class K_Means:

    def __init__(self, n_clusters = 2):
        self.n_clusters = n_clusters
        self.clusters = []
        
        
    def fit(self, X):
        
        self.X = X
        
        #Initialize the values clusters
        width, height = X.shape
        for n in range(self.n_clusters):
            self.clusters.append([randint(0, width-1), randint(0, height-1)])
        
        
        stopped_condition = True
        
        while(stopped_condition):
            #Classify the pixels with the grayscale
            for i in range(width):
                for j in range(height):
                    new = self.smallerDistance([i, j])
                    new = np.array(new)
                
                    w, h =  new[0], new[1]
                   
                    X[i, j] = X[w, h]
        
            #Update the values of the clusters
            clusters = self.updateClusters()
            
            if np.array_equal(self.clusters, clusters):
                stopped_condition = False
                break
            
            self.clusters = clusters
                
        
    def updateClusters(self):
        
        clusters = []
        
        for cluster in self.clusters:
            g = self.X[self.X == self.X[cluster[0], cluster[1]]]
            clusters.append(np.mean(g, axis = 0))
        
        return clusters
    
    def smallerDistance(self, pixel):
        distances = []
        pixel = np.array(pixel)
        
        for cluster in self.clusters:    
            cluster = np.array(cluster)
            distances.append(np.sqrt(np.dot(pixel, cluster.T)))
        
    
        index = np.argmin(distances)
        
        return self.clusters[index]

    
if __name__ == "__main__":
    img = cv2.imread("segmentar.jpg")
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    K = 4
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    
    center = np.uint8(center)
    
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1)
    plt.imshow(img)
    #plt.savefig('Resultados/Topico_7/comp_image.jpg')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(res2)
    plt.savefig('Resultados/Topico_7/comp_image_segmented_4.jpg')
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

    plt.show()
   