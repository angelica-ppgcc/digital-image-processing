import cv2
import numpy as np
from sklearn import datasets
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
import pandas as pd
from skimage.feature.texture import greycomatrix, greycoprops
from sklearn import svm
from sklearn.cluster import KMeans
 
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist


def normalization(data):
    normalized_data = data.copy()
    normalized_data = (normalized_data - np.min(normalized_data))/(np.max(normalized_data - np.min(normalized_data)))
    return normalized_data

digits = datasets.load_digits()
data_features = []
data_targets = []
i = 0

data_base = pd.read_table("./ocr_car_numbers_rotulado.txt", header=None, delim_whitespace=True)

for i in range(len(data_base.iloc[0,:])):
    
	im = np.array(data_base.iloc[i,:-1])
    
	label = data_base.iloc[i,-1]

	im = np.reshape(im, (1,1225))
        
	im[im == 1] = 255
    
	im = np.reshape(im, (35, 35))
	
	im = np.uint8(im)

	#------------------LBP extractor-------------------#
 
	#desc = LocalBinaryPatterns(8, 3)
	#hist = desc.describe(im)
	#hist = normalization(hist)

	#--------------------------------------------------#	
 
	#---------------Hu Moments extractor---------------#
 
	#cv2.imwrite("./image_hu.jpg", im)
	#im = cv2.imread("./image_hu.jpg", 0)
	#hist = cv2.HuMoments(cv2.moments(im)).flatten()
	#hist = normalization(hist)
 
	#--------------------------------------------------#
 
	#------------------GLCM extractor------------------#
	
	props = ['ASM', 'dissimilarity', 'contrast', 'energy', 'correlation', 'homogeneity']
	hist = [greycoprops(greycomatrix(im, distances=[1], angles=[0]), prop)[0,0] for prop in props]
	hist = np.array(hist)
	hist = normalization(hist)
	#--------------------------------------------------#	
 	
	data_features.append(hist)
	data_targets.append(label)
	i += 1
    
    
data_features = np.array(data_features)
data_targets = np.array(data_targets)

#Use Gaussian Naive Bayes Classifier
#clf = GaussianNB()

#Use SVM with kernel RBF Classifier
clf = SVC(C = 300, kernel='rbf', gamma = 100)


realizations = 20
accs_train = []
cms_train = []

accs_test = []
cms_test = []

#Realizations
for i in range(realizations):
	X_train, X_test, y_train, y_test = train_test_split(data_features, data_targets, test_size=0.2, random_state=42)
	
	#Define k-fold = 10
 	kf = KFold(n_splits=10)
	kf.get_n_splits(X_train)

	accuracies = []
	index = 0

	matrixes = []

	#10 iterations of k-fold
 
	for train_index, test_index in kf.split(X_train):
		X_train_ev, X_test_ev = X_train[train_index], X_train[test_index]
		y_train_ev, y_test_ev = y_train[train_index], y_train[test_index]
		
		print("shape x: ", X_train_ev.shape)
		print("shape y: ", y_train_ev.shape)
		
		clf.fit(X_train_ev, y_train_ev)
	
		
		y_pred_ev = clf.predict(X_test_ev)
		print(y_test_ev)
		print(y_pred_ev)
		acc = accuracy_score(y_test_ev, y_pred_ev)
		cm = confusion_matrix(y_test_ev, y_pred_ev)
		print("Acuracia:"+str(index)+": ", acc)
		accuracies.append(acc)
		print("Matriz de Confusao")
		print(cm)
		matrixes.append(cm)
		
		index += 1
 
	y_pred = clf.predict(X_test)
	acc_test = accuracy_score(y_test, y_pred)
 	cm_test = confusion_matrix(y_test, y_pred)
	cms_test.append(cm_test)
	accs_test.append(acc_test)
 
	accs_train.append(np.mean(accuracies))
	mean_cm = np.mean(matrixes, axis = 0)
	mean_cm = np.ceil(mean_cm)
	mean_cm = np.uint8(mean_cm)
	cms_train.append(mean_cm)
 
	

#Results of accuracy and standard deviation

accs_train = np.array(accs_train)
accs_test = np.array(accs_test)

diff = np.abs(accs_train - accs_test)
ind = np.min(diff)

print("Treinamento:")
print("Acuracia media - Treinamento: ", np.mean(accs_train))
print("Desvio padrao- Treinamento: ", np.std(accs_train))
cms_train = np.array(cms_train)
print("shape cm",cms_train.shape)
mean_cm = np.mean(cms_train, axis = 0)
mean_cm = np.ceil(mean_cm)
mean_cm = np.uint8(mean_cm)
print("Matriz de confusao media- Treinamento")
print(mean_cm)
print("")
print("Teste:")
print("Acuracia media - Teste: ", np.mean(accs_test))
print("Desvio padrao - Teste: ", np.std(accs_test))
cms_test = np.array(cms_test)
print("shape cm", cms_test.shape)
mean_cm = np.mean(cms_test, axis = 0)
mean_cm = np.ceil(mean_cm)
mean_cm = np.uint8(mean_cm)
print("Matriz de confusao media - Teste")
print(mean_c)
 
    
    