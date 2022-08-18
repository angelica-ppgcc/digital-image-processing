import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Activation, Conv2D, MaxPool2D, Dense
from tensorflow.keras import backend 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

#Read data_base
data_base = pd.read_table("./ocr_car_numbers_rotulado.txt", header=None, delim_whitespace=True)

X_data = []
y_data = []
for i in range(len(data_base.iloc[0,:])):
    
    im = np.array(data_base.iloc[i,:-1])
    
    label = data_base.iloc[i,-1]
        
    im[im == 1] = 255
    
    im = np.reshape(im, (35, 35))
    
    X_data.append(im)
    y_data.append(label)
    

backend.clear_session()

#Feature extraction stages
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(35, 35, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())

#Fully connected stages (Classification)
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

realizations = 20

accs_train = []
cms_train = []

accs_test = []
cms_test = []

#Realizations
for i in range(realizations):
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    #One-hot-encode: Transform values 1 to 000000001, 2 to 000000010, 3 to 000000100, etc
    y_train_categ = to_categorical(y_train) 
    y_test_categ = to_categorical(y_test)

    X_train = X_train.reshape(-1, 35, 35, 1)
    X_test = X_test.reshape(-1, 35, 35, 1)
    
    #Define k-fold = 10
    kf = KFold(n_splits=10)
	kf.get_n_splits(X_train)

	accuracies = []
	index = 0

	matrixes = []
    
    #10 iterations of k-fold
	for train_index, test_index in kf.split(X_train):
		X_train_ev, X_test_ev = X_train[train_index], X_train[test_index]
		y_train_ev, y_test_ev = y_train_categ[train_index], y_train_categ[test_index]
		
		print("shape x: ", X_train_ev.shape)
		print("shape y: ", y_train_ev.shape)
		
        h = model.fit(X_train_ev, y_train_ev, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
		
		y_pred_ev = model.predict(X_test_ev)
		
        print(y_test_ev)
		print(y_pred_ev)
		acc = accuracy_score(y_test_ev, y_pred_ev)
		cm = confusion_matrix(y_test_ev, y_pred_ev)
		print("Acuracia:"+str(index)+": ", acc)
		accuracies.append(acc)
		print("Matriz de Confusao")
		print(cm)
		matrixes.append(cm)
		
    y_pred = model.predict(X_test)
	acc_test = accuracy_score(y_test_categ, y_pred)
 	cm_test = confusion_matrix(y_test_categ, y_pred)
	cms_test.append(cm_test)
	accs_test.append(acc_test)
 
	accs_train.append(np.mean(accuracies))
	mean_cm = np.mean(matrixes, axis = 0)
	mean_cm = np.ceil(mean_cm)
	mean_cm = np.uint8(mean_cm)
	cms_train.append(mean_cm)


accs_train = np.array(accs_train)
accs_test = np.array(accs_test)

diff = np.abs(accs_train - accs_test)
ind = np.min(diff)

#Results of accuracy and standard deviation

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
print(mean_cm)