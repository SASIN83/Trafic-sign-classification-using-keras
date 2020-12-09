import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D ,MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle
import pandas as pd
import numpy as np
import random

data=pd.read_csv("german-traffic-signs/signnames.csv")

data

with open("german-traffic-signs/train.p",mode='rb') as trainer:
  train=pickle.load(trainer)

with open("german-traffic-signs/valid.p",mode='rb') as validation:
  valid=pickle.load(validation)

with open("german-traffic-signs/test.p",mode='rb') as tester:
  test=pickle.load(tester)
  
x_train,y_train=train['features'],train['labels']
x_valid,y_valid=valid['features'],valid['labels']
x_test,y_test=test['features'],test['labels']

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

index=np.random.randint(1,len(x_train))
plt.imshow(x_train[index])
print("image label: {}".format(y_train[index]))

from sklearn.utils import shuffle
x_train,y_train=shuffle(x_train,y_train)

def preprocess(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Grayscaling

  img=cv2.equalizeHist(img) #Equalizing Histogram

  img=img/255 #Normalization

  return img

x_train_processed=np.array(list(map(preprocess,x_train)))
x_valid_processed=np.array(list(map(preprocess,x_valid)))
x_test_processed =np.array(list(map(preprocess,x_test)))

x_train_processed =x_train_processed.reshape(34799, 32, 32, 1)
x_valid_processed=x_valid_processed.reshape(4410, 32, 32, 1)
x_test_processed=x_test_processed.reshape(12630, 32, 32, 1)

print(x_train_processed.shape)
print(x_test_processed.shape)
print(x_valid_processed.shape)

i = random.randint(1,len(x_train))
plt.imshow(x_train_processed[i].squeeze(),cmap='gray')
plt.figure()
plt.imshow(x_train[i].squeeze())

#Deep CNN model
model= Sequential()

model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,1))) #Add convolution layer 1

model.add(MaxPooling2D(pool_size=(2,2)))#pooling layer

model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5),activation='relu')) #Add convolution layer 2

model.add(MaxPooling2D(pool_size=(2,2))) #pooling layer

model.add(Flatten()) #Image flattener

model.add(Dense(256,activation='relu')) #Image density

model.add(Dropout(0.5)) #dropout rate layer

model.add(Dense(43,activation='softmax')) 

model.summary()

#Model compiler

model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(x_train_processed,y_train,batch_size=500,epochs=50,verbose=1,validation_data=(x_valid_processed,y_valid))

#Evaluation of model, Accuracy score

score = model.evaluate(x_test_processed,y_test)
print(score[1])

hist.history.keys()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training','validation'])
plt.title("Training & Validation losses")
plt.xlabel("Epochs")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title("Training & Validation")
plt.xlabel("Epochs")

prediction=model.predict_classes(x_test_processed)
y_true_label=y_test

from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(y_true_label,prediction)
plt.figure(figsize=(20,20))
sns.heatmap(matrix,annot=True)

L=6
W=6
fig , axes = plt.subplots(L,W,figsize=(12,12))
axes = axes.ravel()
for i in range(0,L*W):
  axes[i].imshow(x_test[i])
  axes[i].set_title('Prediction ={}\n True={}'.format(prediction[i],y_true_label[i]))
  axes[i].axis('off')
plt.subplots_adjust(wspace=1)
