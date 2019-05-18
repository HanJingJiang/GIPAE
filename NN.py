import numpy as np
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization


import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
import csv
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
SampleFeature = []
ReadMyCsv(SampleFeature, "G64SampleFeature.csv")
x = SampleFeature
data = []
data1 = np.ones((1933,1), dtype=int)
data2 = np.zeros((1933,1))
y=np.concatenate((data1,data2),axis=0)
x = np.array(x)
print(x.shape)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)   


x_train=x_train.reshape(-1,1,970,1)
x_test=x_test.reshape(-1,1,970,1)
x = x.reshape(-1,1,970,1)
print(x_train.shape)
print(x_test.shape)
batch_size=32    
epochs=2
model = Sequential() 
return_sequences=True
model.add(BatchNormalization(input_shape=(1,970,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu',  name='Dense-2'))  
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=20, batch_size=10,validation_split=0.1)
from keras.models import Model
dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense-2').output)
dense1_output = dense1_layer_model.predict(x)
print(dense1_output.shape)
print(dense1_output[0])
storFile(dense1_output, 'fdatasetfeature1.csv')
