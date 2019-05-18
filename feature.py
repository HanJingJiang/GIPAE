from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
import pandas as pd
import csv
import math
import random
import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
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
OriginalData = []
ReadMyCsv(OriginalData, "drug-disease-whole.csv")
print(len(OriginalData))
counter = 0
while counter < len(OriginalData):
    OriginalData[counter][0] = OriginalData[counter][0].lower()
    OriginalData[counter][1] = OriginalData[counter][1].lower()
    counter = counter + 1

LncDisease = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    LncDisease.append(Pair)
    counter = counter + 1
AllDisease = []
counter1 = 0
while counter1 < len(OriginalData):
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):  
        if OriginalData[counter1][1] != AllDisease[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))

AllDRUG = []
counter1 = 0
while counter1 < len(OriginalData): 
    counter2 = 0
    flag = 0
    while counter2 < len(AllDRUG):  
        if OriginalData[counter1][0] != AllDRUG[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllDRUG[counter2]:
            flag = 1
            break
    if flag == 0:
        AllDRUG.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllDRUG)', len(AllDRUG))
storFile(AllDRUG, 'AllDRUG.csv')

DiseaseAndDrugBinary = []
counter = 0
while counter < len(AllDisease):
    row = []
    counter1 = 0
    while counter1 < len(AllDRUG):
        row.append(0)
        counter1 = counter1 + 1
    DiseaseAndDrugBinary.append(row)
    counter = counter + 1


print('len(LncDisease)', len(LncDisease))
counter = 0
while counter < len(LncDisease):
    DN = LncDisease[counter][1]
    RN = LncDisease[counter][0]
    counter1 = 0
    while counter1 < len(AllDisease):
        if AllDisease[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllDRUG):
                if AllDRUG[counter2] == RN:
                    DiseaseAndDrugBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(DiseaseAndDrugBinary)', len(DiseaseAndDrugBinary))
lines = [line.strip().split() for line in open("diseaseSimilarity.txt")]
txtSimilarity = []
i = 0
for dis in lines:
    i = i + 1
    if i == 1:
        continue
    txtSimilarity.append(dis[1:])

counter1 = 0
sum1 = 0
while counter1 < (len(AllDisease)):
    counter2 = 0
    while counter2 < (len(AllDRUG)):
        sum1 = sum1 + pow((DiseaseAndDrugBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nd = len(AllDisease)
rdpie = 0.5
rd = rdpie * Nd / Ak
print('disease rd', rd)

DiseaseGaussian = []
counter1 = 0
while counter1 < len(AllDisease):
    counter2 = 0
    DiseaseGaussianRow = []
    while counter2 < len(AllDisease):
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDRUG):
            sum2 = pow((DiseaseAndDrugBinary[counter1][counter3] - DiseaseAndDrugBinary[counter2][counter3]), 2)
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DiseaseGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DiseaseGaussian.append(DiseaseGaussianRow)
    counter1 = counter1 + 1
print('len(DiseaseGaussian)', len(DiseaseGaussian))

from numpy import *
MDiseaseAndDrugBinary = np.array(DiseaseAndDrugBinary)    
DRUGAndDiseaseBinary = MDiseaseAndDrugBinary.T   
DRUGGaussian = []
counter1 = 0
sum1 = 0
while counter1 < (len(AllDRUG)):   
    counter2 = 0
    while counter2 < (len(AllDisease)):    
        sum1 = sum1 + pow((DRUGAndDiseaseBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nm = len(AllDRUG)
rdpie = 0.5
rd = rdpie * Nm / Ak
print('DRUG rd', rd)

counter1 = 0
while counter1 < len(AllDRUG):   
    counter2 = 0
    DRUGGaussianRow = []
    while counter2 < len(AllDRUG):  
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDisease):  
            sum2 = pow((DRUGAndDiseaseBinary[counter1][counter3] - DRUGAndDiseaseBinary[counter2][counter3]), 2)
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DRUGGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DRUGGaussian.append(DRUGGaussianRow)
    counter1 = counter1 + 1
print('type(DRUGGaussian)', type(DRUGGaussian))
workbook = xlrd.open_workbook('newfingerprint.xlsx')
booksheet = workbook.sheet_by_index(0)
booksheet = workbook.sheet_by_name('Sheet1')

nrows = booksheet.nrows
row_list = []

for i in range(0, nrows):
         row=[]
         row_data = booksheet.row_values(i)
         ii = row_data
         for j in ii[0]:
             row.append(j)
         row_list.append(row)

y_test_test=[]
x_train_test=[]
x_test_test=[]
x_train_test_all=[]
x_all=[]
i=0
for row in row_list:
    a = row
    x_train_test.append(a)
    i=i+1
x_train_test =np.array(x_train_test)
x_test_test =np.array(x_test_test)
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train_test = x_train_test.reshape((x_train_test.shape[0], -1))
x_train = x_train_test
x_test = x_test_test

encoding_dim = 64
input_img = Input(shape=(920,))


encoded = Dense(500, activation='relu')(input_img)
encoded = Dense(200, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)
print()
# decoder layers
decoded = Dense(100, activation='relu')(encoder_output)
decoded = Dense(200, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(920, activation='tanh')(decoded)

autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoder_output)


autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=100,
                shuffle=True)


encoded_imgs = encoder.predict(x_train)



from openpyxl import Workbook

workbook = Workbook()
booksheet = workbook.active

drugSimilarity = []
for rows in encoded_imgs:
    booksheet.append(rows.tolist())
    drugSimilarity.append(rows.tolist())
print('drugSimilarity',len(drugSimilarity[0]))

import random
counter1 = 0   
counter2 = 0    
counterP = 0   
counterN = 0    
PositiveSample = []    
PositiveSample = LncDisease
print('PositiveSample)', len(PositiveSample))

NegativeSample = []
counterN = 0
while counterN < len(PositiveSample):                        
    counterD = random.randint(0, len(AllDisease)-1)
    counterR = random.randint(0, len(AllDRUG)-1)     
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(AllDRUG[counterR])
    DiseaseAndRnaPair.append(AllDisease[counterD])
    flag1 = 0
    counter = 0
    while counter < len(LncDisease):
        if DiseaseAndRnaPair == LncDisease[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(AllDRUG[counterR])
        NegativePair.append(AllDisease[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
DiseaseSimilarity = []
counter = 0
while counter < len(AllDisease):
    counter1 = 0
    Row = []
    while counter1 < len(AllDisease):
        v = float(DiseaseGaussian[counter][counter1]
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(txtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DiseaseSimilarity.append(Row)
    counter = counter + 1
DRUGSimilarity = []
DRUGSimilarity=np.hstack((DRUGGaussian, drugSimilarity))
print('len(DRUGSimilarity)', len(DRUGSimilarity))
print('len(DRUGSimilarity[0])', len(DRUGSimilarity[0]))
storFile(DRUGSimilarity, 'DRUGSimilarity.csv')
AllSample = PositiveSample.copy()
AllSample.extend(NegativeSample)        
SampleFeature = []
counter = 0
while counter < len(AllSample):
    counter1 = 0
    while counter1 < len(AllDRUG):
        if AllSample[counter][0] == AllDRUG[counter1]:
            a = []
            counter3 = 0
            # 原本是ALLDrug
            while counter3 <len(DRUGSimilarity[0]):
                v = DRUGSimilarity[counter1][counter3]
                a.append(v)
                counter3 = counter3 + 1
            break
        counter1 = counter1 + 1
    counter2 = 0
    while counter2 < len(AllDisease):
        if AllSample[counter][1] == AllDisease[counter2]:
            b = []
            counter3 = 0
            while counter3 < len(DiseaseSimilarity[0]):
                v = DiseaseSimilarity[counter2][counter3]
                b.append(v)
                counter3 = counter3 + 1
            break
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature, 'G64SampleFeature.csv')

