import feature as ft
from ensemble import AdaBoostClassifier
from sklearn import tree
import numpy as np
from PIL import Image
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import *
from sklearn.metrics import classification_report

def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
if __name__ == "__main__":
    #读取图片转为数组，数组再经过NPD方法处理成浮点数组并缓存到两个对应的文件
    #path = 'E:/github/ML2017-lab/ML2017-lab-03/datasets/original/nonface'
    #filelist = os.listdir(path)
    #feature = np.array([])
    #for i in range(len(filelist)):
    #    photo = Image.open('datasets/original/nonface/'+filelist[i])
    #    data = np.array(photo.resize((24,24)).convert("L"))
    #    npfd = ft.NPDFeature(data)
    #    if(feature.size == 0):
    #        feature = np.append(feature, npfd.extract())
    #    else:
    #        feature = np.row_stack([feature, npfd.extract()])
    
    #读取缓存文件,并整合到X数组集
    file1 = load('face.txt').tolist()
    file1_size = len(file1)
    file2 = load('nonface.txt').tolist()
    file2_size = len(file2)
    X = file1+file2
    #构造结果值Y
    y_face = np.ones(file1_size).tolist()
    y_nonface = (-1*np.ones(file2_size)).tolist()
    y = y_face+y_nonface
    
    #划分训练集和验证集
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=1)
    classifier = tree.DecisionTreeClassifier
    adaBoostClassifier = AdaBoostClassifier(classifier,20)
    adaBoostClassifier.fit(x_train,y_train)
    
    y_true = y_test
    y_pred = adaBoostClassifier.predict(x_test, y_test)
    target_names = ['nonface','face']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    '''
    weight = np.ones(len(x_train)).tolist()
    
    #test = em.AdaBoostClassifier(classifier, 2)
    classifier.fit(x_train,y_train,sample_weight =weight )
    s = classifier.score(x_test,y_test)
    print(s)
    '''      
