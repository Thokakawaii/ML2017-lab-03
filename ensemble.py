import pickle
from sklearn.tree import DecisionTreeClassifier
#import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.alpha_list = []
        self.weak_classifier_list = []
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        weight = ones(len(X))
        weight = weight / weight.sum(0)
        totalEst = 0
        y = array(y)
        for i in range(self.n_weakers_limit):
            classifier = self.weak_classifier(max_depth = 1)
            classifier.fit(X,y,sample_weight = weight)
            # save the weak classifier in memory
            self.weak_classifier_list.append(classifier)
            print(type(classifier))
            _y = sign(classifier.predict(X))
            error_num = (y!=_y).sum()
            error_rate = error_num/y.shape[0]
            alpha = 0.5 * log((1-error_rate)/error_rate)
            # save alpha
            self.alpha_list.append(alpha)
            expon = multiply(-1*alpha*y, _y)
            weight = multiply(weight,exp(expon))
            weight = weight/weight.sum()
            #test the total error
            totalEst += alpha * _y
            total_error_rate = (sign(totalEst)!=y).sum()/y.shape[0]
            
            print(i,"the total error rate is",total_error_rate)
            #当错误率降至0.01停止训练
            if total_error_rate <= 0.01:break
        self.save(self.weak_classifier_list,'./classifier.pkl')
        self.save(self.alpha_list,'./alpha.pkl')
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        pass

    def predict_scores(self, X, y):
        
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, y, threshold=0):
        #将分类器和超参数导出
        weak_classifier_list = self.load('./classifier.pkl')
        alpha_list= self.load('./alpha.pkl')
    
        totalEst = 0
        y = array(y)
        print(type(weak_classifier_list[0]))
        for i in range(len(weak_classifier_list)):
            classifier = weak_classifier_list[i]
            _y = sign(classifier.predict(X))
            error_num = (y!=_y).sum()
            error_rate = error_num/y.shape[0]
            alpha = alpha_list[i]
            expon = multiply(-1*alpha*y, _y)
            totalEst += alpha * _y
            total_error_rate = (sign(totalEst)!=y).sum()/y.shape[0]
            
            print(i,"the total accuracy rate is",1-total_error_rate)
        return sign(totalEst)
            #if total_error_rate <= 0.01:break
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb+") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
