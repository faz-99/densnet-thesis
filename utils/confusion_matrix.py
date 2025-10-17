"""
混淆矩阵
@date: 2022/05/01
@author: wuqiwei
"""

import numpy
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ConfusionMatrix(object):

    def __init__(self, class_num: int):
        self.matrix = numpy.zeros((class_num, class_num))
        self.class_num = class_num
        self.pred=[]
        self.true=[]

    def update(self, pred, label):
        # p代表Predicted label、t代表True label
        for p, t in zip(pred, label):
            self.matrix[p, t] += 1
            self.pred.append(p)
            self.true.append(t)

    def acc(self):
        acc = 0
        for i in range(self.class_num):
            acc += self.matrix[i, i]
        accure = accuracy_score(self.true, self.pred)
        acc = acc / numpy.sum(self.matrix)

        return acc
    def pre(self):
        #print('true',self.true)
        #print('pred',self.pred)
        precision = precision_score(self.true, self.pred,average='micro')
        #print('pre',precision)
        return precision
    def recall(self):
        recall = recall_score(self.true, self.pred,average='micro')
        #print('re',recall)
        return recall
    def f1_score(self):
        f1=f1_score(self.true, self.pred,average='micro')
        #print('f1',f1)
        return f1
    def report(self):
        report=classification_report(self.true,self.pred)
        return report