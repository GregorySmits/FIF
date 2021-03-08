#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:08:30 2020

@author: SHAMAN
"""

import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import sys
import math

class Dataset:

    def __init__(self, df, header, conv, supervised = False, ratio = None):
        """
        Load a csv dataset 
        """
        us=None
        self.supervised=supervised
        
        if supervised:
            us = tuple(range(len(header)))
        
        self.data = np.loadtxt(df, delimiter=",",converters = conv,skiprows=0,usecols=us)#is a np.ndarray
        self.nbPoints = len(self.data)

        self.mins = self.data.min(0)
        self.maxs = self.data.max(0)
        if supervised:
            self.mins = self.mins[:-1]
            self.maxs = self.maxs[:-1]
            

        self.ratio = ratio
        self.trainingData = np.arange(self.nbPoints)
        self.evaluatedData = np.arange(self.nbPoints)

        if self.ratio is not None and self.ratio > 0:
            rng = np.random.default_rng()
            sampleSize = math.floor(self.nbPoints * self.ratio)
            self.trainingData = rng.choice(self.nbPoints, sampleSize, replace=False)
            self.evaluatedData = np.setdiff1d(np.arange(self.nbPoints), self.trainingData)
        
        self.attributes=header #is a list of str
        if supervised:
            self.attributes=header[:-1]

    def getNbPoints(self):
        return self.nbPoints

    def getTrainingDataSet(self):
        """
        Returns a 1D array of point ids used for training
        """
        return self.trainingData

    def getEvalDataSet(self):
        """
        Returns a 1D array of point ids to evaluate
        """
        return self.evaluatedData

    def getData(self, i=None):
        if i is not None:
            return self.data[i]
        else:
            return self.data

    def getClass(self,i):
        classe = None
        if self.supervised:
            classe =  self.data[i][len(self.getAttributes())]
        return classe

    def getAttributes(self):
        return self.attributes
        
    def binClassesVector(self):
        res = np.zeros(len(self.getEvalDataSet()))
        j=0
        for i in self.getEvalDataSet():
            pt = self.data[i]
            res[j] = int(pt[len(pt)-1])
            j=j+1
        return res

if __name__ == "__main__":
#    d = Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True)
    d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.1)
    d = Dataset("../Data/foo.csv",["x","y"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0)},False,0.1)

    print(d.getData())
    print(d.mins)
    print(d.maxs)

    sys.exit(0)
    e = d.getEvalDataSet()
    t = d.getTrainingDataSet()
    print(d.getAttributes())
    print(d.getData(5))
#    print("CLASS ", d.getClass(5))


"""

#  Run each segement to load the particular data (df) and its label (labels) 


#############################     Data D1     ##################################################### 

df = np.genfromtxt("Data8.csv", delimiter=",")
df=pd.DataFrame(df)
df.columns=['c1','c2']

labels = np.genfromtxt("Data8_labels.csv", delimiter=",")


plt.figure(figsize = (12,9))
plt.scatter(df['c1'], df['c2'])
plt.rcParams.update({'font.size': 14})
plt.grid(False)



############################     Data D2  ######################################################

df = np.genfromtxt("Donut.csv", delimiter=",")
df=pd.DataFrame(df)
df.columns=['c1','c2']

labels = np.genfromtxt("Donut_labels.csv", delimiter=",")

plt.figure(figsize = (12,9))
plt.scatter(df['c1'], df['c2'])
plt.rcParams.update({'font.size': 14})
plt.grid(False)


#########################       Data Ionosphere  #############################################


import scipy.io
mat = scipy.io.loadmat('ionosphere.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]





#########################       Data Arrhythmia  #############################################

import scipy.io
mat = scipy.io.loadmat('arrhythmia.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]



#########################       Data Pima  #############################################

df = np.genfromtxt("diabetes.csv", delimiter=",")
df=pd.DataFrame(df)
df = df.iloc[1:]

labels = df.iloc[:,8]
del df[8]

df.columns=['c1','c2','c3','c4','c5','c6','c7','c8']


df = df.reset_index()               # only for the data starting with index 1
df = df.drop(['index'],axis=1)




#########################       Data Glass  #############################################

import scipy.io
mat = scipy.io.loadmat('glass.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]






#################   Data Compound    ##############################################

agg_data = np.loadtxt(fname = "Compound.txt")

agg_label=agg_data[:,2]



df_compound1 = agg_data[agg_data[:,2] ==1]
df_compound2 = agg_data[agg_data[:,2] ==2]


df_compound = np.vstack((df_compound1, df_compound2))

df_compound=pd.DataFrame(df_compound)
df_compound.columns=['c1','c2','labels']

df_compound['labels'][df_compound['labels'] == 2] = 0 


df_labels = df_compound['labels']
df_compound=df_compound.drop(['labels'], axis=1)

fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(df_compound['c1'], df_compound['c2'], s=40, cmap='viridis')


df = df_compound.copy()
labels = df_labels.copy()


"""