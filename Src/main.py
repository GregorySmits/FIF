# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:02:48 2020

@author: veron
"""

import dataset as dt
import FIF as fif
import viewer as vw
import numpy as np
import math
import sys


#The datasets and their dimensions
real_datasets = ["annthyroid", "arrhythmia", "breastw", "cardio", "cover", "hbk", "http", "ionos", "letter", "lympho",
                 "mammography", "musk", "pima", "satellite", "shuttle", "smtp", "wood"]

dimensions = [6, 271, 9, 21, 10, 4, 3, 32, 32, 18, 6, 166, 8, 36, 9, 3, 6]

#You just have to choose the index of the dataset in the table. Example : idx_dataset = 0 <=> annthyroid
idx_dataset = 6
converters = {}
header = []
 
for i in range(dimensions[idx_dataset]):
    converters[i] = lambda s: float(s.strip() or 0)
    header.append(str(i))
  
converters[dimensions[idx_dataset]] = lambda s: int(float(s.strip() or 0))
header.append("CLASS")

#Loading the dataset (do not forget to set skiprows=0 in the __init__ of the Dataset class, 
#because the files do not have headers)
d = dt.Dataset("../Data/"+real_datasets[idx_dataset]+".csv", header, converters, True, 0.8)

#d = dt.Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0) 
#d = dt.Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0.8)
#d = dt.Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#d = dt.Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)


"""
Generate a 100 trees forest and compare the result of the crisp ensemble-based method of IF and
 an individual strongfuzzy interpretation of each tree
"""
METHOD = "crisp"#"strongfuzzy","fuzzy","orthogonal","crisp"

beta=0.05
f = fif.FForest(d,beta)
f.build()

compB={'P':0,'R':0,'F':0}
compW={'P':0,'R':0,'F':0}
moyESF=0
moyEC=0
precMoy=0
precMoySF=0

for treeI in range(len(f.trees)): 
    """
    CRISP INDIVIDUAL TREE EVALUATION
    """ 
    print("TREE :",treeI)
    print("\tCRISP SCORE COMPUTATION:")   
    f.setAlpha(0.5)
    scores = f.computeScores("crisp",treeI)
    pC,rC,fmC, eC = f.evaluate(scores)
    precMoy=precMoy+pC
    msg = "-CRISP PREC:"+str(pC)+" RAP:"+str(rC)+" FM:"+str(fmC)+" ER:"+str(eC)
    print(msg)
    #print("\n***************\n")

    """
    STRONGFUZZY INDIVIDUAL TREE EVALUATION
    """
    #print("TREE :",treeI)
    print("\tFUZZY SCORE COMPUTATION:")
    f.setAlpha(0.9)
    scores = f.computeScores("strongfuzzy",treeI)
    pSF,rSF,fmSF,eSF = f.evaluate(scores)
    precMoySF=precMoySF + pSF
    msg="-FUZZY PREC:"+str(pSF)+" RAP:"+str(rSF)+" FM:"+str(fmSF)+" ER:"+str(eSF)
    print(msg)
    print("\n***************\n")



    if fmSF > fmC:
        compB['F'] = compB['F'] +1
    if fmC > fmSF:
        compW['F'] = compW['F'] +1

    if rSF > rC:
        compB['R'] = compB['R'] +1
    if rC > rSF:
        compW['R'] = compW['R'] +1

    if pSF > pC:
        compB['P'] = compB['P'] +1
    if pC > pSF:
        compW['P'] = compW['P'] +1

    moyESF=moyESF+eSF
    moyEC=moyEC+eC

for i in compB.keys():
    compB[i] = compB[i] *100 / len(f.trees)
    compW[i] = compW[i] *100 / len(f.trees)
moyESF =moyESF /len(f.trees)
moyEC =moyEC /len(f.trees)
precMoySF=precMoySF /len(f.trees)
precMoy=precMoy /len(f.trees)
print("INDIVIDUAL TREE COMPARISON RESULTS :")
print("\tBETTER THAN CRISP PREC=",compB['P'],"RAP=",compB['R'],"FM=",compB['F'],"ER=",moyESF)
print("\tWORSE THAN CRISP PREC=",compW['P'],"RAP=",compW['R'],"FM=",compW['F'],"ER=",moyEC)
print("PRECISION MOYENNE CRISP=",precMoy," FUZZY=",precMoySF)
sys.exit(0)

