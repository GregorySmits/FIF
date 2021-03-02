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

#d = dt.Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8) 
d = dt.Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0.8)
#d = dt.Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#d = dt.Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.7)


"""
Generate a 100 trees forest and compare the result of the crisp ensemble-based method of IF and
 an individual strongfuzzy interpretation of each tree
"""
METHOD = "crisp"#"strongfuzzy","fuzzy","orthogonal","crisp"
ALP=0.5
BET=0
f = fif.Forest(d)
f.build()

compB={'P':0,'R':0,'F':0}
compW={'P':0,'R':0,'F':0}
moyESF=0
moyEC=0
for treeI in range(len(f.trees)):
    """
    STRONGFUZZY INDIVIDUAL TREE EVALUATION
    """
    print("TREE :",treeI)
    print("\tFUZZY ISOLATION TREE:")
    f.setMode("strongfuzzy")
    f.setAlpha(0.84)
    f.setBeta(0.05)
    scores = f.computeScores(treeI)
    pSF,rSF,fmSF,eSF = f.evaluate(scores)
    msg="-FUZZY PREC:"+str(pSF)+" RAP:"+str(rSF)+" FM:"+str(fmSF)+" ER:"+str(eSF)
    print(msg)
 
    """
    CRISP INDIVIDUAL TREE EVALUATION
    """ 
    print("\tCRISP ISOLATION TREE:")   
    f.setMode("crisp")
    f.setAlpha(0.5)
    f.setBeta(0)
    scores = f.computeScores(treeI)
    pC,rC,fmC, eC = f.evaluate(scores)
    msg = "-CRISP PREC:"+str(pC)+" RAP:"+str(rC)+" FM:"+str(fmC)+" ER:"+str(eC)
    print(msg)
    print("\n")
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

print("INDIVIDUAL TREE COMPARISON RESULTS :")
print("\tBETTER THAN CRISP PREC=",compB['P'],"RAP=",compB['R'],"FM=",compB['F'],"ER=",moyESF)
print("\tWORSE THAN CRISP PREC=",compW['P'],"RAP=",compW['R'],"FM=",compW['F'],"ER=",moyEC)

sys.exit(0)

