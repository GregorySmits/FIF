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
import time

d = dt.Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0) 
#d = dt.Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0.8)
#d = dt.Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#d = dt.Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.7)


"""
Generate a 100 trees forest and compare the result of the crisp ensemble-based method of IF and
 an individual strongfuzzy interpretation of each tree
"""
beta=0.05
NBRUN=10

nbT=10
nbTMax=100
moyC={'P':([0]*10),'R':([0]*10),'F':([0]*10),'E':([0]*10),'AUC':([0]*10)}
moySF={'P':([0]*10),'R':([0]*10),'F':([0]*10),'E':([0]*10),'AUC':([0]*10)}

while nbT <= nbTMax: 
    meanTimeC=0    
    meanTimeSF=0    
    cutsC={'minP':0,'maxP':0,'moyP':0,'minN':0,'maxN':0,'moyN':0}
    cutsF={'minP':0,'maxP':0,'moyP':0,'minN':0,'maxN':0,'moyN':0}
    for R in range(NBRUN):
        f = fif.FForest(d,beta,nbT)
        f.build()
        ##IF
        t_beg = time.time()
        f.setAlpha(0.5)
        scores = f.computeScores("crisp")
        pC,rC,fmC, eC = f.evaluate(scores)
        aucC = f.computeAUC(scores)
        print("CRISP AUC",aucC)

        moyC['AUC'][R] = moyC['AUC'][R] + aucC
        t_end = time.time()
        minP,maxP,moyP,minN,maxN,moyN= f.anomalyCuts(scores)
        cutsC['minP']=cutsC['minP']+minP
        cutsC['maxP']=cutsC['maxP']+maxP
        cutsC['moyP']=cutsC['moyP']+moyP
        cutsC['minN']=cutsC['minP']+minN
        cutsC['maxN']=cutsC['minP']+maxN
        cutsC['moyN']=cutsC['moyN']+moyN

        meanTimeC =meanTimeC + round(t_end-t_beg, 3)

       # print("SCORE RUN",R,"nb trees",nbT,":",(pC,rC,fmC,eC))
        moyC['P'][R] = moyC['P'][R] + pC
        moyC['R'][R] = moyC['R'][R] + rC
        moyC['F'][R] = moyC['F'][R] + fmC
        moyC['E'][R] = moyC['E'][R] + eC

        ##FIF
        t_beg = time.time()
        f.setAlpha(0.9)
        scores = f.computeScores("strongfuzzy")
        aucF = f.computeAUC(scores)
        print("FUZZY AUC",aucF)
        moySF['AUC'][R] = moySF['AUC'][R] + aucF
        pC,rC,fmC, eC = f.evaluate(scores)
        t_end = time.time()
        minP,maxP,moyP,minN,maxN,moyN= f.anomalyCuts(scores)
        cutsF['minP']=cutsF['minP']+minP
        cutsF['maxP']=cutsF['maxP']+maxP
        cutsF['moyP']=cutsF['moyP']+moyP
        cutsF['minN']=cutsF['minP']+minN
        cutsF['maxN']=cutsF['minP']+maxN
        cutsF['moyN']=cutsF['moyN']+moyN
        meanTimeSF =meanTimeSF + round(t_end-t_beg, 3)
        #print("Fuzzy evaluation completed = %ss"%round(t_end-t_beg, 3))

        #print("FSCORE RUN",R,"nb trees",nbT,":",(pC,rC,fmC,eC))
        moySF['P'][R] = moySF['P'][R] + pC
        moySF['R'][R] = moySF['R'][R] + rC
        moySF['F'][R] = moySF['F'][R] + fmC
        moySF['E'][R] = moySF['E'][R] + eC
   
    meanTimeC = meanTimeC / NBRUN
    meanTimeSF = meanTimeSF / NBRUN

    print("CRISP evaluation completed in average in = ",meanTimeC)
    #print("CRISP[NBTREE="+str(nbT)+"]",moyC)
    print("CRISP SEP IR:",cutsC['minP'],cutsC['maxP'],cutsC['moyP']," R:",cutsC['minN'],cutsC['maxN'],cutsC['moyN'])
    print("FUZZY evaluation completed in average in = ",meanTimeSF)
    #print("FUZZY[NBTREE="+str(nbT)+"]",moySF)
    print("FUZZY SEP IR:",cutsF['minP'],cutsF['maxP'],cutsF['moyP']," R:",cutsF['minN'],cutsF['maxN'],cutsF['moyN'])
    print("---------------------------------")

    nbT=nbT+10

for k in moyC.keys():
    print("Measure ",k)
    cr=""
    sf=""
    for R in range(NBRUN):
        moyC[k][R]=moyC[k][R]/NBRUN
        cr=cr+","+str(moyC[k][R])
        moySF[k][R]=moySF[k][R]/NBRUN
        sf=sf+","+str(moySF[k][R])
    print("CRISP",cr)
    print("FUZZY",sf)

sys.exit(0)

