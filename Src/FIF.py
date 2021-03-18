#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:40:37 2019

@author: SHAMAN
"""
from dataset import *
import math
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import copy
EPSILONMIN=0.001

    
def c(n):
    """
    Computes the adjustment c(Size) : c(i) = ln(i) + Euler's constant
    
    Parameters
    ----------
    n : integer
        size of the sample.

    Returns
    -------
    float
        adjustment value.

    """
    if n > 2:
        return 2.0 * (np.log(n-1) + np.euler_gamma) - 2.0 * (n-1.0) / n
    elif n == 2:
        return 1.0
    else:
        return 0.0

def symSum(x : float,y : float):
    """
    return the aggregation of x and y using the symSum reinforcement operator

    """
    ret = 0 if x == 0 or y == 0 else (x*y) / (x * y + (1-x) * (1-y))
    return ret

class Node:

    def __init__(self,id,d,a,v,l,r,rd=None,sd=None, bounds = None):
        """
        Creates a new Node (internal or external)

        Parameters
        ----------
        id :dict()
            pointIdx => deg
        d : integer
            depth of the node in the tree.
        a : integer
            index of the split attribute.
        v : float
            split value.
        l : Node
            left child.
        r : Node
            right child.
        rd : float
            reduction degree depending on the number of points in the parent node and the number of points in the node
        sd : float
            separation degree depending on how far points are in the other side of the separation line

        Returns
        -------
        None.

        """
        self.ids = id
        self.depth = d
        self.sepAtt = a
        self.sepVal = v
        self.leftNode = l
        self.rightNode = r
        self.reductionDegree = rd
        self.separationDegree = sd
        self.bounds=bounds
        #print("NEW NODE DEPTH=",self.depth,"sepAtt=",self.sepAtt,"sepVal",self.sepVal,"reductionDegree=",self.reductionDegree,"separationDegree=",self.separationDegree)
        
    def __str__(self):
        ret= ""
        if self.leftNode is None and self.rightNode is None:
            ret= "LEAF | "+" (IDS : "+str(self.ids)+")"
        else:
            ret = "NODE | " + "sepAtt : "+str(self.sepAtt)+" sepVal : "+str(self.sepVal)+" (IDS : "+str(self.ids)+")"
        return ret
    

class FForest:

    def __init__(self, d: Dataset, beta : float,nbT : int = 100):
        """
        Parameters
        ----------
        d : Dataset
            the data
        beta : float
            beta is an hyperparameter about that determines the width of the uncertainty area around separation values. This is used to compute the separation degree
        Returns
        -------
        None.

        """
        random.seed()
        self.dataSet= d
        self.trees = []
        #hyperparameters
        self.NBTREES = nbT
        self.MAXSIZESS = 256
        self.BETA = beta
        self.ALPHA = None

    def setAlpha(self, m:str):
        """
        Set the alpha hyper-parameter
        """
        self.ALPHA = m

    def setBeta(self, m:str):
        """
        Set the beta hyper-parameter
        """
        self.BETA = m
        

    def __str__(self):
        ret = "FOREST | NBTREES : "+str(self.NBTREES)+ " MAXSIZE : "+str(self.MAXSIZESS)+" ALPHA : "+str(self.ALPHA)+" BETA : "+str(self.BETA)
        for i in range(len(self.trees)):
            ret += "\n"+str(self.trees[i])
        return ret
    
    def build(self):
        """
        Builds the isolation forest
        """
        #permutation of the ids
        rng = np.random.default_rng()
        nbPoints = len(self.dataSet.getTrainingDataSet())
        sampleSize = min(self.MAXSIZESS, nbPoints)
        for i in range(self.NBTREES):
            sampleIds = rng.choice(self.dataSet.getTrainingDataSet(), sampleSize, replace=False)#nbPoints, sampleSize, replace=False)
            self.trees.append(FTree(self, self.dataSet, sampleIds))
 
    def normalizeAnomalyScore(self,hx:float):
        """
        normalize the anomaly score
        Parameters
        ----------
        hx : float
            the mean length of the path to isolate a point
        Returns
        -------
        float
            the normalized anomaly score
        """
        nbPoints = len(self.dataSet.getData())
        sampleSize = min(self.MAXSIZESS, nbPoints)
        hx = 2 ** ( -hx / c(sampleSize))
        return hx
    
    def computeAUC(self,scores):
        binData = self.dataSet.binClassesVector()
        scoresSelected = np.take(scores,self.dataSet.getEvalDataSet())
        ruc= roc_auc_score(binData,scoresSelected)
        ns_fpr, ns_tpr, _ = roc_curve(binData, scoresSelected)
        return ruc,ns_fpr,ns_tpr

    def computeAnomalyDegree(self,path, method):
        """
        A point is an anomaly if it mostly falls far from separation line in isolated areas : high density reduction and far from points on the other side
        """
        deg = 1
        if method == "crisp":
            deg = len(path)
        else:
            wSum = 0
            w=1
            rank=len(path)
            w=1/2**rank
            print("ANOMALY DEGREE FOR PATH:",path)
            for a in path:
                pointDeg = a["pointIsolation"]
                areaDeg = (a['nodeReduction'] +  a['nodeSeparation']) /2
                combDeg = pointDeg * areaDeg
                deg = min(deg,   combDeg)
                print('\t-PT ISOL DEG:',pointDeg,"AREA DEG (ND:",a['nodeReduction'],"NS:",a['nodeSeparation'],"->AD:,",areaDeg,"combDeg:",combDeg,"=",deg)
            #deg = deg/rank
            print("DEG FINAL =",deg)
        return deg

    def computeScore(self, x, method : str,treeId : int = None):
        """
        Computes the anomaly scores of a single data point
        using the degrees attached to points in the leaf
        Parameters
        ----------
        x : 1D array of floats
            the coordinates of the data point.
        method : str
            the method to use to compute the scores (crisp, strongfuzzy)
        treeId : int (None by default)
            the tree index in case the score is computed on only one Node
        Returns
        -------
        float
            the anomaly score of the data point in the isolation forest.

        """
        hx = 0.0
        rng = range(self.NBTREES) if treeId is None else range(treeId, treeId+1)
        
        for i in rng:
            allPaths=[]
            self.trees[i].pathLength(x, self.trees[i].root,0, [], allPaths, method)
            if method == "strongfuzzy":
                maxAs = -math.inf
                for path in allPaths:
                    ad = self.computeAnomalyDegree(path, method)
                    if ad > maxAs:
                        maxAs = ad
            if method == "crisp":
                maxAs = len(allPaths[0])
            hx=hx+maxAs

        hx = hx / len(rng)

        if method == "crisp":
            
            hx = self.normalizeAnomalyScore(hx)
        
        return hx

    class ForestParametersNotSet(Exception):
        """Raised when the input value is too large"""
        pass

    def computeScores(self, method : str, treeId : int = None):
        """
        Computes the anomaly scores of all the data points in the dataset
        Parameters
        -------
        method : str
            the method to use to compute the scores (crisp, strongfuzzy)
        treeId : int (None by default)
            the tree index in case the score is computed on only one Node
        Returns
        -------
        1D array of floats
            the anomaly score of each data point.
        """
        if self.ALPHA is None or (method == "strongfuzzy" and self.BETA is None):
            raise ForestParametersNotSet

        res = np.zeros(len(self.dataSet.getData()))
        for x in self.dataSet.getEvalDataSet():
            print("POINT :",self.dataSet.getData(x))
            pt = self.dataSet.getData(x)
            classO = int(pt[len(pt)-1])
            res[x] = self.computeScore(self.dataSet.getData(x),method, treeId)
            deg = res[x]
            if classO == 1:
                print("\t\tANOMALY DEG=%.4f"%deg)
            else:
                print("REGULAR DEG=%.4f"%deg)
            print("**************************************")
        return res

    def anomalyCuts(self, scores):
        """
        Analyses the anomaly scores of regular and irregular points
        Parameters
        -------
        scores : AD array of floats storing the anomaly scores
        Returns
        -------
        minP,maxP,moyP,minN,maxN,moyN : float,float,float,float,float,float 
            minP corresponds to the lowest anomaly score of irregularities
            maxP corresponds to the highest anomaly score of irregularities
            moyP is the mean of anomaly scores of irregularities
            stdP is the standard deviation of anomaly scores of irregularities  
            minN corresponds to the lowest anomaly score of regularities
            maxN corresponds to the highest anomaly score of regularities
            moyN is the mean of anomaly scores of regularities
            stdN is the standard deviation of anomaly scores of regularities
        """
    

        scoresP = []
        scoresN = []
        for pi in self.dataSet.getEvalDataSet():
            pt = self.dataSet.getData(pi)
            classO = int(pt[len(pt)-1])
            deg = scores[pi]
            if classO == 1:
                scoresP.append(deg)
            else:
                scoresN.append(deg)
        return [min(scoresP),min(scoresN)],[max(scoresP),max(scoresN)],[sum(scoresP)/len(scoresP),sum(scoresN)/len(scoresN)],[np.std(scoresP),np.std(scoresN)]


    def evaluate(self, scores):
        """
        In a supervised setting, where anomalies are known, this method evaluates the efficiency of the detection process

        Parameters
        -------
        scores :1D array of floats
            the anomaly score of each data point.
        """
        TP=0
        FP=0
        FN=0
        TN=0
        nbP = 0
        nbN = 0
        errRt=0
        print("WITH SETTINGS ALPHA",self.ALPHA,"BETA",self.BETA)
        for pi in self.dataSet.getEvalDataSet():
            pt = self.dataSet.getData(pi)
            classO = int(pt[len(pt)-1])
            deg = scores[pi]
            if classO == 1:
              #  print("\t\tANOMALY DEG=%.4f"%deg)
                nbP = nbP+1
                if deg >= self.ALPHA:
                    TP = TP + 1
                else:
                    errRt=errRt+1
                    FN = FN + 1
            else:
            #    print("REGULAR DEG=%.4f"%deg)
                nbN=nbN+1
                if deg < self.ALPHA:
                    TN = TN + 1
                else:
                    errRt=errRt+1
                    FP = FP + 1

        precision = 0
        rappel = 0
        fmeasure = 0
        if TP + FN > 0:
            rappel = (TP/(TP+FN))
        if TP + FP >0:
            precision = (TP/(TP+FP))
        if (TP+.5*(FN+FP)) > 0:
            fmeasure = (TP/(TP+.5*(FN+FP)))
        if nbP == 0 and FN + TP == 0:
            fmeasure = 1 
    #    print("STATS (nbPoints:",len(self.dataSet.getEvalDataSet())," nbAnomalie:",nbP,") - (TruePos:",TP,", FalseNeg:",FN,", TrueNeg:",TN,", FalsePos:",FP,")")
        return precision, rappel, fmeasure, errRt*100/len(self.dataSet.getEvalDataSet())

class FTree :

    def __init__(self,f : FForest,d : Dataset,ids):
        self.forest = f
        self.dataSet = d
        self.ids = ids
        self.MAXHEIGHT = math.ceil(math.log(len(ids),2))
        self.root = self.build(self.ids, 0)
        
    
        
    def __str__(self):
        ret="TREE | IDS : "+str(self.ids)+"\n"
        
        return ret + self.recStr(self.root,0)


    def recStr(self, n,i):
        ret = ""
        if n is not None:
            ret="\t"*i+str(n)+"\n"+self.recStr(n.leftNode,i+1)+"\n"+self.recStr(n.rightNode,i+1)
        return ret


    def computeDegree(self,point, node:Node,method:str):
        """
        Compute the membership degree according to the selected mode
        Parameters
        ----------
        point : 1Darray
            the point
        node : Node
            the node concerned by the isolation, degrees returned are the membreship degrees on both sides of the node
        Returns
        -------
        (Float,Float) : the membership degree in the two subsets
        """
        
        ret = (0.0,0.0)
        #attRange = self.dataSet.maxs[node.sepAtt] - self.dataSet.mins[node.sepAtt]
        attRange = node.bounds[1] - node.bounds[0]
        v = point[node.sepAtt]
        sep = node.sepVal
        
        if method == "strongfuzzy":
            r = attRange * self.forest.BETA / 2
            if v < sep + r:
                rl = 1.0 if v <= sep - r else (sep+r-v)/ (2*r)
                rg = 1 - rl
                ret = (rl,rg)
            else:
                ret = (0.0,1.0)
       
        return ret


    def separationDegree(self, attRange : float, sep : float, mean : float):
        """
        Compute the separation degree that is related to the distance between the separation value v and 
        the mean value of the points on the other side of the separation line
        Parameters
        ----------
        attrange : float
            the range of the concerned attribute (max - min)
        sep : float
            the separation value
        meanR : float
            the mean value on the concerned attribute of the points on the other side of the separation value
        Returns
        -------
        float 
            the isolation degree that is computed using a FSB def
        """
        ret = 0
        """
        if mean != 0 or sep!= 0:    
            r = attRange * self.forest.BETA / 2
            if mean <= sep:
                ret = 1.0 if mean <= (sep - r) else (sep+r-mean)/ (2*r)
               
            else:
                ret = 1.0 if mean >= (sep + r) else (mean-sep + r)/ (2*r)
        """
        if mean != 0 or sep!= 0:    
            r = attRange * self.forest.BETA 
            if mean <= sep:
                ret = 1.0 if mean <= (sep - r) else (sep-mean) / r
               
            else:
                ret = 1.0 if mean >= (sep + r) else (mean-sep) / r
        return ret


    def build(self, idsR, currDepth, rd=None,sd=None):
        """
        Builds the tree

        Parameters
        ----------
        idsR : dict()
            pointIdx => deg
        currDepth : integer
            the current depth of the tree in the building process. 
            Starts from 0 for the root of the tree and is incremented as we go deeper in the tree. 
            Used to stop building the tree when we reach the height limit.

        Returns
        -------
        Node
            can be an internal or an external node. An external node has no children and
            is returned when we have only one data point left or when the height limit is 
            reached.

        """
        attributes = self.dataSet.getAttributes()
        
        if len(idsR) <= 1 or currDepth >= self.MAXHEIGHT:
            return Node(idsR, currDepth, None, None, None,None, rd,sd,None)
        else:
            a = random.randint(0, len(attributes)-1)
            mini = math.inf
            maxi = (-math.inf)
            for ix in idsR:#get the max and min values of the points in ix
                data = self.dataSet.getData(ix)
                if data[a] > maxi: maxi = data[a] 
                if data[a] < mini: mini = data[a]
                
            v=random.uniform(mini,maxi)
            
            idsLeft = []
            idsRight = []
            meanL=0
            meanR=0

            for ix in idsR:
                if self.dataSet.getData(ix)[a] <= v:
                    idsLeft.append(ix)
                    meanL=meanL+self.dataSet.getData(ix)[a]
                else:
                    idsRight.append(ix)
                    meanR=meanR+self.dataSet.getData(ix)[a]

            meanL = 0 if len(idsLeft) == 0 else meanL / len(idsLeft)
            meanR = 0 if len(idsRight) == 0 else meanR / len(idsRight)
            attRange = self.dataSet.maxs[a] - self.dataSet.mins[a]
            bounds = (mini,maxi)
     
            if attRange > 0:
#                lrd = ((len(self.dataSet.trainingData) - len(idsLeft))/len(self.dataSet.trainingData) + (v - mini) /attRange)/2
                lrd = min((len(self.dataSet.trainingData) - len(idsLeft))/len(self.dataSet.trainingData), (v - mini) /attRange)

            else:
                lrd=0
                
            lsd = self.separationDegree(attRange,v,meanR)
            rsd = self.separationDegree(attRange,v,meanL)


            print("LEVEL:",currDepth,"ATT",a,"SEP",v)
            print("DENSITY REDUCTION DEGREE LEFT:")
            print("\t CARDINALITY |D|:",len(self.dataSet.trainingData),"|Dleft|:",len(idsLeft),"=",(len(self.dataSet.trainingData) - len(idsLeft))/(len(self.dataSet.trainingData)))
            print("\t SPACE ATT RANGE:",attRange,"SUBSPACE:",(v-mini),"=",( (v - mini) )/(attRange))
            print("\t LEFT AREA DEGREE=",lrd)
            print("\t LEFT AREA SEP DEGREE wrt. RIGHT=",lsd)
#            rrd = ((len(self.dataSet.trainingData) - len(idsRight))/len(self.dataSet.trainingData) + (maxi - v) /attRange)/2 #density reduction 
            rrd = min((len(self.dataSet.trainingData) - len(idsRight))/len(self.dataSet.trainingData), (maxi - v) /attRange) #density reduction 

            #rrd = ((len(idsR) - len(idsRight))/len(idsR) + (maxi - v) /attRange)/2 #density reduction 

            print("DENSITY REDUCTION DEGREE RIGHT:")
            print("\t CARDINALITY |D|:",len(self.dataSet.trainingData),"|Dright|:",len(idsRight),"=",(len(self.dataSet.trainingData) - len(idsRight))/(len(self.dataSet.trainingData)))
            print("\t SPACE ATT RANGE:",attRange,"SUBSPACE:",(maxi - v),"=",((maxi - v) )/(attRange) )
            print("\t RIGHT AREA DEGREE=",rrd)
            print("\t RIGHT AREA SEP DEGREE wrt. LEFT=",rsd)
            print('\n')
            
         

            return Node(idsR, currDepth, a, v, self.build(np.array(idsLeft), currDepth + 1, lrd , lsd), self.build(np.array(idsRight), currDepth + 1, rrd,rsd),rd,sd,bounds)
            
  
    def pathLength(self, point, node,e, curPath, allPaths, method):
        """
        pathC is a list of possible pathes
        """
        if((node.leftNode is not None) or (node.rightNode is not None)):
       # if node is not None:
            degs = self.computeDegree(point, node, method)    
            if method == "strongfuzzy":
                if degs[0] > 0:
                    degsL ={"SIDE":0,'pointIsolation':degs[0],"nodeReduction":node.leftNode.reductionDegree,"nodeSeparation":node.leftNode.separationDegree}
                    if degs[1] > 0:
                        degsR ={"SIDE":1,'pointIsolation':degs[1],"nodeReduction":node.rightNode.reductionDegree,"nodeSeparation":node.rightNode.separationDegree}
                        curPath2 = copy.deepcopy(curPath)
                        curPath.append(degsL)
                        curPath2.append(degsR)
                        self.pathLength(point,node.leftNode,e+1,curPath, allPaths, method)
                        self.pathLength(point,node.rightNode,e+1,curPath2, allPaths, method)

                    else:
                        curPath.append(degsL)
                        self.pathLength(point,node.leftNode,e+1,curPath, allPaths, method)
                else:
                    degsR ={"SIDE":1,'pointIsolation':degs[1],"nodeReduction":node.rightNode.reductionDegree,"nodeSeparation":node.rightNode.separationDegree}
                    curPath.append(degsR)
                    self.pathLength(point,node.rightNode,e+1,curPath, allPaths, method)
                
            if method == "crisp":
                if point[node.sepAtt] <= node.sepVal:
                    degsL ={"SIDE":0,'pointIsolation':1,"nodeReduction":None,"nodeSeparation":None}
                    curPath.append(degsL)
                    self.pathLength(point,node.leftNode,e+1,curPath, allPaths, method)
                else:
                    degsR ={"SIDE":1,'pointIsolation':1,"nodeReduction":None,"nodeSeparation":None}
                    curPath.append(degsR)
                    self.pathLength(point,node.rightNode,e+1,curPath, allPaths, method)  
        else:
            allPaths.append(curPath)
    
if __name__ == "__main__":

    #The datasets and their dimensions
    real_datasets = ["annthyroid", "arrhythmia", "breastw", "cardio", "cover", "hbk", "http", "ionos", "letter", "lympho",
                    "mammography", "musk", "pima", "satellite", "shuttle", "smtp", "wood"]

    dimensions = [6, 271, 9, 21, 10, 4, 3, 32, 32, 18, 6, 166, 8, 36, 9, 3, 6]
    #You just have to choose the index of the dataset in the table. Example : idx_dataset = 0 <=> annthyroid
    idx_dataset = 0
    converters = {}
    header = []
    for i in range(dimensions[idx_dataset]):
        converters[i] = lambda s: float(s.strip() or 0)
        header.append(str(i))
    converters[dimensions[idx_dataset]] = lambda s: int(float(s.strip() or 0))
    header.append("CLASS")
#    d = Dataset("../Data/"+real_datasets[idx_dataset]+".csv", header, converters, True, 0.8)

    d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#    d = Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#    d = Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0.8)
#    d = Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8) 
    d = Dataset("../Data/dataTest.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0)

    e = d.getEvalDataSet()


#    f = Forest(d, "strongfuzzy",0.84,0.05)
    beta=0.2
    nbT=1
    f = FForest(d,beta,nbT)

    f.build()
    aTreeId =random.randint(0,f.NBTREES-1)
   
    import viewer as vw
    #
    
    idds = f.trees[aTreeId].ids
 
    
    td = np.empty([len(e),3])
    for i in range(len(e)):
        td[i] = f.dataSet.getData(e[i])
    vw.viewData(td)
    vw.drawTreeRec(f.trees[aTreeId].root,d )
    plt.show()
    sys.exit(0)
    

    for pte in e:
        allPaths=[]
        pt = f.dataSet.getData(pte)
        f.trees[aTreeId].pathLength(pt, f.trees[aTreeId].root,0, [], allPaths, "strongfuzzy")
        print("POINT ",pt)
        f.computeScore(pt,"strongfuzzy",aTreeId)
        print("\n************\n")

    #
    VIEW=True
   
    if VIEW:
        fig, ax = plt.subplots(2,3)
       
 
    
    A=0.5
    print("CRISP METHOD WITH PARAMETERS ALPHA:",A)
    f.setAlpha(A)
    f.setBeta(0)
        
    scores = f.computeScores("crisp")
   
   
    pC,rC,fmC, eR = f.evaluate(scores)
    print("\tPRECISION:",round(pC,3),"RAPPEL:",round(rC,3),"FMEASURE:",round(fmC,3),"ERRORRATE:",round(eR,3))
    if VIEW:
        vw.viewIsolatedDatasetWithAnomalies(d,f.trees[aTreeId],scores,f.ALPHA,e,ax[0][0],"IT")

    auc,lr_fpr, lr_tpr = f.computeAUC(scores)
    print("\tAUC:",round(auc,3))
    if VIEW:
        vw.displayAUC(lr_fpr,lr_tpr,"Crisp",ax[0][1])

    minPC,maxPC,moyPC,stdPC= f.anomalyCuts(scores)
    if VIEW:
        vw.displayCuts(minPC,maxPC,moyPC,stdPC,ax[0][2])

    print("\tANOMALY SCORES:")
    print("\t\tIRREGULARITIES MIN:,",round(minPC[0],3),"MAX:",round(maxPC[0],3),"MEAN:",round(moyPC[0],3),"STD:",round(stdPC[0],3))
    print("\t\tREGULARITIES MIN:,",round(minPC[1],3),"MAX:",round(maxPC[1],3),"MEAN:",round(moyPC[1],3),"STD:",round(stdPC[1],3))
    
    A=0.9
    f.setBeta(beta)

    print("***********\n***********")
    print("FUZZY METHOD WITH PARAMETERS ALPHA:",A, "BETA:",beta)
    f.setAlpha(A)
    scoresF = f.computeScores("strongfuzzy")

    
    pSF,rSF,fmSF,eRSF = f.evaluate(scoresF)
    print("\tPRECISION:",round(pSF,3),"RAPPEL:",round(rSF,3),"FMEASURE:",round(fmSF,3),"ERRORRATE:",round(eRSF,3))

    auc,lr_fpr, lr_tpr = f.computeAUC(scoresF)
    print("\tAUC:",round(auc,3))
    if VIEW:
        vw.displayAUC(lr_fpr,lr_tpr,"Fuzzy",ax[1][1])

    minP,maxP,moyP,stdP= f.anomalyCuts(scoresF)
    if VIEW:
        vw.displayCuts(minP,maxP,moyP,stdP,ax[1][2])
    print("\tANOMALY SCORES:")
    print("\t\tIRREGULARITIES MIN:,",round(minP[0],3),"MAX:",round(maxP[0],3),"MEAN:",round(moyP[0],3),"STD:",round(stdP[0],3))
    print("\t\tREGULARITIES MIN:,",round(minP[1],3),"MAX:",round(maxP[1],3),"MEAN:",round(moyP[1],3),"STD:",round(stdP[1],3))
    if VIEW:
        vw.viewIsolatedDatasetWithAnomalies(d,f.trees[aTreeId],scoresF,f.ALPHA,e,ax[1][0],"FIT")
        plt.show()
 