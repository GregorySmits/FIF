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
EPSILONMIN=0.001



class Node:

    def __init__(self,id,d,a,v,l,r,rd=None,sd=None):
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
       # print("NEW NODE DEPTH=",self.depth,"sepAtt=",self.sepAtt,"sepVal",self.sepVal,"reductionDegree=",self.reductionDegree,"separationDegree=",self.separationDegree)
        
    def __str__(self):
        ret= ""
        if self.leftNode is None and self.rightNode is None:
            ret= "LEAF | "+" (IDS : "+str(self.ids)+")"
        else:
            ret = "NODE | " + "sepAtt : "+str(self.sepAtt)+" sepVal : "+str(self.sepVal)+" (IDS : "+str(self.ids)+")"
        return ret
    
    
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
        return roc_auc_score(binData,scoresSelected)

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
        if treeId is None:
            for i in range(self.NBTREES):
                hx = hx + self.trees[i].pathLength(x, self.trees[i].root,0,None, method)
        else:
            hx = hx + self.trees[treeId].pathLength(x, self.trees[treeId].root,0,None, method)

        if treeId is None:
            hx = hx / self.NBTREES
       
        return self.normalizeAnomalyScore(hx)

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
            res[x] = self.computeScore(self.dataSet.getData(x),method, treeId)
           
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
            minN corresponds to the lowest anomaly score of regularities
            maxN corresponds to the highest anomaly score of regularities
            moyN is the mean of anomaly scores of regularities
        """
        minN=(math.inf)
        maxN=-(math.inf)
        minP=(math.inf)
        maxP=-(math.inf)
        moyN = 0
        moyP = 0

        n=0
        for pi in self.dataSet.getEvalDataSet():
            pt = self.dataSet.getData(pi)
            classO = int(pt[len(pt)-1])
            deg = scores[pi]
            if classO == 1:
                n=n+1
                if deg < minP:
                    minP = deg
                if deg > maxP:
                    maxP = deg
                moyP = moyP + deg
            else:
                if deg > maxN:
                    maxN = deg
                if deg < minN:
                    minN = deg
                moyN = moyN + deg

        moyP = moyP / n if n > 0 else moyP
        moyN = moyN / (len(self.dataSet.getEvalDataSet()) -n)
        return minP,maxP,moyP,minN,maxN,moyN


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
                #print("\t\tANOMALY DEG=%.4f"%deg)
                nbP = nbP+1
                if deg >= self.ALPHA:
                    TP = TP + 1
                else:
                    errRt=errRt+1
                    FN = FN + 1
            else:
                #print("REGULAR DEG=%.4f"%deg)
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
        # orthogonal | strongfuzzy | fuzzy
        ret = (0.0,0.0)
        attRange = self.dataSet.maxs[node.sepAtt] - self.dataSet.mins[node.sepAtt]
        v = point[node.sepAtt]
        sep = node.sepVal
        if method == "orthogonal" :
            if v <= sep:
                ret = (max((sep-v) / attRange, EPSILONMIN), 0.0)
            else:
                ret = (0.0, max((v-sep)/attRange,EPSILONMIN))
           
        if method == "crisp":
            if v <= sep:
                ret = (1.0,0.0)
            else:
                ret = (0.0,1.0)
                
        if method == "fuzzy" :
            if v == sep:
                ret = (EPSILONMIN,EPSILONMIN)
            else:
                r = attRange * self.forest.BETA / 2
                if v < sep :
                    rl = 1 if v < sep-2*r else (sep - v) / (2*r)
                    ret=(rl,0)
                else:
                    rg = 1 if v > sep+2*r else (v - sep) / (2*r)
                    ret = (0,rg)
        
        if method == "strongfuzzy":
            r = attRange * self.forest.BETA / 2
            if v < sep + r:
                rl = 1.0 if v <= sep - r else (sep+r-v)/ (2*r)
                rg = 1 - rl
                ret = (rl,rg)
            else:
                ret = (0,1)
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
        if mean != 0 or sep!= 0:    
            r = attRange * self.forest.BETA / 2
            if mean <= sep:
                ret = 1.0 if mean <= (sep - r) else (sep+r-mean)/ (2*r)
               
            else:
                ret = 1.0 if mean >= (sep + r) else (mean-sep + r)/ (2*r)
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
            return Node(idsR, currDepth, None, None, None,None, rd,sd)
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
            lrd = 1- len(idsLeft)/len(idsR)
            rrd = 1- len(idsRight)/len(idsR)

         #   print("LEFT REDUCTION DEGREE nbPtParent=",len(idsR),"nbPointsLeft=",len(idsLeft),"RD=",lrd)
          #  print("RIGHT REDUCTION DEGREE nbPtParent=",len(idsR),"nbPointsRight=",len(idsRight),"RD=",rrd)
            lsd = self.separationDegree(attRange,v,meanR)
            rsd = self.separationDegree(attRange,v,meanL)

            return Node(idsR, currDepth, a, v, self.build(np.array(idsLeft), currDepth + 1, lrd , lsd), self.build(np.array(idsRight), currDepth + 1, rrd,rsd),rd,sd)
            

    def pathLength(self, point, node,e,deg, method : str, nbP : int =0):
        """
        Computes the path's length of a data point in the tree from Node node.
        To compute the path's length in the entire tree, set node to the root of the tree.

        Parameters
        ----------
        point : 1-D array of floats
            coordinates of the data point.
        node : Node
            the Node from which the search starts.
        e : integer (0 at start)
            current path length.
        deg : float (None at start)
            current cumulated degrees

        Returns
        -------
        float
            length of the path of the data point.

        """
        if((node.leftNode is not None) or (node.rightNode is not None)):
            degs = self.computeDegree(point, node, method)
    
            if method == "strongfuzzy":
                if degs[0] > 0:
                    leftDeg = self.aggNodeDegrees(deg,degs[0],node.leftNode.reductionDegree,node.leftNode.separationDegree)
                    if degs[1] > 0:
                        rightDeg =self.aggNodeDegrees(deg,degs[1],node.rightNode.reductionDegree,node.rightNode.separationDegree)
                        p1 = self.pathLength(point, node.leftNode,e+1,leftDeg,method)
                        p2 = self.pathLength(point, node.rightNode,e+1,rightDeg,method)

                        return max(p1,p2)#(p1+p2)/2
                        #return min(self.pathLength(point, node.leftNode,e+1,leftDeg,method), self.pathLength(point, node.rightNode,e+1,rightDeg,method))
                        #return max(self.pathLength(point, node.leftNode,e+1,leftDeg,method), self.pathLength(point, node.rightNode,e+1,rightDeg,method))
                    else:
                        p1 = self.pathLength(point, node.leftNode,e+1,leftDeg,method)
                        return p1
                else:
                    rightDeg =self.aggNodeDegrees(deg,degs[1],node.rightNode.reductionDegree,node.rightNode.separationDegree)
                    p2= self.pathLength(point, node.rightNode,e+1,rightDeg,method)

                    return p2

            if method == "crisp":
                if point[node.sepAtt] <= node.sepVal:
                    if method == "crisp":
                        return self.pathLength(point, node.leftNode,e+1,deg,method)

                else:
                    if method == "crisp":
                        return self.pathLength(point, node.rightNode,e+1,deg,method)
        else:
            # (If it is an external node)
            if method == "crisp":
                return e #/ c(len(self.dataSet.trainingData))
            else:                
                return deg #/ c(len(self.dataSet.trainingData))
        
    def aggNodeDegrees(self,curDegree :float,pointIsolation : float,nodeReduction:float,nodeSeparation:float):
        """
        curDegree is None at first call
        """
        #using all degrees and the probabilistic norm

        if curDegree is None: 
            curDegree = 0
        #
        if nodeSeparation > 1:
            print("!!! NODE SEPARATION",nodeSeparation)
        if nodeReduction > 1:
            print("!!! NODE REDUCTION",nodeReduction)

        if pointIsolation > 1:
            print("!!! NODE POINT ISOLATION",pointIsolation)

        nodeDeg = nodeSeparation*nodeReduction

        return curDegree + (1- (pointIsolation + nodeDeg - pointIsolation * nodeDeg))
#first method
 #       return curDegree + (pointIsolation * nodeReduction)
#using Zadeh tconorm
#        return curDegree + (1 - max(pointIsolation, nodeReduction*nodeSeparation) 
#using the probabilistic tconorm between point separation and node degree
#        return curDegree + (pointIsolation + nodeDeg  - pointIsolation * nodeDeg)
#probabilistic tnorm and conorm withou node separation
#        return curDegree + (pointIsolation * nodeReduction) - curDegree * (pointIsolation * nodeReduction)
#tnorm + KLEENE DIENES IMPLICATION
#        if curDegree is None: 
#            curDegree = 1
#        return curDegree * max(1-min(nodeReduction,nodeSeparation),pointIsolation)

if __name__ == "__main__":
    d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#    d = Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
#    d = Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0.9)
#    d = Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0) 
    e = d.getEvalDataSet()


#    f = Forest(d, "strongfuzzy",0.84,0.05)
    beta=0.05
    f = FForest(d,beta)

    f.build()
    #print(f)
   
    import viewer as vw
    fig, ax = plt.subplots(2,5)
    for i in range(5):
        aTreeId =random.randint(0,f.NBTREES-1)
        
        A=0.5        
        print("CRISP METHOD WITH PARAMETERS ALPHA:",A)
        f.setAlpha(A)
        
        scores = f.computeScores("crisp",aTreeId)
        pC,rC,fmC, eR = f.evaluate(scores)
        print("CRISP ROC AUC ",f.computeAUC(scores))
        minP,maxP,moyP,minN,maxN,moyN= f.anomalyCuts(scores)
        print("CRISP SEP IR:",minP,maxP,moyP," R:",minN,maxN,moyN)
        msg = "P"+str(round(pC,1))+" R"+str(round(rC,1))+" F"+str(round(fmC,1))+" R"+str(round(eR,1))
        print(msg)
        vw.viewIsolatedDatasetWithAnomalies(d,f.trees[aTreeId],scores,f.ALPHA,e,ax[0][i],msg)
        
        A=0.9
        print("FUZZY METHOD WITH PARAMETERS ALPHA:",A, "BETA:",beta)
        f.setAlpha(A)
        scoresF = f.computeScores("strongfuzzy",aTreeId)
        pSF,rSF,fmSF,eRSF = f.evaluate(scoresF)
        minP,maxP,moyP,minN,maxN,moyN= f.anomalyCuts(scoresF)
        print("FUZZY ROC AUC ",f.computeAUC(scoresF))

        print("FUZZY SEP IR:",minP,maxP,moyP," R:",minN,maxN,moyN)
        msg="P"+str(round(pSF,1))+" R"+str(round(rSF,1))+" F"+str(round(fmSF,1))+" E"+str(round(eRSF,1))
        print(msg)
        print("\n")
        vw.viewIsolatedDatasetWithAnomalies(d,f.trees[aTreeId],scoresF,f.ALPHA,e,ax[1][i],msg)
    plt.show()
 