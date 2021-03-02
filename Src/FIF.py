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
EPSILONMIN=0.001

class Forest:

    def __init__(self, d: Dataset):
        """
        Parameters
        ----------
        d : Dataset
            the data
            
        m : str
            the method used to build the forest (crisp | orthogonal | fuzzy | strongfuzzy)

        alph : float
            the anomaly threshold

        bet : float
            the imprecision threshold

        Returns
        -------
        None.

        """
        random.seed()
        self.mode = None
        self.dataSet= d
        self.trees = []
        #hyperparameters
        self.NBTREES = 1
        self.MAXSIZESS = 256
        self.BETA = None
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
        
    def setMode(self, m:str):
        """
        Modify the mode of the forest
        """
        self.mode = m

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
 
 
    def computeScore(self, x, treeId : int = None):
        """
        Computes the anomaly scores of a single data point
        using the degrees attached to points in the leaf
        Parameters
        ----------
        x : 1D array of floats
            the coordinates of the data point.

        Returns
        -------
        float
            the anomaly score of the data point in the isolation forest.

        """
        hx = 0.0
        if treeId is None:
            for i in range(self.NBTREES):
                hx = hx + self.trees[i].pathLength(x, self.trees[i].root,0,0)
        else:
            hx = hx + self.trees[treeId].pathLength(x, self.trees[treeId].root,0,0)

        #    print("SCORE FOR POINT",x, "IN TREE",i,":",hx)
        if treeId is None:
            hx = hx / self.NBTREES
        #if self.mode == "crisp":#normalization
        nbPoints = len(self.dataSet.getData())
        sampleSize = min(self.MAXSIZESS, nbPoints)
        hx = 2 ** ( -hx / c(sampleSize))
        return hx

    class ForestParametersNotSet(Exception):
        """Raised when the input value is too large"""
        pass

    def computeScores(self, treeId : int = None):
        """
        Computes the anomaly scores of all the data points in the dataset
      
        Returns
        -------
        1D array of floats
            the anomaly score of each data point.
        """
        if self.mode is None or (self.mode is not None and self.mode != "crisp" and self.BETA is None):
            raise ForestParametersNotSet

        res = np.zeros(len(self.dataSet.getData()))
        for x in self.dataSet.getEvalDataSet():
            res[x] = self.computeScore(self.dataSet.getData(x),treeId)
           
        return res

    def anomalyCuts(self, scores):
        """
        Analyses the anomaly scores of regular and irregular points
        Parameters
        -------
        scores : AD array of floats storing the anomaly scores
        Returns
        -------
        maxN,minP,moy : float,float,float 
            maxN corresponds to the highest anomaly score of regularities
            minP corresponds to the lowest anomaly score of irregularities
            moy is the mean of anomaly scores of irregularities
        """
        maxN=-(math.inf)
        minP=(math.inf)
        moy = 0
        n=0
        for pi in self.dataSet.getEvalDataSet():
            pt = self.dataSet.getData(pi)
            classO = int(pt[len(pt)-1])
            deg = scores[pi]
            if classO == 1:
                n=n+1
                if deg < minP:
                    minP = deg
                moy = moy + deg
            else:
                if deg > maxN:
                    maxN = deg

        moy = moy / n if n > 0 else moy
        return maxN,minP,moy


    def evaluate(self, scores):
        """
        In a supervised setting, where anomalies are known, this method evaluates the efficiency of the detection process

        Parameters
        -------
        scores :1D array of floats
            the anomaly score of each data point.
        """
        if self.mode is None or (self.mode is not None and self.mode != "crisp" and self.BETA is None):
            raise ForestParametersNotSet

        TP=0
        FP=0
        FN=0
        TN=0
        nbP = 0
        nbN = 0
        errRt=0
        #print("WITH SETTINGS ALPHA",self.ALPHA,"BETA",self.BETA)
        for pi in self.dataSet.getEvalDataSet():
            pt = self.dataSet.getData(pi)
            classO = int(pt[len(pt)-1])
            deg = scores[pi]
            if classO == 1:
                print("\t\tANOMALY DEG=",deg)
                nbP = nbP+1
                if deg >= self.ALPHA:
                    TP = TP + 1
                else:
                    errRt=errRt+1
                    FN = FN + 1
            else:
                print("REGULAR DEG=",deg)
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
        print("STATS (nbPoints:",len(self.dataSet.getEvalDataSet())," nbAnomalie:",nbP,") - (TruePos:",TP,", FalseNeg:",FN,", TrueNeg:",TN,", FalsePos:",FP,")")
        return precision, rappel, fmeasure, errRt*100/len(self.dataSet.getEvalDataSet())

class FTree :

    def __init__(self,f : Forest,d : Dataset,ids):
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


    def computeDegree(self,point, node):
        """
        Compute the membership degree according to the selected mode
        Parameters
        ----------
        v : the point value
        sep : the separation value
        min : the min value of the points to separate
        max : the max value of the points to separate

        Returns
        -------
        (Float,Float) : the membership degree in the two subsets
        """
        # orthogonal | strongfuzzy | fuzzy
        ret = (0.0,0.0)
        attRange = self.dataSet.maxs[node.sepAtt] - self.dataSet.mins[node.sepAtt]
        v = point[node.sepAtt]
        sep = node.sepVal
        if self.forest.mode == "orthogonal" :
            if v <= sep:
                ret = (max((sep-v) / attRange, EPSILONMIN), 0.0)
            else:
                ret = (0.0, max((v-sep)/attRange,EPSILONMIN))
           
        if self.forest.mode == "crisp":
            if v <= sep:
                ret = (1.0,0.0)
            else:
                ret = (0.0,1.0)
                
        if self.forest.mode == "fuzzy" :
            if v == sep:
                ret = (EPSILONMIN,EPSILONMIN)
            else:
                r = attRange * self.forest.BETA / 2
                if v < sep :
                    rl = 1 if v < sep-2*r else (sep - v) / 2*r
                    ret=(rl,0)
                else:
                    rg = 1 if v > sep+2*r else (v - sep) / 2*R 
                    ret = (0,rg)
        
        if self.forest.mode == "strongfuzzy":
            r = attRange * self.forest.BETA / 2
            if v < sep + r:
                rl = 1.0 if v <= sep - r else max((sep+r-v)/ 2*r,EPSILONMIN)
                rg = 1 - rl
                ret = (rl,rg)
            else:
                ret = (0,1)
        return ret




    def build(self, idsR, currDepth, rd=None):
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
        
        if len(idsR) <= 1:# or currDepth >= self.MAXHEIGHT:
            return Node(idsR, currDepth, None, None, None, None, None, rd)
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
            cm = None
            cp = None
            
            for ix in idsR:
                if self.dataSet.getData(ix)[a] <= v:
                    idsLeft.append(ix)
                    if cm is None or self.dataSet.getData(ix)[a] < cm:
                        cm = self.dataSet.getData(ix)[a]
                else:
                    idsRight.append(ix)
                    if cp is None or self.dataSet.getData(ix)[a] > cp:
                        cp = self.dataSet.getData(ix)[a]

            return Node(idsR, currDepth, a, v, self.build(np.array(idsLeft), currDepth + 1, 1- len(idsLeft)/len(idsR)), self.build(np.array(idsRight), currDepth + 1, 1- len(idsRight)/len(idsR)),cm,cp,rd)
            

    def pathLength(self, point, node,e,deg):
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
        deg : float (1 at start)
            current cumulated degrees

        Returns
        -------
        float
            length of the path of the data point.

        """
        if((node.leftNode is not None) or (node.rightNode is not None)):
            degs = self.computeDegree(point, node)
            if self.forest.mode == "strongfuzzy":
                if degs[0] > 0:
                    if degs[1] > 0:
                        return max(self.pathLength(point, node.leftNode,e+1,deg+(degs[0]*node.leftNode.reductionDegree)), self.pathLength(point, node.rightNode,e+1,deg+(degs[1]*node.rightNode.reductionDegree)))
                    else:
                        return self.pathLength(point, node.leftNode,e+1,deg+(degs[0]*node.leftNode.reductionDegree))
                else:
                    return self.pathLength(point, node.rightNode,e+1,deg+(degs[1]*node.rightNode.reductionDegree))
            else:
                if point[node.sepAtt] <= node.sepVal:
                    if self.forest.mode == "crisp":
                        return self.pathLength(point, node.leftNode,e+1,deg)
                    else:
                        return self.pathLength(point, node.leftNode,e+1,deg + degs[0])
                else:
                    if self.forest.mode == "crisp":
                        return self.pathLength(point, node.rightNode,e+1,deg)
                    else:
                        return self.pathLength(point, node.rightNode,e+1,deg+degs[1])
        else:
            # (If it is an external node)
            if self.forest.mode == "crisp":
                return e #/ c(len(self.dataSet.trainingData))
            else:
                return deg #/ e
        

class Node:

    def __init__(self,id,d,a,v,l,r,m,M,sd=0):
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
        m : float
            min value amont points in the node for attribute a
        M : float
            max value amont points in the node for attribute a
        sd : float
            reduction degree depending on the number of points in the parent node and the number of points in the node
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
        self.closestP = m
        self.closestM = M
        self.reductionDegree = sd
    
        
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

if __name__ == "__main__":
#    d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
 #   d = Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0)
 #   d = Dataset("../Data/diabetes.csv",["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CLASS"], {0: lambda s: int(s.strip() or 0),1: lambda s: int(s.strip() or 0),2: lambda s: int(s.strip() or 0),3: lambda s: int(s.strip() or 0),4: lambda s: int(s.strip() or 0),5: lambda s: float(s.strip() or 0),6: lambda s: float(s.strip() or 0),7: lambda s: int(s.strip()) or 0, 8: lambda s: int(s.strip() or -1)},True,0)
    d = Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0) 
    e = d.getEvalDataSet()

#    f = Forest(d, "strongfuzzy",0.84,0.05)
    f = Forest(d)

    f.build()
    
    import viewer as vw
    fig, ax = plt.subplots(2)
    A=0.5
    print("CRISP METHOD WITH PARAMETERS ALPHA:",A)
    f.setMode("crisp")
    f.setAlpha(A)
    f.setBeta(0)
    scores = f.computeScores(0)
    pC,rC,fmC, eR = f.evaluate(scores)
    msg = "CRISP PREC:"+str(pC)+" RAP:"+str(rC)+" FM:"+str(fmC)+" ER:"+str(eR)
    print(msg)
    vw.viewIsolatedDatasetWithAnomalies(d,f.trees[0],scores,f.ALPHA,e,ax[0],msg)

    print("\n")
    A=0.84
    B=0.01
    print("FUZZY METHOD WITH PARAMETERS ALPHA:",A, "BETA:",B)
    f.setMode("strongfuzzy")
    f.setAlpha(A)
    f.setBeta(B)
    scores = f.computeScores(0)
    pSF,rSF,fmSF,eRSF = f.evaluate(scores)
    msg="FUZZY PREC:"+str(pSF)+" RAP:"+str(rSF)+" FM:"+str(fmSF)+" ER:"+str(eRSF)
    print(msg)
    vw.viewIsolatedDatasetWithAnomalies(d,f.trees[0],scores,f.ALPHA,e,ax[1],msg)
    plt.show()
 