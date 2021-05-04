#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:08:30 2020
Contains functions to view datasets and forests
@author: SHAMAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import *
from FIF import *
import matplotlib.patches as patches

def displayAUC(lr_fpr,lr_tpr,msg,pl=plt):
    pl.plot(lr_fpr, lr_tpr, marker=',', label=msg)

def displayCuts(mins,maxs,means,stds,pl=plt):
    """
    Display
    """
    # create stacked errorbars:
 

    pl.errorbar(np.arange(len(mins)), means, stds, fmt='ok', lw=3)
    diffsMin=means
    diffsMax=maxs
    for i in range(len(means)):
        diffsMin[i] = diffsMin[i] - mins[i]
        diffsMax[i] = diffsMax[i] - means[i]
 
    #pl.errorbar(np.arange(len(mins)), means, [diffsMin, diffsMax], fmt='.k', ecolor='gray', lw=1)
   # plt.xlim(-1, len(mins)+2)
    

def viewDataset(d:Dataset):
    """
    Used to plot 2D datasets
    Parameters:
    -----------
    d : Dataset
        an instance of the Dataset class
    """
    x=list(d.data[:,0])
    y=list(d.data[:,1])
        
    plt.scatter(x,y)
  

def viewData(d:np.ndarray, p = None):
    """
    Used to plot 2D datasets
    Parameters:
    -----------
    d : np.ndarray
        an array of data
    """
    pp = plt if p is None else p
    x=list(d[:,0])
    y=list(d[:,1])
    p.scatter(x,y)

def viewDatasetWithAnomalies(d:Dataset,scores:np.ndarray,ALPHA:float, filters : np.ndarray =None):
    """
    Used to plot 2D datasets
    scores is a 1D array of anomaly scores
    Parameters:
    -----------
    d : Dataset
        an instance of the Dataset class
    scores : np.ndarray
        an array containing the anomaly scores
    ALPHA : float
        the anomaly threshold
    filters : np.ndarray
        the ids of the points to display (None means all points are displayed)
    """
    for pi in range(len(d.data)):
        if filters is None or pi in filters:
            if ALPHA != 0 and scores[pi] >= ALPHA:
                plt.plot([d.data[pi][0]],[d.data[pi][1]],'ro')
            else:
                plt.plot([d.data[pi][0]],[d.data[pi][1]],'bo',alpha=scores[pi])
    


def viewIsolatedDatasetWithAnomalies(d:Dataset,tree:FTree,scores:np.ndarray,ALPHA:float, filters: np.ndarray =None, pp = plt,msg:str=None):
    """
    Used to plot 2D datasets
    scores is a 1D array of anomaly scores
    Parameters:
    -----------
    d : Dataset
        an instance of the Dataset class
    tree : FTree
        the isolation tree to draw
    scores : np.ndarray
        an array containing the anomaly scores
    ALPHA : float
        the anomaly threshold
    filters : np.ndarray
        the ids of the points to display (None means all points are displayed)
    """
  
    for pi in range(len(d.data)):
        if filters is None or pi in filters:
            if ALPHA != 0 and scores[pi] >= ALPHA:
                pp.plot([d.data[pi][0]],[d.data[pi][1]],'ro')
            else:
                pp.plot([d.data[pi][0]],[d.data[pi][1]],'bo')
    if msg is not None:
        pp.set_title(msg)
    drawTreeRec(tree.root,d,pp)
    

def getColor4Degree(de:float):
    colors=['black','black','dimgray','dimgray','gray','gray','darkgray','darkgray','silver','silver']
   
    if de is None:
        de = 0
    i = 0
    while i < len(colors) -1 and 1 - i * 0.1 > de :
        i=i+1
    print
    return colors[i]

def drawTreeRec(nd:Node,d : Dataset, pl = None):
    """
    Draw a tree for a 2D dataset
    
    Parameters:
    -----------
    nd : Node
        The root node
    d : Dataset
        the dataset
    """
    pp = plt
    if pl is not None:
        pp= pl
    #  pp.add_patch(Rectangle((nd.sepVal, miniY), maxiY-miniY , maxiX-miniX))

    if nd is not None and nd.sepAtt is not None:
        miniX = math.inf
        maxiX = (-math.inf)
        miniY = math.inf
        maxiY = (-math.inf)
        for id in nd.ids:
            if d.data[id][1] > maxiY:
                maxiY = d.data[id][1]
            if d.data[id][1] < miniY:
                miniY = d.data[id][1]
            if d.data[id][0] > maxiX:
                maxiX = d.data[id][0]
            if d.data[id][0] < miniX:
                miniX = d.data[id][0]

        if nd.sepAtt == 0:
             #   pp.add_patch(patches.Rectangle((nd.sepVal, miniY) , maxiX-miniX, maxiY-miniY,edgecolor='r',color=getColor4Degree(nd.anomalyAreaDegree)))
                pp.plot([nd.sepVal, nd.sepVal], [miniY, maxiY],color='black',alpha=max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree ))#,color=getColor4Degree(max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree)) )#,alpha=max(nd.rightNode.reductionDegree,nd.leftNode.reductionDegree))
        else:
           # pp.add_patch(patches.Rectangle((nd.sepVal, miniY), maxiY-miniY , maxiX-miniX,edgecolor='r',color=getColor4Degree(nd.anomalyAreaDegree)))
            pp.plot([miniX,maxiX], [nd.sepVal,nd.sepVal],color='black',alpha=max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree ))#,color=getColor4Degree(max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree)) )#,alpha=max(nd.rightNode.reductionDegree,nd.leftNode.reductionDegree))
        drawTreeRec(nd.leftNode,d,pp)
        drawTreeRec(nd.rightNode,d,pp)
    

if __name__ == "__main__":
    from dataset import Dataset
    d = Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
    #d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
    #d = Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.7)

    viewDataset(d)