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
    plt.show()

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
    plt.show()


def viewIsolatedDatasetWithAnomalies(d:Dataset,tree:FTree,scores:np.ndarray,ALPHA:float, filters: np.ndarray =None, subplt = None,msg:str=None):
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
    pp = plt
    if subplt is not None:
        pp = subplt
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
    color='b'
    if de is None:
        de = 0
    if de > 0.9:
        color= 'k'
    else:
        if de > 0.7:
            color= 'r'
        else:
            if de > 0.5:
                color= 'm'
            else:
                if de > 0.3:
                    color= 'y'
    return color
                

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

    if nd is not None and nd.sepAtt is not None:
        if nd.sepAtt == 0:
            mini = math.inf
            maxi = (-math.inf)
            for id in nd.ids:
                if d.data[id][1] > maxi:
                    maxi = d.data[id][1]
                if d.data[id][1] < mini:
                    mini = d.data[id][1]
                pp.plot([nd.sepVal, nd.sepVal], [mini, maxi],color=getColor4Degree(max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree)),alpha=max(nd.rightNode.reductionDegree,nd.leftNode.reductionDegree))
        else:
            mini = math.inf
            maxi = (-math.inf)
            for id in nd.ids:
                if d.data[id][0] > maxi:
                    maxi = d.data[id][0]
                if d.data[id][0] < mini:
                    mini = d.data[id][0]

            pp.plot([mini,maxi], [nd.sepVal,nd.sepVal],color=getColor4Degree(max(nd.leftNode.reductionDegree,nd.rightNode.reductionDegree)),alpha=max(nd.rightNode.reductionDegree,nd.leftNode.reductionDegree))
        drawTreeRec(nd.leftNode,d,pp)
        drawTreeRec(nd.rightNode,d,pp)
    

if __name__ == "__main__":
    from dataset import Dataset
    d = Dataset("../Data/DonutL.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
    #d = Dataset("../Data/data8S.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.8)
    #d = Dataset("../Data/DataGauss.csv",["x","y","CLASS"], {0: lambda s: float(s.strip() or 0),1: lambda s: float(s.strip() or 0),2: lambda s: int(s.strip() or 0)},True,0.7)

    viewDataset(d)