#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:15:37 2021

@author: veronne
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt



class Displayer:
    
    def __init__(self, precC, precF, rapC, rapF, fmC, fmF, eC, eF, timeC, timeF, aucC, aucF):
        
        
        x     = [0,10,20,30,40,50,60,70,80,90,100]
        fig, ax = plt.subplots(6)
        
        #display precision
        ax[0].plot(x,[0]+precC,'b:',linewidth=2.0,label="crisp")
        ax[0].plot(x,[0]+precF,'r',linewidth=2.0,label="fuzzy")
        ax[0].set_title("PREC (FUZZY is plain red, CRISP is dashed blue")
        
        ax[1].plot(x,[0]+rapC,'b:',linewidth=2.0,label="crisp")
        ax[1].plot(x,[0]+rapF,'r-',linewidth=2.0,label="fuzzy")
        ax[1].set_title("RAP (FUZZY is plain red, CRISP is dashed blue)")
        
        ax[2].plot(x,[0]+fmC,'b:',linewidth=2.0,label="crisp")
        ax[2].plot(x,[0]+fmF,'r-',linewidth=2.0,label="fuzzy")
        ax[2].set_title("FM (FUZZY is plain red, CRISPis dashed blue)")
        
        ax[3].plot(x,[0]+eC,'b:',linewidth=2.0,label="crisp")
        ax[3].plot(x,[0]+eF,'r-',linewidth=2.0,label="fuzzy")
        ax[3].set_title("ER (FUZZY is plain red, CRISP is dashed blue)")
        
        ax[4].plot(x,[0]+aucC,'b:',linewidth=2.0,label="crisp")
        ax[4].plot(x,[0]+aucF,'r-',linewidth=2.0,label="fuzzy")
        ax[4].set_title("AUC(FUZZY is plain red, CRISP is dashed blue)")
        
        ax[5].plot(x,np.concatenate([np.array([0]),np.log(timeC)]),'b:',linewidth=2.0,label="crisp")
        ax[5].plot(x,np.concatenate([np.array([0]),np.log(timeF)]),'r-',linewidth=2.0,label="fuzzy")
        #ax[5].plot(x,np.concatenate([np.array([0]),timeC]),'b:',linewidth=2.0,label="crisp")
        #ax[5].plot(x,np.concatenate([np.array([0]),timeF]),'r-',linewidth=2.0,label="fuzzy")
        ax[5].set_title("TIME (FUZZY is plain red, CRISP is dashed blue) log")
        
        self.figure = plt
        
