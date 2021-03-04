# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

x     = [0,10,20,30,40,50,60,70,80,90,100]

precC = [0,0.7,0.8,0.82,0.97,0.97,0.8,0.79,0.97,0.97,0.94]
precF = [0]+[1.0]*10
rapC  = [0]+[1.0]*10
rapF  = [0]+[1.0]*10
fmC   = [0,0.8,0.89,0.89,0.98,0.98,0.88,0.87,0.98,0.98,0.96]
fmF   = [0]+[1.0]*10
eC    = [0,0.05,0.02,0.02,0.004,0.04,0.02,0.03,0.04,0.04,0.08]
eF    = [0]*11

#computation time 
crisp = [0,3,5,8,11,14,18,20,24,16,30]
fuzzy = [0,18,35,52,71,90,110,130,150,166,187]

fig, ax = plt.subplots(5)

#ax = plt.figure().gca()

#display precision
ax[0].plot(x,precC,'b:',linewidth=2.0,label="crisp")
ax[0].plot(x,precF,'r',linewidth=2.0,label="fuzzy")
ax[0].set_title("PREC (CRISP is plain, FUZZY is dashe")

ax[1].plot(x,rapC,'b:',linewidth=2.0,label="crisp")
ax[1].plot(x,rapF,'r-',linewidth=2.0,label="fuzzy")
ax[1].set_title("RAP (CRISP is plain, FUZZY is dashed)")

ax[2].plot(x,fmC,'b:',linewidth=2.0,label="crisp")
ax[2].plot(x,fmF,'r-',linewidth=2.0,label="fuzzy")
ax[2].set_title("FM (CRISP is plain, FUZZY is dashed)")

ax[3].plot(x,eC,'b:',linewidth=2.0,label="crisp")
ax[3].plot(x,eF,'r-',linewidth=2.0,label="fuzzy")
ax[3].set_title("ER (CRISP is plain, FUZZY is dashed)")

ax[4].plot(x,np.log(crisp),'b:',linewidth=2.0,label="crisp")
ax[4].plot(x,np.log(fuzzy),'r-',linewidth=2.0,label="fuzzy")
ax[4].set_title("TIME (CRISP is plain, FUZZY is dashed) log")


#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True)
plt.show()