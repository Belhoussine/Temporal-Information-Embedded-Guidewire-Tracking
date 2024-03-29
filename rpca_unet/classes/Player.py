# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:43:10 2018

@author: Yi Zhang
@reference: Deep Unfolded Robust PCA With Application to Clutter Suppression in Ultrasound. https://github.com/KrakenLeaf/CORONA

"""

import numpy as np
import matplotlib.pyplot as plt
import torch

minDBdf=-50
class Player:
    def __init__(self,Tpause=None):
        if Tpause==None:
            self.Tpause=0.1
        else:
            self.Tpause=Tpause
        self.fig=None
        self.ax={1:None,2:None,3:None,4:None,5:None,6:None}
        self.axnum=0
    
    def plotmat(self,mvlist,note=None,tit=None,supt=None,
                cmap='gray',ion=True,minDB=None):
        """
        input:matrix dimension
        """
        if minDB is None:
            minDB=minDBdf
        
        if ion:
            plt.ion()
        subp={1:[1,1],2:[1,2],3:[1,3],4:[2,2],5:[2,3],6:[2,3],9:[3,3],12:[4,3]}
        p1,p2=subp[len(mvlist)]
        if p1*p2!=self.axnum or self.fig is None\
           or not(plt.fignum_exists(self.fig.number)):
            self.fig,(self.ax)=plt.subplots(p1,p2) 
            self.axnum=p1*p2
            if self.axnum==1:
                self.ax=np.array([self.ax])   
            self.ax=self.ax.reshape([-1])
                    
        for i in range(len(mvlist)):            
            US=mvlist[i]
            if US is None:
                continue
            if US.dtype is torch.float32:
                US=US.detach().numpy()
            US=np.abs(US).reshape([-1,mvlist[i].shape[-1]])
            if np.sum(np.abs(US))!=0:
                US=US/np.max(US)
            if note=='db':
                US[US<10**(minDB/20)]=10**(minDB/20)
                US=20*np.log10(US)
            vmin,vmax=[minDB,0] if note=='db' else [0,1]
            self.ax[i].clear()
            self.ax[i].imshow(US,cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
            if not(tit is None):
                self.ax[i].set_title(tit[i])
        if not(supt is None):
            self.fig.suptitle(supt)
        if ion:
            plt.pause(self.Tpause)
        return self.fig
        
    def play(self,mvlist,note=None,tit=None,supt=None,cmap='gray',minDB=None):
        """
        input:movie dimension
        """
        if minDB is None:
            minDB=minDBdf
        
        subp={1:[1,1],2:[1,2],3:[1,3],4:[2,2],5:[2,3],6:[2,3],9:[3,3]}
        p1,p2=subp[len(mvlist)]
        T=mvlist[0].shape[-1]
        fig,ax=plt.subplots(p1,p2) 
        if p1*p2==1:
            ax=np.array([ax]) 
        ax=ax.reshape([-1])
        
        plt.ion()
        
        for i in range(len(mvlist)):
            US=mvlist[i]
            if US is None:
                continue
            if US.dtype is torch.float32:
                US=US.detach().numpy().squeeze()
            
            US=np.abs(US)
            if np.sum(np.abs(US))!=0:
                US=US/np.max(US)
            if note=='db':
                US[US<10**(minDB/20)]=10**(minDB/20)
                US=20*np.log10(US)
            mvlist[i]=US
        
        for t in range(T):
            for i in range(len(mvlist)):  
                if mvlist[i] is None:
                    continue
                vmin,vmax=[minDB,0] if note=='db' else [0,1]
                ax[i].clear()
                ax[i].imshow(mvlist[i][:,:,t],cmap=cmap,aspect='auto',
                             vmin=vmin,vmax=vmax)
                if not(tit is None):
                    ax[i].set_title(tit[i])
            if supt==None:
                supt=''
            fig.suptitle('%dth Frame,'%(t+1)+supt)
            plt.pause(self.Tpause)
        