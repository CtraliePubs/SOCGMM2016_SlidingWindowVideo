import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import scipy.io as sio
from sklearn.decomposition import PCA

def doPCA(XOrig, ncomponents = 3):
    X = XOrig - np.mean(XOrig, 0)
    D = (X.T).dot(X)
    D = 0.5*(D + D.T)
    (lam, eigvecs) = np.linalg.eig(D)
    print lam
    lam = np.abs(lam)
    varExplained = lam[0:ncomponents]
    PCs = eigvecs[:, 0:ncomponents]
    Y = X.dot(PCs)
    return (PCs, Y, varExplained)

def getSlidingWindowEmbedding(x, W):
    N = len(x)
    M = N-W+1
    y = np.zeros((M, W))
    for i in range(W):
        y[:, i] = x[i:i+M]
    return y

def output1DSliding(x, y, N, W, lims):
    h = 1.3*np.max(np.abs(x))
    M = N-W+1
    #cm = plt.get_cmap('cool')
    cm = plt.get_cmap('jet')
    C = [cm(int(np.round(255.0*i/M))) for i in range(M)]
    C = np.array(C)
    plt.figure(figsize=(18, 9))
    for i in range(M):
        plt.clf()
        plt.subplot(121)
        plt.plot(x, 'k')
        plt.hold(True)
        t = np.arange(N)
        t = t[i:i+W+1]
        plt.plot(t, x[t], color=C[i, :])
        plt.plot([i, i], [-h, h], color=C[i, :])
        plt.plot([i+W, i+W], [-h, h], color=C[i, :])
        plt.axis('off')
        ax2 = plt.subplot(122)
        ax2.scatter(y[0:i, 0], y[0:i, 1], 50, C, linewidths=0)
        ax2.set_aspect('equal')
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        plt.axis('off')
        plt.savefig("%i.png"%i, bbox_inches='tight', dpi=100)   

def output1DSliding_SignalAlone(x, N, W, prefix):
    h = 1.3*np.max(np.abs(x))
    M = N-W+1
    #cm = plt.get_cmap('cool')
    cm = plt.get_cmap('jet')
    C = [cm(int(np.round(255.0*i/M))) for i in range(M)]
    C = np.array(C)
    plt.figure(figsize=(9, 9))
    for i in range(M):
        plt.clf()
        plt.plot(x, 'k')
        plt.hold(True)
        t = np.arange(N)
        t = t[i:i+W+1]
        plt.plot(t, x[t], color=C[i, :])
        plt.plot([i, i], [-h, h], color=C[i, :])
        plt.plot([i+W, i+W], [-h, h], color=C[i, :])
        plt.axis('off')
        plt.savefig("%s%i.png"%(prefix, i), bbox_inches='tight', dpi=100)   
