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
    cm = plt.get_cmap('cool')
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
        ax2 = plt.subplot(122)
        ax2.scatter(y[0:i, 0], y[0:i, 1], 50, C, linewidths=0)
        ax2.set_aspect('equal')
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        plt.savefig("%i.png"%i, bbox_inches='tight', dpi=100)    

def do1SineSliding():
    T = 60
    N = 600
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t)
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    pca = PCA(n_components=2)
    y = pca.fit_transform(y)
#    for i in range(PCs.shape[1]):
#        plt.plot(PCs[:, i])
#        plt.show()
    output1DSliding(x, y, N, W, (-6, 6))
    subprocess.call(["rm", "1sine.ogg"])
    subprocess.call(["avconv", "-r", "60", "-i", "%d.png", "-r", "60", "-b", "30000k", "1sine.ogg"])
    for i in range(M):
        os.remove("%i.png"%i)
    

def do2SinesSliding():
    T = 60
    N = 600
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t) + np.cos(3*t)
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    sio.savemat("2Sines.mat", {"X":y})
    pca = PCA(n_components=2)
    y = pca.fit_transform(y)
    output1DSliding(x, y, N, W, (-11, 11))
    subprocess.call(["rm", "2sines.ogg"])
    subprocess.call(["avconv", "-r", "60", "-i", "%d.png", "-r", "60", "-b", "30000k", "2sines.ogg"])
    for i in range(M):
        os.remove("%i.png"%i)

def do2SinesNoncommensurate():
    T = 60
    N = 600
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t) + np.cos(np.pi*t)
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    sio.savemat("2SinesNC.mat", {"X":y})
    y = y - np.mean(y, 0)
    (PCs, y, _) = doPCA(y, 2)
    output1DSliding(x, y, N, W, (-11, 11))
    subprocess.call(["rm", "2sinesNC.ogg"])
    subprocess.call(["avconv", "-r", "60", "-i", "%d.png", "-r", "60", "-b", "30000k", "2sinesNC.ogg"])
    for i in range(M):
        os.remove("%i.png"%i)

def do3SinesSlidingNoncommensurate():
    T = 30
    N = 1200
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t) + np.cos(np.pi/2*t) + np.cos(np.exp(1)*t)
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    sio.savemat("3Sines.mat", {"X":y})
    y = y - np.mean(y, 0)
    (PCs, y, _) = doPCA(y, 2)
    output1DSliding(x, y, N, W, (-10, 10))
    subprocess.call(["rm", "3sines.ogg"])
    subprocess.call(["avconv", "-r", "60", "-i", "%d.png", "-r", "60", "-b", "30000k", "3sines.ogg"])
    for i in range(M):
        os.remove("%i.png"%i)

if __name__ == '__main__':
    do1SineSliding()
    do2SinesSliding()
    do2SinesNoncommensurate()
    #do3SinesSlidingNoncommensurate()
