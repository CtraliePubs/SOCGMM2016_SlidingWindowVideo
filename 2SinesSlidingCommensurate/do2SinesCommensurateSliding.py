import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from SlidingWindow1D import *
from PCAGL import *
from sklearn.decomposition import PCA 
import numpy as np
import scipy

if __name__ == '__main__':
    T = 60
    N = 600
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t) + np.cos(3*t)
    plt.plot(x)
    plt.title('cos(t) + cos(3t)')
    plt.axis('off')
    plt.show()
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    sio.savemat("2Sines.mat", {"X":y})
    pca = PCA()
    Y = pca.fit_transform(y)
    Y = Y/np.max(np.abs(Y))
    sio.savemat("Y.mat", {"Y":Y})
    np.savetxt("Y.txt", Y, fmt='%g', delimiter=' ', newline='\n')
    Y = Y[:, 0:3]
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    
    #Step 1: Ouput 3D PCA
    angles = np.pi/2*np.ones((Y.shape[0], 2))
    angles[:, 0] = np.linspace(0, np.pi/4, Y.shape[0])
    angles[:, 1] = np.linspace(np.pi/2 - np.pi/4, np.pi/2, Y.shape[0])
    doPCAGLPlot(Y, C, angles, "Points")
    
    #Step 2: Output sliding window
    output1DSliding_SignalAlone(x, N, W, "Signal")
    
    
    #Step 3: Combine Sliding Window with 3D PCA in one set of plots
    plt.figure(figsize=(12, 6))
    for starti in range(M):
        plt.clf()
        f = scipy.misc.imread("Signal%i.png"%starti)
        plt.subplot(121)
        plt.imshow(f)
        plt.axis('off')
        plt.subplot(122)
        P = scipy.misc.imread("Points%i.png"%starti)
        plt.imshow(P)
        plt.axis('off')
        plt.savefig("Frames%i.png"%starti, dpi=150, bbox_inches = 'tight')
    subprocess.call(["avconv", "-r", "60", "-i" , "Frames%d.png", "-r", "60", "-b", "30000k", "Frames.ogg"]) 
