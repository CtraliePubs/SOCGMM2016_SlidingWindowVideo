import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.signal
import subprocess

import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from VideoTools import *
from PCAGL import *

def getTimeDerivative(I, Win):
    dw = np.floor(Win/2)
    t = np.arange(-dw, dw+1)
    sigma = 0.4*dw
    xgaussf = t*np.exp(-t**2  / (2*sigma**2))
    #Normalize by L1 norm to control for length of window
    xgaussf = xgaussf/np.sum(np.abs(xgaussf)) 
    xgaussf = xgaussf[None, :]
    IRet = scipy.signal.convolve2d(I, xgaussf, 'valid')
    validIdx = np.arange(dw, I.shape[1]-dw, dtype='int64')
    return [IRet, validIdx]

if __name__ == '__main__':
    WinLen = 34
    
    (Vid, IDims) = loadCVVideo('myneck.ogg')
    #Vid = Vid[:, 0:200]
    #(Vid, idx) = getTimeDerivative(Vid, 7)
    (data_subspace, s_I) = tde_tosubspace(Vid, Vid.shape[1])
    s_I_mean = tde_mean(s_I,WinLen)
    (Y, S) = tde_rightsvd(s_I,WinLen,s_I_mean)
#    plt.imshow(Y[:, 0:20], interpolation = 'none', aspect = 'auto')
#    plt.show()
    #Get rid of first 4 principal components because they're drift
    S = S[4:]
    Y = Y[:, 4:]
    Y = S[None, :]*Y
    Y = Y[:, 0:3]
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    angles = math.pi/2*np.ones((Y.shape[0], 2))
    angles[:, 0] = np.linspace(0, np.pi/4, Y.shape[0])
    angles[:, 1] = np.linspace(np.pi/2 - np.pi/4, np.pi/2, Y.shape[0])
    
    print "Variance Explained: ", 100*np.sum(S[0:3])/np.sum(S), "%"
    
    Y = Y/np.max(np.abs(Y))
    sio.savemat("Y.mat", {"Y":Y})
    
    #Step 1: Ouput 3D PCA
    doPCAGLPlot(Y, C, angles, "Points")
    
    
    N = Vid.shape[1]
    M = N-WinLen+1
    skip = 2
    
    movex = 0.3
    movey = 0.25
    
    [H, W] = [IDims[0], IDims[1]]

    #Step 2: Output 3D PCA synchronized with original video
    subprocess.call(["avconv", "-i", "myneck.ogg", "-f" , "image2", "frames%00d.png"]) 
    plt.figure(figsize=(12, 6))
    for starti in range(M):
        plt.clf()
        f = scipy.misc.imread("frames%i.png"%(starti+1))
        plt.subplot(121)
        plt.imshow(f)
        #plt.imshow(I/255.0)
        plt.axis('off')
        plt.subplot(122)
        P = scipy.misc.imread("Points%i.png"%starti)
        plt.imshow(P)
        plt.axis('off')
        plt.savefig("Original%i.png"%starti, dpi=150, bbox_inches = 'tight')
    subprocess.call(["avconv", "-r", "25", "-i" , "Original%d.png", "-r", "25", "-b", "30000k", "Original.ogg"]) 

    #Step 3: Output 3D PCA synchronized with amplified video
    subprocess.call(["avconv", "-i", "myneck-FIRWindowBP-band0.50-3.00-sr25-alpha30-mp0-sigma3-scale0.67-frames1-400-octave.avi", "-f" , "image2", "frames%00d.png"]) 
    plt.figure(figsize=(12, 6))
    for starti in range(M):
        plt.clf()
        f = scipy.misc.imread("frames%i.png"%(starti+1))
        plt.subplot(121)
        plt.imshow(f)
        #plt.imshow(I/255.0)
        plt.axis('off')
        plt.subplot(122)
        P = scipy.misc.imread("Points%i.png"%starti)
        plt.imshow(P)
        plt.axis('off')
        plt.savefig("Amplified%i.png"%starti, dpi=150, bbox_inches = 'tight')
    subprocess.call(["avconv", "-r", "25", "-i" , "Amplified%d.png", "-r", "25", "-b", "30000k", "Amplified.ogg"])
