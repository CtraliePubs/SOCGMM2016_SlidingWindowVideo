import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from VideoTools import *
from PCAGL import *

if __name__ == '__main__':
    WinLen = 34
    
    (Vid, IDims) = loadCVVideo('jumpingjackscropped.avi')
    (data_subspace, s_I) = tde_tosubspace(Vid, Vid.shape[1])
    s_I_mean = tde_mean(s_I,WinLen)
    (Y, S) = tde_rightsvd(s_I,WinLen,s_I_mean)
    Y = S[None, :]*Y
    Y = Y[:, 0:3]
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    angles = math.pi/2*np.ones((Y.shape[0], 2))
    angles[:, 0] = np.linspace(0, np.pi/2, Y.shape[0])
    
    print "Variance Explained: ", 100*np.sum(S[0:3])/np.sum(S), "%"
    
    Y = Y/np.max(np.abs(Y))
    plt.scatter(Y[:, 0], Y[:, 1], 20, C)
    plt.show()
    
    doPCAGLPlot(Y, C, angles, "Points")
    
    
    N = Vid.shape[1]
    M = N-WinLen+1
    skip = 2
    
    movex = 0.3
    movey = 0.25
    
    [H, W] = [IDims[0], IDims[1]]

    plt.figure(figsize=(12, 6))
    for starti in range(M):
        plt.clf()
        idxs = range(starti, starti+WinLen, skip)
        print "len(idxs) = ", len(idxs)
        outH = int(np.ceil(H + H*movey*(len(idxs)-1)))
        outW = int(np.ceil(W + W*movex*(len(idxs)-1)))
        I = 255*np.ones((outH, outW, 3))
        for i in range(len(idxs)):
            print "i = ", i
            idx = idxs[i]
            f = np.reshape(Vid[:, idx], IDims)
            f = f[:, :, [2, 1, 0]] #CV stores as BGR
            f = f*255
            for c in range(3):
                f[:, :, c] = np.fliplr(f[:, :, c])
            startx = int(np.floor(i*movex*W))
            starty = int(np.floor(i*movey*H))
            I[starty:starty+H, startx:startx+W, :] = f
        for c in range(3):
            I[:, :, c] = np.fliplr(I[:, :, c])
        #scipy.misc.imsave("VideoStack%i.png"%starti, I)
        plt.subplot(121)
        plt.imshow(I/255.0)
        plt.axis('off')
        plt.subplot(122)
        P = scipy.misc.imread("Points%i.png"%starti)
        plt.imshow(P)
        plt.axis('off')
        plt.savefig("%i.png"%starti, dpi=150, bbox_inches = 'tight')
