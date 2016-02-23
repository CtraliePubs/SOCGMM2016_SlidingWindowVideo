import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import sys
sys.path.append("../")
from VideoTools import *

if __name__ == '__main__':
    (Vid, IDims) = loadCVVideo('jumpingjackscropped.avi')
    (data_subspace, s_I) = tde_tosubspace(Vid, Vid.shape[1])
    
    WinLen = 34
    N = Vid.shape[1]
    M = N-WinLen+1
    skip = 2
    
    movex = 0.3
    movey = 0.25
    
    [H, W] = [IDims[0], IDims[1]]

    for starti in range(M):
        idxs = range(starti, starti+WinLen, skip)
        outH = int(np.ceil(H + H*movey*(len(idxs)-1)))
        outW = int(np.ceil(W + W*movex*(len(idxs)-1)))
        I = 255*np.ones((outH, outW, 3))
        for i in range(len(idxs)):
            idx = idxs[i]
            f = np.reshape(Vid[:, idx], IDims)
            f = f[:, :, [2, 1, 0]] #CV stores as BGR
            f = f*255
            for c in range(3):
                f[:, :, c] = np.fliplr(f[:, :, c])
            startx = i*movex*W
            starty = i*movey*H
            I[starty:starty+H, startx:startx+W, :] = f
        for c in range(3):
            I[:, :, c] = np.fliplr(I[:, :, c])
        #scipy.misc.imsave("VideoStack%i.png"%starti, I)
        plt.subplot(121)
        plt.imshow(I/255.0)
        plt.sublpot(122)
        
