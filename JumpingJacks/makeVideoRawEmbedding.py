import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from VideoTools import *
from PCAGL import *

if __name__ == '__main__':
    WinLen = 1
    
    (Vid, IDims) = loadCVVideo('jumpingjackscropped.avi')
    (data_subspace, s_I) = tde_tosubspace(Vid, Vid.shape[1])
    s_I_mean = tde_mean(s_I,WinLen)
    (Y, S) = tde_rightsvd(s_I,WinLen,s_I_mean)
    Y = S[None, :]*Y
    Y = Y/np.max(np.abs(Y))
    sio.savemat("YRaw.mat", {"Y":Y})
    np.savetxt("YRaw.txt", Y, fmt='%g', delimiter=' ', newline='\n')
    Y = Y[:, 0:3]
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    angles = math.pi/2*np.ones((Y.shape[0], 2))
    angles[:, 0] = np.linspace(0, np.pi/2, Y.shape[0])
    
    fout = open("VarExplainedRaw.txt", "w")
    fout.write("Variance Explained: %g Percent"%(100*np.sum(S[0:3])/np.sum(S)))
    fout.close()
    

    plt.scatter(Y[:, 0], Y[:, 1], 20, C)
    plt.show()
    
    doPCAGLPlot(Y, C, angles, "PointsRaw")
    
    
    N = Vid.shape[1]
    M = N-WinLen+1

    plt.figure(figsize=(12, 6))
    for starti in range(M):
        plt.clf()
        f = np.reshape(Vid[:, starti], IDims)
        f = f[:, :, [2, 1, 0]] #CV stores as BGR
        f = f*255
        #scipy.misc.imsave("VideoStack%i.png"%starti, I)
        plt.subplot(121)
        plt.imshow(f/255.0)
        plt.axis('off')
        plt.subplot(122)
        P = scipy.misc.imread("PointsRaw%i.png"%starti)
        plt.imshow(P)
        plt.axis('off')
        plt.savefig("Raw%i.png"%starti, dpi=150, bbox_inches = 'tight')
