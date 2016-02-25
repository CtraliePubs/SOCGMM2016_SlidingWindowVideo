import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from VideoTools import *
from PCAGL import *

if __name__ == '__main__':
    (Vid, IDims) = loadCVVideo('jumpingjackscropped.avi')
    N = Vid.shape[1]
    loc = [70, 323]
    
    vals = np.zeros((N, 3))
    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.clf()
        f = np.reshape(Vid[:, i], IDims)
        f = f[:, :, [2, 1, 0]] #CV stores as BGR
        f = f*255
        thisvals = np.zeros((100, 3))
        idx = 0
        for ii in range(-5, 5):
            for jj in range(-5, 5):
                thisvals[idx, :] = np.array(f[loc[0]+ii, loc[1]+jj, :])
                f[loc[0]+ii, loc[1]+jj, :] = [0, 255.0, 0]
                idx += 1
        thisvals = np.mean(thisvals, 0)
        vals[i, :] = thisvals.flatten()
        #scipy.misc.imsave("VideoStack%i.png"%i, I)
        plt.subplot(121)
        plt.imshow(f/255.0)
        plt.axis('off')
        plt.subplot(122)
        plt.hold(True)
        plt.plot(vals[:, 0], 'r')
        plt.plot(vals[:, 1], 'g')
        plt.plot(vals[:, 2], 'b')
        plt.ylim([0, 255])
        plt.xlabel('Frame Number')
        plt.ylabel('RGB Value')
        plt.savefig("Pixel%i.png"%i, dpi=150, bbox_inches = 'tight')
