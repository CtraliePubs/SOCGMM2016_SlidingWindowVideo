#Purpose: 
import sys
sys.path.append("../")
from VideoTools import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    (I, IDims) = loadCVVideo('jumpingjackscropped.avi')
    Ws = [1, 34]
    (data_subspace, s_I) = tde_tosubspace(I, I.shape[1])
    for W in Ws:
        # perform amplification of the reduced representation
        s_I_mean = tde_mean(s_I,W)
        (Y, S) = tde_rightsvd(s_I,W,s_I_mean)
        Y = S.flatten()[None, :]*Y
        ax = plt.subplot(131)
        ax.imshow(Y[:, 0:10], interpolation='none', aspect='auto', cmap=plt.get_cmap('seismic'))
        ax.set_xlabel('RHSV Number')
        ax.set_ylabel('Frame Number')
        ax.set_title('Right Hand Singular Vectors')
        YSum = np.sum(Y**2, 1)
        #Compute SSM
        D = YSum[:, None] + YSum[None, :] - 2*(Y).dot(Y.T)
        D[D < 0] = 0
        D = np.sqrt(D)
        ax = plt.subplot(132)
        ax.plot(Y[:, 0], Y[:, 1], '.')
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_title('2D PCA')
        ax = plt.subplot(133)
        ax.imshow(D, cmap=plt.get_cmap('afmhot'))
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Frame Number')
        ax.set_title('Self-Similarity Matrix')
        plt.show()
