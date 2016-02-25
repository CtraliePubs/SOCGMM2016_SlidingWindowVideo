#Purpose: To show what the eigen modes are for a perfectly periodic video
#Output the principal component videos (dynamic range scaled) for the supplemental video
#For the paper, output an XT slice, showing where that XT slice is
import sys
sys.path.append("../")
from VideoTools import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__2':
    (I, IDims) = loadCVVideo('jumpingjackscropped.avi')
    W = 30
    IMean = np.mean(I, 1)[:, None]
    Alpha = np.ones((1, 8))
    Alpha[0:6] = 0
    IAmp = subspace_tde_amplification(I, IDims, W, np.ones((1, 20)), I.shape[1])
    print "np.min(IAmp) = ", np.min(IAmp)
    print "np.max(IAmp) = ", np.max(IAmp)
    saveCVVideo(IAmp + IMean, IDims, 'First2PCs.avi', FrameRate = 25.0)

def plotXTSlice(V, IDims, startx, endx, y, ax):
    im = np.zeros((V.shape[1], endx-startx, 3))
    V = V - np.min(V)
    V = V/np.max(V)
    for k in range(V.shape[1]):
        ITemp = np.reshape(V[:, k], IDims)
        im[k, :, :] = ITemp[y, startx:endx, :]
    ax.imshow(im[:, :, [2, 1, 0]], interpolation = 'none', aspect = 'auto')
    ax.get_xaxis().set_visible(False)
    ax.set_yticks([])
    #ax.get_yaxis().set_visible(False)

if __name__ == '__main__':
    (I, IDims) = loadCVVideo('jumpingjackscropped.avi')
    NPCs = 8
    W = 34
    (data_subspace, s_I) = tde_tosubspace(I, I.shape[1])
    # perform amplification of the reduced representation
    s_I_mean = tde_mean(s_I,W)
    (s_I_PCs,s_eigenvalues) = tde_rightsvd(s_I,W,s_I_mean)
    PCs = tde_getPCs(s_I, W, s_I_mean, s_I_PCs, NPCs)
    NPCs = PCs.shape[1]/W
    PCs = PCs + np.repeat(s_I_mean, PCs.shape[1]/W, axis=1) #Add back mean for visualization
    meanI = np.mean(I, 1)[:, None]
    PCs = data_subspace.dot(PCs) #+ meanI
#    plt.imshow(np.reshape(meanI, IDims))
#    plt.show()
    
    #Make principal component videos for supplementary video
    prefix = "PC"
    for i in range(NPCs):
        print "Outputting principal component %i of %i"%(i, NPCs)
        saveCVVideo(PCs[:, i*W:(i+1)*W], IDims, "%s%i.avi"%(prefix, i), FrameRate = W, Normalize = True)
    
    #Extract an XT slice each principal component for the figure in the paper
    startx = 0#93
    endx = IDims[1]#184
    y = 350
    
    #Pull-up
#    startx = 33
#    endx = 354
#    y = 160
    #Plot the X slice location on the left of the plot
    im = np.array(np.reshape(I[:, 0], IDims))[:, :, [2, 1, 0]]
    im[y, startx:endx, :] = 1
    ax = plt.subplot2grid((NPCs/2+1, 3), (0, 0), rowspan = NPCs+1)
    ax.imshow(im)
    ax.axis('off')
    
    #Plot XT slice for the first W frames of the original video
    ax = plt.subplot2grid((NPCs/2+1, 3), (0, 1))
    plotXTSlice(I[:, 0:W], IDims, startx, endx, y, ax)
    ax.set_ylabel("Orig", rotation = 0, labelpad = 20)
    
    #Plot the principal component XT slices stacked on the right
    for i in range(NPCs):
        V = np.array(PCs[:, i*W:(i+1)*W])
        ax = plt.subplot2grid((NPCs/2+1, 3), (i/2+1, 1 + i%2))
        plotXTSlice(V, IDims, startx, endx, y, ax)
        ax.set_ylabel("PC %i"%(i+1), rotation = 0, labelpad = 20)
    plt.show()
