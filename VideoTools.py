#Programmer: Chris Tralie (with help from Matt Berger at US ARFL loading videos
#and doing subspace projections)
#Purpose: Tools for efficiently manipulating sliding windows of videos
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import time
import random
import os
import subprocess
import matplotlib
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import cv2
import sys

AVCONV_BIN = 'avconv'
TEMP_STR = "pymeshtempprefix"

#############################################################
####                  VIDEO I/O TOOLS                   #####
#############################################################

def loadCVVideo(path, pyr_level=0, show_video=False):
    #(Matt Berger wrote most of this function)
    if not os.path.exists(path):
        print "ERROR: Video path not found: %s"%path
        return None
    video_reader = cv2.VideoCapture(path)
    num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fdx = 0
    all_frames = None
    pyramid = []
    while video_reader.isOpened():
        validity,frame = video_reader.read()
        if frame == None:
            break

        #frame = cv2.pyrDown(frame)
        IDims = frame.shape
        pyr_frame = frame.copy()
        pyramid = pyr_frame.flatten()/255.0
        pyr_scale = 1.0
        for p in range(pyr_level):
            pyr_frame = cv2.pyrDown(pyr_frame)
            blurred_frame = cv2.pyrUp(cv2.pyrDown(pyr_frame))
            laplace_frame = pyr_frame - blurred_frame[:pyr_frame.shape[0],:pyr_frame.shape[1],:]
            #laplace_frame = cv2.subtract(pyr_frame,cv2.pyrUp(next_pyr_frame))
            pyr_scale*=4.0
            pyramid = np.concatenate((pyramid,(pyr_frame.flatten())/255.0))
            #pyramid = np.concatenate((pyramid,(laplace_frame.flatten())/255.0))

        # store frame
        #flattened_frame = frame.flatten()
        if all_frames == None:
            all_frames = np.zeros((len(pyramid),num_frames))
        #all_frames[:,fdx] = flattened_frame/255.0
        all_frames[:,fdx] = pyramid
        fdx=fdx+1

        # optionally show it as we load it
        if show_video:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xff == ord('q'):
                #break

    video_reader.release()
    if show_video:
        cv2.destroyAllWindows()

    return ((all_frames), IDims)

def saveCVVideo(V, IDims, path, show_video=True, FrameRate = 30.0, Normalize = False):
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), FrameRate, (IDims[1],IDims[0]), True if IDims[2]==3 else False)
    if Normalize:
        V = V-np.min(V)
        V = V/np.max(V)
    for fidx in range(V.shape[1]):
        proper_frame = np.reshape(V[:,fidx], IDims)
        proper_frame = np.minimum(proper_frame,np.ones(proper_frame.shape))
        proper_frame = np.maximum(proper_frame,np.zeros(proper_frame.shape))
        proper_frame = (255*proper_frame).astype(np.uint8)
        video_writer.write(proper_frame)
        # optionally show it as we save it
        if show_video:
            cv2.imshow('frame', proper_frame)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xff == ord('q'):
                #break

    video_writer.release()
    if show_video:
        cv2.destroyAllWindows()

#W: Window length
def savePCsColor(PCs, W, prefix, FrameRate = 30):
    NPCs = PCs.shape[1]/W
    for i in range(NPCs):
        print "Outputting principal component %i of %i"%(i, NPCs)
        saveCVVideo(PCs[:, i*W:(i+1)*W], IDims, "%s%i.avi"%(prefix, i), FrameRate = FrameRate, Normalize = True)


#############################################################
####           TIME DELAY EMBEDDING TOOLS               #####
#############################################################
#Input: I: P x N Video with frames along the columns
#W: Windows
#Ouput: Mu: P x W video with mean frames along the columns
def tde_mean(I, W):
    IOut = np.array(I)
    IOut[IOut > 1] = 1
    IOut[IOut < 0] = 0
    start_time = time.time()
    N = I.shape[1]
    P = I.shape[0]
    Mu = np.zeros((P, W))
    for i in range(W):
        Mu[:, i] = np.mean(I[:, np.arange(N-W+1) + i], 1)
    end_time = time.time()
    #print "tde_mean elapsed time ", end_time-start_time, " seconds, I.shape = ", I.shape, ", W = ", W
    return Mu

#Frames assumed to be in each column
#Stacked frames are also in one column
#The delay frames are in a matrix I call "ID" which is never explicitly
#stored
#Return a tuple of (right hand singular vectors, singular values)
def tde_rightsvd(I, W, Mu):
    start_time = time.time()
    N = I.shape[1] #Number of frames in the video
    
    ## Step 1: Precompute frame and mean correlations
    B = I.T.dot(I);
    MuFlat = Mu.flatten()
    MuFlat = np.reshape(MuFlat, [len(MuFlat), 1])
    MuTMu = MuFlat.T.dot(MuFlat)
    C = Mu.T.dot(I) #A WxN matrix
    
    ## Step 2: Use precomputed information to compute (ID-Mu)^T*(ID-Mu)
    #Compute the ID^TID part
    ND = N-W+1
    IDTID = np.zeros((ND, ND))
    #Use the fact that a delay embedding is just a moving average along
    #all diagonals
    for i in range(N-W+1):
        b = np.diag(B, i)
        b2 = np.cumsum(b)
        bend = b2[W-1:]
        bbegin = np.zeros(len(bend))
        bbegin[1:] = b2[0:len(bend)-1]
        b2 = bend - bbegin
        IDTID[np.arange(len(b2)), i + np.arange(len(b2))] = b2
    IDTID = IDTID + IDTID.T
    np.fill_diagonal(IDTID, 0.5*np.diag(IDTID)) #Main diagonal was counted twice
    
    #Compute the Mu^TID part to subtract off mean
    MuTID = np.zeros((1, ND))
    for i in range(ND):
        MuTID[0, i] = np.sum(np.diag(C, i))
    ATA = IDTID - MuTID
    ATA = ATA - MuTID.T
    ATA = ATA + MuTMu
    #Handle numerical precision issues and keep it symmetric
    ATA = 0.5*(ATA + ATA.T)
    
    ## Step 3: Compute right singular vectors
    [S, Y] = linalg.eigh(ATA)
    idx = np.argsort(-S)
    S[S < 0] = 0 #Numerical precision
    S = np.sqrt(S[idx])
    Y = Y[:, idx]
    end_time = time.time()
    return (Y, S)

#Unpack the principal components
#I: Video frames PxN (each frame of P dimensions in each column)
#W: Sliding Window size
#Mu: Mean sliding window (PxW)
#Y: Right singular vectors (in columns)
#NPCs: The number of principal components to return
#Return a P x (W*NPCs) matrix of principal components
def tde_getPCs(I, W, Mu, Y, NPCs):
    N = I.shape[1]
    P = I.shape[0]
    M = N-W+1
    IRet = np.zeros((P, W*NPCs))
    YSub = Y[:, 0:NPCs]
    for k in range(W):
        idx = np.arange(k, k+M)
        #Fill in the kth frame of each principal component
        IRet[:, k + W*np.arange(NPCs)] = (I[:, idx] - np.reshape(Mu[:, k], [Mu.shape[0], 1])).dot(YSub)
    return IRet

#I: Video frames PxN (each frame of P dimensions in each column)
#A: Dimensionality of subspace
#Returns: Tuple: (subspace, subspace coordinates)
def tde_tosubspace(I, A, lam = 0):
    #(Matt Berger wrote part of this function)
    N = I.shape[1]
    P = I.shape[0]
    s_dim = A

    I_mean = (1.0/N)*I.dot(np.ones(N))
    I_centered = I - np.asarray([I_mean]*N).T
    # first, compute subspace for I, and corresponding subspace representations
    print 'forming input covariance matrix: ',
    start_time = time.time()
    input_covariance = I_centered.T.dot(I_centered)
    end_time = time.time()
    print end_time-start_time, " seconds"
    
    print 'computing data subspace: ',
    start_time = time.time()
    [data_eigenvalues, data_eigenvectors] = linalg.eigh(input_covariance)
    print 'input eigenvalues:',data_eigenvalues
    end_time = time.time()
    print end_time-start_time, " seconds"

    max_s_dim = N
    for idx,eigenvalue in enumerate(data_eigenvalues):
        if eigenvalue > 1e-12:
            break
        max_s_dim=max_s_dim-1
    if max_s_dim < A:
        s_dim = max_s_dim
        print 'subspace of dimensionality:',s_dim,'NOT',A
    data_subspace = I_centered.dot(data_eigenvectors[0:N,-s_dim:])
    data_subspace = data_subspace/np.sqrt(np.sum(data_subspace**2, 0))
    #print 'data eigenvalues:',data_eigenvalues

    if lam > 0:
        import prox_tv as ptv
        #Perform total variation denoising on the subspace columns
        #TODO: This is no longer an orthogonal subspace
        for i in range(data_subspace.shape[1]-A, data_subspace.shape[1]):
            print "Doing i = %i"%i
            v = np.reshape(data_subspace[:, i], IDims)
            vnew = np.zeros(v.shape)
            for k in range(3):
                vnew[:, :, k] = ptv.tv1_2d(v[:, :, k], lam)
            
            plt.subplot(121)
            v = np.reshape(data_subspace[:, i], IDims)
            plt.imshow(v/np.max(v))
            plt.title("Component %i"%i)
            plt.subplot(122)
            plt.imshow(vnew/np.max(vnew))
            plt.savefig("TV%i.png"%i, dpi=200)
            
            data_subspace[:, i] = vnew.flatten()

    # project data
    print 'projecting data...'
    s_I = data_subspace.T.dot(I_centered)
    return (data_subspace, s_I)    


#Output principal components of a video
if __name__ == '__main__':
    (I, IDims) = loadCVVideo('BeatingHeartSynthetic/heartcrop.avi')
    NPCs = 20
    W = 30
    (data_subspace, s_I) = tde_tosubspace(I, I.shape[1])
    # perform amplification of the reduced representation
    s_I_mean = tde_mean(s_I,W)
    (s_I_PCs,s_eigenvalues) = tde_rightsvd(s_I,W,s_I_mean)
    PCs = tde_getPCs(s_I, W, s_I_mean, s_I_PCs, NPCs)
    PCs = data_subspace.dot(PCs)
    savePCsColor(PCs, W, "BeatingHeartCrop")
