import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

if __name__ == '__main__':
    startframe = 6
    endframe = 30
    skip = 2
    idxs = range(startframe, endframe+skip, skip)
    
    movex = 0.3
    movey = 0.25
    
    f0 = scipy.misc.imread("frames/%i.png"%startframe)
    [H, W] = [f0.shape[0], f0.shape[1]]
    outH = int(np.ceil(H + H*movey*(len(idxs)-1)))
    outW = int(np.ceil(W + W*movex*(len(idxs)-1)))
    I = 255*np.ones((outH, outW, 3))
    for i in range(len(idxs)):
        idx = idxs[i]
        f = scipy.misc.imread("frames/%i.png"%idx)
        for c in range(3):
            f[:, :, c] = np.fliplr(f[:, :, c])
        startx = i*movex*W
        starty = i*movey*H
        I[starty:starty+H, startx:startx+W, :] = f
    for c in range(3):
        I[:, :, c] = np.fliplr(I[:, :, c])
    scipy.misc.imsave("VideoStack.png", I)
