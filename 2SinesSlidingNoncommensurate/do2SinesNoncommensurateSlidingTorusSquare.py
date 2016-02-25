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
    phi1 = t
    phi2 = np.pi*t
    x = np.cos(phi1) + np.cos(phi2)
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
    
    
    #Combine Sliding Window with torus in plots
    h = 1.3*np.max(np.abs(x))
    M = N-W+1
    #cm = plt.get_cmap('cool')
    cm = plt.get_cmap('jet')
    C = [cm(int(np.round(255.0*i/M))) for i in range(M)]
    C = np.array(C)
    plt.figure(figsize=(18, 9))
    phi1 = np.mod(180*phi1/np.pi, 360.0)
    phi2 = np.mod(180*phi2/np.pi, 360.0)
    for i in range(M):
        plt.clf()
        plt.subplot(121)
        plt.plot(x, 'k')
        plt.hold(True)
        plt.scatter([i], [x[i]], 50, C[i, :])
        plt.axis('off')
        for k in range(i+1):
            xs = [k, k+1]
            ys = x[k:k+2]
            plt.plot(xs, ys, color=C[k, :])
        
        plt.subplot(122)
        plt.hold(True)
        for k in range(i):
            xs = np.array(phi1[k:k+2])
            ys = np.array(phi2[k:k+2])
            if np.abs(xs[1]-xs[0]) < 180 and np.abs(ys[1]-ys[0]) < 180:
                plt.plot(xs, ys, color=C[k, :])
            else:
                #Split line segment into two parts
                if np.abs(xs[1]-xs[0]) >= 180:
                    xs[1] += 360
                if np.abs(ys[1]-ys[0]) >= 180:
                    ys[1] += 360
                m = (ys[1]-ys[0])/(xs[1]-xs[0])
                minv = 1/m
                if xs[1] > 360:
                    plt.plot([xs[0], 360], [ys[0], ys[0]+m*(360-xs[0])], color = C[k, :])
                    plt.plot([0, xs[1]-360], [ys[0]+m*(360-xs[0]), ys[1]], color = C[k, :])
                if ys[1] > 360:
                    plt.plot([xs[0], xs[0]+minv*(360-ys[0])], [ys[0], 360], color = C[k, :])
                    plt.plot([xs[0]+minv*(360-ys[0]), xs[1]], [0, ys[1]-360], color = C[k, :])
        plt.scatter([phi1[i]], [phi2[i]], 50, C[i, :])
        plt.xlim([0, 360])
        plt.ylim([0, 360])
        plt.xlabel('Phase 1')
        plt.ylabel('Phase 2')
        plt.savefig("Torus%i.png"%i, bbox_inches='tight', dpi=100)
        if i == 0:
            plt.savefig("Torus0.svg", bbox_inches='tight')
    subprocess.call(["avconv", "-r", "60", "-i" , "Torus%d.png", "-r", "60", "-b", "30000k", "Torus.ogg"]) 
