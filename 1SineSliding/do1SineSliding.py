import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from SlidingWindow1D import *
from sklearn.decomposition import PCA 
from PCAGL import *

if __name__ == '__main__':
    T = 60
    N = 600
    NPeriods = N/T
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    x = np.cos(t)
    plt.plot(x)
    plt.title('cos(t)')
    plt.axis('off')
    plt.show()
    W = T
    M = N-W+1
    y = getSlidingWindowEmbedding(x, W)
    pca = PCA(n_components=2)
    y = pca.fit_transform(y)
    output1DSliding(x, y, N, W, (-6, 6))
    subprocess.call(["rm", "1sine.ogg"])
    subprocess.call(["avconv", "-r", "60", "-i", "%d.png", "-r", "60", "-b", "30000k", "1sine.ogg"])
