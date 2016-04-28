#Code to sample points from a polygon mesh
import sys
sys.path.append("../")
sys.path.append("../S3DGLPy")
from SlidingWindow1D import *
from sklearn.decomposition import PCA 
import numpy as np
import scipy
from Primitives3D import *
from PolyMesh import *
from MeshCanvas import *
import matplotlib.pyplot as plt

#########################################################
##                UTILITY FUNCTIONS                    ##
#########################################################

class PCAGLCanvas(BasicMeshCanvas):
    def __init__(self, parent, Y, C, angles, stds, prefix):
        BasicMeshCanvas.__init__(self, parent)
        self.Y = Y #Geometry
        self.YMean = np.mean(Y, 0)
        self.C = C #Colors
        
        self.YBuf = vbo.VBO(np.array(self.Y, dtype=np.float32))
        self.CBuf = vbo.VBO(np.array(self.C, dtype=np.float32))
        
        self.angles = angles #Angles to rotate the camera through as the trajectory is going
        self.stds = stds
        self.prefix = prefix
        self.frameNum = 0
        #Initialize sphere mesh
        self.bbox = BBox3D()
        self.bbox.fromPoints(Y)
        self.camera.centerOnBBox(self.bbox, theta = angles[0, 0], phi = angles[0, 1]) #theta = -math.pi/2, phi = math.pi/2)
        self.Refresh()
    
    def setupPerspectiveMatrix(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(180.0*self.camera.yfov/M_PI, float(self.size.x)/self.size.y, 0.001, 100)
    
    def repaint(self):
        self.setupPerspectiveMatrix()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        self.YBuf.bind()
        glVertexPointerf(self.YBuf)
        self.CBuf.bind()
        glColorPointerf(self.CBuf)
        
        #Rotate camera
        self.camera.theta = self.angles[self.frameNum, 0]
        self.camera.phi = self.angles[self.frameNum, 1]
        self.camera.updateVecsFromPolar()

        #Set up modelview matrix
        self.camera.gotoCameraFrame()

        glDrawArrays(GL_POINTS, 0, self.Y.shape[0])
        
        #First principal component
        if self.frameNum > 200 and self.frameNum < 360:
            glLineWidth(10.0)
        else:
            glLineWidth(2.0)
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(self.YMean[0], self.YMean[1], self.YMean[2])
        glVertex3f(self.YMean[0], self.YMean[1]-self.stds[1]*2, self.YMean[1])
        glEnd()
        
        if self.frameNum >= 360 and self.frameNum < 470:
            glLineWidth(10.0)
        else:
            glLineWidth(2.0)
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(self.YMean[0], self.YMean[1], self.YMean[2])
        glVertex3f(self.YMean[0]-self.stds[0]*3, self.YMean[1], self.YMean[1])
        glEnd()
        
        glLineWidth(3.0)
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(self.YMean[0], self.YMean[1], self.YMean[2])
        glVertex3f(self.YMean[0], self.YMean[1], self.YMean[1]+self.stds[2]*3)
        glEnd()
        
        self.CBuf.unbind()
        self.YBuf.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        saveImageGL(self, "%s%i.png"%(self.prefix, self.frameNum))
        if self.frameNum < self.Y.shape[0] - 1:
            self.frameNum += 1
            self.Refresh()
        else:
            self.parent.Destroy()
        
        self.SwapBuffers()

def doPCAGLPlot(Y, C, angles, stds, prefix):
    app = wx.PySimpleApp()
    frame = wx.Frame(None, wx.ID_ANY, "PCA GL Canvas", DEFAULT_POS, (800, 800))
    g = PCAGLCanvas(frame, Y, C, angles, stds, prefix)
    frame.canvas = g
    frame.Show()
    app.MainLoop()
    app.Destroy()

if __name__ == '__main__':  
    NRandSamples = 10000 #You can tweak this number
    np.random.seed(100) #For repeatable results randomly sampling
    
    m = PolyMesh()
    m.loadFile("sword2.off")
    
    (Y, Ns) = m.randomlySamplePoints(NRandSamples)
    Y = Y.T
    Y = Y - np.mean(Y, 0)[None, :]
    C = np.array(Y)
    C = C - np.min(C, 0)[None, :]
    C = C/np.max(C, 0)[None, :]
    

    pca = PCA()
    Y = pca.fit_transform(Y)
    Y = Y/np.max(np.abs(Y))
    
    plt.scatter(Y[:, 0], Y[:, 1], 20, C, edgecolors='none')
    plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('First Principal Axis')
    plt.xlabel('Second Principal Axis')
    plt.title('2D PCA of Sword Point Cloud')
    plt.savefig("SwordPCA.png", dpi=300, bbox_inches='tight')
    
    Y = Y[:, [1, 0, 2]]
    stds = np.std(Y, 0)
    
    #Output rotation video
    NFrames = 500
    angles = np.pi/1.5*np.ones((NFrames, 2))
    angles[:, 0] = np.linspace(0, 2*np.pi, NFrames)
    angles[:, 1] = np.linspace(np.pi/1.5, np.pi/1.3, NFrames)
    doPCAGLPlot(Y, C, angles, stds, "Points")


