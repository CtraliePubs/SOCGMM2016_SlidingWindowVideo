import sys
sys.path.append("S3DGLPy")
import numpy as np
import matplotlib.pyplot as plt
from PolyMesh import *
from Cameras3D import *
from MeshCanvas import *

SPHERE_RADIUS = 0.05

class PCAGLCanvas(BasicMeshCanvas):
    def __init__(self, parent, Y, C, angles, prefix):
        BasicMeshCanvas.__init__(self, parent)
        self.Y = Y #Geometry
        self.C = C #Colors
        self.angles = angles #Angles to rotate the camera through as the trajectory is going
        self.prefix = prefix
        self.frameNum = 0
        #Initialize sphere mesh
        self.mesh = getSphereMesh(SPHERE_RADIUS, 3)
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
        
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        
        #Rotate camera
        self.camera.theta = self.angles[self.frameNum, 0]
        self.camera.phi = self.angles[self.frameNum, 1]
        self.camera.updateVecsFromPolar()
        
        for i in range(self.frameNum):
            #Set up modelview matrix
            self.camera.gotoCameraFrame()
            
            #Update position and color buffers for this point
            self.mesh.VPos = self.mesh.VPos + self.Y[i, :]
            self.mesh.VColors = self.C[i, :]*np.ones(self.mesh.VPos.shape)
            self.mesh.needsDisplayUpdate = True
            glLightfv(GL_LIGHT0, GL_POSITION, np.array([0, 0, 0, 1]))
            self.mesh.renderGL(False, False, True, False, False, True, False)
            
            #Undo the translation
            self.mesh.VPos = self.mesh.VPos - self.Y[i, :]
        saveImageGL(self, "%s%i.png"%(self.prefix, self.frameNum))
        if self.frameNum < self.Y.shape[0] - 1:
            self.frameNum += 1
            self.Refresh()
        else:
            self.parent.Destroy()
        
        self.SwapBuffers()

def doPCAGLPlot(Y, C, angles, prefix):
    app = wx.PySimpleApp()
    frame = wx.Frame(None, wx.ID_ANY, "PCA GL Canvas", DEFAULT_POS, (800, 800))
    g = PCAGLCanvas(frame, Y, C, angles, prefix)
    frame.canvas = g
    frame.Show()
    app.MainLoop()
    app.Destroy()

#Test with a helix
if __name__ == '__main__':
    N = 200
    t = np.linspace(0, 6*np.pi, N)
    Y = np.zeros((N, 3))
    Y[:, 0] = np.cos(t)
    Y[:, 1] = np.sin(t)
    Y[:, 2] = t/4
    
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, N)), dtype=np.int64))
    C = C[:, 0:3]
    
    angles = math.pi/2*np.ones((N, 2))
    angles[:, 0] = np.linspace(0, np.pi, N)
    
    doPCAGLPlot(Y, C, angles, "Helix")
