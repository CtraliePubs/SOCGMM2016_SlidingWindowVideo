#!/usr/bin/env python

from    cvxopt          import spmatrix, matrix
from    cvxopt.blas     import copy
from    lsqr            import lsqr
from    sys             import argv, exit
import  os.path

def smooth(boundary_list, cocycle_list):
    dimension = max((max(d[1], d[2]) for d in boundary_list))
    dimension += 1

    # NB: D is a coboundary matrix; 1 and 2 below are transposed
    D = spmatrix([d[0] for d in boundary_list],
                 [d[2] for d in boundary_list],
                 [d[1] for d in boundary_list], (dimension, dimension))

           
    z = spmatrix([zz[0] for zz in cocycle_list],
                 [zz[1] for zz in cocycle_list],
                 [0     for zz in cocycle_list], (dimension, 1))

    v1 = D * z
    # print "D^2 is zero:", not bool(D*D)
    # print "D*z is zero:", not bool(v1)
    z = matrix(z)

    def Dfun(x,y,trans = 'N'):
        if trans == 'N':
            copy(D * x, y)
        elif trans == 'T':
            copy(D.T * x, y)
        else:
            assert False, "Unexpected trans parameter"

    tol = 1e-10
    show = False
    maxit = None
    solution = lsqr(Dfun, matrix(z), show = show, atol = tol, btol = tol, itnlim = maxit)
    
    v = z - D*solution[0]

    # print sum(v**2)
    # assert sum((D*v)**2) < tol and sum((D.T*v)**2) < tol, "Expected a harmonic cocycle"
    if not (sum((D*v)**2) < tol and sum((D.T*v)**2) < tol):
        print "Expected a harmonic cocycle:", sum((D*v)**2), sum((D.T*v)**2) 

    return solution[0], v


def vertex_values(solution, vertices):
    values = [None]*len(vertices)
    for i,v in vertices:
        values[v] = solution[i]
    return values
    

def read_list_file(filename):
    list = []
    with open(filename) as fp:
        for line in fp.xreadlines():
            if line.startswith('#'): continue
            list.append(map(int, line.split()))
    return list


if __name__ == '__main__':
    if len(argv) < 4:
        print "Usage: %s BOUNDARY COCYCLE VERTEXMAP" % argv[0]
        exit()

    boundary_filename = argv[1]
    cocycle_filename = argv[2]
    vertexmap_filename = argv[3]

    boundary_list = read_list_file(boundary_filename)
    cocycle_list = read_list_file(cocycle_filename)
    vertexmap_list = read_list_file(vertexmap_filename)

    solution, v = smooth(boundary_list, cocycle_list)
    values = vertex_values(solution, vertexmap_list)

    outfn = os.path.splitext(cocycle_filename)[0] + '.val'
    with open(outfn, 'w') as f:
        for i,v in enumerate(values):
            f.write('%d %f\n' % (i,v))
