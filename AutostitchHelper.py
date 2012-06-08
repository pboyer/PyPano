import imageIO
from imageIO import *
import numpy
from numpy import *
import random
import StitchTools
from StitchTools import *

def imageFrom1Channel(a):
    out=empty([a.shape[0], a.shape[1], 3])
    for i in xrange(3):
        out[:, :,i]=a
    return out

def writeGrey(L, path='test-'):
    out = constantIm(L.shape[0], L.shape[1], 0)
    for i in xrange(3):
        out[:, :, i]=L
    imwriteSeq(out, path)


def loadListOfImages(path):
    out=[]
    i=1
    while True:
        try: 
            im=imread(path+str(i)+'.png')
            out.append(im)
            i+=1
        except IOError:
            if i==1: print 'problem, could not load:'+path+str(i)+'.png'
            break
    return out


def drawPoint(im, P, color, size=1):
    im[P[0]-size:P[0]+size+1, P[1]-size:P[1]+size+1]=color

def visualizeCorners(im, cornerList):
    out=im*0.2
    for c in cornerList:
        drawPoint(out, 1, 0)
    imwrite(out, 'corners-')

def concatenate(im1, im2):
    out=zeros([max(im1.shape[0], im2.shape[0]), im1.shape[1] + im2.shape[1], 3])
    out[0:im1.shape[0], 0:im1.shape[1], :]=im1
    out[0:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1], :]=im2
    return out
    
    
def drawLine(out, P1, P2, color=1):
    drawPoint(out, P1, 1, 1)
    drawPoint(out, P2, 1, 1)
    
    #if P1[0]>P2[0]: P1, P2=P2*1.0, P1*1.0
    
    for y in xrange(int(P1[0])+1, int(P2[0])):
        t=(y-P1[0])*1.0/(P2[0]-P1[0])
        x=P1[1]*(1-t)+t*P2[1]
        out[y, x]=color
    if P1[1]>P2[1]: P1, P2=P2*1.0, P1*1.0
    for x in xrange(int(P1[1])+1, int(P2[1])):
        t=(x-P1[1])*1.0/(P2[1]-P1[1])
        y=P1[0]*(1-t)+t*P2[0]
        out[y, x]=color

def visualizePairs(im1, im2, pairs):
    out=concatenate(im1, im2)*0.2
    offset=array([0, im1.shape[1]])
    for p in pairs:
        drawLine(out, p[0], p[1]+offset, 1)
    imwriteSeq(out, 'pairs-')
    
def visualizePairsWithInliers(im1, im2, pairs, isInlier):
    out=concatenate(im1, im2)*0.2
    out1=im1*0.1
    out2=im2*0.1

    offset=array([0, im1.shape[1]])
    for p, isIn in zip(pairs, isInlier):
        if isIn: drawLine(out, p[0], p[1]+offset, array([0, 1, 0]))
        else: drawLine(out, p[0], p[1]+offset,  array([1, 0, 0]))
        c1, c2=p[0], p[1]
        color=numpy.random.rand(3)**2
        if not isIn:
            color=array([1, 0, 0])
            drawPoint(out1, c1, color, 1)
            drawPoint(out2, c2, color, 1)
        else:
            if color[1]+color[2]<0.4: color [0]=0
            drawPoint(out1, c1, color, 4)
            drawPoint(out2, c2, color, 4)

    imwriteSeq(out, 'pairs-with-inliers-')
    imwriteSeq(out1, 'myinliers-')
    imwriteSeq(out2, 'myinliers-')

def visualizeInliers(im1, im2, nn, isInlier):
    out1=im1*0.1
    out2=im2*0.1
    for p, inl in zip(nn, isInlier):
        c1, c2=p[0], p[1]
        color=numpy.random.rand(3)**2
        if not inl:
            color=array([0.2, 0, 0])
            drawPoint(out1, c1, color, 0)
            drawPoint(out2, c2, color, 0)
        else:
            color [0]*=0.0
            drawPoint(out1, c1, color, 4)
            drawPoint(out2, c2, color, 4)

    imwriteSeq(out1, 'inliers-')
    imwriteSeq(out2, 'inliers-')
     

def visualizeCorners(im, cL):
    out=im*0.2
    for c in cL: drawPoint(out, c, array([1, 1, 1]), 0)
    imwrite(out, 'corners-')

def visualizeFeatures(LF, radiusDescriptor, im ):
    out=im*0.1
    for f in LF:
        y, x=f[0][0], f[0][1]
        out[y-radiusDescriptor:y+1+radiusDescriptor, x-radiusDescriptor:x+1+radiusDescriptor, 1]=0.5*f[1]
        out[y-radiusDescriptor:y+1+radiusDescriptor, x-radiusDescriptor:x+1+radiusDescriptor, 0]=-0.5*f[1]
    imwriteSeq(out, 'features-')

def homogenize(P3D):
    if (P3D[2]!=0):
        return array([P3D[0]/P3D[2], P3D[1]/P3D[2]])
    else: return array([0, 0])

def applyH3d(H, P):
    return homogenize(dot(H, P))

def applyH2d(H, y, x):
    return homogenize(dot(H, array([y, x, 1.0])))

def applyHP2d(H, P):
    return homogenize(dot(H, array([P[0], P[1], 1.0])))

def visualizeReprojection(im1, im2, pairs, isInlier, H):
    out1=im1*0.2
    out2=im2*0.2
    green=array([0, 1, 0])
    red=array([1, 0, 0])
    yellow=array([1, 1, 0])
    blue=array([0, 0, 1])
    for p, isIn in zip(pairs, isInlier):
        p12=applyHP2d(linalg.inv(H), p[0])
        p21=applyHP2d(H, p[1])
        if isIn:
            drawPoint(out1, p[0], green, 2)
            drawPoint(out1, p21, red, 1)
            drawPoint(out2, p[1], green, 2)
            drawPoint(out2, p12, red, 1)
        else:
            drawPoint(out1, p[0], yellow, 2)
            drawPoint(out1, p21, blue, 1)
            drawPoint(out2, p[1], yellow, 2)
            drawPoint(out2, p12, blue, 1)
    imwriteSeq(out1, 'reproject-')
    imwriteSeq(out2, 'reproject-')
