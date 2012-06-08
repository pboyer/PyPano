import numpy
from numpy import *
import sys
from imageIO import *
import time
from scipy import *
from a7help import *

def imIter(im):
	for y in xrange(im.shape[0]):
        	for x in xrange(im.shape[1]): 
			yield y, x

def getBlackPadded(im, y, x, isWeightMap = False):
    if (x<0) or (x>=im.shape[1]) or (y<0) or (y>= im.shape[0]): 
        if isWeightMap:
            return numpy.array([0])
        else: 
            return numpy.array([0, 0, 0])

    else: return im[y, x]

def outside(im, y, x):
    return (x<0) or (x>=im.shape[1]) or (y<0) or (y>= im.shape[0])

def width(im):
    return im.shape[1]

def height(im):
    return im.shape[0]
    
def clipX(im, x): 
	return min(width(im)-1, max(x, 0))

def clipY(im, y):
	return min(height(im)-1, max(y, 0))

def getSafePix(im, y, x):
	return im[clipY(im, y), clipX(im, x)]

def scaleNN(im, k):
    out = constantIm(im.shape[0]*k, im.shape[1]*k, 0.0)
    for y, x in imIter(out):
        out[y,x]=im[y/k, x/k]
    return out

def interpolateLin(im, y, x, isWeightMap = False):
    yt = math.ceil(y)
    yb = math.floor(y)

    xt = math.ceil(x)
    xb = math.floor (x)

    tl = getBlackPadded(im, yb, xb, isWeightMap)
    tr = getBlackPadded(im, yb, xt, isWeightMap)
    bl = getBlackPadded(im, yt, xb, isWeightMap)
    br = getBlackPadded(im, yt, xt, isWeightMap)

    xint = x - xb
    yint = y - yb
    
    # interpolate between bl&tl and br and tr
    # obtaining two end values
    xvalintl = yint * bl + (1-yint) * tl
    xvalintr = yint * br + (1-yint) * tr

    valint = xint * xvalintr + (1-xint) * xvalintl

    return valint

#bilinear interpolation on coordinates y, x
def interpolateLinSafe(im, y, x):
    yt = math.ceil(y)
    yb = math.floor(y)

    xt = math.ceil(x)
    xb = math.floor (x)

    tl = getSafePix(im, yb, xb)
    tr = getSafePix(im, yb, xt)
    bl = getSafePix(im, yt, xb)
    br = getSafePix(im, yt, xt)

    xint = x - xb
    yint = y - yb
    
    # interpolate between bl&tl and br and tr
    # obtaining two end values
    xvalintl = yint * bl + (1-yint) * tl
    xvalintr = yint * br + (1-yint) * tr

    valint = xint * xvalintr + (1-xint) * xvalintl

    return valint

def applyhomography(source, out, H, bilinear=True):

    out2 = out

    for y, x in imIter(out):

        # get the source pixel that maps to this pixel
        pos = H.dot( array( [ [y], [x], [1] ] ) )
        pos = pos / pos[2]
        
        if bilinear:
            if not outside(source, pos[0], pos[1]):
                out2[y,x] = interpolateLinSafe( source, pos[0], pos[1] )  
        else:
            if not outside( source, int( round( pos[0] ) ), int( round( pos[1] ) ) ):
                out2[y,x] = source[ int( round(pos[0])), int( round(pos[1]) ) ]

    return out2

def computehomography(listOfPairs):
    
    # listOfPairs[pair index][image][coordinate]
    # create A

    numPairs = len(listOfPairs)
    A = zeros( (numPairs * 2, 9) )
    
    for pairIndex in range(numPairs):
        # add two indices for each pair to A
        y1 = listOfPairs[pairIndex][0][0]
        x1 = listOfPairs[pairIndex][0][1]
    
        y1p = listOfPairs[pairIndex][1][0]
        x1p = listOfPairs[pairIndex][1][1]

        A[2*pairIndex + 0, :] = array( [y1, x1, 1, 0, 0, 0, -y1*y1p, -x1*y1p, -y1p] )
        A[2*pairIndex + 1, :] = array( [0, 0, 0, y1, x1, 1, -y1*x1p, -x1*x1p, -x1p] )          

    U, s, V = linalg.svd(A)
    x = V[-1,:]

    #this provides the solution homography
    out = x.reshape(3,3)
    return out

def computeTransformedBBox(im, H):
    
    tl = array( [0, 0, 1] )
    bl = array( [ height(im)-1, 0, 1] )
    tr = array( [0, width(im) - 1, 1] )
    br = array( [ height(im) - 1, width(im) - 1, 1] )

    Hinv = linalg.inv(H)

    tlH = Hinv.dot(tl)
    blH = Hinv.dot(bl)
    trH = Hinv.dot(tr)
    brH = Hinv.dot(br)
    
    yps = array( [ tlH[0] / tlH[2], blH[0] / blH[2], trH[0] / trH[2], brH[0] / brH[2] ] )
    xps = array( [ tlH[1] / tlH[2], blH[1] / blH[2], trH[1] / trH[2], brH[1] / brH[2] ] )

    ymin = min(yps)
    xmin = min(xps)
    ymax = max(yps)
    xmax = max(xps)
    
    return [[ymin, xmin], [ymax, xmax]]

def bboxUnion(B1, B2):

    ymin = floor( min( B1[0][0], B2[0][0] ) )
    xmin = floor( min( B1[0][1], B2[0][1] ) )

    ymax = ceil( max( B1[1][0], B2[1][0] ) )
    xmax = ceil( max( B1[1][1], B2[1][1] ) )
    
    return [[ymin, xmin], [ymax, xmax]]

def translate(bbox):

    ty = -bbox[0][0]
    tx = -bbox[0][1]
    out = array( [[1, 0, ty], [0, 1, tx], [0, 0, 1]] )

    return out

def stitch(out, source, listOfPairs):

    H = computehomography(listOfPairs)

    return stitchWithH(out, source, H)

def translateOnto(imTarget, imSource, tx, ty):
    # here i'm assuming that tx and ty are positive!
    # this is true for our this
    
    imOut = imTarget
    shape = imSource.shape[:2]
    
    #print imSource.shape
    #print imOut[ ty : ty + shape[0], tx : tx + shape[1]].shape
    
    imOut[ ty : ty + shape[0], tx : tx + shape[1]] = imSource
    
    return imOut

def stitchWithH(out, source, H):

    bboxIm2 = computeTransformedBBox(source, H)
    bboxIm1 = [[0, 0] , [height(out) - 1, width(out) - 1]]
    
    bboxFinal = bboxUnion(bboxIm2, bboxIm1)
    
    tr = translate(bboxFinal)

    widthFinal = bboxFinal[1][1] - bboxFinal[0][1] + 1
    heightFinal = bboxFinal[1][0] - bboxFinal[0][0] + 1
    
    imOut = constantIm( heightFinal, widthFinal, 0.0)

    outTarget = translateOnto(imOut, out, tr[1,2], tr[0,2] )

    tr[0,2] = tr[0,2] * -1 
    tr[1,2] = tr[1,2] * -1 

    Ht = H.dot(tr) 
    
    # get the bounding box for the source
    bboxSource = computeTransformedBBox(source, Ht)
    finalComposite = applyhomography2(source, outTarget, Ht, bboxSource, True)

    return finalComposite


def stitchWithH2(target, source, H, targetWeightMap, highFrequency = False):

    bboxIm2 = computeTransformedBBox(source, H)
    bboxIm1 = [[0, 0] , [height(target) - 1, width(target) - 1]]
    
    # gets the union of the bounding boxes for target and source
    bboxFinal = bboxUnion(bboxIm2, bboxIm1)
    
    # give the transform matrix to accomodate the bounding box, as we are translating
    # the target im, we need to account for this in the source
    tr = translate(bboxFinal)

    widthFinal = bboxFinal[1][1] - bboxFinal[0][1] + 1
    heightFinal = bboxFinal[1][0] - bboxFinal[0][0] + 1
    
    # create target im in which to translate and place targetWeightMap and target
    imOut = constantIm( heightFinal, widthFinal, 0.0)
    weightMapOut = zeros( (heightFinal, widthFinal) )
    
    # translate target to accomodate the bounding box of source after H transfom
    outTarget = translateOnto(imOut, target, tr[1,2], tr[0,2] )
    outTargetWeightMap = translateOnto(weightMapOut, targetWeightMap, tr[1,2], tr[0,2] )

    tr[0,2] = tr[0,2] * -1 
    tr[1,2] = tr[1,2] * -1 

    Ht = H.dot(tr) 
    
    # get the bounding box for the source
    bboxSource = computeTransformedBBox(source, Ht)

    # apply homography3 needs to also update the weightmap
    (finalComposite, finalWeightMap) = applyhomography3(source, outTarget, Ht, bboxSource, True, outTargetWeightMap, highFrequency)

    return (finalComposite, finalWeightMap)


def getBoundingBoxTrans(out, source, H):
    
    bboxIm2 = computeTransformedBBox(source, H)
    bboxIm1 = [[0, 0] , [height(out) - 1, width(out) - 1]]
    
    bboxFinal = bboxUnion(bboxIm2, bboxIm1)
    tr = translate(bboxFinal)
    return tr

def stitchN(listOfImages, listOfListOfPairs, refIndex):
    
    hs = []

    for lIndex in range( len(listOfListOfPairs) ):
        
        listOfPairs = listOfListOfPairs[lIndex]
        H = computehomography(listOfPairs)
        hs.append(H)

    H = hs[0]
    out = listOfImages[0]
    source = listOfImages[1]
    finalComp = stitchWithH(out, source, H)
    
    tr = getBoundingBoxTrans(out, source, H)
    
    for ind in range(len(hs) - 1):
        
        H = (hs[ind + 1] ).dot(H)
        tr[0,2] = tr[0,2] * -1 
        tr[1,2] = tr[1,2] * -1 
        H = H.dot(tr)

        finalComp = stitchWithH( finalComp, listOfImages[ind + 2], H )
        tr = getBoundingBoxTrans(finalComp, listOfImages[ind + 2], H )
        
    return finalComp


def applyhomography2(source, target, H, bbox, bilinear=True, targetWeights = None ):

    # accelerate the warping function by restricting the warping loop using the
    # bounding box of each image

    # update stitch n to use it
    # give the bbox of the source image
    # only loop through the source image

    targetOut = target
    
    # we have to compute the blending coefficients for the target and
    # the source
    
    # the target is repeatedly transformed, so this is kinda annoying, no?
    # get the blending values for the target

    for y in range(int( floor(bbox[0][0]) ), int( ceil(bbox[1][0] + 1) ) ):
        for x in range(int( floor(bbox[0][1]) ), int( ceil(bbox[1][1] + 1) ) ):

            # get the source pixel that maps to this pixel
            pos = H.dot( array( [ [y], [x], [1] ] ) )
            pos = pos / pos[2]
        
            if bilinear:
                if not outside(source, pos[0], pos[1]):
                    targetOut[y,x] = interpolateLinSafe( source, pos[0], pos[1] )  
            else:
                if not outside( source, int( round( pos[0] ) ), int( round( pos[1] ) ) ):
                    targetOut[y,x] = source[ int( round(pos[0])), int( round(pos[1]) ) ]

    return targetOut

def getBilinearArray(length):

    lengthF = float(length-1)

    intervalSize = 1 / lengthF
    a = arange( -0.5, 0.5 + intervalSize/2, intervalSize)
    b = arange( 0.5, -0.5 - intervalSize/2, -intervalSize)

    c = ones( (length) ) - abs(a - b)

    c = matrix(c)

    return c

def getLinearWeightMap(height, width):

    wa = getBilinearArray(width)
    ha = getBilinearArray(height)
    
    return dot(ha.T, wa)

def getBlendedPixelValue(targetWeightMap, sourceWeightMap, targetIm, sourceIm, yTarg, xTarg, ySource, xSource, highFrequency = False):
    
    targetWeight = targetWeightMap[yTarg, xTarg]
    sourceWeight = interpolateLin(sourceWeightMap, ySource, xSource, True)

    sumWeight = targetWeight + sourceWeight
 
    targetPix = interpolateLinSafe(targetIm, yTarg, xTarg)
    sourcePix = interpolateLinSafe(sourceIm, ySource, xSource)

    if sumWeight[0] < 1e-8:
        return (sourcePix, sourceWeight)
    else:  
        if highFrequency:
            return ( (1/sumWeight) * (targetWeight * targetPix + sourceWeight * sourcePix), max(targetWeight, sourceWeight) )
        else:
            # return the normalized average of the pixels and the average of the source and target weight   ????
            return ( (1/sumWeight) * (targetWeight * targetPix + sourceWeight * sourcePix), sumWeight)

def applyhomography3(source, target, H, bbox, bilinear=True, targetWeightMap = None, highFrequency = False):
    
    targetOut = target
    targetWeightMapOut = targetWeightMap

    sourceWeightMap = getLinearWeightMap(source.shape[0],source.shape[1])

    # iterate over bounding box of source image
    for y in range(int( floor(bbox[0][0]) ), int( ceil(bbox[1][0] + 1) ) ):
        for x in range(int( floor(bbox[0][1]) ), int( ceil(bbox[1][1] + 1) ) ):
            
             # get the source pixel that maps to this pixel
            samplePos = H.dot( array( [ [y], [x], [1] ] ) )
            samplePos = samplePos / samplePos[2]
            
            if bilinear:
                if not outside(source, samplePos[0], samplePos[1]):
                    sy = samplePos[0]
                    sx = samplePos[1]

                    (targetOut[y,x], targetWeightMapOut[y,x]) = getBlendedPixelValue(targetWeightMap, sourceWeightMap, target, source, y, x, sy, sx, highFrequency)

            else:
                if not outside( source, int( round( pos[0] ) ), int( round( pos[1] ) ) ):
                    sy = int( round( samplePos[0] ) )
                    sx = int( round( samplePos[1] ) ) 

                    (targetOut[y,x], targetWeightMapOut[y,x]) = getBlendedPixelValue(targetWeightMap, sourceWeightMap, target, source, y, x, sy, sx, highFrequency )
    
    return (targetOut, targetWeightMapOut)



