from numpy import *
import sys
from imageIO import *
from AutostitchHelper import *
from scipy import ndimage, signal
from StitchTools import *
import random as rnd

# A convolution kernel for obtaining a gradient image
Sobel=array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Get the luminance from an image
def lum(im):

    imLum = zeros( ( height(im), width(im) ) )

    lumDot = array( [0.3, 0.6, 0.1] )
    imLum = im.dot(lumDot)
    
    return imLum
    

def computeTensor(im, sigmaG=1, factorSigma=4):

    # returns 3d array of the size of the image
    # yx stores xx, yx, and yy components of the tensor

    # get the luminance of the image, use [0.3, 0.6, 0.1]
    # use numpy's dot
    # blur the image
    imLum = lum(im)
    
    imLumBlurred = zeros( ( height(im), width(im) ) )

    ndimage.filters.gaussian_filter( imLum, sigmaG, 0, imLumBlurred )

    gradX = signal.convolve(imLumBlurred, Sobel, mode='same')
    gradY = signal.convolve(imLumBlurred, transpose(Sobel), mode='same')
    
    # construct 3 2d arrays of the elements of the tensor
    gradXX = gradX*gradX
    gradYY = gradY*gradY
    gradXY = gradX*gradY

    ndimage.filters.gaussian_filter( gradXX, sigmaG * factorSigma, 0, gradXX )
    ndimage.filters.gaussian_filter( gradXY, sigmaG * factorSigma, 0, gradXY )
    ndimage.filters.gaussian_filter( gradYY, sigmaG * factorSigma, 0, gradYY )


    # construct RGB image based on these vals
    out = constantIm(height(im), width(im), 0.0)

    out[:,:,0] = gradXX
    out[:,:,1] = gradXY
    out[:,:,2] = gradYY

    return out

def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=4):

    # compute corner response, a formula given in the paper
    # we need to form M, the structure tensor at each point
    
    # this might be expensive, how to do it?
    # only a 2x2 matrix at each pixel
    
    R = zeros( ( height(im), width(im) ) )

    R = (im[:,:,0] * im[:,:,2] - im[:,:,1] * im[:,:,1]) - k * (im[:,:,0] * im[:,:,2] * im[:,:,0] * im[:,:,2] )
    
    maxFiltered = zeros( ( height(im), width(im) ) )

    data_max = ndimage.filters.maximum_filter(R, maxiDiam)
    maxima = (R == data_max) # 1 if a maximum, 0 if not
    
    imwrite(im*0.2 + imageFrom1Channel(maxima), 'output')
    # get all the pixels that are 1
    labeled, num_objects = ndimage.label(maxima)
    
    #returns array slices of the objects that are maximums
    slices = ndimage.find_objects(labeled)
    x, y = [], []

    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2    

        #make sure center of slice is in allowable location
        if ( width(im) - x_center > boundarySize and x_center > boundarySize):
            if (height(im) - y_center > boundarySize ) and ( y_center > boundarySize):
                x.append(float(x_center))
                y.append(float(y_center))

    return zip(y,x)

def descriptor(blurredIm, P, radiusDescriptor) :
    
    r = radiusDescriptor

    # get a subimage representing a window around the pixel
    window = blurredIm[  P[0] - r : P[0] + r + 1 ,   P[1] - r : P[1] + r + 1  ] 

    # subtract the mean from the window
    window = window - mean(window)
    
    # scale the window  the standard deviation
    window = ( 1/std(window) ) * window

    return window

# compute the harris corners, then define a descriptor for each corner
def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
   
    # get the luminance of the image
    imLum = lum(im)

    # blur the image first
    imLumBlurred = ndimage.filters.gaussian_filter( imLum, sigmaBlurDescriptor, 0 )

    descriptors = []

    # for each corner, get its descriptor
    for corner in cornerL:
        d = descriptor(imLumBlurred, corner, radiusDescriptor) 
        descriptors.append( d )

    return zip(cornerL, descriptors)

def getL2Norm(feature1, feature2):

    dist = feature1[1] - feature2[1]
    v = dist.flatten() 
    return dot(v,v)

def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):

    pair1 = []
    pair2 = []

    for feature1 in listFeatures1:

        # get list of l2norms corresponding to this feature1 and all other features in 
        # featureList2
        l2Norms = [getL2Norm(feature1, feature2) for feature2 in listFeatures2]
       
        # get the minimum two
        minArg = argmin(l2Norms)
        minVal = min(l2Norms)

        # remove the minimal element
        l2Norms.pop(minArg) 

        minArg2 = argmin(l2Norms)
        minVal2 = min(l2Norms)  

        # check min 2 points for correspondence ratio
        if (minVal2/minVal > threshold):
            pair1.append( feature1[0] )
            pair2.append( listFeatures2[ minArg ][0] )
    
    return zip(pair1, pair2)

def mapHomo(H, pair, epsilon):

    y = pair[0][0]
    x = pair[0][1]

    yp = pair[1][0]
    xp = pair[1][1]

    # map to out position using homography
    pos = H.dot( array( [ [y], [x], [1] ] ) )
    pos = pos / pos[2]

    posH = pos[ [0,1] ]
    posP = array( [ [yp], [xp] ] )

    c = posH - posP
    v = c.flatten() 

    dist = sqrt( dot(v, v) )

    
    if (dist < epsilon):
        return True
    else:
        return False
    
def RANSAC(listOfCorrespondences, Niter=1000, epsilon=3, acceptableProbFailure=1e-9):

    cn = len(listOfCorrespondences)
    maxNumInliers = 0
    Hbest = array( (3,3) )

    for i in xrange(Niter):
        
        #obtain 4 samples from the list of Correspondences
        samples = rnd.sample(listOfCorrespondences, 4)

        # using these samples, compute the homography using SVD
        H = computehomography(samples)
    
        # figure out how many inliers there are
        inlierBools = map(mapHomo, [H] * cn, listOfCorrespondences, [epsilon] * cn  )
        
        # must count the number of bools
        numInliers = inlierBools.count(True)
        
        if (maxNumInliers < numInliers):
            Hbest = H
            maxNumInliers = numInliers

        x = float(maxNumInliers) / cn
        probFailure = pow( (1 - pow(x, 4)), i+1)

        print 'i: ' + repr(i) + ',' + 'inliers: ' + repr(maxNumInliers) +  'cn: ' + repr(cn) + ' probability of failure : ' + repr(probFailure)

        if (probFailure < acceptableProbFailure):
            break
 
    # would be cool to return the inliers
    return ( Hbest, map(mapHomo, [Hbest] * cn, listOfCorrespondences, [epsilon] * cn  ) )


def getFeatures(im, blurDescriptor, radiusDescriptor):

    imOut = computeTensor(im)
    
    # use visualize corners to see the output
    cornerList = HarrisCorners(imOut)
    LF = computeFeatures(im, cornerList, blurDescriptor, radiusDescriptor)

    return LF

def trueIfGoodPair(x):
    if x[1] is True:
        return True

def getBestH(refFeatures, sourceFeatures):

    pairs = findCorrespondences(refFeatures, sourceFeatures)

    (hRansac, boolInliers) = RANSAC(pairs)
    
    pairsForExtraction = zip(pairs, boolInliers)
    filteredInlierPairs = filter(trueIfGoodPair, pairsForExtraction)

    goodPairs, bools = zip(*filteredInlierPairs)

    # least squares on the ransac output
    # i.e. use svd to solve for homography using all 'good' points
    bestH = computehomography(goodPairs)
    
    return hRansac

# stitches together a list of images from left to right
# assumes first image is the target image
def stitchListSeparate(listOfImages, features, hs, trSoFar = eye(3), startImage = None, startImageWeightMap = None, highFrequency = False):
    
    #initialize H to identity
    H = eye(3)
    
    # replace the target with an already translated version
    # that may already have had other images stitched onto it
    target = listOfImages[0]
    
    if startImage is not None:
        target = startImage
        targetWeightMap = startImageWeightMap

    tr = trSoFar

    for i in xrange( len(hs) ):

        # set the source image
        source = listOfImages[i + 1]

        # compile the homographies sequentially
        H = ( hs[i] ).dot(H) 

        # remove the translation on the target
        tr = linalg.inv( tr ) 
        H = H.dot( tr ) 
    
        # perform stitching
        writeGrey(targetWeightMap)
        (target, targetWeightMap) = stitchWithH2( target, source, H, targetWeightMap, highFrequency ) 
        tr = getBoundingBoxTrans( target, source, H )  

        # add on the total translation which has occurred so far
        trSoFar = trSoFar.dot(tr)
    
    return (target, trSoFar, targetWeightMap) 

# stitches together a list of images from left to right
# assumes first image is the target image
def stitchList(listOfImages, features, trSoFar = eye(3), startImage = None, startImageWeightMap = None, highFrequency = False):
    
    # gets hs from left to right
    hs = map( getBestH, features[:-1], features[1:] )

    #initialize H to identity
    H = eye(3)
    
    # replace the target with an already translated version
    # that may already have had other images stitched onto it
    target = listOfImages[0]
    
    if startImage is not None:
        target = startImage
        targetWeightMap = startImageWeightMap

    tr = trSoFar

    for i in xrange( len(hs) ):

        # set the source image
        source = listOfImages[i + 1]

        # compile the homographies sequentially
        H = ( hs[i] ).dot(H) 

        # remove the translation on the target
        tr = linalg.inv( tr ) 
        H = H.dot( tr ) 
    
        # perform stitching
        writeGrey(targetWeightMap)
        (target, targetWeightMap) = stitchWithH2( target, source, H, targetWeightMap, highFrequency ) 
        tr = getBoundingBoxTrans( target, source, H )  

        # add on the total translation which has occurred so far
        trSoFar = trSoFar.dot(tr)
    
    return (target, trSoFar, targetWeightMap) 

def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
        
    # go through all of the images
    imFeatures = map( getFeatures, L, [blurDescriptor] * len(L), [radiusDescriptor] * len(L) )
    
    #initialize these so they are in function scope
    trSoFar = eye(3)
    stitchSoFar = L[refIndex]
    
    weightMapSoFar = getLinearWeightMap( stitchSoFar.shape[0], stitchSoFar.shape[1] )

    # if refIndex is not rightmost image, stitch rightwards from refIndex
    if refIndex is not (len(L) - 1):
        
        rFeatures = imFeatures[refIndex:]
        rIms = L[refIndex:]

        (stitchSoFar, trSoFar, weightMapSoFar) = stitchList( rIms, rFeatures, trSoFar, stitchSoFar, weightMapSoFar )

    # if refIndex is not first im, stitch leftwards from refIndex
    if refIndex is not 0:
        
        lFeatures = imFeatures[:(refIndex+1)]
        lIms = L[:(refIndex+1)]

        # reverse the list because we stitch from left to right on list
        lFeatures.reverse()
        lIms.reverse()

        (stitchSoFar, trSoFar, weightMapSoFar) = stitchList( lIms, lFeatures, trSoFar, stitchSoFar, weightMapSoFar)
    
    return stitchSoFar

def blurImage(im, sigmaG):
    
    return ndimage.filters.gaussian_filter( im, sigmaG, 0 )

def getHighFrequency(im, sigmaG):

    return im - ndimage.filters.gaussian_filter( im, sigmaG, 0 )

def autostitchSeparate(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
        
    # go through all of the images
    imFeatures = map( getFeatures, L, [blurDescriptor] * len(L), [radiusDescriptor] * len(L) )

    sigmaLow = 2

    Llow = map(blurImage, L, [sigmaLow] * len(L))  # kinda slow but I think works
    Lhigh = map(getHighFrequency, L, [sigmaLow] * len(L))

    #initialize these so they are in function scope
    trSoFar = eye(3)
    trSoFarHigh = eye(3)

    stitchSoFar = Llow[refIndex]
    stitchSoFarHigh = Lhigh[refIndex]

    wentRight = False

    weightMapSoFar = getLinearWeightMap( stitchSoFar.shape[0], stitchSoFar.shape[1] )
    weightMapSoFarHigh = getLinearWeightMap( stitchSoFar.shape[0], stitchSoFar.shape[1] )

    # if refIndex is not rightmost image, stitch rightwards from refIndex
    if refIndex is not (len(L) - 1):
        
        rFeatures = imFeatures[refIndex:]
        #rIms = L[refIndex:]
        rImsLow = Llow[refIndex:]
        rImsHigh = Lhigh[refIndex:]
       
        # compute hs here so they are the same for both frequencies
        hs = map( getBestH, rFeatures[:-1], rFeatures[1:] )

        (stitchSoFarHigh, trSoFarHigh, weightMapSoFarHigh)  = stitchListSeparate( rImsHigh, rFeatures, hs, trSoFar, stitchSoFarHigh, weightMapSoFarHigh, True)
        (stitchSoFar, trSoFar, weightMapSoFar)              = stitchListSeparate( rImsLow, rFeatures, hs, trSoFar, stitchSoFar, weightMapSoFar)
        
        wentRight = True

        imwrite(stitchSoFarHigh, 'highFreq.png')
        imwrite(stitchSoFar, 'lowFreq.png')

        writeGrey(weightMapSoFar)
        writeGrey(weightMapSoFarHigh)

    # if refIndex is not first im, stitch leftwards from refIndex
    if refIndex is not 0:

        lFeatures = imFeatures[:(refIndex+1)]

        lImsHigh = Lhigh[:(refIndex+1)]
        lImsLow = Llow[:(refIndex+1)]

        # reverse the list because we stitch from left to right on list
        lFeatures.reverse()
        lImsHigh.reverse()
        lImsLow.reverse()
        
        hs = map( getBestH, lFeatures[:-1], lFeatures[1:] )

        # we need to generate 
        (stitchSoFarHigh, trSoFarHigh, weightMapSoFarHigh)  = stitchListSeparate( lImsHigh, lFeatures, hs, trSoFar, stitchSoFarHigh, weightMapSoFarHigh, True)
        (stitchSoFar, trSoFar, weightMapSoFar)              = stitchListSeparate( lImsLow, lFeatures, hs, trSoFar, stitchSoFar, weightMapSoFar)

        imwrite(stitchSoFarHigh, 'highFreqLeft.png')
        imwrite(stitchSoFar, 'lowFreqLeft.png')

        writeGrey(weightMapSoFar)
        writeGrey(weightMapSoFarHigh)

        

    out = stitchSoFarHigh + stitchSoFar

    return out





