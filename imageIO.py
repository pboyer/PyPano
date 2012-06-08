import png
import numpy

baseInputPath='Input/'
baseOutputPath='Output/'

def imread(path='in.png'):
    '''reads a PNG RGB image at baseInputPath+path and return a numpy array organized along Y, X, channel.
    The values are encoded as float and are linearized (i.e. gamma is decoded)'''
    global baseInputPath
    reader=png.Reader(baseInputPath+path)
    im=reader.asFloat()
    a=numpy.vstack(im[2])
    if im[3]['greyscale']:
        raise NameError( 'Expected an RGB image, given a greyscale one')        
    x, y=im[0], im[1]
    a.resize(y, x, 3)
    a**=2.2
    return a

def imreadGrey(path='raw.png'):
    '''reads a PNG greyscale image at baseInputPath+path and return a numpy array organized along Y, X.
    The values are encoded as float and are assumed to be linear in the input file (gamma is NOT decoded)'''
    global baseInputPath    
    reader=png.Reader(baseInputPath+path)
    im=reader.asFloat()
    a=numpy.vstack(im[2])
    if not im[3]['greyscale']:
        raise NameError( 'Expected a greyscale image, given an RGB one')
    x, y=im[0], im[1]
    a.resize(y, x)
    return a

def imwrite(im, path='out.png'):
    '''takes a 3D numpy array organized along Y, X, channel and writes it to a PNG file.
    The values are assumed to be linear between 0 and 1 and are gamma encoded before writing.'''
    global baseOutputPath
    y,x=im.shape[0], im.shape[1]
    im=numpy.clip(im, 0, 1)
    im=im.reshape(y, x*3)
    writer = png.Writer(x,y)
    f=open(baseOutputPath+path, 'wb')
    writer.write(f, 255*im**(1/2.2))
    f.close()
    im.resize(y, x, 3)

seqCount=0

def imwriteSeq(im, initial):
    global seqCount
    path=initial+str(seqCount)+'.png'
    imwrite(im, path)
    seqCount+=1


def imwriteGrey(im, path='raw.png'):
    '''takes a 2D numpy array organized along Y, X and writes it to a PNG file.
    The values are assumed to be linear between 0 and 1 and are NOT gamma encoded before writing.'''
    global baseOutputPath
    y,x=im.shape[0], im.shape[1]
    im=numpy.clip(im, 0, 1)
    writer = png.Writer(x,y)
    f=open(baseOutputPath+path, 'wb')
    writer.write(f, im*255)
    f.close()

    

def constantIm(y, x, color):
    out = numpy.empty([y, x, 3])
    out[:, :]=color
    return out
    
