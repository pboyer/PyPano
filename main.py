import numpy
from numpy import *
from imageIO import *
from Autostitch import *

# Main entry point for autostitch functions
def main():

    ims = loadListOfImages('pano/room')
    out = autostitch(ims, 1)
    imwrite(out)


if __name__ == "__main__":
    main()




