import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import sys
import struct
from scipy import misc
from mayavi import mlab

def main()


    return 1

def textureread(fname):
    im_texture=misc.imread(fname)
    im_texture = np.rot90(im_texture)
    return im_texture

def cybread(fname):
    file_hndl = open(fname,mode='rb')
    print(file_hndl)    
    for ii in range(21):
        print(file_hndl.readline())
    image = np.zeros([512*512,1])
    #print(struct.unpack('i',file_hndl.read(4)))
    image = np.fromfile(file_hndl,dtype=np.uint16)
    image = image.reshape([512,512])
    image = image.T
    image = image.astype(np.uint32)
    file_hndl.close()
    for ii in range(512):
        for jj in range(512):
            if image[ii,jj] != -32768:
                image[ii,jj] = image[ii,jj]<<4
            else:
                image[ii,jj] = 0    
    return image
