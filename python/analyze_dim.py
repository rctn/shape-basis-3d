#This will be python2.x code

import numpy as np
from scipy.misc import imread
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs
import glob
from matplotlib import pyplot as plt

#Load self movements
path='/media/mudigonda/Gondor/Data/sensorimotor/'
self_path = path + 'self/*.png'

#list all files in the directory above
self_files=glob.glob(self_path)
print('Total number of self images is %d')%(len(self_files))
self_images = np.zeros([len(self_files),512*512])
for ii in range(len(self_files)):
    tmp=imread(self_files[ii])
    self_images[ii,:]=tmp.flatten()
print('Successfully loaded')

#Now let's try doing PCA
self_images_cov = np.cov(self_images)
self_images_vals = eigs(self_images_cov,k=15,return_eigenvectors=False)
print self_images_vals
print "Eig energy of 6"
num1 = np.sum(self_images_vals[-2:])
den1 = np.sum(self_images_vals)
print num1, den1
print np.real(num1/den1)

#Load world movements
world_path = path + 'world/*.png'
#list all files in the directory above
world_files=glob.glob(world_path)
print('Total number of world images is %d')%(len(world_files))
world_images = np.zeros([len(world_files),512*512])
for ii in range(len(world_files)):
    tmp=imread(world_files[ii])
    world_images[ii,:]=tmp.flatten()
print('Successfully loaded')

#Now let's try doing PCA
world_images_cov = np.cov(world_images)
world_images_vals = eigs(world_images_cov,k=15,return_eigenvectors=False)
print world_images_vals
num2= np.sum(np.real(world_images_vals[-2:]))
den2= np.sum(np.real(world_images_vals))
print num2/den2

#Now let's stack the two matrices above and then do PCA
both_images=np.vstack((self_images,world_images))
both_images_cov = np.cov(both_images)
both_images_val = eigs(both_images_cov,k=15,return_eigenvectors=False)
print both_images_val
num3= np.sum(both_images_val[-2:])
den3= np.sum(both_images_val)
print np.real(num3/den3)

