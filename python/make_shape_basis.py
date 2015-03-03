'''
Load Shapes, Compute basis
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

#Environment Variables
DATA = os.getenv('DATA')
proj_path = DATA + '3dFace/'
write_path = proj_path + 'shape_basis/PCA/'
#Load self movements
path= proj_path + 'matfiles/*'

#list all files in the directory above
print('The path for glob search is', path)
matfiles=glob.glob(path)
print('Total number of mat files  is %d',len(matfiles))


shapes = np.zeros([len(matfiles),512*512])
for ii in range(len(matfiles)):
    if np.mod(ii,10)==0:
        print('Files Loaded --- ',ii)
    matfile = scio.loadmat(matfiles[ii])
    geometry = matfile['geometry']
    shapes[ii,:]=geometry.flatten()
print('Successfully loaded')

#Compute Mean Face
mean_face = np.mean(shapes,axis=0)
mean_plt = plt.figure()
plt.imshow(mean_face.reshape([512,512]))
plt.colorbar()
mean_plt.savefig(write_path+'Mean_Face.png',bbox_inches='tight',pad_inches=0)


#Now let's try doing PCA
print('Compute Covariance Matrix')
shapes_subtr_mean_face = shapes - mean_face
shapes_cov = np.cov(shapes_subtr_mean_face)
print('Compute Eigen Vectors')
shape_vals,shape_vectors = eigs(shapes_cov,k=50,return_eigenvectors=True)


eig_val_fig = plt.figure()
plt.plot(shape_vals)
plt.legend('eigen value energy')
eig_val_fig.savefig(write_path+'Eig_Value_Plot.png',bbox_inches='tight',pad_inches=0)

#Now let's scale the Eigen Vectors back to original shape
shape_vectors_face = np.dot(shape_vectors.T, shapes_subtr_mean_face)


for ii in range(50):
    eig_vec_fig = plt.figure()
    plt.imshow(np.real(shape_vectors_face[ii,:].reshape([512,512])))
    plt.colorbar()
    plt.title('Eigen Vector '+str(ii)+'Rendered as an image')
    eig_vec_fig.savefig(write_path+'Eigen_Vector_'+str(ii)+'_.png',bbox_inches='tight',pad_inches=0)
    plt.close()
