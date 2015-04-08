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
import ipdb
import pcd2im

#Environment Variables
DATA = os.getenv('DATA')
proj_path = DATA + '3dFace/'
write_path = proj_path + 'shape_basis/PCA/'
path= proj_path + 'matfiles/*'

def my_pca(shapes):
    #Compute Mean Face
    mean_face = np.mean(shapes,axis=0)
    #Now let's try doing PCA
    print('Compute Covariance Matrix')
    shapes_subtr_mean_face = shapes - mean_face
    shapes_cov = np.cov(shapes_subtr_mean_face)
    print('Compute Eigen Vectors')
    shape_vals,shape_vectors = eigs(shapes_cov,k=50,return_eigenvectors=True)
    shape_vectors = np.real(shape_vectors)
    #Now let's scale the Eigen Vectors back to original shape
    shape_vectors_face = np.dot(shape_vectors.T, shapes_subtr_mean_face)
    eig_val_fig = plt.figure()
    plt.plot(shape_vals)
    plt.legend('eigen value energy')
    eig_val_fig.savefig(write_path+'Eig_Value_Plot.png',bbox_inches='tight',pad_inches=0)

    return shape_vals, shape_vectors_face 

def sklearn_pca(shapes):
    pca =PCA(n_components=50,whiten=True)
    pca.fit(shapes)
    shape_vectors_face = pca.components_
    shape_vals = pca.explained_variance_ratio_
    eig_val_fig = plt.figure()
    plt.plot(shape_vals)
    plt.legend('eigen value energy')
    eig_val_fig.savefig(write_path+'Eig_Value_Plot.png',bbox_inches='tight',pad_inches=0)
    return shape_vals,shape_vectors_face

def load_data():
    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + '3dFace/'
    write_path = proj_path + 'shape_basis/PCA/'
    path= proj_path + 'matfiles/*'

    #list all files in the directory above
    print('The path for glob search is', path)
    matfiles=glob.glob(path)
    print('Total number of mat files  is %d',len(matfiles))
    #rows_range,cols_range = np.meshgrid(np.arange(128,512-128),np.arange(156,512-156),indexing='ij')
    rows_range,cols_range = np.meshgrid(np.arange(156,512-156),np.arange(156,512-156),indexing='ij')

    shapes = np.zeros([len(matfiles),rows_range.shape[0]*rows_range.shape[1]])
    #shapes = []
    for ii in range(len(matfiles)):
        if np.mod(ii,10)==0:
            print('Files Loaded --- ',ii)
        matfile = scio.loadmat(matfiles[ii])
        geometry = matfile['vertices']
        actual_geometry = np.reshape(geometry[:,2],[512,512])
        actual_geometry = actual_geometry[rows_range,cols_range]
        try:
            #shapes.append(geometry)
            shapes[ii,:] = actual_geometry.flatten()
        except:
            pdb.set_trace()
    print('Successfully loaded')
    vertices = matfile['vertices']
    X=vertices[:,0]
    Y=vertices[:,1]
    X = np.reshape(X,[512,512])
    Y = np.reshape(Y,[512,512])
    X = X[rows_range,cols_range]
    Y = Y[rows_range,cols_range]
    X = X.flatten()
    Y = Y.flatten()
    mean_face = np.mean(shapes,axis=0)
    return shapes,X,Y, mean_face 


if __name__ == "__main__":

    shapes, X, Y, mean_face = load_data()
    shape_vals, shape_vecs = my_pca(shapes)

    #sklearn_pca(shapes)

    print("saving Basis")
    shape_basis = {
    'mean_face': mean_face,
    'X': X,
    'Y': Y,
    #'shapes_subtr_mean_face':shapes_subtr_mean_face,
    #'shapes_cov':shapes_cov,
    'shape_eig_vals':shape_vals,
    #'shape_eig_vectors':shape_vectors,
    'shape_eig_vectors_face':shape_vecs,
    #'shapes':shapes
    }
    scio.savemat(write_path+'basis.mat',shape_basis)

    '''
    print("Reconstructing from basis")
    pca_proj = pca.transform(shapes)
    pca_proj_orig = pca.inverse_transform(pca_proj)

    print("Now computing error")
    residual = ((pca_proj_orig - shapes)**2).sum(axis=1)
    residual = np.sqrt(residual).mean()
    print('Average Reconstruction Error is ',residual)
    '''
