'''
Load Shapes, Compute basis
These basis be sparse
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs
import glob
from scipy.optimize import minimize 
import os
import pdb
import sparse_code_gpu
import time

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + '3dFace/'
    write_path = proj_path + 'shape_basis/sparse/'
    path= proj_path + 'matfiles/*'

    #Inference Variables
    LR = 0.1
    #eta = 0.1
    training_iter = 1000
    #InferIter = 300
    #lamb=0.05
    lam = 0.1
    #adapt = .9
    patchdim = 512
    batch = 30
    basis_no =50
    basis = np.zeros([patchdim**2,basis_no])
    coeff = np.random.randn(batch,basis_no)
    matfile_write_path = write_path+'LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis_'

    #list all files in the directory above
    print('The path for glob search is', path)
    matfiles=glob.glob(path)
    print('Total number of mat files  is %d',len(matfiles))
    shapes= np.zeros([len(matfiles),patchdim**2])
    for ii in range(len(matfiles)):
        if np.mod(ii,10)==0:
            print('Files Loaded --- ',ii)
        matfile = scio.loadmat(matfiles[ii])
        geometry = matfile['vertices']
        actual_geometry = np.reshape(geometry[:,2],[patchdim,patchdim])
        try:
            #shapes.append(geometry)
            shapes[ii,:] = actual_geometry.flatten()
        except:
            pdb.set_trace()
    print('Successfully loaded')
    vertices = matfile['vertices']
    X=np.reshape(vertices[:,0],[patchdim,patchdim])
    Y=np.reshape(vertices[:,1],[patchdim,patchdim])
    #Compute Mean Face
    mean_face = np.mean(shapes,axis=0)
    #Now let's try doing PCA
    print('Compute Covariance Matrix')
    shapes_subtr_mean_face = shapes - mean_face
    shapes_cov = np.cov(shapes_subtr_mean_face)
    print('Compute Eigen Vectors')
    [d,V] = np.linalg.eigh(shapes_cov)
    fudge = 1e-6 
    #Scaling variance matrix
    D = np.diag(1 / np.sqrt(d + fudge))
    #Whitening matrix
    W = np.dot(np.dot(V,D),V.T)
    print('Shape of W ',W.shape)
    #Whitened Shape
    shapes_whitened = np.dot(W,shapes_subtr_mean_face)
    
    #Create object
    lbfgs_sc = sparse_code_gpu.LBFGS_SC(LR=LR,lam=lam,batch=batch,basis_no=basis_no,patchdim=patchdim,savepath=matfile_write_path)

    residual_list=[]
    for ii in np.arange(training_iter):
    #Make data
        gen_rand_ind = np.random.randint(0,shapes_whitened.shape[0],size=batch)
        #Note this way, each column is a data vector
        tm1 = time.time()
        data = shapes_whitened[gen_rand_ind,:].T 
        tm2 = time.time()
        lbfgs_sc.load_new_batch(data)
        tm3 = time.time()
        lbfgs_sc.infer_coeff()
        tm4 = time.time()
        #residual = lbfgs_sc.update_basis()
        res=lbfgs_sc.update_basis()
        tm5 = time.time()
        res = res.eval()
        print(res)
        del(res)
        tm6 = time.time()
        print('Temporary data variable Cost in seconds', tm2-tm1)
        print('Load new batch cost in seconds', tm3-tm2)
        print('Infer coefficients cost in seconds', tm4-tm3)
        print('Updating basis cost in seconds',tm5-tm4)
        print('Cost for evaluating results. I think this is some GPU to CPU BS', tm6-tm5)
        '''
        #residual_list.append(residual)
        if np.mod(ii,10)==0:
            print('Saving the basis now, for iteration ',ii)
            shape_basis = {
            'mean_face': mean_face,
            'shapes_subtr_mean_face':shapes_subtr_mean_face,
            'sparse_shape_basis': lbfgs_sc.basis.eval(),
            #'residuals':residual_list,
            'coeff':lbfgs_sc.coeff,
            'whitening_matrix': W,
            'vertices':vertices,
            }
            scio.savemat(matfile_write_path,shape_basis)
            print('Saving basis visualizations now')
            lbfgs_sc.visualize_basis(ii)
            print('Visualizations done....back to work now')
        '''

