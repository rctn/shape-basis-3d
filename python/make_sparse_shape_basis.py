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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from LCAversions.LCAnumbaprog import lca as lcanumb
from LCAversions.LCAnumpy import lca as lcanumpy
from scipy.optimize import minimize 
import os
import pdb

#Update basis

def update_basis(basis,coeff,LR,data):
    Residual = data - np.dot(basis,coeff.T)
    dbasis = LR * (np.dot(Residual,coeff))
    basis = basis + dbasis
    #Normalize basis
    norm_basis = np.diag(1/np.sqrt(np.sum(basis**2,axis=0)))
    basis = np.dot(basis,norm_basis)
    print('The norm of the basis is',np.linalg.norm(basis)) 
    basis=np.asarray(basis)
    Residual= np.mean(np.sqrt(np.sum(Residual**2,axis=0)))
    return basis,Residual

def objective_fn(coeff,data,basis,lam):
    coeff_size = np.zeros(2)
    coeff_size[0] = data.shape[1]
    coeff_size[1] = basis.shape[1]
    coeff = np.reshape(coeff,coeff_size)
    residual = np.mean(np.linalg.norm( data - np.dot(basis,coeff.T),axis=0))
    penalty = lam * np.linalg.norm(coeff,ord=1)
    E = residual + penalty
    E = np.array(E)
    return E

def objective_grad(coeff,data,basis,lam):
    coeff_size = np.zeros(2)
    coeff_size[0] = data.shape[1]
    coeff_size[1] = basis.shape[1]
    coeff = np.reshape(coeff,coeff_size)
    residual1 = -2*(data - np.dot(basis,coeff.T))
    residual2 = np.dot(residual1,basis.T)
    grad = residual2.T
    return grad

def grad_check(coeff,data,basis,lam):
    #grad_orig
    grad = np.zeros([coeff.shape[1],1])
    #delta x
    delta = 1e-5
    #compute each dimension of grad using method of deltas
    tmp1 = objective_fn(coeff,data,basis,lam)
    for ii in np.arange(coeff.shape[1]):
        #compute grad
        coeff[:,ii]= coeff[:,ii] + delta
        tmp2 = objective_fn(coeff,data,basis,lam)
        coeff[:,ii]= coeff[:,ii] - delta
        grad[ii] = (tmp2-tmp1)/(delta)
    obj_grad = objective_grad(coeff,data,basis,lam)

    print('The norm of the difference of gradients is --', np.linalg.norm(grad-obj_grad))
    return err

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + '3dFace/'
    write_path = proj_path + 'shape_basis/sparse/'
    path= proj_path + 'matfiles/*'

    #Inference Variables
    LR = 0.75
    eta = 0.1
    training_iter = 1000
    InferIter = 300
    lamb=0.05
    lam = 0.2
    adapt = .9
    patchdim = 512
    batch = 30
    basis_no =150
    basis = np.zeros([patchdim**2,basis_no])
    coeff = np.random.randn(batch,basis_no)

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
    fudge = 1e-5 
    #Scaling variance matrix
    D = np.diag(1 / np.sqrt(d + fudge))
    #Whitening matrix
    W = np.dot(np.dot(V,D),V.T)
    print('Shape of W ',W.shape)
    #Whitened Shape
    shapes_whitened = np.dot(W,shapes_subtr_mean_face)
    residual_list=[]
    for ii in np.arange(training_iter):
    #Make data
        gen_rand_ind = np.random.randint(0,shapes_whitened.shape[0],size=batch)
        #Note this way, each column is a data vector
        data = shapes_whitened[gen_rand_ind,:].T 
    #Do learning step here
        [basis,residual] = update_basis(basis,coeff,LR,data)
        residual_list.append(residual)
        if np.any(np.isnan(basis)):
            print('Basis are nans')
        print('The value of the residual is --  ', np.linalg.norm(residual))
    #Do Inference step here
        #[coeff,u,thresh] = lcanumpy.infer(basis.T,data.T,eta,lamb,InferIter,adapt)
        coeff_init = np.random.randn(batch,basis_no)
        pdb.set_trace()
        res = minimize(objective_fn,x0=coeff_init,args=(data,basis,lam),tol=1e-3,method='L-BFGS-B',options={'maxiter':15}) 
        coeff = np.reshape(res.x,[batch,basis_no])
        if np.any(np.isnan(coeff)):
            print('coeff are nans')
        if np.mod(ii,10)==0:
            print('Saving the basis now, for iteration ',ii)
            shape_basis = {
            'mean_face': mean_face,
            'shapes_subtr_mean_face':shapes_subtr_mean_face,
            'sparse_shape_basis': basis,
            'residuals':residual_list,
            'coeff':coeff,
            'whitening_matrix': W,
            'vertices':vertices,
            }
            scio.savemat(write_path+'LR_'+str(LR)+'_batch_'+str(batch)+'_basisno_'+str(basis_no)+'_lam_'+str(lam)+'_basis.mat',shape_basis)

