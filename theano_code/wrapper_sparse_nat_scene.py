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
import ipdb
import sparse_code_gpu
import time
import theano.tensor as T
import theano

def create_theano_fn(patchdim,lam,basis_no,batch):
    coeff_flat = T.dvector('coeffs')
    coeff = T.cast(coeff_flat,'float32')
    coeff = T.reshape(coeff,[basis_no,batch])
    basis_flat = T.dvector('basis')
    basis = T.cast(basis_flat,'float32')
    basis = T.reshape(basis,[patchdim[0]*patchdim[1],basis_no])
    data_flat = T.dvector('data')
    data = T.cast(data_flat,'float32')
    data = T.reshape(data,[patchdim[0]*patchdim[1],batch])
    tmp = (data - basis.dot(coeff))**2
    tmp = 0.5*tmp.sum(axis=0)
    tmp = tmp.mean()
    sparsity = lam * T.abs_(coeff).sum(axis=0).mean()
    obj = tmp + sparsity
    grads = T.grad(obj,coeff_flat)
    f = theano.function([coeff_flat,basis_flat,data_flat],[obj.astype('float64'),grads.astype('float64')])
    return f 

def infer_coeff(func_hndl,data,basis,basis_no,batch):
    init_coeff = np.zeros(basis_no*batch).astype('float32')
    res = minimize(fun=func_hndl,x0=init_coeff,args=(basis.flatten(),data.flatten()),method='L-BFGS-B',jac=True,options={'disp':False})
    print('Value of objective fun after doing inference',res.fun)
    active = len(res.x[np.abs(res.x)> 1e-3])/float(basis_no*batch)
    print('Number of Active coefficients is ...',active)
    return res.x 

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    data_path = DATA + 'scene-sparse/'
    write_path = DATA + '3dFace/shape_basis/sparse/nat_scene'
    IMAGES = scio.loadmat(data_path+'IMAGES.mat')
    IMAGES = IMAGES['IMAGES']

    (imsize, imsize, num_images) = np.shape(IMAGES)


    #Inference Variables
    LR = 0.2
    training_iter = 10000 
    lam = 0.15
    step = 0.05
    border = 4
    orig_patchdim = 512
    patchdim = np.zeros(2)
    patchdim[0] = 8
    patchdim[1] = 8
    print('patchdim is ---',patchdim)
    sz = 8 
    batch = 100
    test_batch = 15
    basis_no = 4*(8**2)
    data = np.zeros((patchdim[0]*patchdim[1],batch))
    matfile_write_path = write_path+'LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

    
    #Making and Changing directory
    try:
        print('Trying to see if directory exists already')
        os.stat(matfile_write_path)
    except:
        print('Nope nope nope. Making it now')
        os.mkdir(matfile_write_path)

    try:
        print('Navigating to said directory for data dumps')
        os.chdir(matfile_write_path)
    except:
        print('Unable to navigate to the folder where we want to save data dumps')

    #Create object
    lbfgs_sc = sparse_code_gpu.LBFGS_SC(LR=LR,lam=lam,batch=batch,basis_no=basis_no,patchdim=patchdim,savepath=matfile_write_path)
    residual_list=[]
    sparsity_list=[]
    avg_r_error = []
    #print('Compiling Inference function locally')
    #infer_func_hndl = create_theano_fn(patchdim,lam,basis_no,test_batch)
    for ii in np.arange(training_iter):
    # Choose a random image
        imi = np.ceil(num_images * np.random.uniform(0, 1))

        for i in range(batch):
            r = border + np.ceil((imsize-sz-2*border) * np.random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * np.random.uniform(0, 1))

            data[:,i] = np.reshape(IMAGES[r:r+patchdim[0], c:c+patchdim[1], imi-1], patchdim[0]*patchdim[1], 1)
            #Make data
        print('Training iteration -- ',ii)
        if np.mod(ii,100)==0:
            print('*****************Modifying LR so our model converges**************************')
            LR = LR - step
            print('New Value of LR is ', LR)
            step = step*0.5
            lbfgs_sc.adjust_LR(LR)
        lbfgs_sc.load_data(data)
        #Note this way, each column is a data vector
        tm3 = time.time()
        active,result = lbfgs_sc.infer_coeff()
        print('Value of  Objective function after Inference ---',result)
        print('Number of active coefficients after Inference ---',active)
        sparsity_list.append(active)
        tm4 = time.time()
        residual,active=lbfgs_sc.update_basis()
        print('Value of Residual after Updating basis ---',residual)
        print('Value of active coefficients after updating basis ---', active)
        residual_list.append(residual)
        tm5 = time.time()    
        print('Infer coefficients cost in seconds', tm4-tm3)
        print('Updating basis cost in seconds',tm5-tm4)
        #residual_list.append(residual)
        if np.mod(ii,100)==0:
            print('Saving the basis now, for iteration ',ii)
            shape_basis = {
            'sparse_shape_basis': lbfgs_sc.basis.get_value(),
            'residuals':residual_list,
            'sparsity':sparsity_list,
            }
            scio.savemat('basis',shape_basis)
            print('Saving basis visualizations now')
            lbfgs_sc.visualize_basis(ii)
            print('Visualizations done....back to work now')
        '''
        if np.mod(ii,5)==0:
            print('testing out test error')
            data = shapes_whitened[batch:,:].T
            print('Moving Basis from GPU to CPU')
            tm1 = time.time()
            basis = lbfgs_sc.basis.get_value()
            tm2 = time.time()
            print('Getting basis costed ', tm2-tm1)
            print('Calling Inference')
            coeff = infer_coeff(infer_func_hndl,data,basis,basis_no,test_batch)
            coeff = np.reshape(coeff,[basis_no,test_batch])
            residual = data - basis.dot(coeff)
            residual = residual**2
            residual = np.sqrt(residual.sum(axis=0))
            r_error = residual.mean()
            avg_r_error.append(r_error)
            print('Test Error is',r_error)
            data = shapes_whitened[0:batch,:].T
            tm6 = time.time()
            lbfgs_sc.load_data(data) 
            tm7 = time.time()
            print('Time to load back original data is',tm7-tm6)
         '''
