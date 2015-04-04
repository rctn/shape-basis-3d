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
    proj_path = DATA + '3dFace/'
    write_path = proj_path + 'shape_basis/sparse/'
    path= proj_path + 'matfiles/*'

    #Inference Variables
    LR = 0.15
    training_iter = 31
    lam = 0.0
    orig_patchdim = 512
    rows_range,cols_range = np.meshgrid(np.arange(128,512-128),np.arange(156,512-156),indexing='ij')
    patchdim = np.asarray([0,0])
    patchdim[0] = rows_range.shape[0]
    patchdim[1] = cols_range.shape[1]
    print('patchdim is ---',patchdim)
    batch = 185
    test_batch = 15
    basis_no =50
    basis = np.zeros([patchdim[0]*patchdim[1],basis_no])
    matfile_write_path = write_path+'LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

    #list all files in the directory above
    print('The path for glob search is', path)
    matfiles=glob.glob(path)
    print('Total number of mat files  is %d',len(matfiles))
    shapes= np.zeros([len(matfiles),patchdim[0]*patchdim[1]])
    for ii in range(len(matfiles)):
        if np.mod(ii,10)==0:
            print('Files Loaded --- ',ii)
        matfile = scio.loadmat(matfiles[ii])
        geometry = matfile['vertices']
        actual_geometry = np.reshape(geometry[:,2],[orig_patchdim,orig_patchdim])
        actual_geometry = actual_geometry[rows_range,cols_range]
        try:
            #shapes.append(geometry)
            shapes[ii,:] = actual_geometry.flatten()
        except:
            pdb.set_trace()
    print('Successfully loaded')
    vertices = matfile['vertices']
    X=np.reshape(vertices[:,0],[orig_patchdim,orig_patchdim])
    Y=np.reshape(vertices[:,1],[orig_patchdim,orig_patchdim])
    X=X[rows_range,cols_range]
    Y=Y[rows_range,cols_range]
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
    #gen_rand_ind = np.random.randint(0,shapes_whitened.shape[0],size=batch)
    data = shapes_whitened[0:batch,:].T 
    lbfgs_sc.load_data(data)
    residual_list=[]
    sparsity_list=[]
    avg_r_error = []
    print('Compiling Inference function locally')
    infer_func_hndl = create_theano_fn(patchdim,lam,basis_no,test_batch)
    for ii in np.arange(training_iter):
    #Make data
        print('Training iteration -- ',ii)
        #Note this way, each column is a data vector
        tm3 = time.time()
        active = lbfgs_sc.infer_coeff()
        sparsity_list.append(active)
        tm4 = time.time()
        residual=lbfgs_sc.update_basis_wrapper()
        residual_list.append(residual)
        tm5 = time.time()    
        print('Infer coefficients cost in seconds', tm4-tm3)
        print('Updating basis cost in seconds',tm5-tm4)
        #residual_list.append(residual)
        if np.mod(ii,10)==0:
            print('Saving the basis now, for iteration ',ii)
            shape_basis = {
            'mean_face': mean_face,
            'shapes_subtr_mean_face':shapes_subtr_mean_face,
            'sparse_shape_basis': lbfgs_sc.basis.get_value(),
            'residuals':residual_list,
            'sparsity':sparsity_list,
            'whitening_matrix': W,
            'vertices':vertices,
            'X':X,
            'Y':Y,
            'avg_r_error':avg_r_error
            }
            scio.savemat('basis',shape_basis)
            print('Saving basis visualizations now')
            lbfgs_sc.visualize_basis(ii)
            print('Visualizations done....back to work now')
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
         
