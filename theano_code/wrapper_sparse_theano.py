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

def norm_basis(data):
    norm_data = np.diag(1.0/np.sqrt(np.sum(data**2,axis=1)))
    data = np.dot(norm_data,data)
    return data

def compute_whitening(data):
    shapes_cov = np.cov(data)
    print('Compute Eigen Vectors')
    [d,V] = np.linalg.eigh(shapes_cov)
    fudge = 1e-3 
    #Scaling variance matrix
    D = np.diag(1 / np.sqrt(d + fudge))
    #Whitening matrix
    W = np.dot(np.dot(V,D),V.T)
    print('Shape of W ',W.shape)
    #Whitened Shape
    shapes_whitened = np.dot(W,data)

    return shapes_whitened,W

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + '3dFace/'
    write_path = proj_path + 'shape_basis/sparse/'
    path= proj_path + 'matfiles/*'

    #Inference Variables
    LR = 0.15
    step = 0.05
    training_iter = 10000 
    lam = 0.10 
    orig_patchdim = 512
    rows_range,cols_range = np.meshgrid(np.arange(156,512-156),np.arange(156,512-156),indexing='ij')
    patchdim = np.asarray([0,0])
    patchdim[0] = rows_range.shape[0]
    patchdim[1] = cols_range.shape[1]
    print('patchdim is ---',patchdim)
    batch = 200 
    test_batch = 15
    basis_no =15
    matfile_write_path = write_path+'Faces_r_theta_norm_data_LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

    #list all files in the directory above
    print('The path for glob search is', path)
    matfiles=glob.glob(path)
    print('Total number of mat files  is %d',len(matfiles))
    shapes= np.zeros([len(matfiles),patchdim[0]*patchdim[1]])
    for ii in range(len(matfiles)):
        if np.mod(ii,10)==0:
            print('Files Loaded --- ',ii)
        matfile = scio.loadmat(matfiles[ii])
        #geometry = matfile['vertices']
        #actual_geometry = np.reshape(geometry[:,2],[orig_patchdim,orig_patchdim])
        geometry = matfile['geometry']
        actual_geometry = np.reshape(geometry,[orig_patchdim,orig_patchdim])
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
    print('Subtractign Mean')
    shapes_subtr_mean_face = shapes - mean_face
    #Whitening the Data
    shapes_whitened,W = compute_whitening(shapes_subtr_mean_face)
    #Normalizing(sphering) the data
    sphered_data = norm_basis(shapes_whitened)
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
    #data = shapes_whitened[0:batch,:].T
    data = sphered_data[0:batch,:].T
    lbfgs_sc.load_data(data)
    residual_list=[]
    sparsity_list=[]
    avg_r_error = []
    print('Compiling Inference function locally')
    infer_func_hndl = create_theano_fn(patchdim,lam,basis_no,test_batch)
    SNR_I_2 = (data**2)
    SNR_I_2 = 0.5*(SNR_I_2).sum(axis=0)
    SNR_I_2 = SNR_I_2.mean()
    for ii in np.arange(training_iter):
        
        if np.mod(ii,10)==0:
            print('*****************Adjusting Learning Rate*******************')
            LR = LR * 0.5 
            #step  = step*0.75
            print('New Learning Rate is .........',LR)
            lbfgs_sc.adjust_LR(LR)
        
    #Make data
        print('Training iteration -- ',ii)
        #Note this way, each column is a data vector
        tm3 = time.time()
        for jj in np.arange(100):
            obj,active_infer = lbfgs_sc.infer_coeff_gd()
            if np.mod(jj,10)==0:
                print('Value of objective function from previous iteration of coeff update',obj)
        sparsity_list.append(active_infer)
        tm4 = time.time()
        residual,active,basis=lbfgs_sc.update_basis()
        residual_list.append(residual)
        tm5 = time.time()
        print('Infer coefficients cost in seconds', tm4-tm3)
        print('The value of active coefficients after we do inference ', active_infer)
        print('The value of residual after we do learning ....', residual)
        print('The SNR for the model is .........',SNR_I_2/residual)
        print('The value of active coefficients after we do learning ....',active)
        print('The mean norm of the basis is .....',np.mean(np.linalg.norm(basis,axis=0)))
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
            lbfgs_sc.visualize_basis(ii,[3,5])
            print('Visualizations done....back to work now')
