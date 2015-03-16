import numpy as np
from scipy.optimize import minimize
import theano 
from theano import tensor as T
import pylearn2
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class LBFGS_SC:

    def __init__(self,savepath=None,LR=None,lam=None,batch=None,basis_no=None,patchdim=None):
        if LR is None:
            self.LR = 0.1
        else:
            self.LR = LR
        if lam is None:
            self.lam = 0.1
        else:
            self.lam = lam
        if batch is None:
            self.batch = 30
        else:
            self.batch = batch
        if basis_no is None:
            self.basis_no = 50
        else:
            self.basis_no = basis_no
        if patchdim is None:
            self.patchdim = 512
        else:
            self.patchdim = patchdim
        if savepath is None:
            self.savepath = os.getcwd() 
        else:
            self.savepath = savepath
        self.data = np.random.randn(self.patchdim**2,self.batch).astype('float32')
        self.basis = np.random.randn(self.patchdim**2,self.basis_no).astype('float32')
        self.data = theano.shared(self.data)
        self.basis = theano.shared(self.basis)
        self.residual = [] 
        #self.residual = theano.shared(self.residual)
        print('Compiling theano function')
        self.f = self.create_theano_fn()
        return 

    def create_theano_fn(self):
        coeff_flat = T.dvector('coeffs')
        coeff = T.cast(coeff_flat,'float32')
        coeff = T.reshape(coeff,[self.basis_no,self.batch])
        tmp = (self.data - self.basis.dot(coeff))**2
        tmp = 0.5*tmp.sum(axis=0)
        tmp = tmp.mean()
        sparsity = self.lam * T.abs_(coeff).sum()
        obj = tmp + sparsity
        grads = T.grad(obj,coeff_flat)
        f = theano.function([coeff_flat],[obj.astype('float64'),grads.astype('float64')])
        return f 

    def infer_coeff(self):
        #init_coeff = np.zeros(4000).astype('float32')
        init_coeff = np.zeros(self.basis_no*self.batch).astype('float32')
        #print(self.f(self.coeff))
        res = minimize(fun=self.f,x0=init_coeff,method='L-BFGS-B',jac=True)
        self.coeff = np.reshape(res.x,[self.basis_no,self.batch])
        return res.x 

    def load_new_batch(self,data):
        #Update self.data to have new data
        self.data.set_value(data.astype('float32'))
        return

    def update_basis(self):
        basis = self.basis.get_value()
        data = self.data.get_value()
        #Update basis with the right update steps
        Residual = data - basis.dot(self.coeff)
        dbasis = self.LR * Residual.dot(self.coeff.T)
        basis = basis + dbasis
        #Normalize basis
        #norm_basis = np.diag(1/np.sqrt(np.sum(self.basis**2,axis=0)))
        #self.basis = np.dot(self.basis,norm_basis)
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = np.sqrt(norm_basis)
        norm_basis = np.diag(1.0/norm_basis)
        basis = basis.dot(norm_basis)
        self.basis.set_value(basis.astype('float32'))
        #print('The norm of the basis is',np.linalg.norm(self.basis)) 
        #self.basis=np.asarray(self.basis)
        #Residual= np.mean(np.sqrt(np.sum(Residual**2,axis=0)))
        tmp = Residual**2
        tmp = tmp.sum(axis=0)
        Residual = tmp.mean()
        self.residual.append(Residual)
        return 
        #return

    def visualize_basis(self,iteration,image_shape=None):
        #Use function we wrote previously
        if image_shape is None:
            f, ax = plt.subplots(self.basis_no/10,10,sharex=True,sharey=True)
        else:
            f, ax = plt.subplots(image_shape[0],image_shape[1],sharex=True,sharey=True)
        for ii in np.arange(ax.shape[0]):
            for jj in np.arange(ax.shape[1]):
                tmp = self.basis[:,ii*ax.shape[1]+jj]
                im = tmp.reshape([self.patchdim,self.patchdim])
                im = im.eval()
                ax[ii,jj].imshow(im)
        savepath_image=self.savepath + '_iterations_' + str(iteration) + '_visualize_.png'
        f.savefig(savepath_image)
        f.close()
        return

