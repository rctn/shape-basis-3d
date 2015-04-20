import numpy as np
from scipy.optimize import minimize
import theano 
from theano import tensor as T
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
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
        self.data = np.random.randn(self.patchdim[0]*self.patchdim[1],self.batch).astype('float32')
        self.coeff = np.random.randn(self.basis_no,self.batch).astype('float32') 
        self.basis = np.random.randn(self.patchdim[0]*self.patchdim[1],self.basis_no).astype('float32')
        self.coeff = theano.shared(self.coeff)
        self.data = theano.shared(self.data)
        self.basis = theano.shared(self.basis)
        self.residual = [] 
        self.infer_iter = 100
        print('Compiling theano function')
        self.f = self.create_theano_fn()
        self.infer_coeff_gd = self.create_infer_coeff_gd()
        self.update_basis = self.create_update_basis()
        return 

    def create_theano_fn(self):
        coeff_flat = T.dvector('coeffs')
        coeff = T.cast(coeff_flat,'float32')
        coeff = T.reshape(coeff,[self.basis_no,self.batch])
        tmp = (self.data - self.basis.dot(coeff))**2
        tmp = 0.5*tmp.sum(axis=0)
        tmp = tmp.mean()
        sparsity = self.lam * T.abs_(coeff).sum(axis=0).mean()
        obj = tmp + sparsity
        grads = T.grad(obj,coeff_flat)
        f = theano.function([coeff_flat],[obj.astype('float64'),grads.astype('float64')])
        return f 

    def infer_coeff(self):
        #init_coeff = np.zeros(4000).astype('float32')
        init_coeff = np.zeros(self.basis_no*self.batch).astype('float32')
        #print(self.f(self.coeff))
        res = minimize(fun=self.f,x0=init_coeff,method='L-BFGS-B',jac=True,options={'disp':False})
        self.coeff.set_value(np.reshape(res.x,[self.basis_no,self.batch]).astype('float32'))
        print('Value of objective fun after doing inference',res.fun)
        active = len(res.x[np.abs(res.x)>1e-2])/float(self.basis_no*self.batch)
        #active = (np.abs(res.x).sum())/float(self.basis_no*self.batch)
        print('Number of Active coefficients is ...',active)
        return active,res.fun 

    def create_infer_coeff_gd(self):
        r_err = (self.data - self.basis.dot(self.coeff))**2
        r_err = 0.5*r_err.sum(axis=0)
        r_err = r_err.mean()
        sparsity = self.lam * T.abs_(self.coeff).sum(axis=0).mean()
        obj = r_err + sparsity
        grads = T.grad(obj, self.coeff)
        coeff_update = self.coeff - self.LR*1e-3*grads
        updates = {self.coeff:coeff_update.astype('float32')}
        active = T.abs_(self.coeff).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([],[obj.astype('float64'),active.astype('float64')],updates=updates)
        return f


    def load_data(self,data):
        print('The norm of the data is ', np.mean(np.linalg.norm(data,axis=0)))
        #Update self.data to have new data
        self.data.set_value(data.astype('float32'))
        return
   
    def adjust_LR(self,LR):
        self.LR = LR
        return
        

    def create_update_basis(self):
        #basis_flat = T.dvector('basis')
        #basis = T.cast(basis_flat,'float32')
        #basis = T.reshape(basis,[self.patchdim[0]*self.patchdim[1],self.basis_no])
        #Update basis with the right update steps
        Residual = self.data - self.basis.dot(self.coeff)
        dbasis = self.LR * Residual.dot(self.coeff.T)
        basis = self.basis + dbasis
        #Normalize basis
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = T.sqrt(norm_basis)
        norm_basis = T.nlinalg.diag(1.0/norm_basis)
        basis = basis.dot(norm_basis)
        updates = {self.basis: basis.astype('float32')}
        tmp = Residual**2
        tmp = 0.5*tmp.sum(axis=0)
        Residual = tmp.mean()
        #grads = T.grad(Residual,basis_flat)
        num_on = T.abs_(self.coeff).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([],[Residual.astype('float32'),num_on,basis], updates=updates)
        #f = theano.function([basis_flat],[Residual.astype('float64'),grads.astype('float64')] )
        return f 

    def update_basis_wrapper(self):
        init_basis = np.random.randn(self.patchdim[0]*self.patchdim[1]*self.basis_no)
        res = minimize(fun=self.update_basis,x0=init_basis,method='L-BFGS-B',jac=True,options={'disp':False})
        #Normalize stuff here
        basis = np.reshape(res.x,[self.patchdim[0]*self.patchdim[1],self.basis_no]) 
        norm_basis = np.diag(1.0/np.sqrt(np.sum(basis**2,axis=0)))
        basis = np.dot(basis,norm_basis)
        self.basis.set_value(np.reshape(res.x,[self.patchdim[0]*self.patchdim[1],self.basis_no]).astype('float32'))
        print('value of Objective function after updating basis is ',res.fun)
        return res.fun 

    def visualize_basis(self,iteration,image_shape=None):
        #Use function we wrote previously
        if image_shape is None:
            f, ax = plt.subplots(16,16,sharex=True,sharey=True)
        else:
            f, ax = plt.subplots(image_shape[0],image_shape[1],sharex=True,sharey=True)
        for ii in np.arange(ax.shape[0]):
            for jj in np.arange(ax.shape[1]):
                tmp = self.basis.get_value()
                tmp = tmp[:,ii*ax.shape[1]+jj]
                im = tmp.reshape([self.patchdim[0],self.patchdim[1]])
                ax[ii,jj].imshow(im,interpolation='nearest',aspect='equal')
                ax[ii,jj].axis('off')
        savepath_image= '_iterations_' + str(iteration) + '_visualize_.png'
        f.savefig(savepath_image)
        f.clf()
        plt.close()
        return

