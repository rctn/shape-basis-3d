import numpy as np
from scipy.optimize import minimize
import theano 
from theano import tensor as T
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from math import radians


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
        self.data = np.random.randn(self.patchdim**2,200).astype('float32')
        self.X = np.random.randn(self.patchdim**2,1).astype('float32')
        self.Y = np.random.randn(self.patchdim**2,1).astype('float32')
        self.coeff = np.random.randn(self.basis_no,self.batch).astype('float32')
        self.proj_matrix = np.random.randn(3,3).astype('float32')
        self.basis = np.random.randn(self.patchdim**2,self.basis_no).astype('float32')
        self.rand_idx = np.random.randint(0,200)
        #Making them shared variable
        self.data = theano.shared(self.data,'self.data')
        self.basis = theano.shared(self.basis,'self.basis')
        self.X = theano.shared(self.X,'self.X')
        self.Y = theano.shared(self.Y,'self.Y')
        self.proj_matrix = theano.shared(self.proj_matrix,'self.proj_matrix')
        self.rand_idx = theano.shared(self.rand_idx,'self.rand_idx')
        self.coeff = theano.shared(self.coeff,'self.coeff')
        self.residual = [] 
        print('Compiling Projection matrix function')
        self.proj_data = self.create_proj_data_matrix()
        print('Compiling Projection matrix function')
        self.proj_basis = self.create_proj_basis_matrix()
        #print('Compiling inference theano function')
        #self.f = self.create_inference_fn()
        print('Compiling inference theano function')
        self.f_proj = self.create_inference_proj_fn()
        #print('Compiling update basis theano function')
        #self.update_basis = self.create_update_basis()
        print('Compiling update basis theano function')
        self.update_basis = self.create_update_proj_basis()
        return 

    def create_inference_fn(self):
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


    def create_inference_proj_fn(self):
        coeff_flat = T.dvector('coeffs')
        coeff = T.cast(coeff_flat,'float32')
        coeff = T.reshape(coeff,[self.basis_no,self.batch])
        idx = self.rand_idx.eval()
        idx = np.uint8(idx)
        var1 = self.proj_data(idx)
        var1 = np.asarray(var1)
        var1 = var1.T
        proj_basis = T.zeros((self.patchdim**2,self.basis_no))
        for ii in np.arange(self.basis_no):
            proj_basis_tmp = self.proj_basis(ii)
            proj_basis_tmp = np.asarray(proj_basis_tmp)
            proj_basis = T.set_subtensor(proj_basis[:,ii],proj_basis_tmp.flatten())
       
        tmp = (var1 - proj_basis.dot(coeff))**2
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
        res = minimize(fun=self.f_proj,x0=init_coeff,method='L-BFGS-B',jac=True,options={'disp':False})
        self.coeff = np.reshape(res.x,[self.basis_no,self.batch])
        print np.sum(np.abs(self.coeff))
        return res.x 

    def new_sample(self):
        #Update New Random Idx
        rand_idx = np.random.randint(0,200)
        self.rand_idx.set_value(rand_idx)
        return

    def new_proj_matrix(self):
        angle = 60*np.random.rand() -30
        new_matrix = np.zeros((3,3))
        new_matrix[0,0] = np.cos(np.deg2rad(angle))
        new_matrix[0,2] = np.sin(np.deg2rad(angle))
        new_matrix[1,1] = 1.0
        new_matrix[2,0] = -np.sin(np.deg2rad(angle))
        new_matrix[2,2] = np.cos(np.deg2rad(angle))
        self.proj_matrix = T.set_subtensor(self.proj_matrix[:,:],new_matrix)
        return

    def load_all_data(self,X,Y,data):
        self.X.set_value(X.astype('float32'))
        self.Y.set_value(Y.astype('float32'))
        self.data.set_value(data.astype('float32'))
        return

    def create_proj_data_matrix(self):
        #Input parameter is the data that we want to do projections on
        idx_64 = T.scalar('dtype_flag')
        idx = T.cast(idx_64,'uint8')
        input_T = self.data[:,idx]
        #We then Compute Transformation
        all_vertices = T.zeros((3,self.patchdim**2))
        all_vertices = T.set_subtensor(all_vertices[0,:],self.X[:,0])
        all_vertices = T.set_subtensor(all_vertices[1,:],self.Y[:,0])
        all_vertices = T.set_subtensor(all_vertices[2,:],input_T)
        transformed_input = self.proj_matrix.dot(all_vertices)
        X= transformed_input[0,:]
        Y= transformed_input[1,:]
        Z= transformed_input[2,:]
        #We then compute 3D to 2D Image (Pulkit's code goes here)
	#Add to Z so things are a little far away
        #Shift data so it scales well
	Z = Z - T.min(Z) + 1000.0
	#Get the perspective (x,y) coordinates
	x    = X/Z
	y    = Y/Z
	#Create the image bins location
	mnX, mxX = T.min(x), T.max(x)
	mnY, mxY = T.min(y), T.max(y)
	mn       = T.min((mnX, mnY))
	mx       = T.max((mxX, mxY))
	#Create the image
	im  = T.zeros((self.patchdim, self.patchdim))
        new_x = (self.patchdim -1.0)*(x - mn)/(mx-mn)
        new_y = (self.patchdim -1.0)*(y - mn)/(mx-mn)
        im=T.set_subtensor(im[new_y.astype('uint8'),new_x.astype('uint8')], Z)
        #We return projected input
        f = theano.function([idx_64],[im.flatten().astype('float32')])
        #Create theano function
        return f


    def create_proj_basis_matrix(self):
        #Input parameter is the data that we want to do projections on
        idx_64 = T.scalar('dtype_flag')
        idx = T.cast(idx_64,'uint8')
        #check what type of data it is -- data or basis?
        input_T = self.basis[:,idx]
        #We then Compute Transformation
        all_vertices = T.stack([self.X[:,0],self.Y[:,0],input_T])
        transformed_input = self.proj_matrix.dot(all_vertices)
        X= transformed_input[0,:]
        Y= transformed_input[1,:]
        Z= transformed_input[2,:]
        #We then compute 3D to 2D Image (Pulkit's code goes here)
	#Add to Z so things are a little far away
        #Shift data so it scales well
	Z = Z - T.min(Z) + 1000.0
	#Get the perspective (x,y) coordinates
	x    = X/Z
	y    = Y/Z
	#Create the image bins location
	mnX, mxX = T.min(x), T.max(x)
	mnY, mxY = T.min(y), T.max(y)
	mn       = T.min((mnX, mnY))
	mx       = T.max((mxX, mxY))
	#Create the image
	im  = T.zeros((self.patchdim, self.patchdim))
        new_x = (self.patchdim -1.0)*(x - mn)/(mx-mn)
        new_y = (self.patchdim -1.0)*(y - mn)/(mx-mn)
        im=T.set_subtensor(im[new_y.astype('uint8'),new_x.astype('uint8')], Z)
        #We return projected input
        f = theano.function([idx_64],[im.flatten().astype('float32')])
        #Create theano function
        return f

    def create_update_basis(self):
        coeff_f64 = T.dmatrix('coeff')
        coeff = T.cast(coeff_f64,'float32')
        #Update basis with the right update steps
        Residual = (self.data) - self.basis.dot(coeff)
        dbasis = self.LR * Residual.dot(coeff.T)
        basis = self.basis + dbasis
        #Normalize basis
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = T.sqrt(norm_basis)
        norm_basis = T.nlinalg.diag(1.0/norm_basis)
        basis = basis.dot(norm_basis)
        updates = {self.basis: basis}
        tmp = Residual**2
        tmp = 0.5*tmp.sum(axis=0)
        Residual = tmp.mean()
        num_on = (T.abs_(coeff)).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([coeff_f64],[Residual.astype('float32'),num_on], updates=updates)
        return f 


    def create_update_proj_basis(self):
        #coeff_f64 = T.dmatrix('coeff')
        #coeff = T.cast(coeff_f64,'float32')
        #Update basis with the right update steps
        idx = self.rand_idx.eval()
        idx = np.uint8(idx)
        var1 = self.proj_data(idx)
        var1 = np.asarray(var1)
        var1 = var1.T
        proj_basis = T.zeros((self.patchdim**2,self.basis_no))
        for ii in np.arange(self.basis_no):
            proj_basis_tmp = self.proj_basis(ii)
            proj_basis_tmp = np.asarray(proj_basis_tmp)
            proj_basis = T.set_subtensor(proj_basis[:,ii],proj_basis_tmp.flatten())

        Residual = var1 - proj_basis.dot(self.coeff)
        tmp = Residual**2
        tmp = 0.5*tmp.sum(axis=0)
        tmp = tmp.mean()
        sparsity = self.lam * T.abs_(self.coeff).sum(axis=0).mean()
        #Compute gradient here
        #dbasis = T.grad(obj,self.basis)
        dbasis = self.LR*Residual.dot(self.coeff.T)
        basis = self.basis + dbasis
        #Normalize basis
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = T.sqrt(norm_basis)
        norm_basis = T.nlinalg.diag(1.0/norm_basis)
        basis = basis.dot(norm_basis)
        updates = {self.basis: basis}
        Residual = tmp.mean()
        num_on = (T.abs_(self.coeff)).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([],[Residual.astype('float32'),num_on], updates=updates)
        return f 

    def visualize_basis(self,iteration,image_shape=None):
        #Use function we wrote previously
        if image_shape is None:
            f, ax = plt.subplots(self.basis_no/10,10,sharex=True,sharey=True)
        else:
            f, ax = plt.subplots(image_shape[0],image_shape[1],sharex=True,sharey=True)
        for ii in np.arange(ax.shape[0]):
            for jj in np.arange(ax.shape[1]):
                tmp = self.basis.get_value()
                tmp = tmp[:,ii*ax.shape[1]+jj]
                im = tmp.reshape([self.patchdim,self.patchdim])
                im = im
                ax[ii,jj].imshow(im)
        savepath_image=self.savepath + '_iterations_' + str(iteration) + '_visualize_.png'
        f.savefig(savepath_image)
        f.clf()
        plt.close()
        return

