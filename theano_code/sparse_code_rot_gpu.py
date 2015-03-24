import numpy as np
from scipy.optimize import minimize
import theano 
from theano import tensor as T
import pdb
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
        self.data = np.random.randn(self.patchdim**2,self.batch).astype('float32')
        self.X = np.random.randn(self.patchdim**2,1).astype('float32')
        self.Y = np.random.randn(self.patchdim**2,1).astype('float32')
        self.proj_matrix = np.random.randn(3,3)
        self.basis = np.random.randn(self.patchdim**2,self.basis_no).astype('float32')
        self.data = theano.shared(self.data)
        self.basis = theano.shared(self.basis)
        self.X = theano.shared(self.X)
        self.Y = theano.shared(self.Y)
        self.proj_matrix = theano.shared(self.proj_matrix)
        self.residual = [] 
        #self.residual = theano.shared(self.residual)
        print('Compiling inference theano function')
        self.f = self.create_theano_fn()
        print('Compiling update basis theano function')
        self.update_basis = self.create_update_basis()
        print('Compiling Projection matrix function')
        self.proj_data = self.create_proj_matrix()
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
        self.coeff = np.reshape(res.x,[self.basis_no,self.batch])
        print np.sum(np.abs(self.coeff))
        return res.x 

    def load_new_batch(self,data,X,Y):
        #Update self.data to have new data
        self.data.set_value(data.astype('float32'))
        self.X.set_value(X.astype('float32'))
        self.Y.set_value(Y.astype('float32'))
        return

    def set_proj_matrix(self,angle):
        self.proj_matrix[0,0].set_value(np.cos(np.deg2rad(angle)))
        self.proj_matrix[0,2].set_value(np.sin(np.deg2rad(angle)))
        self.proj_matrix[1,1].set_value(1.0)
        self.proj_matrix[2,0].set_value(-np.sin(np.deg2rad(angle)))
        self.proj_matrix[2,2].set_value(np.cos(np.deg2rad(angle)))

        return

    def create_proj_matrix(self):
        #Input parameter is the angle and data that we want to do projections on
        #angle_f64 = T.scalar('angle_f64')
        #angle = T.cast(angle_f64,'uint8')
        input_f64  = T.dmatrix('input')
        input_for_transform = T.cast(input_f64,'float32')

        #We then Compute Transformation
        all_vertices = [self.X,self.Y,input_for_transform]
        transformed_input = self.proj_matrix.dot(all_vertices)
        X= transformed_input[0,:]
        Y= transformed_input[1,:]
        Z= transformed_input[2,:]

        #We then compute 3D to 2D Image (Pulkit's code goes here)
	#Add to Z so things are a little far away
        #Shift data so it scales well
	Z      = Z - T.min(Z) + 1000.0

	#Get the perspective (x,y) coordinates
	x    = X/Z
	y    = Y/Z
	#Create the image bins location
	mnX, mxX = T.min(x), T.max(x)
	mnY, mxY = T.min(y), T.max(y)
	mn       = T.min((mnX, mnY))
	mx       = T.max((mxX, mxY))

        #Write my own binning code because fucking Theano can't do this right
	#bins     = np.linspace(mn, mx, self.patchdim)
        #bins     = bins.flatten()
        bins= T.arange(mn,mx,(mx-mn)/self.patchdim)

	#Create the image
	im  = np.zeros((self.patchdim, self.patchdim))
	xFlat, yFlat, ZFlat = x.flatten(), y.flatten(), Z.flatten()
	count = 0
	for (xv, yv) in zip(xFlat, yFlat):
		bx = find_bins(xv, bins)
		by = find_bins(yv, bins)
		im[by, bx] = ZFlat[count]
		count      += 1

        #We return projected input
        f = theano.function([angle_f64,input_f64],[im.astype('float32')],updates=updates)
        #Create theano function

        return f

    def create_update_basis(self):
        coeff_f64 = T.dmatrix('coeff')
        coeff = T.cast(coeff_f64,'float32')
        #Update basis with the right update steps
        Residual = self.data - self.basis.dot(coeff)
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
        num_on = (T.abs_(coeff)>0).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([coeff_f64],[Residual.astype('float32'),num_on], updates=updates)
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
        return

