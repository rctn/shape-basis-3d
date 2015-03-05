'''
This function remaps the image to a new axes
'''

import numpy as np
import pdb
import matplotlib.pyplot as plt

def proj_3d_2d(X,Y,Z):
	#flatten arrays, reshape them later if needede
	new_im = np.zeros(Z.shape)
	Z = Z + np.abs(np.min(Z)) + 1.0
	X = X+ np.abs(np.min(X))
	Y = Y+ np.abs(np.min(Y))
	new_x = X/Z
	new_y = Y/Z
	print('The min and max values of new_x and new_y are')
	print np.min(new_x),np.min(new_y),np.max(new_x),np.max(new_y)
	max_scaling = np.max([np.max(new_x),np.max(new_y)])
	print('The Max Scaling value is -- ', max_scaling)
	#new_x = new_x *(511.0/max_scaling)
	#new_y = new_y *(511.0/max_scaling)
	
	[freq_x,bin_x]=np.histogram(new_x,510)
	idx_bin_x = np.digitize(new_x.flatten(),bin_x)
	idx_bin_x = np.reshape(idx_bin_x,[512,512])
	[freq_y,bin_y]=np.histogram(new_y,510)
	idx_bin_y = np.digitize(new_y.flatten(),bin_y)
	idx_bin_y = np.reshape(idx_bin_y,[512,512])
	pdb.set_trace()
	for ii in np.arange(Z.shape[0]):
		for jj in np.arange(Z.shape[0]):
			#new_im[np.int(np.round(new_x[ii,jj])),np.int(np.round(new_y[ii,jj]))]=Z[ii,jj]
				#if freq_x[ii]!=0 and freq_y[jj]!=0:
				#try:
						new_im[idx_bin_x[ii,jj],idx_bin_y[ii,jj]] = Z[ii,jj]
				#except:
				#pdb.set_trace()
			
	#Okay, now we iterate through our indices and create a new image
	return new_im
