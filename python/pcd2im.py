import numpy as np 
import scipy.io as sio
import pdb

def load_data(fName):
	dat   = sio.loadmat(fName)
	X,Y,Z = dat['X'], dat['Y'], dat['Z']
	return X,Y,Z


def find_bins(x, bins):
	return np.where(bins >= x)[0][0]

def make_im(X, Y, Z, imSz=256):
	#Add to Z so things are a little far away
	ignoreIdx = (Z<=5.0).flatten()
	minZ   = np.min(Z)
	Z      = Z - np.min(Z) + 2.0
	nr, nc = Z.shape
	print np.min(Z), np.max(Z)

	#Get the perspective (x,y) coordinates
	x    = X/Z
	y    = Y/Z
	
	#Find the image bins
	mnX, mxX = np.min(x), np.max(x)
	mnY, mxY = np.min(y), np.max(y)
	mn       = np.min((mnX, mnY))
	mx       = np.max((mxX, mxY))
	bins     = np.linspace(mn, mx, imSz)
	bins     = bins.flatten()

	#Fast and slightly hacky version to create the image
	#scale    = 1000
	#mnBin    = floor(scale * bins[0])
	#bnSz     = floor(scale * (bins[1] - bins[0]))

	#Create the image
	im  = np.zeros((imSz, imSz))
	
	xFlat, yFlat, ZFlat = x.flatten(), y.flatten(), Z.flatten()
	count = 0
	for (xv, yv) in zip(xFlat, yFlat):
		bx = find_bins(xv, bins)
		by = find_bins(yv, bins)
		if not ignoreIdx[count]:
			im[by, bx] = ZFlat[count]
		count      += 1
	
	return im
	
def run_main(imSz=256):
	fName = 'xyz.mat'
	X,Y,Z = load_data(fName)
	im    = make_im(X, Y, Z, imSz=imSz)
	return im
