import numpy as np 
import scipy.io as sio
import pdb

def load_data(fName):
	dat   = sio.loadmat(fName)
	X,Y,Z = dat['X'], dat['Y'], dat['Z']
	return X,Y,Z


def load_vertices_texture(fName):
	dat = sio.loadmat(fName)
	print('Successfully loaded file ',fName)
	vertices = dat['vertices']
	texture  = dat['texture']
	return vertices, texture


def find_bins(x, bins):
	return np.where(bins >= x)[0][0]


def make_im(X, Y, Z, imSz=256):
	'''
		Renders the image when X,Y,Z coordinates are provided
	'''
	#Add to Z so things are a little far away
	#ignoreIdx = (Z<=5.0).flatten()
	minZ   = np.min(Z)
	Z      = Z - np.min(Z) + 1000.0
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
		im[by, bx] = ZFlat[count]
		count      += 1
	
	return im

def make_im_noloop(X,Y,Z, binSz =256):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    Z = Z - Z.min() + 1300.0
    x = X/Z
    y = Y/Z

    mnX, mxX =  x.min(), x.max()
    mnY, mxY =  y.min(), y.max()
    mn = np.min((mnX, mnY))
    mx = np.max((mxX, mxY))

    im = np.zeros((binSz*binSz))
    new_x = (binSz-1)*(x - mn)/(mx-mn)
    new_y = (binSz-1)*(y - mn)/(mx-mn)
    '''
    for ii in np.arange(x.shape[0]):
        im[new_x[ii],new_y[ii]] = Z[ii]
    '''
    lin_idx = new_x*binSz + new_y
    pdb.set_trace()
    im[lin_idx.astype('uint8')] = Z

    return im, new_x,new_y

    


def render_im(vertices, texture, imSz=256):
	'''
		verts  : N x 3 where 3 is X,Y,Z
		texture: N x 3, where 3 is R,G,B
		The values where Z = 0, the data will assumed to missing
	'''
	assert vertices.shape[0]==texture.shape[0], "Improper number of datapoints"
	X,Y,Z   = vertices[:,0],vertices[:,1],vertices[:,2]

	'''	
	nZIdx  = ~(Z==0)
	Z      = Z[nZIdx,:]
	X      = X[nZIdx,:]
	Y      = Y[nZIdx,:]
	texture  = texture[nZIdx,:]
	'''

	minZ    = np.min(Z)
	Z       = Z - minZ + 500.0

	#Project to image Coordinates
	x       = X/Z
	y       = Y/Z
	
	#Find the image bins
	mnX, mxX = np.min(x), np.max(x)
	mnY, mxY = np.min(y), np.max(y)
	mn       = np.min((mnX, mnY))
	mx       = np.max((mxX, mxY))
	print('Min and Max value of vertices are -- ', mn, mx)
	bins     = np.linspace(mn, mx, imSz)
	bins     = bins.flatten()

	#Mantain a Z-buffer
	zBuf = np.max(Z) * np.ones((imSz, imSz))
	im   = np.zeros((imSz, imSz,3))
	xFlat, yFlat, ZFlat = x.flatten(), y.flatten(), Z.flatten()
	count = 0
	for (xv, yv, Zv) in zip(xFlat, yFlat, ZFlat):
		bx = find_bins(xv, bins)
		by = find_bins(yv, bins)
		#If the Z coordinate to the point is closer then update it. 
		if Zv < zBuf[by, bx]:
			#zBuf[by,bx]   = Zv 
			im[by, bx, 0] = texture[count,0]
			im[by, bx, 1] = texture[count,1]
			im[by, bx, 2] = texture[count,2]
		count      += 1
	
	return im


	
def run_main(fName,imSz=256,flag=0):
	'''
	fName = 'xyz.mat'
	X,Y,Z = load_data(fName)
	im    = make_im(X, Y, Z, imSz=imSz)
	'''
	#fName = 'ADCM370.mat'
	verts, texture = load_vertices_texture(fName)
	if flag == 0:
		im = render_im(verts, texture)
	else:
		X = verts[:,0].reshape([512,512])
		Y = verts[:,1].reshape([512,512])
		Z = verts[:,2].reshape([512,512])
		im = make_im(X,Y,Z,imSz)
	return im
