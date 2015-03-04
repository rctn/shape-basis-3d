'''
Load Shapes, Compute basis
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigs
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb

#Environment Variables
DATA = os.getenv('DATA')
proj_path = DATA + '3dFace/'
write_path = proj_path + 'shape_basis/PCA/'
path= proj_path + 'matfiles/*'



#list all files in the directory above
print('The path for glob search is', path)
matfiles=glob.glob(path)
print('Total number of mat files  is %d',len(matfiles))


shapes = np.zeros([len(matfiles),512*512])
#shapes = []
for ii in range(len(matfiles)):
    if np.mod(ii,10)==0:
        print('Files Loaded --- ',ii)
    matfile = scio.loadmat(matfiles[ii])
    geometry = matfile['vertices']
    actual_geometry = np.reshape(geometry[:,2],[512,512])
    try:
        #shapes.append(geometry)
        shapes[ii,:] = actual_geometry.flatten()
    except:
        pdb.set_trace()
print('Successfully loaded')
vertices = matfile['vertices']
lon=np.reshape(vertices[:,0],[512,512])
lat=np.reshape(vertices[:,1],[512,512])


#Compute Mean Face
mean_face = np.mean(shapes,axis=0)
#mean_plt = plt.figure()
#ax = Axes3D(mean_plt) 
#ax.imshow(mean_face.reshape([512,512]))
#ax.plot_surface(lon,lat,mean_face.reshape([512,512]),rstride=1,cstride=1,linewidth=0,antialiased=False,cmap=cm.Greys_r)
#ax.view_init(elev=0,azim=90)
#ax.set_axis_off()
#points3d(lon,lat,mean_face.reshape([512,512]))
#mean_plt.set_size_inches(5.12,5.12)
#mean_plt.savefig(write_path+'Mean_Face.png',bbox_inches='tight',pad_inches=0,dpi=100)


#Now let's try doing PCA
print('Compute Covariance Matrix')
shapes_subtr_mean_face = shapes - mean_face
shapes_cov = np.cov(shapes_subtr_mean_face)
print('Compute Eigen Vectors')
shape_vals,shape_vectors = eigs(shapes_cov,k=50,return_eigenvectors=True)

'''
eig_val_fig = plt.figure()
plt.plot(shape_vals)
plt.legend('eigen value energy')
eig_val_fig.savefig(write_path+'Eig_Value_Plot.png',bbox_inches='tight',pad_inches=0)
'''

#Now let's scale the Eigen Vectors back to original shape
try:
    shape_vectors_face = np.dot(shape_vectors.T, shapes_subtr_mean_face)
except:
    pdb.set_trace()

'''
for ii in range(50):
    eig_vec_fig = plt.figure()
    plt.imshow(np.real(shape_vectors_face[ii,:].reshape([512,512])))
    plt.colorbar()
    plt.title('Eigen Vector '+str(ii)+'Rendered as an image')
    eig_vec_fig.savefig(write_path+'Eigen_Vector_'+str(ii)+'_.png',bbox_inches='tight',pad_inches=0)
    plt.close()
'''
shape_basis = {
'mean_face': mean_face,
'shapes_subtr_mean_face':shapes_subtr_mean_face,
'shapes_cov':shapes_cov,
'shape_eig_vals':shape_vals,
'shape_eig_vectors':shape_vectors,
'shape_eig_vectors_face':shape_vectors_face,
'vertices':vertices,
}
scio.savemat(write_path+'basis.mat',shape_basis)
