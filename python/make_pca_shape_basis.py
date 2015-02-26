import numpy as np
import scipy.io as as scio
from matplotlib import pylot as plt
from sklearn.decomposition import PCA
import glob
import os

def face_PCA_basis(path):
    os.chdir(path)
    files=glob.glob("*")
    load_all_geometry=np.zeros(len(files),512**2)
    for ii in enumerate(files):
        if np.mod(ii,10)==0:
            print("Loaded 10 more files, ",ii)
        tmp=scio.loadmat(files[ii])
        load_all_geometry[ii,:]=tmp.flatten()

    #Compute the covariance matrix
    geometry_cov = np.cov(load_all_geometry)
    [shape_eig_vals,shape_eig_vec]=np.eigs(geometry_cov,k=30)
    
    return 1

