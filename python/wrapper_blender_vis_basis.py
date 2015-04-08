#Wrapper script that you can just call for visualizing basis
import vis_basis
import glob
import scipy.io as scio
import pdb

FLAG = 0 
LR = 0.15
batch = 185
basis_no = 50
lam = 0.01

fname = 'LR_'+str(LR) + '_batch_' + str(batch) + '_basis_no_' + str(basis_no) + '_lam_' + str(lam) + '_basis'

version = 0.001 

if FLAG==0:
    files = glob.glob('/media/mudigonda/Gondor/Data/3dFace/shape_basis/PCA/split_mat/*')
    savepath = '/media/mudigonda/Gondor/Data/3dFace/shape_basis/PCA/'

    BL = vis_basis.blenderLoad()

    for ii in range(len(files)):
        print('Visualizing Eigen Head -------'+ str(ii))
        #BL.load_face(mean_vertices,files[ii],savepath+'Eig_Face_'+str(ii)+'.png')
        BL.load_face(files[ii],savepath+files[ii].split('/')[-1]+'.png')

    print('All Eig Faces Visualized')

else:

    files = glob.glob('/media/mudigonda/Gondor/Data/3dFace/shape_basis/sparse/' + fname + '/split_mat/*')
    savepath = '/media/mudigonda/Gondor/Data/3dFace/shape_basis/sparse/'

    BL = vis_basis.blenderLoad()

    for ii in range(len(files)):
        print('Visualizing Sparse Head -------'+ str(ii))
        BL.load_face(files[ii],savepath+'Sparse_Face_'+str(ii)+'.png')

