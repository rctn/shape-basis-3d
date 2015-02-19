#creating lists of the file name
import string as str
import os
import numpy as np
import h5py
from scipy import ndimage
import shutil
import glob
import pdb
from subprocess import call
import random

def make_fname_lists(path='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/'):
    files = os.listdir(path)
    identity = []
    x_rot = []
    y_rot = []
    z_rot = []
    counter = 0
    for fname in files:
        if np.mod(counter,100)==0:
            print('The value of counter is %d',counter)

        split_filename = str.split(fname,'_')
        try:
            identity.append(split_filename[0])
            x_rot.append(int(split_filename[3]))
            y_rot.append(int(split_filename[6]))
            z_rot.append(int(str.split(split_filename[9],'.')[0]))
            counter = counter + 1
        except:
            print(split_filename)
            print("Something went wrong")
    print('Creating lists done')
    return files,identity,x_rot,y_rot,z_rot 

#Now write out the HDF5
def write_hdf5(outputfname,inputpath='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/'):
    files = os.listdir(inputpath)
    no_of_files = len(files)
    #File handle
    try:
        hfl = h5py.File(outputfname,"w")
        dset = hfl.create_dataset("Faces",(no_of_files,196608),'i')
        counter = 0
        for fname in files:
            if np.mod(counter,100)==0:
                print('The value of the counter is %d',counter)
            im = ndimage.imread(inputpath+fname)
            im_wo_alpha = im[:,:,0:3]
            dset[counter,] = im_wo_alpha.flatten()
            counter = counter + 1
    except:
        print('Something went wrong')
    hfl.flush()
    hfl.close()

    return 1

def write_train_hdf5(outputfnames=None,inputpath=None,no_of_files=100):

    #First we create three file handles
    #Let's assume there are training folders and test folders
    if inputpath is None:
        inputpath='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/train/'
    if outputfnames is None:
        outputfnames ='/media/mudigonda/Gondor/Data/3dFace/train.hdf5'

    [files,face_id,xx,yy,zz] = make_fname_lists(inputpath)
    print('Trying to create training file handles')
    try:
        train = h5py.File(outputfnames,"w")
        train_inp_dset = train.create_dataset("Input",(no_of_files,120000))
        train_op_dset = train.create_dataset("Output",(no_of_files,120000))
        train_transf_dset = train.create_dataset("Transformation",(no_of_files,3))
    except:
        print("It looks like I couldn't create the file handles for the training set")
        train.flush()
        train.close()

    #Then we iterate through the files list that we get tcalling making list
    print('Okay, here goes -- trying to write stuff into files')
    #for ii in np.arange(len(face_id)):
    for ii in np.arange(100):
        if np.mod(ii,1000)==0:
            print('Written images = ',ii)

        #Read File
        try:
            im_ip = ndimage.imread(inputpath+files[ii])
            im_ip_wo_alpha = im_ip[:,:,0:3]
        except:
            print("Unable to read training input image")
        try:
            #For face chosen find all other entries of faces
            faces_idx = [i for i,x in enumerate(face_id) if x==face_id[ii]]
            #Generate random number
        except:
            print("Unable to find similar Faces. Fail",face_id[ii])
        try:
            rand_int = random.randint(0,len(faces_idx))
        except:
            print("Failed to randint. Values of rand_int and faces_idx[rand_int] are",rand_int,faces_idx[rand_int]) 
        try:
            im_op = ndimage.imread(inputpath+files[faces_idx[rand_int]])
            im_op_wo_alpha = im_op[:,:,0:3]
        except:
            print("Unable to load file, this is the file path",inputpath+files[faces_idx[rand_int]])
        try:
            #The transformation between the two is just Pose2-Pose1
            xx_diff = xx[faces_idx[rand_int]]-xx[ii]
            yy_diff = yy[faces_idx[rand_int]]-yy[ii]
            zz_diff = zz[faces_idx[rand_int]]-zz[ii]
        except:
            print("Unable to transform input to output because of bad indexing")
        try:
            train_inp_dset[ii,] = im_ip_wo_alpha.flatten()
            train_op_dset[ii,] = im_op_wo_alpha.flatten()
        except:
            print("Unable to Write out the inputs to the file handle")
        try:
            train_transf_dset[ii,] = np.array([xx_diff,yy_diff,zz_diff]).flatten()
        except:
            print("Unable to write this tuple of transformations into the dataset")
    train.flush()
    train.close()
    return 1


def move_train_test(src=None,ignore=None):
    if src is None:
        src='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/'
    if ignore is None:
        ignore=['MACF312.mat','test','train','bad']
    #Find the list of files from src
    files,face_id,xx,yy,zz = make_fname_lists(src)

    #Find unique Face ID
    unique_items=[]
    for item in face_id:
        if item not in unique_items:
           unique_items.append(item)
    
    #Take 75% of them and put them in train, verify ignore
    for ii in range(150):
        if unique_items[ii] not in ignore:
            print('Moving File',unique_items[ii])
            mv_files = glob.glob(src+unique_items[ii]+'*')
            for file_to_move in mv_files:
                shutil.move(file_to_move,src+'/train')
    #Take the remaining and put them in test, verify ignore
    for ii in range(150,len(unique_items)): 
        if unique_items[ii] not in ignore:
            print('Moving File',unique_items[ii])
            mv_files = glob.glob(src+unique_items[ii]+'*')
            for file_to_move in mv_files:
                shutil.move(file_to_move,src+'/test')


    return 1


def crop_images(path):
    os.chdir(path)
    files_to_crop=glob.glob("*")
    for ii in range(len(files_to_crop)):
        if np.mod(ii,1000)==0:
            print('The value of the counter is ',ii)
        try:
            call('convert -crop 200X200+25+0 '+files_to_crop[ii]+' '+files_to_crop[ii],shell=True)
        except:
            print('could not convert')
    os.chdir('/media/mudigonda/Gondor/Projects/shape-basis-3d/python')
    return 1
